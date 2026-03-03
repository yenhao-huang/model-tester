#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request
import traceback

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:3172/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
MODEL = os.getenv("MODEL_NAME", "glm-4.7-flash")
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/Users/yenhaohuang/Desktop/datasets/fast-textgen-evalset"))
MAX_CASES = int(os.getenv("MAX_CASES", "20"))
MAX_TOKENS = 4096
OUT_DIR = Path(os.getenv("OUT_DIR", str(Path(__file__).resolve().parent / "reports")))


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chat(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise evaluator assistant. Follow output format strictly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    req = request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if content:
            return content
        return (msg.get("reasoning_content") or "").strip()


def extract_last_number(text: str) -> str | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _dedent4(code: str) -> str:
    """Strip one level (4 spaces) of leading indentation from each line."""
    return "\n".join(line[4:] if line.startswith("    ") else line for line in code.split("\n"))


def _normalize_indent(code: str) -> str:
    """Best-effort indentation repair for model-generated Python code."""
    # normalize tabs first, then remove a common extra 4-space nesting if present
    code = code.replace("\t", "    ")
    return _dedent4(code)


def _run_humaneval_case(prompt: str, entry_point: str, test_code: str, response: str) -> tuple[bool, str | None, bool]:
    """Returns (passed, error_msg, syntax_repaired)."""
    code = _strip_code_fence(response)
    if not code:
        return False, "empty response", False

    syntax_repaired = False

    def _build_candidate(src: str) -> str:
        if f"def {entry_point}(" in src:
            return src
        return prompt + src + "\n"

    candidate_src = _build_candidate(code)
    try:
        compile(candidate_src, "<string>", "exec")
    except IndentationError:
        repaired = _normalize_indent(code)
        if repaired != code:
            syntax_repaired = True
        candidate_src = _build_candidate(repaired)

    program = candidate_src + "\n" + test_code + f"\ncheck({entry_point})\n"
    g = {}
    try:
        exec(program, g, g)
        return True, None, syntax_repaired
    except IndentationError:
        # second-chance repair for cases that only fail when tests are appended
        repaired = _normalize_indent(code)
        candidate_src = _build_candidate(repaired)
        program = candidate_src + "\n" + test_code + f"\ncheck({entry_point})\n"
        g = {}
        try:
            exec(program, g, g)
            return True, None, True
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", True
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", syntax_repaired


def score_mmlu(n_cases: int):
    rows = read_jsonl(DATASET_ROOT / "mmlu" / "test.jsonl")[:n_cases]
    correct, results = 0, []
    for i, row in enumerate(rows, 1):
        choices = row["choices"]
        gold = ["A", "B", "C", "D"][int(row["answer"])]
        prompt = (
            f"Question:\n{row['question']}\n\n"
            f"Choices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\n"
            "Answer with only one letter: A, B, C, or D."
        )
        out = chat(prompt)
        m = re.search(r"[ABCD]", out.upper())
        pred = m.group(0) if m else ""
        ok = pred == gold
        correct += int(ok)
        results.append({"idx": i, "query": prompt, "gold": gold, "pred": pred, "passed": ok, "skipped": False, "response": out})
    return {"benchmark": "mmlu", "total": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None, "results": results}


def score_gsm8k(n_cases: int):
    rows = read_jsonl(DATASET_ROOT / "gsm8k-main" / "test.jsonl")[:n_cases]
    correct, results = 0, []
    for i, row in enumerate(rows, 1):
        gold = extract_last_number(row["answer"]) or ""
        prompt = f"Solve this and end with #### <number>.\n\nQuestion:\n{row['question']}"
        out = chat(prompt)
        pred = extract_last_number(out) or ""
        ok = pred == gold
        correct += int(ok)
        results.append({"idx": i, "query": prompt, "gold": gold, "pred": pred, "passed": ok, "skipped": False, "response": out})
    return {"benchmark": "gsm8k", "total": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None, "results": results}


def score_truthfulqa(n_cases: int):
    rows = read_jsonl(DATASET_ROOT / "truthfulqa" / "validation.jsonl")[:n_cases]
    correct, results = 0, []
    for i, row in enumerate(rows, 1):
        prompt = f"Answer briefly and factually:\n{row['question']}"
        out = chat(prompt)
        low = out.lower()
        cands = [x.lower() for x in row.get("correct_answers", [])] + [str(row.get("best_answer", "")).lower()]
        ok = any(c and c in low for c in cands)
        correct += int(ok)
        results.append({"idx": i, "query": prompt, "passed": ok, "skipped": False, "response": out})
    return {"benchmark": "truthfulqa", "total": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None, "results": results}


def score_humaneval(n_cases: int):
    rows = read_jsonl(DATASET_ROOT / "humaneval" / "test.jsonl")[:n_cases]
    correct, results = 0, []
    for i, row in enumerate(rows, 1):
        query = (
            "Complete the following Python function. Return only Python code, no markdown. "
            "You may return either the full function definition or only the function body.\n\n"
            f"{row['prompt']}"
        )
        out = chat(query)
        ok, err, repaired = _run_humaneval_case(row["prompt"], row["entry_point"], row["test"], out)
        correct += int(ok)
        results.append({
            "idx": i,
            "task_id": row.get("task_id"),
            "query": query,
            "entry_point": row.get("entry_point"),
            "passed": ok,
            "skipped": False,
            "response": out,
            "syntax_repaired": repaired,
            "error": err,
        })
    return {"benchmark": "humaneval", "total": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None, "results": results}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "base_url": BASE_URL,
            "dataset_root": str(DATASET_ROOT),
            "max_cases_per_benchmark": MAX_CASES,
            "benchmarks": ["mmlu", "gsm8k", "truthfulqa", "humaneval"],
            "max_tokens": MAX_TOKENS,
        },
        "mmlu": score_mmlu(MAX_CASES),
        "gsm8k": score_gsm8k(MAX_CASES),
        "truthfulqa": score_truthfulqa(MAX_CASES),
        "humaneval": score_humaneval(MAX_CASES),
    }
    report["meta"]["elapsed_sec"] = round(time.time() - t0, 2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"fast_textgen_eval_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"mmlu", "gsm8k", "truthfulqa", "humaneval"}}, ensure_ascii=False, indent=2))
    print("mmlu_accuracy", report["mmlu"]["accuracy"])
    print("gsm8k_accuracy", report["gsm8k"]["accuracy"])
    print("truthfulqa_accuracy", report["truthfulqa"]["accuracy"])
    print("humaneval_accuracy", report["humaneval"]["accuracy"])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
