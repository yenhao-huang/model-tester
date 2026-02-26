#!/usr/bin/env python3
"""Universal fast-textgen eval runner.

Benchmarks: mmlu, gsm8k, humaneval
Backend  : any OpenAI-compatible /v1/chat/completions endpoint

Usage examples
--------------
# local llama-server / ollama
python utils/eval_fast_textgen_eval.py --model qwen3.5

# specific benchmarks only
python utils/eval_fast_textgen_eval.py --model qwen3.5 --benchmarks mmlu gsm8k

# remote endpoint
python utils/eval_fast_textgen_eval.py \\
    --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o

# save to custom dir, more cases
python utils/eval_fast_textgen_eval.py --model mymodel --out-dir /tmp/reports --n 50
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request


ALL_BENCHMARKS = ["mmlu", "gsm8k", "humaneval"]
_DEFAULT_DATASET_ROOT = "/Users/yenhaohuang/Desktop/datasets/fast-textgen-evalset"
_DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / "reports")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def chat(prompt: str, base_url: str, api_key: str, model: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise evaluator assistant. Follow output format strictly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if content:
                return content
            return (msg.get("reasoning_content") or "").strip()
    except TimeoutError:
        return ""


# ---------------------------------------------------------------------------
# HumanEval helpers
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _dedent4(code: str) -> str:
    """Strip one level (4 spaces) of indentation from each line.

    Models sometimes return function-body completions with an extra level of
    indentation (as if they were writing inside a def block).  This repairs
    the most common case before exec().
    """
    return "\n".join(line[4:] if line.startswith("    ") else line for line in code.split("\n"))


def _run_humaneval_case(
    prompt: str, entry_point: str, test_code: str, response: str
) -> tuple[bool, str | None, bool]:
    """Execute a HumanEval case.

    Returns
    -------
    passed : bool
    error  : str | None   – exception string on failure
    syntax_repaired : bool – True when indentation was auto-fixed
    """
    code = _strip_code_fence(response)
    if not code:
        return False, "empty response", False

    syntax_repaired = False
    if f"def {entry_point}(" in code:
        candidate_src = code
    else:
        body = code
        candidate_src = prompt + body + "\n"
        try:
            compile(candidate_src, "<string>", "exec")
        except IndentationError:
            body = _dedent4(code)
            candidate_src = prompt + body + "\n"
            syntax_repaired = True

    program = candidate_src + "\n" + test_code + f"\ncheck({entry_point})\n"
    g: dict = {}
    try:
        exec(program, g, g)  # noqa: S102
        return True, None, syntax_repaired
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", syntax_repaired


# ---------------------------------------------------------------------------
# Benchmark scorers
# ---------------------------------------------------------------------------

def score_mmlu(cfg: dict) -> dict:
    rows = read_jsonl(cfg["dataset_root"] / "mmlu" / "test.jsonl")[: cfg["n"]]
    correct, results = 0, []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        choices = row["choices"]
        gold = ["A", "B", "C", "D"][int(row["answer"])]
        prompt = (
            f"Question:\n{row['question']}\n\n"
            f"Choices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\n"
            "Answer with only one letter: A, B, C, or D."
        )
        out = chat(prompt, **cfg["chat_kwargs"])
        m = re.search(r"[ABCD]", out.upper())
        pred = m.group(0) if m else ""
        ok = pred == gold
        correct += int(ok)
        results.append({"idx": i, "gold": gold, "pred": pred, "passed": ok, "skipped": False, "response": out})
        print(f"  mmlu {i}/{len(rows)}  gold={gold} pred={pred} {'OK' if ok else 'FAIL'}")
    acc = correct / len(rows) if rows else None
    return {
        "benchmark": "mmlu",
        "total": len(rows),
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def score_gsm8k(cfg: dict) -> dict:
    rows = read_jsonl(cfg["dataset_root"] / "gsm8k-main" / "test.jsonl")[: cfg["n"]]
    correct, results = 0, []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        gold = _extract_last_number(row["answer"]) or ""
        prompt = f"Solve this math problem. End your answer with #### <number>.\n\nQuestion:\n{row['question']}"
        out = chat(prompt, **cfg["chat_kwargs"])
        pred = _extract_last_number(out) or ""
        ok = pred == gold
        correct += int(ok)
        results.append({"idx": i, "gold": gold, "pred": pred, "passed": ok, "skipped": False, "response": out})
        print(f"  gsm8k {i}/{len(rows)}  gold={gold} pred={pred} {'OK' if ok else 'FAIL'}")
    acc = correct / len(rows) if rows else None
    return {
        "benchmark": "gsm8k",
        "total": len(rows),
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }



def score_humaneval(cfg: dict) -> dict:
    rows = read_jsonl(cfg["dataset_root"] / "humaneval" / "test.jsonl")[: cfg["n"]]
    correct, results = 0, []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        query = (
            "Complete the following Python function. Return only Python code, no markdown fences. "
            "You may return either the full function definition or only the function body.\n\n"
            f"{row['prompt']}"
        )
        out = chat(query, **cfg["chat_kwargs"])
        ok, err, repaired = _run_humaneval_case(row["prompt"], row["entry_point"], row["test"], out)
        correct += int(ok)
        results.append({
            "idx": i,
            "task_id": row.get("task_id"),
            "entry_point": row.get("entry_point"),
            "passed": ok,
            "skipped": False,
            "syntax_repaired": repaired,
            "response": out,
            "error": err,
        })
        print(f"  humaneval {i}/{len(rows)}  {row.get('task_id', '')} {'OK' if ok else 'FAIL'}{' [repaired]' if repaired else ''}")
    acc = correct / len(rows) if rows else None
    return {
        "benchmark": "humaneval",
        "total": len(rows),
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def _extract_last_number(text: str) -> str | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


SCORERS = {
    "mmlu": score_mmlu,
    "gsm8k": score_gsm8k,
    "humaneval": score_humaneval,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal fast-textgen eval runner")
    p.add_argument("--benchmarks", nargs="+", choices=ALL_BENCHMARKS, default=ALL_BENCHMARKS,
                   metavar="BENCH", help="benchmarks to run (default: all)")
    p.add_argument("--model", required=True, help="model name passed to the API")
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                   help="OpenAI-compatible endpoint base URL (default: http://127.0.0.1:8080/v1)")
    p.add_argument("--api-key", default="dummy", help="API key (default: dummy)")
    p.add_argument("--n", type=int, default=20, help="max cases per benchmark (default: 20)")
    p.add_argument("--max-tokens", type=int, default=4096, help="max tokens per response (default: 4096)")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="output directory")
    p.add_argument("--dataset-root", default=_DEFAULT_DATASET_ROOT, help="dataset root path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = {
        "n": args.n,
        "dataset_root": Path(args.dataset_root),
        "chat_kwargs": {
            "base_url": args.base_url,
            "api_key": args.api_key,
            "model": args.model,
            "max_tokens": args.max_tokens,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    report: dict = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "base_url": args.base_url,
            "dataset_root": args.dataset_root,
            "max_cases_per_benchmark": args.n,
            "max_tokens": args.max_tokens,
            "benchmarks": args.benchmarks,
        },
    }

    for bench in args.benchmarks:
        print(f"\n[{bench}]")
        report[bench] = SCORERS[bench](cfg)

    report["meta"]["elapsed_sec"] = round(time.time() - t0, 2)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"fast_textgen_eval_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- results ---")
    for bench in args.benchmarks:
        r = report[bench]
        print(f"  {bench:12s}  {r.get('accuracy_pct', 'N/A'):>7s}  ({r['correct']}/{r['total']})")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
