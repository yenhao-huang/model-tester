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


ALL_BENCHMARKS = ["mmlu", "geo-mmlu-high-school", "law-mmlu-professional", "gsm8k", "humaneval"]
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
        with urllib_request.urlopen(req, timeout=10) as resp:
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


def _normalize_indent(code: str) -> str:
    """Best-effort indentation repair for model-generated Python code."""
    code = code.replace("\t", "    ")
    return _dedent4(code)


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

    def _build_candidate(src: str) -> str:
        if f"def {entry_point}(" in src:
            return src
        body_lines = src.splitlines()
        if any(line.strip() and not line.startswith((" ", "\t")) for line in body_lines):
            src = "\n".join(("    " + line) if line.strip() else line for line in body_lines)
        return prompt + src + "\n"

    candidate_src = _build_candidate(code)
    try:
        compile(candidate_src, "<string>", "exec")
    except SyntaxError:
        repaired = _normalize_indent(code)
        if repaired != code:
            syntax_repaired = True
        candidate_src = _build_candidate(repaired)

    program = candidate_src + "\n" + test_code + f"\ncheck({entry_point})\n"
    g: dict = {}
    try:
        exec(program, g, g)  # noqa: S102
        return True, None, syntax_repaired
    except IndentationError:
        repaired = _normalize_indent(code)
        candidate_src = _build_candidate(repaired)
        program = candidate_src + "\n" + test_code + f"\ncheck({entry_point})\n"
        g = {}
        try:
            exec(program, g, g)  # noqa: S102
            return True, None, True
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", True
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", syntax_repaired


# ---------------------------------------------------------------------------
# Benchmark scorers
# ---------------------------------------------------------------------------

def _score_mmlu_like(cfg: dict, benchmark: str, dataset_subdir: str) -> dict:
    rows = read_jsonl(cfg["dataset_root"] / dataset_subdir / "test.jsonl")[: cfg["n"]]
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
        results.append({"idx": i, "gold": gold, "pred": pred, "passed": ok, "skipped": False, "prompt": prompt, "response": out})
        print(f"  {benchmark} {i}/{len(rows)}  gold={gold} pred={pred} {'OK' if ok else 'FAIL'}")
    acc = correct / len(rows) if rows else None
    return {
        "benchmark": benchmark,
        "total": len(rows),
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def score_mmlu(cfg: dict) -> dict:
    return _score_mmlu_like(cfg, "mmlu", "mmlu")


def score_geo_mmlu_high_school(cfg: dict) -> dict:
    return _score_mmlu_like(cfg, "geo-mmlu-high-school", "geo-mmlu-high-school")


def score_law_mmlu_professional(cfg: dict) -> dict:
    return _score_mmlu_like(cfg, "law-mmlu-professional", "law-mmlu-professiona")


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
        results.append({"idx": i, "gold": gold, "pred": pred, "passed": ok, "skipped": False, "prompt": prompt, "response": out})
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
            "prompt": query,
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
    "geo-mmlu-high-school": score_geo_mmlu_high_school,
    "law-mmlu-professional": score_law_mmlu_professional,
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

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for bench in args.benchmarks:
        existing = sorted(out_dir.glob(f"*{bench}.json"))
        if existing:
            cached_path = existing[-1]
            print(f"\n[{bench}] skipping — cache found: {cached_path.name}")
            report[bench] = json.loads(cached_path.read_text(encoding="utf-8"))
            r = report[bench]
            print(f"  => {r.get('accuracy_pct', 'N/A')} ({r['correct']}/{r['total']})")
            continue

        print(f"\n[{bench}]")
        try:
            report[bench] = SCORERS[bench](cfg)
        except Exception as e:
            report[bench] = {"benchmark": bench, "error": f"{type(e).__name__}: {e}"}
            print(f"  ERROR: {e}")
            continue

        bench_path = out_dir / f"fast_textgen_eval_{ts}_{bench}.json"
        bench_path.write_text(json.dumps(report[bench], ensure_ascii=False, indent=2), encoding="utf-8")
        r = report[bench]
        print(f"  => {r.get('accuracy_pct', 'N/A')} ({r['correct']}/{r['total']})  saved: {bench_path.name}")

    report["meta"]["elapsed_sec"] = round(time.time() - t0, 2)

    all_path = out_dir / f"fast_textgen_eval_{ts}_all.json"
    all_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- results ---")
    for bench in args.benchmarks:
        r = report[bench]
        if "error" in r:
            print(f"  {bench:12s}  ERROR: {r['error']}")
        else:
            print(f"  {bench:12s}  {r.get('accuracy_pct', 'N/A'):>7s}  ({r['correct']}/{r['total']})")
    print(f"\nConsolidated: {all_path}")


if __name__ == "__main__":
    main()
