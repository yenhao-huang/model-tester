#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import InferenceClient

MODEL = os.getenv("MODEL_NAME", "MiniMaxAI/MiniMax-M2.5")
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/Users/yenhaohuang/Desktop/datasets/my-textgen-evalset"))
MAX_CASES = int(os.getenv("MAX_CASES", "20"))
OUT_DIR = Path(os.getenv("OUT_DIR", str(Path(__file__).resolve().parent / "reports")))

# Rate limit: 4 question-requests per 10 seconds (global across all benchmarks)
REQUESTS_PER_WINDOW = int(os.getenv("REQUESTS_PER_WINDOW", "4"))
REQUEST_WINDOW_SEC = float(os.getenv("REQUEST_WINDOW_SEC", "10"))
_request_timestamps = deque()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _throttle_question_requests() -> None:
    now = time.monotonic()
    while _request_timestamps and (now - _request_timestamps[0]) >= REQUEST_WINDOW_SEC:
        _request_timestamps.popleft()

    if len(_request_timestamps) >= REQUESTS_PER_WINDOW:
        wait_sec = REQUEST_WINDOW_SEC - (now - _request_timestamps[0])
        if wait_sec > 0:
            time.sleep(wait_sec)
        now = time.monotonic()
        while _request_timestamps and (now - _request_timestamps[0]) >= REQUEST_WINDOW_SEC:
            _request_timestamps.popleft()

    _request_timestamps.append(time.monotonic())


def ask(client: InferenceClient, prompt: str) -> str:
    _throttle_question_requests()
    r = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a precise evaluator assistant. Follow output format strictly."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    return (r.choices[0].message.content or "").strip()


def is_402(err_text: str) -> bool:
    return "402 Payment Required" in err_text or "Credit balance is depleted" in err_text


def score_mmlu(client: InferenceClient):
    rows = read_jsonl(DATASET_ROOT / "mmlu" / "test.jsonl")[:MAX_CASES]
    results = []
    scored_total = skipped = correct = errors = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        choices = row["choices"]
        gold_letter = ["A", "B", "C", "D"][int(row["answer"])]
        prompt = (
            f"Question:\n{row['question']}\n\n"
            f"Choices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\n"
            "Answer with only one letter: A, B, C, or D."
        )
        try:
            out = ask(client, prompt)
            pred = (re.search(r"[ABCD]", out.upper()) or [None])[0] if re.search(r"[ABCD]", out.upper()) else ""
            passed = pred == gold_letter
            scored_total += 1
            if passed:
                correct += 1
            results.append({"idx": i, "gold": gold_letter, "pred": pred, "passed": passed, "skipped": False, "response": out})
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if is_402(err):
                skipped += 1
                results.append({"idx": i, "gold": gold_letter, "pred": None, "passed": False, "skipped": True, "response": f"[ERROR] {err}"})
            else:
                errors += 1
                results.append({"idx": i, "gold": gold_letter, "pred": None, "passed": False, "skipped": False, "response": f"[ERROR] {err}"})

    acc = (correct / scored_total) if scored_total else None
    return {
        "benchmark": "mmlu",
        "total": len(rows),
        "scored_total": scored_total,
        "skipped": skipped,
        "errors": errors,
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def extract_last_number(text: str) -> str | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


def score_gsm8k(client: InferenceClient):
    rows = read_jsonl(DATASET_ROOT / "gsm8k-main" / "test.jsonl")[:MAX_CASES]
    results = []
    scored_total = skipped = correct = errors = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        gold = extract_last_number(row["answer"]) or ""
        prompt = (
            f"Solve this math word problem and end with '#### <number>'.\n\nQuestion:\n{row['question']}"
        )
        try:
            out = ask(client, prompt)
            pred = extract_last_number(out)
            passed = (pred == gold)
            scored_total += 1
            if passed:
                correct += 1
            results.append({"idx": i, "gold": gold, "pred": pred, "passed": passed, "skipped": False, "response": out})
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if is_402(err):
                skipped += 1
                results.append({"idx": i, "gold": gold, "pred": None, "passed": False, "skipped": True, "response": f"[ERROR] {err}"})
            else:
                errors += 1
                results.append({"idx": i, "gold": gold, "pred": None, "passed": False, "skipped": False, "response": f"[ERROR] {err}"})

    acc = (correct / scored_total) if scored_total else None
    return {
        "benchmark": "gsm8k",
        "total": len(rows),
        "scored_total": scored_total,
        "skipped": skipped,
        "errors": errors,
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def score_truthfulqa(client: InferenceClient):
    rows = read_jsonl(DATASET_ROOT / "truthfulqa" / "validation.jsonl")[:MAX_CASES]
    results = []
    scored_total = skipped = correct = errors = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        prompt = f"Answer briefly and factually:\n{row['question']}"
        try:
            out = ask(client, prompt)
            low = out.lower()
            cands = [x.lower() for x in row.get("correct_answers", [])] + [str(row.get("best_answer", "")).lower()]
            passed = any(c and c in low for c in cands)
            scored_total += 1
            if passed:
                correct += 1
            results.append({"idx": i, "passed": passed, "skipped": False, "response": out})
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if is_402(err):
                skipped += 1
                results.append({"idx": i, "passed": False, "skipped": True, "response": f"[ERROR] {err}"})
            else:
                errors += 1
                results.append({"idx": i, "passed": False, "skipped": False, "response": f"[ERROR] {err}"})

    acc = (correct / scored_total) if scored_total else None
    return {
        "benchmark": "truthfulqa",
        "total": len(rows),
        "scored_total": scored_total,
        "skipped": skipped,
        "errors": errors,
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def main():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model=MODEL, token=token, timeout=30)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    mmlu_report = score_mmlu(client)
    gsm8k_report = score_gsm8k(client)
    truthfulqa_report = score_truthfulqa(client)

    all_reports = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "dataset_root": str(DATASET_ROOT),
            "max_cases_per_benchmark": MAX_CASES,
            "hf_token_present": bool(token),
            "request_rate_limit": {
                "requests_per_window": REQUESTS_PER_WINDOW,
                "window_seconds": REQUEST_WINDOW_SEC,
                "note": "question-level throttle across all benchmarks",
            },
        },
        "mmlu": mmlu_report,
        "gsm8k": gsm8k_report,
        "truthfulqa": truthfulqa_report,
    }

    out = OUT_DIR / f"my_textgen_evalset_{ts}.json"
    out.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_reports, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
