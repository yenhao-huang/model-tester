#!/usr/bin/env python3
"""Evaluate gpt-oss-20b on GSM8K jsonl dataset."""

import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from scripts.run_benchmarks import build_client, call_model


def extract_num(text: str):
    if not text:
        return None
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to GSM8K jsonl")
    args = parser.parse_args()

    load_dotenv(Path('.env'))
    client = build_client()

    dataset = Path(args.dataset).expanduser().resolve()
    rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]

    system_prompt = "You are a careful math solver. Solve step by step, and end with one line: #### <final_number>."

    total = len(rows)
    scored_total = 0
    skipped = 0
    errors = 0
    correct = 0
    results = []
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        q = row["question"]
        gold = extract_num(row.get("answer", ""))
        resp = call_model(client, system_prompt, q)

        skipped_case = False
        passed = False
        if resp.startswith("[ERROR]"):
            if "402 Payment Required" in resp:
                skipped_case = True
                skipped += 1
            else:
                errors += 1
        else:
            pred = extract_num(resp)
            passed = (pred is not None and gold is not None and pred == gold)

        if not skipped_case:
            scored_total += 1
            if passed:
                correct += 1

        results.append({
            "idx": i,
            "passed": passed,
            "skipped": skipped_case,
            "gold": gold,
            "response": resp,
        })

    elapsed = round(time.time() - t0, 1)
    acc = (correct / scored_total) if scored_total else 0
    out = {
        "benchmark": "gsm8k",
        "dataset_path": str(dataset),
        "total": total,
        "scored_total": scored_total,
        "skipped": skipped,
        "errors": errors,
        "correct": correct,
        "accuracy": round(acc, 4),
        "accuracy_pct": f"{acc * 100:.1f}%",
        "elapsed_sec": elapsed,
        "avg_latency_ms": round(elapsed * 1000 / total) if total else 0,
        "results": results,
    }

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"gsm8k_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: out[k] for k in ["total", "scored_total", "skipped", "errors", "correct", "accuracy_pct", "elapsed_sec"]}, ensure_ascii=False, indent=2))
    print(f"REPORT_PATH={report_path}")


if __name__ == "__main__":
    main()
