#!/usr/bin/env python3
"""TruthfulQA MCQ eval for OpenAI-compatible /v1/chat/completions.

Uses the prebuilt deprecated/gpt-oss-20b TruthfulQA MCQ JSONL format:
{id,prompt,expected,...}. Keeps raw response for each item.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chat(prompt: str, base_url: str, api_key: str, model: str, max_tokens: int, timeout_sec: int) -> dict:
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
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            return {
                "content": (msg.get("content") or "").strip(),
                "thinking": (msg.get("reasoning_content") or "").strip(),
                "usage": body.get("usage"),
                "elapsed_sec": round(time.time() - t0, 2),
                "error": None,
            }
    except TimeoutError as e:
        return {"content": "", "thinking": "", "usage": None, "elapsed_sec": round(time.time() - t0, 2), "error": f"TimeoutError: {e}"}
    except Exception as e:  # preserve error string in response/error fields
        return {"content": f"[ERROR] {type(e).__name__}: {e}", "thinking": "", "usage": None, "elapsed_sec": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"}


def extract_choice(text: str) -> str | None:
    if not text:
        return None
    t = text.strip().upper()
    marked = re.findall(r"(?im)^(?:final\s*answer|answer)\s*[:：]?\s*([A-E])\b", t)
    if marked:
        return marked[-1]
    single_line = re.findall(r"(?im)^\s*([A-E])\s*$", t)
    if single_line:
        return single_line[-1]
    if t and t[0] in "ABCDE":
        return t[0]
    tokens = re.findall(r"(?<![A-Z])[A-E](?![A-Z])", t)
    return tokens[-1] if tokens else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8098/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--dataset", default="/Users/yenhaohuang/Desktop/model-tester/deprecated/gpt-oss-20b/eval/datasets/truthfulqa.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout-sec", type=int, default=180)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rows = read_jsonl(Path(args.dataset))[: args.n]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    correct = 0
    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        out_obj = chat(row["prompt"], args.base_url, args.api_key, args.model, args.max_tokens, args.timeout_sec)
        out = out_obj.get("content", "")
        pred = extract_choice(out) or ""
        gold = row.get("expected", "")
        ok = pred == gold
        correct += int(ok)
        results.append({
            "idx": i,
            "id": row.get("id"),
            "gold": gold,
            "pred": pred,
            "passed": ok,
            "skipped": False,
            "prompt": row.get("prompt"),
            "response": out,
            "thinking": out_obj.get("thinking", ""),
            "usage": out_obj.get("usage"),
            "elapsed_sec": out_obj.get("elapsed_sec"),
            "error": out_obj.get("error"),
        })
        print(f"  truthfulqa {i}/{len(rows)} gold={gold} pred={pred} {'OK' if ok else 'FAIL'}")
    acc = correct / len(rows) if rows else None
    report = {
        "benchmark": "truthfulqa",
        "total": len(rows),
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": f"{acc*100:.1f}%" if acc is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "base_url": args.base_url,
            "dataset": args.dataset,
            "max_cases": args.n,
            "timeout_sec": args.timeout_sec,
        },
    }
    path = out_dir / f"truthfulqa_mcq_{ts}_truthfulqa.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  => {report['accuracy_pct']} ({correct}/{len(rows)}) saved: {path.name}")


if __name__ == "__main__":
    main()
