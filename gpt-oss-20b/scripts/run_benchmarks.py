#!/usr/bin/env python3
"""Run all benchmarks against gpt-oss-20b and produce a score report."""

import json
import os
import re
import sys
import time
import textwrap
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ── Config ──────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval" / "datasets"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_v1.txt"

BENCHMARKS = ["mmlu", "hellaswag", "ifeval", "humaneval", "truthfulqa"]
# Sample sizes per benchmark (set None for full). Reduce for faster testing.
SAMPLE_SIZES = {
    "mmlu": 200,
    "hellaswag": 200,
    "ifeval": None,  # only 541
    "humaneval": None,  # only 164
    "truthfulqa": None,  # only 817
}


# ── Model Client ────────────────────────────────────────────────────────
def build_client():
    model = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("⚠️  HF_TOKEN not set. API calls may fail for gated models.")
    return InferenceClient(model=model, token=token)


def call_model(client, system_prompt: str, user_prompt: str) -> str:
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    except Exception as e:
        return f"[ERROR] {e}"
    return ""


# ── Scoring Functions ───────────────────────────────────────────────────
def score_mcq(response: str, expected: str) -> bool:
    """Check if response contains the correct letter choice."""
    resp = response.strip().upper()
    # Direct letter match
    if resp == expected:
        return True
    # First letter
    if resp and resp[0] == expected:
        return True
    # Pattern: "The answer is X" or just "X."
    patterns = [
        rf'\b{expected}\b',
        rf'answer\s*(?:is|:)\s*{expected}',
    ]
    for p in patterns:
        if re.search(p, resp, re.IGNORECASE):
            return True
    return False


def score_code(response: str, meta: dict) -> bool:
    """Execute generated code against HumanEval test cases."""
    prompt_code = meta.get("prompt_code", "")
    test_code = meta.get("test", "")
    entry_point = meta.get("entry_point", "")

    # Clean response: strip markdown fences
    code = response.strip()
    code = re.sub(r'^```(?:python)?\n?', '', code)
    code = re.sub(r'\n?```$', '', code)

    # Build full program
    full_code = prompt_code + code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=10,
        )
        os.unlink(tmp)
        return result.returncode == 0
    except Exception:
        return False


def score_ifeval(response: str, meta: dict) -> bool:
    """Basic IFEval scoring: check non-empty response (full eval needs instruction parser)."""
    # Simplified: just check response is substantial
    return len(response.strip()) > 20


# ── Main Runner ─────────────────────────────────────────────────────────
def run_benchmark(name: str, client, system_prompt: str) -> dict:
    jsonl_path = EVAL_DIR / f"{name}.jsonl"
    if not jsonl_path.exists():
        return {"error": f"File not found: {jsonl_path}"}

    cases = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # Sample if needed
    sample_size = SAMPLE_SIZES.get(name)
    if sample_size and len(cases) > sample_size:
        import random
        random.seed(42)
        cases = random.sample(cases, sample_size)

    total = len(cases)
    correct = 0
    errors = 0
    results = []
    t0 = time.time()

    for i, case in enumerate(cases):
        response = call_model(client, system_prompt, case["prompt"])

        if response.startswith("[ERROR]"):
            errors += 1
            passed = False
        elif name in ("mmlu", "hellaswag", "truthfulqa"):
            passed = score_mcq(response, case.get("expected", ""))
        elif name == "humaneval":
            passed = score_code(response, case.get("meta", {}))
        elif name == "ifeval":
            passed = score_ifeval(response, case.get("meta", {}))
        else:
            passed = len(response.strip()) > 0

        if passed:
            correct += 1

        results.append({
            "id": case["id"],
            "passed": passed,
            "response": response,
        })

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            pct = correct / (i + 1) * 100
            print(f"  [{name}] {i+1}/{total} — running accuracy: {pct:.1f}%")

    elapsed = time.time() - t0
    accuracy = correct / total if total else 0

    return {
        "benchmark": name,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "elapsed_sec": round(elapsed, 1),
        "avg_latency_ms": round(elapsed / total * 1000) if total else 0,
        "results": results,
    }


def main():
    load_dotenv()
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    client = build_client()

    print(f"Model: {os.getenv('MODEL_NAME', 'openai/gpt-oss-20b')}")
    print(f"Benchmarks: {', '.join(BENCHMARKS)}")
    print("=" * 60)

    all_results = {}
    for name in BENCHMARKS:
        print(f"\n▶ Running {name}...")
        result = run_benchmark(name, client, system_prompt)
        all_results[name] = result
        if "error" not in result:
            print(f"  ✅ {name}: {result['accuracy_pct']} ({result['correct']}/{result['total']}) in {result['elapsed_sec']}s")
        else:
            print(f"  ❌ {name}: {result['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<20} {'Score':>10} {'Correct':>10} {'Total':>8} {'Time':>8}")
    print("-" * 60)
    for name in BENCHMARKS:
        r = all_results[name]
        if "error" in r:
            print(f"{name:<20} {'ERROR':>10}")
        else:
            print(f"{name:<20} {r['accuracy_pct']:>10} {r['correct']:>10} {r['total']:>8} {r['elapsed_sec']:>7}s")
    print("=" * 60)

    # Save full report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"benchmark_{ts}.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 Full report: {report_path}")


if __name__ == "__main__":
    main()
