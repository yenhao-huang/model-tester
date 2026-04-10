#!/usr/bin/env python3
"""Eval failed questions from a local benchmark report on OpenRouter.

Reads a fast-textgen-eval all.json report, collects questions that the local
model got wrong, randomly samples k of them, sends each to OpenRouter, and
saves one JSON record per question under results/openrouter/<model_slug>/.

Usage
-----
python utils/eval_on_openrouter.py \
    --input gemma-4-26b-a4b-gguf/text/results/summary/fast_textgen_eval_20260411_0555_all.json \
    --or-model "google/gemma-3-27b-it" \
    --k 5

# Override output folder
python utils/eval_on_openrouter.py \
    --input path/to/all.json \
    --or-model "anthropic/claude-3.5-sonnet" \
    --k 10 \
    --out-dir results/openrouter/my_run
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

# project root = one level up from this file (utils/eval_on_openrouter.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(_PROJECT_ROOT))
from utils.providers.call_openrouter_api import call_openrouter

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_MODEL_SLUG = "gemma4-26b-a4b"
_DEFAULT_OR_MODEL         = "qwen/qwen3-235b-a22b-2507"
_MAX_QUESTIONS            = 30  # hard cap — never eval more than this per run


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def collect_failures(input_path: Path) -> list[dict]:
    """Return all rows where passed=False from every benchmark (includes skipped).

    Handles two formats:
    - all.json  : { "mmlu": { "benchmark": "mmlu", "results": [...] }, ... }
    - per-bench : { "benchmark": "law-mmlu-professional", "results": [...] }
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))
    failures: list[dict] = []

    # Detect per-benchmark format: top-level has a "results" list directly
    if "results" in data and isinstance(data["results"], list):
        benches = {data.get("benchmark", "unknown"): data}
    else:
        benches = {k: v for k, v in data.items() if isinstance(v, dict) and "results" in v}

    for bench_key, bench in benches.items():
        for row in bench["results"]:
            if row.get("passed") == False:
                failures.append({
                    "benchmark": bench_key,
                    "idx":       row.get("idx"),
                    "gold":      row.get("gold"),
                    "local_pred":row.get("pred", ""),
                    "prompt":    row.get("prompt", ""),
                    # humaneval extras
                    "task_id":       row.get("task_id"),
                    "entry_point":   row.get("entry_point"),
                    "local_error":   row.get("error"),
                    "local_thinking":row.get("thinking", ""),
                })
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample failed questions from a benchmark report and re-eval on OpenRouter."
    )
    p.add_argument(
        "--input", required=True,
        help="Path to the fast-textgen-eval all.json (or per-benchmark .json) report file.",
    )
    p.add_argument(
        "--or-model", default=_DEFAULT_OR_MODEL,
        help=f"OpenRouter model ID (default: {_DEFAULT_OR_MODEL})",
    )
    p.add_argument(
        "--k", type=int, default=5,
        help="Number of failed questions to sample (default: 5)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling (default: None)",
    )
    p.add_argument(
        "--local-model-slug", default=_DEFAULT_LOCAL_MODEL_SLUG,
        help=f"Slug used for the output subfolder (default: {_DEFAULT_LOCAL_MODEL_SLUG})",
    )
    p.add_argument(
        "--out-dir", default=None,
        help=(
            "Output directory for JSON records "
            "(default: results/openrouter/<local-model-slug>)"
        ),
    )
    p.add_argument(
        "--reasoning-effort", default="medium",
        choices=["", "low", "medium", "high"],
        help="OpenRouter reasoning effort (default: medium)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=4096,
    )
    p.add_argument(
        "--api-key", default="",
        help="OpenRouter API key. Defaults to 'openrouter' in .env",
    )
    p.add_argument(
        "--benchmarks", nargs="*", default=None,
        help="Limit sampling to these benchmarks (default: all failing benchmarks)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.is_file():
        logger.error("Report not found: %s", input_path)
        raise SystemExit(1)

    # Collect failures
    failures = collect_failures(input_path)
    logger.info("Total failures in report: %d", len(failures))

    # Optionally filter by benchmark
    if args.benchmarks:
        failures = [f for f in failures if f["benchmark"] in args.benchmarks]
        logger.info("After benchmark filter (%s): %d", args.benchmarks, len(failures))

    if not failures:
        logger.warning("No failures found — nothing to eval.")
        return
    # Sample k
    if args.seed is not None:
        random.seed(args.seed)
    if args.k > _MAX_QUESTIONS:
        logger.error("--k %d exceeds MAX_QUESTIONS=%d", args.k, _MAX_QUESTIONS)
        raise SystemExit(1)
    k = min(args.k, len(failures))
    sampled = random.sample(failures, k)
    logger.info("Sampled %d / %d failures to send to OpenRouter (%s)", k, len(failures), args.or_model)

    # Sanity-check: ping OpenRouter with a trivial question before the real eval
    logger.info("Sanity check: pinging OpenRouter (%s) …", args.or_model)
    _ping = call_openrouter(
        question="Reply with the single word OK.",
        model=args.or_model,
        api_key=args.api_key,
        reasoning_effort=args.reasoning_effort,
        max_tokens=16,
        out_dir="",
    )
    if _ping["error"]:
        logger.error("Sanity check FAILED: %s", _ping["error"])
        raise SystemExit(1)
    logger.info("Sanity check passed — response: %r", _ping["response"])

    # Resolve output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _PROJECT_ROOT / "results" / "openrouter" / args.local_model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_summary: list[dict] = []

    for i, item in enumerate(sampled, 1):
        bench = item["benchmark"]
        idx   = item["idx"]
        label = f"[{i}/{k}] {bench} #{idx}"

        # Cache check: skip if a result file already exists for this bench+idx
        bench_slug = bench.replace("-", "_")
        idx_str    = f"q{idx:04d}" if idx is not None else f"q{str(idx)}"
        cached = list(out_dir.glob(f"*_{bench_slug}_{idx_str}.json"))
        if cached:
            cached_record = json.loads(cached[0].read_text(encoding="utf-8"))
            logger.info("%s  [CACHE HIT] → %s", label, cached[0].name)
            run_summary.append({
                "benchmark":   bench,
                "idx":         idx,
                "gold":        item["gold"],
                "local_pred":  item["local_pred"],
                "or_response": (cached_record.get("or_response") or "")[:120],
                "or_error":    cached_record.get("or_error"),
                "saved_to":    str(cached[0]),
            })
            continue

        logger.info("%s  →  %s", label, args.or_model)

        result = call_openrouter(
            question=item["prompt"],
            model=args.or_model,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            out_dir="",           # we handle saving ourselves below
        )

        record = {
            # provenance
            "run_timestamp_utc":  run_ts,
            "report_file":        str(input_path),
            "local_model_slug":   args.local_model_slug,
            "benchmark":          bench,
            "question_idx":       idx,
            "task_id":            item.get("task_id"),
            # question
            "prompt":             item["prompt"],
            "gold":               item["gold"],
            # local model result
            "local_pred":         item["local_pred"],
            "local_thinking":     item["local_thinking"],
            "local_error":        item["local_error"],
            # openrouter result
            "or_model":           result["model"],
            "or_thinking":        result["thinking"],
            "or_response":        result["response"],
            "or_usage":           result["usage"],
            "or_elapsed_sec":     result["elapsed_sec"],
            "or_error":           result["error"],
            # settings
            "reasoning_effort":   args.reasoning_effort,
            "max_tokens":         args.max_tokens,
        }

        fname = f"{run_ts}_{bench_slug}_{idx_str}.json"
        save_path = out_dir / fname
        save_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("%s  saved → %s", label, save_path)

        run_summary.append({
            "benchmark":   bench,
            "idx":         idx,
            "gold":        item["gold"],
            "local_pred":  item["local_pred"],
            "or_response": result["response"][:120] if result["response"] else "",
            "or_error":    result["error"],
            "saved_to":    str(save_path),
        })

    # Print summary table
    print(f"\n=== OpenRouter eval summary  ({args.or_model}) ===")
    print(f"{'#':<4} {'bench':<26} {'idx':>4}  {'gold':>4}  {'local':>5}  {'or_resp[:60]'}")
    print("-" * 90)
    for j, s in enumerate(run_summary, 1):
        resp_preview = (s["or_response"] or s.get("or_error") or "")[:60]
        idx_str  = str(s["idx"])  if s["idx"]  is not None else "-"
        gold_str = str(s["gold"]) if s["gold"] is not None else "-"
        pred_str = str(s["local_pred"]) if s["local_pred"] is not None else "-"
        print(f"{j:<4} {s['benchmark']:<26} {idx_str:>4}  {gold_str:>4}  {pred_str:>5}  {resp_preview}")
    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
