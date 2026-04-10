#!/usr/bin/env bash
# Eval gemma4-26b-a4b failed questions on OpenRouter
# Run from project root: bash exp/gemma4-26b-a4b/eval.sh
set -euo pipefail

RESULTS_DIR="results/gemma-4-26b-a4b-gguf_text_sources"
OR_MODEL="google/gemma-4-26b-a4b-it"
LOCAL_SLUG="gemma4-26b-a4b"
K=10
SEED=42

run() {
    local input="$1"
    echo "=== $input ==="
    python utils/eval_on_openrouter.py \
        --input        "$input" \
        --or-model     "$OR_MODEL" \
        --out-dir      "results/openrouter/exp-gemma4-26b-a4b" \
        --local-model-slug "$LOCAL_SLUG" \
        --k            "$K" \
        --seed         "$SEED" \
        --reasoning-effort medium
}

run "$RESULTS_DIR/fast_textgen_eval_20260405_023141_mmlu.json"
run "$RESULTS_DIR/fast_textgen_eval_20260405_023141_geo-mmlu-high-school.json"
run "$RESULTS_DIR/fast_textgen_eval_20260406_023200_humaneval.json"
run "$RESULTS_DIR/fast_textgen_eval_20260406_165225_gsm8k.json"
run "$RESULTS_DIR/fast_textgen_eval_20260410_004700_law-mmlu-professional.json"

echo ""
echo "All done. Results in: results/openrouter/exp-gemma4-26b-a4b/"
