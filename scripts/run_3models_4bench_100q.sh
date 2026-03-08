#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$HOME/Desktop/model-tester"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_DIR/batch_runs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/${TS}_3models_4bench_100q.log"

run_combo() {
  local key="$1"
  local launch="$2"
  local port="$3"
  local model_name="$4"
  local bench="$5"

  echo "[$(date '+%F %T')] START key=${key} bench=${bench}" | tee -a "$MASTER_LOG"
  "$BASE_DIR/run_one_bench_fast2.sh" "$key" "$launch" "$port" "$model_name" "$bench" 2>&1 | tee -a "$MASTER_LOG"
  echo "[$(date '+%F %T')] DONE  key=${key} bench=${bench}" | tee -a "$MASTER_LOG"
}

# model: gpt-oss-20b
for bench in mmlu gsm8k truthfulqa humaneval; do
  run_combo "gpt-oss-20b-gguf" "$HOME/Desktop/llama-bash/gptoss20b.sh" 3172 "gptoss20b" "$bench"
done

# model: glm4.7-flash
for bench in mmlu gsm8k truthfulqa humaneval; do
  run_combo "glm-4.7-flash-fp4" "$HOME/Desktop/llama-bash/run_glm47_flash_fp4.sh" 3172 "glm4.7-flash" "$bench"
done

# model: qwen3.5-35b-a3b
for bench in mmlu gsm8k truthfulqa humaneval; do
  run_combo "qwen3.5_35b_a3b" "$HOME/Desktop/llama-bash/qwen35a3b.sh" 3172 "qwen3.5-a3b" "$bench"
done

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$MASTER_LOG"
