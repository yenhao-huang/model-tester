#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-hf_api}"
DATASET="${2:-eval/datasets/smoke.jsonl}"
PROMPT="${3:-prompts/system_v1.txt}"
OUT="${4:-reports/${MODE}_report.json}"

if [[ ! -f ".env" ]]; then
  echo "[error] .env not found. Copy .env.example to .env first."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "[error] .venv not found. Create it first: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

if [[ "$MODE" == "chat" ]]; then
  BACKEND="${2:-hf_api}"
  CHAT_PROMPT="${3:-prompts/system_v1.txt}"
  export INFERENCE_BACKEND="$BACKEND"

  echo "[info] mode=chat"
  echo "[info] backend=$INFERENCE_BACKEND"
  echo "[info] prompt=$CHAT_PROMPT"

  python -m src.gpt_oss20b_eval.chat --prompt "$CHAT_PROMPT"
  exit 0
fi

export INFERENCE_BACKEND="$MODE"

echo "[info] backend=$INFERENCE_BACKEND"
echo "[info] dataset=$DATASET"
echo "[info] prompt=$PROMPT"
echo "[info] out=$OUT"

python -m src.gpt_oss20b_eval.runner \
  --dataset "$DATASET" \
  --prompt "$PROMPT" \
  --out "$OUT"

echo "[done] report written to $OUT"
