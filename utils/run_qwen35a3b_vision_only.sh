#!/usr/bin/env bash
set -euo pipefail
BASE="/Users/yenhaohuang/Desktop/model-tester"
OUT="$BASE/qwen3.5-35b-a3b-vision/results-rerun-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
API_BASE="http://127.0.0.1:3172"
API="$API_BASE/v1/chat/completions"
MODEL_PATH="/Users/yenhaohuang/Desktop/models/unsloth-qwen35/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE.gguf"
MMPROJ="/Users/yenhaohuang/Desktop/models/unsloth-qwen35/Qwen3.5-35B-A3B-GGUF/mmproj-BF16.gguf"
MODEL_NAME="Qwen3.5-35B-A3B"

pkill -f "llama-server.*--port 3172" >/dev/null 2>&1 || true
sleep 3

echo "[$(date '+%F %T')] starting $MODEL_NAME" | tee -a "$OUT/progress.log"
llama-server -m "$MODEL_PATH" --mmproj "$MMPROJ" --port 3172 --ctx-size 16384 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --chat-template-kwargs '{"enable_thinking": false}' > "$OUT/llama-server.log" 2>&1 &
echo $! > "$OUT/llama-server.pid"

ready=0
for i in {1..30}; do
  code=$(curl -s -o /tmp/qwen35a3b_ready.json -w "%{http_code}" "$API" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"temperature\":0,\"max_tokens\":8}" || true)
  if [ "$code" = "200" ]; then
    ready=1
    cp /tmp/qwen35a3b_ready.json "$OUT/ready_probe.json"
    echo "[$(date '+%F %T')] ready gate passed" | tee -a "$OUT/progress.log"
    break
  fi
  echo "[$(date '+%F %T')] ready gate retry $i http=$code" | tee -a "$OUT/progress.log"
  sleep 3
done

if [ "$ready" -ne 1 ]; then
  echo "[$(date '+%F %T')] blocked: model not ready" | tee -a "$OUT/progress.log"
  exit 2
fi

~/Desktop/python-venvs/qwen35vision/bin/python "$BASE/utils/eval_vision.py" --api "$API" --model "$MODEL_NAME" --num 100 --examples 5 --tasks ocr,classification,detection --ocr-ds /Users/yenhaohuang/Desktop/datasets/vision-dataset/PokemonCards_train_300 --cls-ds /Users/yenhaohuang/Desktop/datasets/vision-dataset/cifar10_classification --det-ds /Users/yenhaohuang/Desktop/datasets/vision-dataset/object_detection --out-dir "$OUT" | tee "$OUT/eval_vision.log"

grep -E "prompt eval time|eval time|total time" "$OUT/llama-server.log" | tail -n 3 > "$OUT/speed_metrics.txt" || true
pkill -f "llama-server.*--port 3172" >/dev/null 2>&1 || true

echo "$OUT" > "$BASE/qwen35a3b_latest_outdir.txt"
echo "[$(date '+%F %T')] done" | tee -a "$OUT/progress.log"
