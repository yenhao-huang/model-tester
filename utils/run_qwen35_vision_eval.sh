#!/usr/bin/env bash
set -euo pipefail

BASE="/Users/yenhaohuang/Desktop/model-tester"
EVAL_PY="$BASE/utils/eval_vision.py"
API_BASE="http://127.0.0.1:3172"
API="$API_BASE/v1/chat/completions"

run_model () {
  local key="$1"
  local model_path="$2"
  local mmproj_path="$3"
  local model_name="$4"

  local out_dir="$BASE/$key/results"
  mkdir -p "$out_dir"

  echo "[$(date '+%F %T')] Starting $key" | tee -a "$out_dir/progress.log"

  pkill -f "llama-server.*--port 3172" >/dev/null 2>&1 || true
  sleep 2

  llama-server \
    -m "$model_path" \
    --mmproj "$mmproj_path" \
    --port 3172 \
    --ctx-size 16384 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.0 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    > "$out_dir/llama-server.log" 2>&1 &
  local spid=$!
  echo "$spid" > "$out_dir/llama-server.pid"

  for i in {1..60}; do
    if curl -s "$API_BASE/v1/models" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] Server ready for $key" | tee -a "$out_dir/progress.log"
      break
    fi
    sleep 2
  done

  # Image smoke test
  curl -s "$API" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe image in one word\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/256px-Cat03.jpg\"}}]}],\"temperature\":0,\"max_tokens\":20}" \
    > "$out_dir/smoke_test.json" || true

  echo "[$(date '+%F %T')] Running eval_vision for $key" | tee -a "$out_dir/progress.log"
  ~/Desktop/python-venvs/qwen35vision/bin/python "$EVAL_PY" \
    --api "$API" \
    --model "$model_name" \
    --num 100 \
    --examples 5 \
    --tasks ocr,classification,detection \
    --ocr-ds /Users/yenhaohuang/Desktop/datasets/vision-dataset/PokemonCards_train_300 \
    --cls-ds /Users/yenhaohuang/Desktop/datasets/vision-dataset/cifar10_classification \
    --det-ds /Users/yenhaohuang/Desktop/datasets/vision-dataset/object_detection \
    --out-dir "$out_dir" \
    | tee "$out_dir/eval_vision.log"

  echo "[$(date '+%F %T')] Eval complete for $key" | tee -a "$out_dir/progress.log"

  # speed sample
  curl -s "$API" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"請總結這段文章重點並給出三點行動建議：生成式AI在企業導入時，常見問題包含資料治理、隱私合規、模型漂移與成本控制。成功落地需要先定義業務目標，再建立可觀測的評估流程，並透過持續監控與人機協作迭代。\"}],\"temperature\":0.0,\"max_tokens\":180}" \
    > "$out_dir/speed_sample_response.json" || true

  rg "prompt eval time|eval time|total time" "$out_dir/llama-server.log" | tail -n 3 > "$out_dir/speed_metrics.txt" || true

  pkill -f "llama-server.*--port 3172" >/dev/null 2>&1 || true
  echo "[$(date '+%F %T')] Stopped server for $key" | tee -a "$out_dir/progress.log"
}

run_model \
  "qwen3.5-27b-vision" \
  "/Users/yenhaohuang/Desktop/models/unsloth-qwen35/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_1.gguf" \
  "/Users/yenhaohuang/Desktop/models/unsloth-qwen35/Qwen3.5-27B-GGUF/mmproj-BF16.gguf" \
  "Qwen3.5-27B"

run_model \
  "qwen3.5-35b-a3b-vision" \
  "/Users/yenhaohuang/Desktop/models/unsloth-qwen35/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE.gguf" \
  "/Users/yenhaohuang/Desktop/models/unsloth-qwen35/Qwen3.5-35B-A3B-GGUF/mmproj-BF16.gguf" \
  "Qwen3.5-35B-A3B"

echo "[$(date '+%F %T')] All done"
