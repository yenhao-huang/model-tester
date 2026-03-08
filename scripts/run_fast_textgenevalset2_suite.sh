#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="$HOME/Desktop/datasets/fast-textgen-evalset2"
BASE_DIR="$HOME/Desktop/model-tester"
TS="$(date +%Y%m%d_%H%M%S)"

cleanup_all() {
  pkill -f llama-server >/dev/null 2>&1 || true
}
trap cleanup_all EXIT

run_one() {
  local key="$1"
  local launch="$2"
  local port="$3"
  local model_name="$4"

  local run_dir="$BASE_DIR/$key/runs/${TS}_fasttextgenevalset2_500q"
  mkdir -p "$run_dir"

  echo "==== [$key] launch: $launch (port $port) ====" | tee -a "$run_dir/run.log"

  # ensure no stale server
  pkill -f llama-server >/dev/null 2>&1 || true
  sleep 1

  "$launch" > "$run_dir/server.log" 2>&1 &
  local spid=$!
  echo "server_pid=$spid" | tee -a "$run_dir/run.log"

  local ready=0
  for i in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${port}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"'"$model_name"'","messages":[{"role":"user","content":"say READY"}],"temperature":0.0,"max_tokens":8}' \
      | grep -q '"choices"'; then
      ready=1
      break
    fi
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "[$key] server process exited before ready." | tee -a "$run_dir/run.log"
      break
    fi
    sleep 2
  done

  if [[ "$ready" -ne 1 ]]; then
    echo "[$key] server not ready, aborting." | tee -a "$run_dir/run.log"
    kill "$spid" >/dev/null 2>&1 || true
    wait "$spid" >/dev/null 2>&1 || true
    return 1
  fi

  echo "[$key] server ready. start eval..." | tee -a "$run_dir/run.log"
  python "$BASE_DIR/utils/eval_fast_textgen_eval.py" \
    --model "$model_name" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --dataset-root "$DATASET_ROOT" \
    --n 100 \
    --out-dir "$run_dir" \
    > "$run_dir/eval.log" 2>&1

  curl -s "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$model_name"'","messages":[{"role":"user","content":"請總結這段文章重點並給出三點行動建議：Open-source model evaluation requires consistency in prompts, dataset versions, and logging format. Without reproducible scripts and fixed seeds, benchmark comparisons become noisy and misleading. Teams should keep exact run manifests, raw responses, and error taxonomy so regressions can be detected quickly and discussed objectively."}],"temperature":0.0,"max_tokens":180}' \
    > /dev/null 2>&1 || true

  {
    echo "\n[$key] speed sample lines:";
    grep -E "prompt eval time|eval time|total time" "$run_dir/server.log" | tail -n 6;
  } >> "$run_dir/run.log" || true

  echo "[$key] eval done. stop server..." | tee -a "$run_dir/run.log"
  kill "$spid" >/dev/null 2>&1 || true
  wait "$spid" >/dev/null 2>&1 || true

  echo "==== [$key] finished ====" | tee -a "$run_dir/run.log"
}

source "$HOME/Desktop/python-venvs/model-tester/bin/activate"

run_one "lfm2-24b" "$HOME/Desktop/llama-bash/lfm2.sh" 3176 "lfm2"
run_one "gpt-oss-20b-gguf" "$HOME/Desktop/llama-bash/gptoss20b.sh" 3172 "gptoss20b"
run_one "qwen3.5_35b_a3b" "$HOME/Desktop/llama-bash/qwen35a3b.sh" 3172 "qwen3.5-a3b"
run_one "glm-4.7-mxfp4-gguf" "$HOME/Desktop/llama-bash/glm4.7.sh" 3179 "glm4.7"

echo "ALL_DONE TS=$TS"
