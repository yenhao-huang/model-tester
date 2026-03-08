#!/usr/bin/env bash
set -euo pipefail

KEY="$1"
LAUNCH="$2"
PORT="$3"
MODEL_NAME="$4"

DATASET_ROOT="$HOME/Desktop/datasets/fast-textgen-evalset2"
BASE_DIR="$HOME/Desktop/model-tester"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BASE_DIR/$KEY/runs/${TS}_fasttextgenevalset2_500q"
mkdir -p "$RUN_DIR"

pkill -f llama-server >/dev/null 2>&1 || true
sleep 1

"$LAUNCH" > "$RUN_DIR/server.log" 2>&1 &
SPID=$!
echo "server_pid=$SPID" | tee "$RUN_DIR/run.log"

READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"say READY"}],"temperature":0.0,"max_tokens":8}' \
    | grep -q '"choices"'; then
    READY=1; break
  fi
  if ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "server not ready" | tee -a "$RUN_DIR/run.log"
  kill "$SPID" >/dev/null 2>&1 || true
  wait "$SPID" >/dev/null 2>&1 || true
  exit 1
fi

echo "server ready; eval start" | tee -a "$RUN_DIR/run.log"
source "$HOME/Desktop/python-venvs/model-tester/bin/activate"
python "$BASE_DIR/utils/eval_fast_textgen_eval.py" \
  --model "$MODEL_NAME" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --dataset-root "$DATASET_ROOT" \
  --n 100 \
  --out-dir "$RUN_DIR" \
  > "$RUN_DIR/eval.log" 2>&1

echo "eval done" | tee -a "$RUN_DIR/run.log"
kill "$SPID" >/dev/null 2>&1 || true
wait "$SPID" >/dev/null 2>&1 || true

echo "done run_dir=$RUN_DIR" | tee -a "$RUN_DIR/run.log"
