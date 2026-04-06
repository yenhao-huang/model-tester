#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/Desktop/model-tester/gemma-4-26b-a4b-gguf"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$BASE/text/results/rerun_ctx16384_${TS}"
RUNLOG="$BASE/rerun_ctx16384_${TS}.log"
SERVERLOG="$BASE/rerun_ctx16384_${TS}_server.log"
READY_JSON="$BASE/rerun_ctx16384_${TS}_ready.json"

mkdir -p "$OUT"

echo "ts=$TS" > "$RUNLOG"
echo "out=$OUT" >> "$RUNLOG"

pkill -f "llama-server.*--port 8092" >/dev/null 2>&1 || true
sleep 2

"$HOME/Desktop/others/llama.cpp/build/bin/llama-server" \
  -m "$HOME/Desktop/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" \
  --mmproj "$HOME/Desktop/models/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf" \
  --port 8092 \
  --ctx-size 16384 \
  --temp 0.0 \
  --top-p 0.95 \
  > "$SERVERLOG" 2>&1 &
SPID=$!
echo "server_pid=$SPID" >> "$RUNLOG"

READY=0
for i in $(seq 1 120); do
  code=$(curl -s -o "$READY_JSON" -w "%{http_code}" http://127.0.0.1:8092/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf","messages":[{"role":"user","content":"Only answer READY"}],"temperature":0,"max_tokens":8}' || true)
  if [ "$code" = "200" ]; then READY=1; echo "ready_attempt=$i" >> "$RUNLOG"; break; fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "blocked=ready_failed" >> "$RUNLOG"
  exit 2
fi

source "$HOME/Desktop/python-venvs/model-tester/bin/activate"
python "$HOME/Desktop/model-tester/utils/eval_fast_textgen_eval.py" \
  --model gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --base-url http://127.0.0.1:8092/v1 \
  --dataset-root "$HOME/Desktop/datasets/full-textgen-evalset" \
  --benchmarks humaneval gsm8k law-mmlu-professional \
  --n 100 \
  --max-tokens 16384 \
  --out-dir "$OUT" \
  > "$OUT/eval.log" 2>&1

echo "text_done=1" >> "$RUNLOG"
kill "$SPID" >/dev/null 2>&1 || true
wait "$SPID" >/dev/null 2>&1 || true
echo "server_stopped=1" >> "$RUNLOG"
