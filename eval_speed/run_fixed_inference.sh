#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_fixed_inference.sh <port> <model-name>
# Example:
#   ./run_fixed_inference.sh 3176 lfm2-24b

PORT="${1:-3176}"
MODEL="${2:-lfm2-24b}"
PROMPT_FILE="$(dirname "$0")/fixed_article_prompt.txt"
OUT_FILE="$(dirname "$0")/fixed_article_response.json"

PROMPT_CONTENT=$(python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/Users/yenhaohuang/Desktop/model-tester/eval_speed/fixed_article_prompt.txt")
print(json.dumps(p.read_text(encoding='utf-8')))
PY
)

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":${PROMPT_CONTENT}}],\"temperature\":0.0,\"max_tokens\":180}" \
  > "${OUT_FILE}"

echo "Saved response to: ${OUT_FILE}"
echo "Now check llama-server terminal log for:"
echo "  prompt eval time / eval time / total time"
