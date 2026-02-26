# GLM-4.7-MXFP4 (GGUF) Evaluation

This folder contains local evaluation scripts and reports for `glm-4.7-mxfp4.gguf`.

## Model
- File: `~/Desktop/models/glm-4.7-mxfp4/GLM-4.7-Flash-MXFP4_MOE.gguf`
- Runtime: `llama-server` (OpenAI-compatible API)

## Dataset
- Root: `~/Desktop/datasets/fast-textgen-evalset`
- Benchmarks:
  - `mmlu/test.jsonl`
  - `gsm8k-main/test.jsonl`
  - `truthfulqa/validation.jsonl`
  - `humaneval/test.jsonl`

## How to Eval

### 1) Start model server
Use your zsh PATH (contains `llama-server`):

```bash
zsh -lc "source ~/.zshrc; llama-server \
  -m ~/Desktop/models/glm-4.7-mxfp4/GLM-4.7-Flash-MXFP4_MOE.gguf \
  --jinja -a gpt-3.5-turbo \
  -t 8 -c 8192 -n 4096 -ngl 99 \
  --seed 3407 --temp 0.2 --top-p 0.95 --top-k 40 \
  --ubatch-size 32 --port 29172 --host 127.0.0.1"
```

Health check:

```bash
curl -s http://127.0.0.1:29172/health
```

### 2) Run evaluation (20 per benchmark)

```bash
cd ~/Desktop/model-tester/glm-4.7-mxfp4-gguf
PYTHONUNBUFFERED=1 MAX_CASES=20 MAX_TOKENS=96 \
python eval_fast_textgen_eval_local.py
```

## Output
- Reports are saved to: `./reports/fast_textgen_eval_YYYYmmdd_HHMMSS.json`
- Each benchmark runs `MAX_CASES` questions independently (e.g., 20 each).
- Each result item records:
  - `idx`
  - `query`
  - `gold` (if applicable)
  - `pred` (if applicable)
  - `passed`
  - `skipped`
  - `response`

## Notes
- If `content` is empty in OpenAI-compatible response, evaluator falls back to `reasoning_content`.
- Keep this project under `~/Desktop/model-tester/<model-name>/` as required by model-tester skill.
