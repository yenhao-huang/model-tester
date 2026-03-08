# LFM-2 (lfm2-24b GGUF) Fast TextGen Eval Report

## Setup
- Project dir: `~/Desktop/model-tester/lfm-2`
- Python env: `~/Desktop/python-venvs/lfm2`
- Model path: `~/Desktop/models/lfm2-24b/LFM2-24B-A2B-Q8_0.gguf`
- Dataset: `~/Desktop/datasets/fast-textgen-evalset`

Create env:
```bash
python3 -m venv ~/Desktop/python-venvs/lfm2
source ~/Desktop/python-venvs/lfm2/bin/activate
```

## Model Launch
- Launch script: `~/Desktop/llama-bash/lfm2.sh`
- Start command:
```bash
~/Desktop/llama-bash/lfm2.sh
```
- Ready check:
```bash
curl http://127.0.0.1:3176/health
```
Proceed when response contains `{"status":"ok"}`.

## Benchmark Command
```bash
source ~/Desktop/python-venvs/lfm2/bin/activate
python ~/Desktop/model-tester/utils/eval_fast_textgen_eval.py \
  --model lfm2-24b \
  --base-url http://127.0.0.1:3176/v1 \
  --api-key dummy \
  --dataset-root ~/Desktop/datasets/fast-textgen-evalset \
  --out-dir ~/Desktop/model-tester/lfm-2/reports \
  --n 20
```

## Results (2026-03-05)
- mmlu: **45.0%** (9/20)
- geo-mmlu-high-school: **80.0%** (16/20)
- law-mmlu-professional: **55.0%** (11/20)
- gsm8k: **90.0%** (18/20)
- humaneval: **100.0%** (20/20)

Scoring accounting:
- total: 100
- scored_total: 100
- skipped: 0

## Artifacts
- Consolidated: `reports/fast_textgen_eval_20260305_023220_all.json`
- Per benchmark JSON files under `reports/`
- Each item includes: `idx`, `query`, `response`, `pred`, `passed`, `skipped` (and benchmark-specific fields)

## Known Limitations
- This run is local GGUF inference (`llama-server`) and depends on quantization + runtime params, so results are not directly equivalent to cloud FP16 baselines.
- HumanEval includes automatic indentation repair in the evaluator; repaired passes are still counted as pass.
