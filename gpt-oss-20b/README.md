# gpt-oss-20b-dev

Lightweight development scaffold for testing and iterating `gpt-oss-20b` with Hugging Face.

## What this includes (v0)
- Dataset schema (`jsonl`)
- Prompt template management
- Model client wrapper (Hugging Face Inference API)
- Guardrail checks (basic)
- Evaluation runner + metrics output

## Quick start (HF API)

```bash
cd gpt-oss-20b-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# INFERENCE_BACKEND=hf_api
python -m src.gpt_oss20b_eval.runner \
  --dataset eval/datasets/smoke.jsonl \
  --prompt prompts/system_v1.txt \
  --out reports/smoke_report.json
```

## Quick start (Local Transformers)

```bash
cd gpt-oss-20b-dev
source .venv/bin/activate
# set in .env: INFERENCE_BACKEND=local, DEVICE=cpu (or mps/cuda:0)
python -m src.gpt_oss20b_eval.runner \
  --dataset eval/datasets/smoke.jsonl \
  --prompt prompts/system_v1.txt \
  --out reports/local_smoke_report.json
```

## Env vars
- `HF_TOKEN` (optional for public models; needed for gated/private access)
- `MODEL_NAME` (default: `openai/gpt-oss-20b`)
- `INFERENCE_BACKEND` (`hf_api` or `local`, default `hf_api`)
- `DEVICE` (`cpu` / `mps` / `cuda:0`, used in local mode)

## Dataset format
Each line is JSON:

```json
{
  "id": "q1",
  "category": "core",
  "prompt": "Explain what overfitting is in one paragraph.",
  "expected": "Optional expected answer or keywords",
  "checks": ["non_empty", "max_len_1200"]
}
```

## One-command runner

```bash
cd gpt-oss-20b-dev
chmod +x run.sh
./run.sh hf_api
./run.sh local
# custom: ./run.sh local eval/datasets/smoke.jsonl prompts/system_v1.txt reports/local.json
```

## Next milestones
1. Add pairwise A/B benchmark mode
2. Add async/concurrency load test
3. Add safety red-team suite
4. Add dashboard export (CSV + markdown)
```