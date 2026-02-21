# gpt-oss-20b-dev

`gpt-oss-20b-dev` 是一個用來快速測試與迭代 **gpt-oss-20b** 的輕量開發專案，支援 Hugging Face API 與本地推論。

## 專案介紹

這個專案主要提供：
- 評測資料集（JSONL）
- Prompt 模板管理
- 模型呼叫封裝（HF API / local）
- 基本檢查與報告輸出

適合用來做 smoke test、prompt 調整與基本評估流程驗證。

## 安裝

```bash
cd gpt-oss-20b-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

可在 `.env` 設定：
- `MODEL_NAME`（預設：`openai/gpt-oss-20b`）
- `INFERENCE_BACKEND`（`hf_api` 或 `local`）
- `HF_TOKEN`（使用私有/受限模型時需要）
- `DEVICE`（local 模式可用：`cpu`/`mps`/`cuda:0`）

## 使用方式

### 1) Hugging Face API

```bash
python -m src.gpt_oss20b_eval.runner \
  --dataset eval/datasets/smoke.jsonl \
  --prompt prompts/system_v1.txt \
  --out reports/smoke_report.json
```

> 請先確認 `.env` 內 `INFERENCE_BACKEND=hf_api`。

### 2) 本地 Transformers

```bash
python -m src.gpt_oss20b_eval.runner \
  --dataset eval/datasets/smoke.jsonl \
  --prompt prompts/system_v1.txt \
  --out reports/local_smoke_report.json
```

> 請先確認 `.env` 內 `INFERENCE_BACKEND=local`，並設定 `DEVICE`。

### 3) 一鍵執行腳本

```bash
chmod +x run.sh
./run.sh hf_api
./run.sh local
```

以上即可完成最基本的安裝與評測流程。