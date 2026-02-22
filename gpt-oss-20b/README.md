# gpt-oss-20b-dev

`gpt-oss-20b-dev` 是一個用來快速測試與迭代 **gpt-oss-20b** 的輕量開發專案，支援 Hugging Face API 與本地推論（`AutoTokenizer/AutoModelForCausalLM.from_pretrained`）。

## 專案介紹

這個專案主要提供：
- 評測資料集（JSONL）
- Prompt 模板管理
- 模型呼叫封裝（HF API / local）
- 基本檢查與報告輸出

適合用來做 smoke test、prompt 調整與基本評估流程驗證。

## 安裝

```bash
cd gpt-oss-20b
python3 -m venv ../python-venvs/.gpt-oss
source ../python-venvs/gpt-oss/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

可在 `.env` 設定：
- `MODEL_NAME`（預設：`openai/gpt-oss-20b`）
- `LOCAL_MODEL_PATH`（local 模式優先使用；例如 `/Users/yenhaohuang/Desktop/models/llama3-8b`）
- `INFERENCE_BACKEND`（`hf_api` 或 `local`）
- `HF_TOKEN`（使用私有/受限模型時需要）
- `DEVICE`（local 模式可用：`cpu`/`mps`/`cuda:0`）
- `TOP_P`（local 生成可選，預設 `0.9`）

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
# 1) 先下載到本機
huggingface-cli download <model-name> \
  --local-dir ~/Desktop/models/<model-name-abbr> \
  --local-dir-use-symlinks False

# 2) 跑評測
python -m src.gpt_oss20b_eval.runner \
  --dataset eval/datasets/smoke.jsonl \
  --prompt prompts/system_v1.txt \
  --out reports/local_smoke_report.json
```
> 請先確認 `.env` 內 `INFERENCE_BACKEND=local`。

### 3) 一鍵執行腳本

```bash
chmod +x run.sh
./run.sh hf_api
./run.sh local
```

### 4) 互動式問答（Interactive Chat）

```bash
# 直接啟動互動問答（backend 可選 hf_api / local）
./run.sh chat hf_api prompts/system_v1.txt
# 或
python -m src.gpt_oss20b_eval.chat --prompt prompts/system_v1.txt
```

互動模式支援指令：
- `/exit` 或 `/quit`：離開
- `/reset`：清空當前對話歷史（保留 system prompt）
- `/history`：查看目前 turn 數

以上即可完成最基本的安裝、評測與互動測試流程。

## 評估結果

> 評估規則（共用）：`402 Payment Required` 視為配額/計費限制，標記為 `skipped`，不納入 accuracy 分母。

### HumanEval（Random sample 20 題）
- Report: `reports/humaneval_sample20_20260222_222029.json`
- Total: `20`
- Scored total: `16`
- Skipped (402): `4`
- Correct: `16`
- Accuracy: `100.0%`
- Syntax repaired and passed: `11`

### GSM8K
- Report: `reports/gsm8k_20260222_224244.json`
- Dataset: `~/Desktop/datasets/common-text-gen-evalset/math-reasoning/gsm8k/test.jsonl`
- Total: `50`
- Scored total: `15`
- Skipped (402): `35`
- Errors (non-402): `0`
- Correct: `14`
- Accuracy: `93.3%`
- Elapsed: `26.9s`

## 評估過程
### code-generation: HumanEval
```py
data = load("eval/humaneval.jsonl")
model_output = llm(data["prompt"])
full_code = data["prompt_code"] + model_output
exec(full_code, namespace)     # 定義 function
exec(test_code, namespace)     # 定義 check()

namespace["check"](namespace["has_close_elements"])
```