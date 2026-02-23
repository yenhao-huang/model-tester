# minmax-m2.5

新增 **local inference** 能力（`huggingface transformers.from_pretrained`）。

## 1) 建立環境

```bash
python3 -m venv ~/Desktop/python-venvs/minmax-m25
source ~/Desktop/python-venvs/minmax-m25/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) 執行本地推論

> 若模型需要自訂程式碼，請加上 `--trust-remote-code`。

```bash
python run_local_inference.py \
  --model MiniMaxAI/MiniMax-M2.5 \
  --prompt "用三點整理 MiniMax-M2.5 的定位" \
  --max-new-tokens 256 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --trust-remote-code \
  --save-json outputs/run-$(date +%Y%m%d-%H%M%S).json
```

## 3) 輸出與紀錄

- 終端機會印出生成結果
- `--save-json` 會保存：時間、模型、裝置、參數、prompt、output

## 4) 評估方式（my-textgen-evalset）

使用 `eval_my_textgen_evalset.py` 進行基準評估，預設會跑：
- MMLU
- GSM8K
- TruthfulQA

### 4.1 資料集位置

- `~/Desktop/datasets/my-textgen-evalset`
  - `mmlu/test.jsonl`
  - `gsm8k-main/test.jsonl`
  - `truthfulqa/validation.jsonl`

### 4.2 執行指令（每個 benchmark 取前 20 題）

```bash
source ~/Desktop/python-venvs/minmax-m25/bin/activate
MAX_CASES=20 python eval_my_textgen_evalset.py
```

### 4.3 速率限制設定（避免過快打 API）

目前腳本內建「問題級別」節流：
- 每 10 秒最多 4 個請求（跨 benchmark 共用）

可用環境變數覆蓋：

```bash
REQUESTS_PER_WINDOW=4 REQUEST_WINDOW_SEC=10 MAX_CASES=20 python eval_my_textgen_evalset.py
```

### 4.4 評分與 skipped 規則

- 報告欄位：`total`、`scored_total`、`skipped`、`correct`、`accuracy`
- 若遇到 `402 Payment Required`（HF credits 不足）：
  - 該題標記為 `skipped=true`
  - **不納入 accuracy 分母**（即不算在 `scored_total`）
- 其他錯誤會計入 `errors`

### 4.5 輸出位置

評估結果會輸出到：

- `reports/my_textgen_evalset_YYYYMMDD_HHMMSS.json`

可從 `meta` 查看本次參數（model、max_cases、request_rate_limit 等）。

## 5) 已知限制

- 模型體積可能很大，首次下載時間長
- 本機記憶體/VRAM 不足時會 OOM
- 某些模型必須 `trust_remote_code=True`
- Apple Silicon（MPS）精度與速度可能與 CUDA 不同
- 若 HF credits 不足會大量出現 402，導致 `skipped` 偏高、有效樣本偏少
