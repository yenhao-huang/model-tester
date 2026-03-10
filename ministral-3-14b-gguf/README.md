# Ministral-3-14B-Instruct-2512 (GGUF Q8_K_XL) 測試紀錄

- Model path: `~/Desktop/models/ministral-3-14b-gguf/Ministral-3-14B-Instruct-2512-UD-Q8_K_XL.gguf`
- 啟動腳本: `~/Desktop/llama-bash/ministral314b.sh`
- 實際啟動命令:
  - `bash ~/Desktop/llama-bash/ministral314b.sh`
  - 內部轉呼叫 `~/Desktop/llama-bash/run_ministral314b.sh`（port `3172`）

## full-textgen-evalset（每 benchmark 100 題，final）

| Benchmark | Result | Status |
|---|---:|---|
| mmlu | 48.0% (48/100) | ✅ 完成 |
| geo-mmlu-high-school | 88.0% (88/100) | ✅ 完成 |
| law-mmlu-professional | 48.0% (48/100) | ✅ 完成 |
| gsm8k | 2.0% (2/100) | ✅ 完成 |
| humaneval (rerun) | 88.0% (88/100) | ✅ 完成 |

輸出檔案（`reports_100_rerun`）：
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100_rerun/fast_textgen_eval_20260310_054415_mmlu.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100_rerun/fast_textgen_eval_20260310_054415_geo-mmlu-high-school.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100_rerun/fast_textgen_eval_20260310_054415_law-mmlu-professional.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100_rerun/fast_textgen_eval_20260310_062817_gsm8k.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100_rerun/fast_textgen_eval_20260310_090157_humaneval.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100_rerun/fast_textgen_eval_20260310_090157_all.json`

> 每題輸出已包含 `prompt`（query）、`response`、`error`（成功時通常為 `null`）。

## 速度樣本（llama-server）

固定請求：`/v1/chat/completions`, `temperature=0.0`, `max_tokens=180`

```text
prompt eval time =    3969.39 ms /   540 tokens (    7.35 ms per token,   136.04 tokens per second)
       eval time =    4488.20 ms /    31 tokens (  144.78 ms per token,     6.91 tokens per second)
      total time =    8457.58 ms /   571 tokens
```

- 原始回應: `~/Desktop/model-tester/ministral-3-14b-gguf/speed_sample_response.json`
- 指標摘錄: `~/Desktop/model-tester/ministral-3-14b-gguf/speed_metrics.txt`

## 備註

1. evaluator HTTP timeout 已設為 180 秒（`utils/eval_fast_textgen_eval.py`）。
2. HumanEval 單題執行有 timeout 保護（`signal.alarm(3)`），避免模型回傳程式碼造成無限執行。
