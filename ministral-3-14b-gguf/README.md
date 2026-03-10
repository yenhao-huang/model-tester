# Ministral-3-14B-Instruct-2512 (GGUF Q8_K_XL) 測試紀錄

- Model path: `~/Desktop/models/ministral-3-14b-gguf/Ministral-3-14B-Instruct-2512-UD-Q8_K_XL.gguf`
- 啟動腳本: `~/Desktop/llama-bash/ministral314b.sh`
- 實際啟動命令:
  - `bash ~/Desktop/llama-bash/ministral314b.sh`
  - 內部轉呼叫 `~/Desktop/llama-bash/run_ministral314b.sh`（port `3172`）

## fast-textgen-evalset（每 benchmark 100 題）

| Benchmark | Result |
|---|---:|
| mmlu | 48.0% (48/100) |
| geo-mmlu-high-school | 88.0% (88/100) |
| law-mmlu-professional | 48.0% (48/100) |
| gsm8k | 未完成（長時間執行，尚未產出 100 題最終檔） |
| humaneval | 未完成（等待 gsm8k 完成後才會開始） |

輸出檔案：
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100/fast_textgen_eval_20260310_034611_mmlu.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100/fast_textgen_eval_20260310_034611_geo-mmlu-high-school.json`
- `~/Desktop/model-tester/ministral-3-14b-gguf/reports_100/fast_textgen_eval_20260310_034611_law-mmlu-professional.json`

> 每題輸出已包含 `prompt`（query）、`response`、`error`（成功時為 `null`）。

## 速度樣本（llama-server）

固定請求：`/v1/chat/completions`, `temperature=0.0`, `max_tokens=180`

```text
prompt eval time =    3969.39 ms /   540 tokens (    7.35 ms per token,   136.04 tokens per second)
       eval time =    4488.20 ms /    31 tokens (  144.78 ms per token,     6.91 tokens per second)
      total time =    8457.58 ms /   571 tokens
```

- 原始回應: `~/Desktop/model-tester/ministral-3-14b-gguf/speed_sample_response.json`
- 指標摘錄: `~/Desktop/model-tester/ministral-3-14b-gguf/speed_metrics.txt`

## 已知限制

1. 在本機 CPU llama-server 配置下，`gsm8k` 單題耗時過長；100 題完整跑完需要數小時，容易被 timeout/中斷。
2. `humaneval` 排在 `gsm8k` 後面執行，因此在同一輪 full-run 中尚未開始。
3. 為避免長時間卡住，已把 evaluator 的 HTTP timeout 由 120 秒調整為 10 秒：
   - `~/Desktop/model-tester/utils/eval_fast_textgen_eval.py`
