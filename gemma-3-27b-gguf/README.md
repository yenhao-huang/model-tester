# Gemma-3-27B-IT (GGUF Q4_1) 測試紀錄

- Model path: `~/Desktop/models/gemma3-gguf/gemma-3-27b-it-Q4_1.gguf`
- 啟動腳本: `~/Desktop/llama-bash/gemma27b.sh`
- 實際啟動命令:
  - `bash ~/Desktop/llama-bash/gemma27b.sh`
  - 內部轉呼叫 `~/Desktop/llama-bash/run_gemma27b.sh`（port `3172`）

## README benchmarks（fast-textgen-evalset，每項 20 題）

| Benchmark | Result |
|---|---:|
| mmlu | 75.0% (15/20) |
| geo-mmlu-high-school | 85.0% (17/20) |
| law-mmlu-professional | 55.0% (11/20) |
| gsm8k | blocked（目前執行流程卡住，尚未產出最終檔） |
| humaneval | blocked（目前執行流程卡住，尚未產出最終檔） |

輸出檔案：
- `~/Desktop/model-tester/gemma-3-27b-gguf/reports/fast_textgen_eval_20260309_013236_mmlu.json`
- `~/Desktop/model-tester/gemma-3-27b-gguf/reports/fast_textgen_eval_20260309_013236_geo-mmlu-high-school.json`
- `~/Desktop/model-tester/gemma-3-27b-gguf/reports/fast_textgen_eval_20260309_013236_law-mmlu-professional.json`

## 速度評估（llama-server）

固定請求：`/v1/chat/completions`, `temperature=0.0`, `max_tokens=180`

原始 log（三行）：

```text
prompt eval time =    2093.85 ms /   114 tokens (   18.37 ms per token,    54.45 tokens per second)
       eval time =   34892.32 ms /   180 tokens (  193.85 ms per token,     5.16 tokens per second)
      total time =   36986.17 ms /   294 tokens
```

## 已知限制

1. 使用目前 `utils/eval_fast_textgen_eval.py` 跑 `gsm8k` / `humaneval` 時，流程會卡在長時間請求，尚未完成 20 題輸出。
2. `max_tokens` 預設較大時（4096）在 27B 模型上單題耗時偏長；即使降低後仍有卡住情況，需要在 evaluator 增加更嚴格 timeout / retry 才能穩定跑完。
