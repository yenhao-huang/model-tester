# model-tester

這個目錄用來測試各種模型，尤其是：
- 熱門模型（社群常用、討論度高）
- 新模型（剛發布、值得快速驗證）

## 目的
- 快速建立測試環境
- 驗證模型基本能力與穩定性
- 比較不同模型在相同任務上的表現

## 建議做法
- 每個模型使用獨立子目錄
- 保留最小可重現的測試流程（安裝、執行、輸出）
- 把關鍵結果整理在各模型目錄的 README

## Eval Pipeline

所有 text-gen 模型統一用 `utils/eval_fast_textgen_eval.py` 評估。

### Benchmarks
| Benchmark | 題型 | 評分方式 |
|---|---|---|
| MMLU | 數學題 (大學) | 字母完全匹配 |
| GSM8K | 數學題 (高中) | 最後數字匹配 |
| HumanEval | 程市題 | `exec()` 執行測試通過 |

### 快速使用
```bash
# 本地 llama-server / ollama（預設 http://127.0.0.1:8080/v1）
python utils/eval_fast_textgen_eval.py --model <model-name>

# 指定 endpoint
python utils/eval_fast_textgen_eval.py \
    --model qwen3.5 \
    --base-url http://127.0.0.1:3172/v1

# 只跑部分 benchmark
python utils/eval_fast_textgen_eval.py --model qwen3.5 --benchmarks mmlu gsm8k

# 遠端 API
python utils/eval_fast_textgen_eval.py \
    --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o
```

### 全部參數
| 參數 | 預設值 | 說明 |
|---|---|---|
| `--model` | **(必填)** | 傳給 API 的 model name |
| `--base-url` | `http://127.0.0.1:8080/v1` | OpenAI-compatible endpoint |
| `--api-key` | `dummy` | API key |
| `--benchmarks` | 全部 | 選擇要跑的 benchmark |
| `--n` | `20` | 每個 benchmark 的題數上限 |
| `--max-tokens` | `4096` | 每次推理的 max tokens |
| `--dataset-root` | `~/Desktop/datasets/fast-textgen-evalset` | 資料集根目錄 |
| `--out-dir` | `<repo>/reports` | 報告輸出目錄 |

Report 自動存為 `<out-dir>/fast_textgen_eval_<timestamp>.json`。ß

## Results
### Speed
qwen3.5-27b
--- results ---
prompt eval time =  275075.89 ms / 15057 tokens (   18.27 ms per token,    54.74 tokens per second)
       eval time =   28220.04 ms /   112 tokens (  251.96 ms per token,     3.97 tokens per second)
      total time =  303295.93 ms / 15169 tokens

glm-4.7-flash-fp4
--- results ---
prompt eval time =  138437.94 ms / 14785 tokens (    9.36 ms per token,   106.80 tokens per second)
       eval time =    6395.09 ms /    83 tokens (   77.05 ms per token,    12.98 tokens per second)
      total time =  144833.02 ms / 14868 tokens


### Accuracy
gpt-oss-20b-gguf
--- results ---
  mmlu            90.0%  (18/20)
  geo-mmlu-high-school    85.0%  (17/20)
  law-mmlu-professional    55.0%  (11/20)
  gsm8k           95.0%  (19/20)
  humaneval       95.0%  (19/20)


qwen3.5-27b
--- results ---
  mmlu            70.0%  (14/20)
  geo-mmlu-high-school    90.0%  (18/20)
  law-mmlu-professional    85.0%  (17/20)
  gsm8k           90.0%  (18/20)
  humaneval       95.0%  (19/20)


glm-4.7-flash-fp4
--- results ---
  mmlu            40.0%  (8/20)
  geo-mmlu-high-school    70.0%  (14/20)
  law-mmlu-professional    45.0%  (9/20)
  gsm8k           90.0%  (18/20)ß
  humaneval       95.0%  (19/20)