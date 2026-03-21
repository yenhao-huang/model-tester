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
| MMLU | 通識 | 字母完全匹配 |
| GSM8K | 數學題 (高中) | 最後數字匹配 |
| HumanEval | 程市題 | `exec()` 執行測試通過 |
| geo-mmlu| 地理 | x |
| law-mmlu | 法律 | x |

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
device; mac-mini m4 chip
| Model | Prompt Eval Time (ms) | Prompt Tokens | Prompt ms/token | Prompt tok/s | Eval Time (ms) | Eval Tokens | Eval ms/token | Eval tok/s | Total Time | Total Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5-27b | 275075.89  | 15057 | 18.27 | 54.74 | 28220.04  | 112 | 251.96 | 3.97 | 303295.93  | 15169 |
| glm-4.7-flash-fp4 | 138437.94  | 14785 | 9.36 | 106.80 | 6395.09  | 83 | 77.05 | 12.98 | 144833.02  | 14868 |
| lfm2-24b | 2354.28  | 511 | 4.61 | 217.05 | 4984.87  | 180 | 27.69 | 36.11 | 7339.15  | 691 |
| gemma-3-27b-gguf | 2093.85 | 114 | 18.37 | 54.45 | 34892.32 | 180 | 193.85 | 5.16 | 36986.17 | 294 |
| ministral-3-14b-gguf | 3969.39 | 540 | 7.35 | 136.04 | 4488.20 | 31 | 144.78 | 6.91 | 8457.58 | 571 |

### Accuracy

#### Text (all)
> Latest 100-question results per benchmark (5 benchmarks, total 500/model).
> Note: `gpt-oss-20b-gguf` / `qwen3.5-35b-a3b` / `glm-4.7-flash-fp4` are from the unified run on 2026-03-07; `lfm2-24b` is from the latest complete prior run (same benchmark set and question count).

| Model | MMLU | GSM8K | Geo MMLU High School | Law MMLU Professional | HumanEval | Total |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.5-35b-a3b | 83% (83/100) | 95% (95/100) | 95% (95/100) | 68% (68/100) | 92% (92/100) | **86.6% (433/500)** |
| gpt-oss-20b-gguf | 83% (83/100) | 94% (94/100) | 88% (88/100) | 51% (51/100) | 96% (96/100) | **82.4% (412/500)** |
| glm-4.7-flash-fp4 | 72% (72/100) | 92% (92/100) | 86% (86/100) | 54% (54/100) | 76% (76/100) | **76.0% (380/500)** |
| lfm2-24b | 74% (74/100) | 86% (86/100) | 81% (81/100) | 57% (57/100) | 80% (80/100) | **75.6% (378/500)** |
| gemma-3-27b-gguf | 79% (79/100) | 76% (76/100) | 86% (86/100) | 57% (57/100) | 63% (63/100) | **72.2% (361/500)** |
| ministral-3-14b-gguf | 48% (48/100) | 2% (2/100) | 88% (88/100) | 48% (48/100) | 88% (88/100) | **54.8% (274/500)** |

#### Vision (100 題/任務)
> Tasks: OCR (PokemonCards), Classification (CIFAR10), Detection (CPPE5)

| Model | OCR | Classification | Detection | Avg (3 tasks) | Result Path |
|---|---:|---:|---:|---:|---|
| qwen3.5-27b | 89% (89/100) | 79% (79/100) | 60% (60/100) | **76.0%** | `qwen3.5-27b/vision/results/summary_100.json` |
| qwen3.5-35b-a3b | 92% (92/100) | 70% (70/100) | 59% (59/100) | **73.7%** | `qwen3.5-35b-a3b/vision/results-rerun-20260321_225509/summary_100.json` |
| ministral-3-14b-gguf | 89% (89/100) | 82% (82/100) | 51% (51/100) | **74.0%** | `ministral-3-14b-gguf/vision/results/summary_100.json` |

Run folders:
- `/Users/yenhaohuang/Desktop/model-tester/results/`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5-27b/vision/results/`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5-35b-a3b/vision/results-rerun-20260321_225509/`
