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
| Model | Prompt Eval Time | Prompt Tokens | Prompt ms/token | Prompt tok/s | Eval Time | Eval Tokens | Eval ms/token | Eval tok/s | Total Time | Total Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5-27b | 275075.89 ms | 15057 | 18.27 | 54.74 | 28220.04 ms | 112 | 251.96 | 3.97 | 303295.93 ms | 15169 |
| glm-4.7-flash-fp4 | 138437.94 ms | 14785 | 9.36 | 106.80 | 6395.09 ms | 83 | 77.05 | 12.98 | 144833.02 ms | 14868 |
| lfm2-24b | 2354.28 ms | 511 | 4.61 | 217.05 | 4984.87 ms | 180 | 27.69 | 36.11 | 7339.15 ms | 691 |

### Accuracy

#### all
> Latest 100-question results per benchmark (5 benchmarks, total 500/model).
> Note: `gpt-oss-20b-gguf` / `qwen3.5_35b_a3b` / `glm-4.7-flash-fp4` are from the unified run on 2026-03-07; `lfm2-24b` is from the latest complete prior run (same benchmark set and question count).

| Model | MMLU | GSM8K | Geo MMLU High School | Law MMLU Professional | HumanEval | Total |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.5_35b_a3b | 83% (83/100) | 95% (95/100) | 95% (95/100) | 68% (68/100) | 92% (92/100) | **86.6% (433/500)** |
| gpt-oss-20b-gguf | 83% (83/100) | 94% (94/100) | 88% (88/100) | 51% (51/100) | 96% (96/100) | **82.4% (412/500)** |
| glm-4.7-flash-fp4 | 72% (72/100) | 92% (92/100) | 86% (86/100) | 54% (54/100) | 76% (76/100) | **76.0% (380/500)** |
| lfm2-24b | 74% (74/100) | 86% (86/100) | 81% (81/100) | 57% (57/100) | 80% (80/100) | **75.6% (378/500)** |

Run folders:
- `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_125451_fasttextgenevalset2_mmlu`
- `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_131727_fasttextgenevalset2_gsm8k`
- `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_133912_fasttextgenevalset2_geo-mmlu-high-school`
- `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_135147_fasttextgenevalset2_law-mmlu-professional`
- `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_145155_fasttextgenevalset2_humaneval`
- `/Users/yenhaohuang/Desktop/model-tester/glm-4.7-flash-fp4/runs/20260307_152832_fasttextgenevalset2_mmlu`
- `/Users/yenhaohuang/Desktop/model-tester/glm-4.7-flash-fp4/runs/20260307_152958_fasttextgenevalset2_gsm8k`
- `/Users/yenhaohuang/Desktop/model-tester/glm-4.7-flash-fp4/runs/20260307_154146_fasttextgenevalset2_geo-mmlu-high-school`
- `/Users/yenhaohuang/Desktop/model-tester/glm-4.7-flash-fp4/runs/20260307_154245_fasttextgenevalset2_law-mmlu-professional`
- `/Users/yenhaohuang/Desktop/model-tester/glm-4.7-flash-fp4/runs/20260307_154444_fasttextgenevalset2_humaneval`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5_35b_a3b/runs/20260307_155157_fasttextgenevalset2_mmlu`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5_35b_a3b/runs/20260307_155337_fasttextgenevalset2_gsm8k`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5_35b_a3b/runs/20260307_162211_fasttextgenevalset2_geo-mmlu-high-school`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5_35b_a3b/runs/20260307_162324_fasttextgenevalset2_law-mmlu-professional`
- `/Users/yenhaohuang/Desktop/model-tester/qwen3.5_35b_a3b/runs/20260307_162542_fasttextgenevalset2_humaneval`
- `/Users/yenhaohuang/Desktop/model-tester/lfm2-24b/runs/20260306_114503_fasttextgenevalset2_500q`

#### fast
| Model | MMLU | Geo MMLU High School | Law MMLU Professional | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| gpt-oss-20b-gguf | 90.0% (18/20) | 85.0% (17/20) | 55.0% (11/20) | 95.0% (19/20) | 95.0% (19/20) |
| qwen3.5-27b | 70.0% (14/20) | 90.0% (18/20) | 85.0% (17/20) | 90.0% (18/20) | 95.0% (19/20) |
| qwen3.5_35b_a3b | 55.0% (11/20) | 95.0% (19/20) | 85.0% (17/20) | 95.0% (19/20) | 95.0% (19/20) |
| glm-4.7-flash-fp4 | 40.0% (8/20) | 70.0% (14/20) | 45.0% (9/20) | 90.0% (18/20) | 95.0% (19/20) |
| lfm2-24b | 45.0% (9/20) | 80.0% (16/20) | 55.0% (11/20) | 90.0% (18/20) | 100.0% (20/20) |