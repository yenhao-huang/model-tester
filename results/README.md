# 4 Models × 5 Benchmarks (100 questions each)
Generated at: 2026-03-08T09:52:11.020554

## Folder structure
- One folder per model
- Inside each model: one folder per benchmark
- Each benchmark folder contains:
  - `predictions.csv` (clear tabular predictions)
  - `predictions.md` (readable sample + stats)
  - `raw_*.json` (original evaluator output)

## Note
- `gpt-oss-20b`, `glm4.7-flash`, `qwen3.5-35b-a3b` come from the latest unified 5-benchmark batch.
- `lfm2-24b` comes from the latest complete prior run (same 5 benchmarks, 100 questions each), not the same timestamped batch.

## gpt-oss-20b
- MMLU: **83.0%** (83/100)  \n  - folder: `gpt-oss-20b/MMLU__mmlu`
- GSM8K: **94.0%** (94/100)  \n  - folder: `gpt-oss-20b/GSM8K__gsm8k`
- geo-mmlu: **88.0%** (88/100)  \n  - folder: `gpt-oss-20b/geo-mmlu__geo-mmlu-high-school`
- law-mmlu: **51.0%** (51/100)  \n  - folder: `gpt-oss-20b/law-mmlu__law-mmlu-professional`
- HumanEval: **96.0%** (96/100)  \n  - folder: `gpt-oss-20b/HumanEval__humaneval`

## glm4.7-flash
- MMLU: **72.0%** (72/100)  \n  - folder: `glm4.7-flash/MMLU__mmlu`
- GSM8K: **92.0%** (92/100)  \n  - folder: `glm4.7-flash/GSM8K__gsm8k`
- geo-mmlu: **86.0%** (86/100)  \n  - folder: `glm4.7-flash/geo-mmlu__geo-mmlu-high-school`
- law-mmlu: **54.0%** (54/100)  \n  - folder: `glm4.7-flash/law-mmlu__law-mmlu-professional`
- HumanEval: **76.0%** (76/100)  \n  - folder: `glm4.7-flash/HumanEval__humaneval`

## qwen3.5-35b-a3b
- MMLU: **83.0%** (83/100)  \n  - folder: `qwen3.5-35b-a3b/MMLU__mmlu`
- GSM8K: **95.0%** (95/100)  \n  - folder: `qwen3.5-35b-a3b/GSM8K__gsm8k`
- geo-mmlu: **95.0%** (95/100)  \n  - folder: `qwen3.5-35b-a3b/geo-mmlu__geo-mmlu-high-school`
- law-mmlu: **68.0%** (68/100)  \n  - folder: `qwen3.5-35b-a3b/law-mmlu__law-mmlu-professional`
- HumanEval: **92.0%** (92/100)  \n  - folder: `qwen3.5-35b-a3b/HumanEval__humaneval`

## lfm2-24b
- MMLU: **74.0%** (74/100)  \n  - folder: `lfm2-24b/MMLU__mmlu`
- GSM8K: **86.0%** (86/100)  \n  - folder: `lfm2-24b/GSM8K__gsm8k`
- geo-mmlu: **81.0%** (81/100)  \n  - folder: `lfm2-24b/geo-mmlu__geo-mmlu-high-school`
- law-mmlu: **57.0%** (57/100)  \n  - folder: `lfm2-24b/law-mmlu__law-mmlu-professional`
- HumanEval: **80.0%** (80/100)  \n  - folder: `lfm2-24b/HumanEval__humaneval`

## Average score (simple mean of 5 benchmarks)
1. **qwen3.5-35b-a3b**: 86.60% (N=5)
2. **gpt-oss-20b**: 82.40% (N=5)
3. **glm4.7-flash**: 76.00% (N=5)
4. **lfm2-24b**: 75.60% (N=5)
