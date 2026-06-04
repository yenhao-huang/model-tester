# Model Tester

這個目錄用來測試各種模型，尤其是：
- 熱門模型（社群常用、討論度高）
- 新模型（剛發布、值得快速驗證）

**目的**
- 快速建立測試環境
- 驗證模型基本能力與穩定性
- 比較不同模型在相同任務上的表現

**建議做法**
- 每個模型使用獨立子目錄
- 保留最小可重現的測試流程（安裝、執行、輸出）
- 把關鍵結果整理在各模型目錄的 README

## Text Benchmark

目前 text 評測固定以下任務（各 100 題）：

| Benchmark | 題型 | 評分方式 |
|---|---|---|
| MMLU | 通識 | 字母完全匹配 |
| GSM8K | 數學題 (高中) | 最後數字匹配 |
| HumanEval | 程式題 | `exec()` 執行測試通過 |
| geo-mmlu | 地理 | 字母完全匹配 |
| law-mmlu | 法律 | 字母完全匹配 |

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

**全部參數**
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

Report 自動存為 `<out-dir>/fast_textgen_eval_<timestamp>.json`。

## Vision Benchmark

目前 vision 評測固定三個任務（各 100 題）：

| 任務 | 資料集 | 評分方式 |
|---|---|---|
| OCR | `PokemonCards_train_300` | 判斷卡片文字（name/hp）是否命中 |
| Classification | `cifar10_classification` | 10 類單標籤分類 |
| Detection | `object_detection`（CPPE5）| PPE 類別集合完全匹配 |

資料集預設路徑：`~/Desktop/datasets/vision-dataset/<dataset-name>`

### 快速使用

#### 1) 啟動模型（OpenAI-compatible + mmproj）
```bash
llama-server \
  -m <model.gguf> \
  --mmproj <mmproj.gguf> \
  --port 3172
```

#### 2) 跑三項 benchmark（各 100 題）
```bash
~/Desktop/python-venvs/qwen35vision/bin/python ~/Desktop/model-tester/utils/eval_vision.py \
  --api http://127.0.0.1:3172/v1/chat/completions \
  --model <model-name> \
  --num 100 \
  --tasks ocr,classification,detection \
  --out-dir ~/Desktop/model-tester/<model-name>/vision/results
```

#### 3) 看結果
- 彙總：`summary_100.json`
- 逐題：`ocr_*.jsonl` / `classification_*.jsonl` / `detection_*.jsonl`
- 範例圖與回覆：`examples/<task>/`

## Results

### Speed
Device: Mac-Mini M4 Chip

| Model | Prompt ms/token | Prompt token/s | Eval ms/token | Eval token/s | Alpaca eval tok/s |
|---|---:|---:|---:|---:|---:|
| lfm2.5-vl-1.6b | 0.79 | 1266.33 | 24.19 | 41.33 | — |
| lfm2-24b | 4.61 | 217.05 | 27.69 | 36.11 | — |
| gemma-e4b (no think) | 4.64 | 215.50 | 34.11 | 29.32 | — |
| qwen3.6-35b-a3b (no think) | 4.74 | 227.20 | 40.74 | 24.73 | — |
| gemma-4-26b-a4b-gguf | 8.97 | 111.46 | 40.07 | 24.96 | — |
| qwen3.6-35b-a3b-mtp (no think, draft-mtp n=6) | 14.85 | 67.35 | 69.23 | 14.44 | — |
| qwen3.6-35b-a3b-mtp | pending | pending | pending | pending | — |
| gemma-4-26b-a4b-mtp | pending | pending | pending | pending | — |
| qwen3.5-35b-a3b | 6.05 | 165.15 | 53.54 | 18.68 | — |
| gemma-3-12b-it-gguf | 8.90 | 112.40 | 76.35 | 13.10 | — |
| glm-4.7-flash-fp4 | 9.36 | 106.80 | 77.05 | 12.98 | — |
| ministral-3-14b-gguf | 7.35 | 136.04 | 144.78 | 6.91 | — |
| qwen3.6-27b-mtp (no think, draft-mtp n=1) | — | — | — | — | **6.42** |
| gemma-4-31b (no think) | 39.30 | 29.44 | 179.40 | 5.63 | — |
| qwen3.6-27b (no think) | 21.13 | 47.34 | 182.99 | 5.46 | **5.87** |
| qwen3.6-27b-mtp (no think, draft-mtp n=6) | 55.72 | 17.95 | 362.87 | 2.76 | — |
| gemma-3-27b-gguf | 18.37 | 54.45 | 193.85 | 5.16 | — |
| qwen3.5-27b | 18.27 | 54.74 | 251.96 | 3.97 | — |

> 備註：`gemma-4-31b (no think)`、`qwen3.6-35b-a3b (no think)` 與 `qwen3.6-27b (no think)` 速度皆來自 llama-server log 的 token 加權平均。`qwen3.6-27b (no think)` 使用 `qwen36_27b_server.log` 中本輪 Text-Englsih 500 題對應的 timing pair。
>
> 備註（2026-05-19）：`qwen3.6-35b-a3b-mtp (no think, draft-mtp n=6)` 來源為 `qwen3.6-35b-a3b-mtp/results/speed_20260519_120520.jsonl`；3 samples 平均 eval `14.44 tok/s`，在此 Mac / llama.cpp build 上低於 non-MTP baseline `24.73 tok/s`。`qwen3.6-27b-mtp (no think, draft-mtp n=6)` 來源為 `qwen3.6-27b-mtp/results/speed_20260519_160115.jsonl`；3 samples 平均 eval `2.76 tok/s`，低於 non-MTP baseline `5.46 tok/s`。`gemma-4-26b-a4b-mtp` draft 權重已下載並 checksum 驗證，但 llama.cpp 載入失敗：`unknown model architecture: 'gemma4_assistant'`，未列入速度表。
>
> 2026-05-19 MTP run: `qwen3.6-35b-a3b-mtp` and `gemma-4-26b-a4b-mtp` launch scripts are prepared and llama.cpp MTP flags were verified in `~/Desktop/others/llama.cpp/build-mtp-noui/bin/llama-server`, but timing is pending because required MTP GGUF downloads are still incomplete. See each model README and `speed/logs/` for source notes.


#### Alpaca speed — MTP evaluation

Dataset: `yahma/alpaca-cleaned` (`~/Desktop/datasets/alpaca-cleaned/alpaca_data_cleaned.json`), `N=50`, `MAX_TOKENS=128`, seed `42`, temperature `0`, no-think (`--reasoning off`; no `<think>` observed).

| Model | Variant | Eval tok/s avg | Eval tok/s median | Eval tok/s p95 | Prompt tok/s avg | Draft acceptance | Errors | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3.6-27b-mtp | draft-mtp n=1 | **6.42** | 6.37 | 7.01 | 36.87 | 83.6% | 0/50 | `qwen3.6-27b-mtp/results/alpaca_speed_mtp_n1_20260519_175549/alpaca_speed_20260519_175558_summary.json` |
| qwen3.6-27b | non-MTP baseline | 5.87 | 5.54 | 6.97 | 38.93 | — | 0/50 | `qwen3.6-27b/results/alpaca_speed_no_mtp_20260519_184507/alpaca_speed_20260519_184516_summary.json` |

MTP `draft-mtp n=1` averaged **1.09x** the non-MTP baseline on Alpaca generation speed (`6.42` vs `5.87` eval tok/s). Source index: `results/alpaca_speed_sources_20260519.json`; MTP-only source index: `results/qwen3.6-27b-mtp_alpaca_speed_sources.json`.

### Accuracy

#### Text-Englsih (100 題/任務)

| Model | MMLU | GSM8K | Geo MMLU HS | Law MMLU Prof | HumanEval | Total |
|---|---:|---:|---:|---:|---:|---:|
| gemma-4-31b (thinking) | 96% | 89% | 92% | 80% | 92% | **89.8%** |
| gemma-4-31b (no think) | 83% | **98%** | 89% | 80% | 97% | **89.4%** |
| gemma-4-26b-a4b (no think) | 91% | 93% | 91% | 70% | **99%** | **88.8%** |
| qwen3.5-27b (no think) | 79% | **97%** | 91% | 75% | 95% | **87.4%** |
| qwen3.5-35b-a3b (thinking) | 83% | 95% | 95% | 68% | 92% | **86.6%** |
| qwen3.6-27b (no think) | 65% | **97%** | 95% | 73% | 94% | **84.8%** |
| qwen3.6-35b-a3b (no think) | 60% | **98%** | 92% | 71% | 98% | **83.8%** |
| qwen3.5-35b-a3b (no think) | 63% | **98%** | 93% | 68% | 92% | **82.8%** |
| gpt-oss-20b | 83% | 94% | 88% | 51% | 96% | **82.4%** |
| glm-4.7-flash-fp4 | 72% | 92% | 86% | 54% | 76% | **76.0%** |
| lfm2-24b | 74% | 86% | 81% | 57% | 80% | **75.6%** |
| omnicoder-9b-gguf | 57% | 86% | 78% | 62% | 84% | **73.4%** |
| gemma-e4b (no think) | 46% | 88% | 83% | 51% | 93% | **72.2%** |
| gemma-3-27b-gguf | 79% | 76% | 86% | 57% | 63% | **72.2%** |
| gemma-3-12b-it-gguf | 33% | 79% | 81% | 38% | 89% | **64.0%** |
| ministral-3-14b-gguf | 48% | 2% | 88% | 48% | 88% | **54.8%** |
| lfm2.5-vl-1.6b | 51% | 65% | 52% | 28% | 56% | **50.4%** |

> 備註（2026-04-20）：`gemma-4-31b (no think)` 來自 `gemma-4-31b-gguf/text/results/no_think_20260420_151309/`，五項 text benchmark 皆為 100 題。
>
> 備註（2026-04-21）：`qwen3.6-35b-a3b (no think)` 來自 `qwen3.6-35b-a3b/results/no_think_20260421_153833/`，五項 text benchmark 皆為 100 題；此輪需使用 **server nohup + eval nohup** 才能避免 OpenClaw session 中斷影響結果。
>
> 備註（2026-04-30）：`qwen3.6-27b (no think)` 來自 `qwen3.6-27b/results/no_think_20260430_101759/`，五項 Text-Englsih benchmark 皆為 100 題。
>
> 備註（2026-05-21）：`gemma-e4b (no think)` 來自 `gemma-e4b/text/results/text_eval_20260521_182757/fast_textgen_eval_20260521_110507_all.json`；source index 為 `results/gemma-e4b_text_sources.json`，五項 Text-English benchmark 皆為 100 題。速度來源為 `gemma-e4b/logs/speed_metrics_20260521_182757.txt`。
>
> 備註（2026-06-04）：`gemma-3-12b-it-gguf` 來自 `gemma-3-12b-it-gguf/text/results/full_dataset_20260604_155307/fast_textgen_eval_20260604_075307_all.json`；source index 為 `results/gemma-3-12b-it-gguf_text_sources.json`，五項 Text-English benchmark 皆為 100 題。速度來源為 `gemma-3-12b-it-gguf/text/results/speed_llama_log_tail.txt`。

#### Text-Arch (100 題/任務)

| Benchmark | Model | Accuracy | Correct/Total | Result Path | Notes |
|---|---|---:|---:|---|---|
| OpsEval | qwen3.6-27b (no think) | **70.0%** | 70/100 | `qwen3.6-27b/results/chinese_question_eval_20260430_113155/chinese_question_eval_20260430_052413_all.json` | 選項字母 exact match；少選、多選、錯選都算錯。 |
| OpsEval | gemma4-31b (no think) | **65.0%** | 65/100 | `gemma-4-31b-gguf/text/results/opseval_20260501_092600/chinese_question_eval_20260501_012711_all.json` | 以串行 llama-server queue 執行；為避免記憶體不足，獨立接續跑。 |
| OpsEval | gemma4-26b-a4b (no think) | **57.0%** | 57/100 | `gemma-4-26b-a4b-gguf/results/opseval_20260501_092600/chinese_question_eval_20260501_012606_all.json` | 以串行 llama-server queue 執行；為避免記憶體不足，獨立接續跑。 |
| OpsEval | qwen3.6-35b-a3b (no think) | **18.0%** | 18/100 | `qwen3.6-35b-a3b/results/opseval_20260501_092600/chinese_question_eval_20260501_013213_all.json` | 結果顯著異常，log 顯示模型常輸出多選（如 `AC` / `ACB` / `ACDB`），需另做 prompt / extraction 對齊檢查。 |
| NetBench | qwen3.6-27b (no think) | N/A | N/A | N/A | `NetoAISolutions/NetBench` 仍回傳 gated 403 / awaiting review，尚未取得 `T-NetEval.csv`。 |
| QuArch | qwen3.6-27b (no think) | N/A | N/A | N/A | 尚未找到 public bulk dataset endpoint。 |

#### Vision (100 題/任務)
> Tasks: OCR (PokemonCards), Classification (CIFAR10), Detection (CPPE5)

| Model | OCR | Classification | Detection | Avg (3 tasks) | Result Path |
|---|---:|---:|---:|---:|---|
| gemma-4-26b-a4b (think) | 92% (92/100) | 88% (88/100) | 61% (61/100) | **80.3%** | `gemma-4-26b-a4b-gguf/vision/results/full_20260404_094301/summary_100.json` |
| qwen3.5-27b (think) | 89% (89/100) | 79% (79/100) | 60% (60/100) | **76.0%** | `qwen3.5-27b/vision/results/summary_100.json` |
| ministral-3-14b| 89% (89/100) | 82% (82/100) | 51% (51/100) | **74.0%** | `ministral-3-14b-gguf/vision/results/summary_100.json` |
| qwen3.5-35b-a3b | 92% (92/100) | 70% (70/100) | 59% (59/100) | **73.7%** | `qwen3.5-35b-a3b/vision/results-rerun-20260321_225509/summary_100.json` |
| gemma-3-27b | 91% (91/100) | 82% (82/100) | 47% (47/100) | **73.3%** | `gemma-3-27b-gguf/vision/results-20260322_002456/summary_100.json` |
| lfm2.5-vl-1.6b | 87% (87/100) | 76% (76/100) | 18% (18/100) | **60.3%** | `lfm2.5-vl-1.6b/vision/results-20260322_083607/summary_100.json` |
