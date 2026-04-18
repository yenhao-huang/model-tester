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

| Model | Prompt ms/token | Prompt token/s | Eval ms/token | Eval token/s |
|---|---:|---:|---:|---:|
| lfm2.5-vl-1.6b | 0.79 | 1266.33 | 24.19 | 41.33 |
| lfm2-24b | 4.61 | 217.05 | 27.69 | 36.11 |
| gemma-4-26b-a4b-gguf | 8.97 | 111.46 | 40.07 | 24.96 |
| qwen3.5-35b-a3b | 6.05 | 165.15 | 53.54 | 18.68 |
| glm-4.7-flash-fp4 | 9.36 | 106.80 | 77.05 | 12.98 |
| ministral-3-14b-gguf | 7.35 | 136.04 | 144.78 | 6.91 |
| gemma-3-27b-gguf | 18.37 | 54.45 | 193.85 | 5.16 |
| qwen3.5-27b | 18.27 | 54.74 | 251.96 | 3.97 |

### Accuracy

#### Text (100 題/任務)

| Model | MMLU | GSM8K | Geo MMLU HS | Law MMLU Prof | HumanEval | Total |
|---|---:|---:|---:|---:|---:|---:|
| gemma-4-31b (thinking) | 96% | 89% | 92% | 80% | 92% | **89.8%** |
| gemma-4-26b-a4b (no think) | 91% | 93% | 91% | 70% | **99%** | **88.8%** |
| qwen3.5-27b (no think) | 79% | **97%** | 91% | 75% | 95% | **87.4%** |
| qwen3.5-35b-a3b (thinking) | 83% | 95% | 95% | 68% | 92% | **86.6%** |
| qwen3.5-35b-a3b (no think) | 63% | **98%** | 93% | 68% | 92% | **82.8%** |
| gpt-oss-20b | 83% | 94% | 88% | 51% | 96% | **82.4%** |
| glm-4.7-flash-fp4 | 72% | 92% | 86% | 54% | 76% | **76.0%** |
| lfm2-24b | 74% | 86% | 81% | 57% | 80% | **75.6%** |
| omnicoder-9b-gguf | 57% | 86% | 78% | 62% | 84% | **73.4%** |
| gemma-3-27b-gguf | 79% | 76% | 86% | 57% | 63% | **72.2%** |
| ministral-3-14b-gguf | 48% | 2% | 88% | 48% | 88% | **54.8%** |
| lfm2.5-vl-1.6b | 51% | 65% | 52% | 28% | 56% | **50.4%** |


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
