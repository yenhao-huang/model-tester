# qwen3.5-27b-gguf 測試紀錄

## 模型資訊
- 模型檔：`/Users/yenhaohuang/Desktop/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_1.gguf`
- 檔案大小：`~16 GiB`
- GGUF metadata architecture：`qwen35`

## 環境
- 專案目錄：`~/Desktop/model-tester/qwen3.5-27b-gguf`
- venv：`~/Desktop/python-venvs/qwen35-27b`
- 套件：`llama-cpp-python==0.3.16`

## 安裝步驟
```bash
python3 -m venv ~/Desktop/python-venvs/qwen35-27b
source ~/Desktop/python-venvs/qwen35-27b/bin/activate
pip install -U pip setuptools wheel
CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python==0.3.16
```

## 可直接執行的測試指令
```bash
cd ~/Desktop/model-tester/qwen3.5-27b-gguf
source ~/Desktop/python-venvs/qwen35-27b/bin/activate
python smoke_test.py
```

## 測試結果（smoke test）
- 狀態：`blocked`
- 原因：推論引擎不支援此模型架構（非模型能力失敗）
- 原始錯誤：

```text
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'qwen35'
ValueError: Failed to load model from file: /Users/yenhaohuang/Desktop/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_1.gguf
```

## 已知限制
- 目前安裝的 `llama-cpp-python/llama.cpp` 版本無法載入 `qwen35` architecture。
- 因此尚未進入 MMLU/GSM8K/TruthfulQA/HumanEval 的 20 題評測。

## 下一步建議
1. 升級到支援 `qwen35` 的 llama.cpp/llama-cpp-python 版本（或 nightly/source build）。
2. 或改測可被目前推論引擎識別的 GGUF 模型。
3. 引擎可成功載入後，再執行四個 benchmark 各 20 題，逐題保留 `query/response`。

## 參考 glm-4.7-flash 的評測流程
已複製評測腳本：`eval_fast_textgen_eval_local.py`（來源：`glm-4.7-mxfp4-gguf`）。

當 qwen3.5-27b 的 server 可正常啟動後可直接跑：

```bash
cd ~/Desktop/model-tester/qwen3.5-27b-gguf
OPENAI_BASE_URL=http://127.0.0.1:29172/v1 \
MODEL_NAME=qwen3.5-27b \
MAX_CASES=20 \
python eval_fast_textgen_eval_local.py
```
