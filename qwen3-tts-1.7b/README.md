# Qwen3-TTS-12Hz-1.7B-CustomVoice 測試專案

## 專案位置
- `~/Desktop/model-tester/qwen3-tts-12hz-1.7b-customvoice`

## 模型
- 已下載模型路徑：`/Users/yenhaohuang/Desktop/models/qwen3-tts`
- 來源：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

## 環境
```bash
python3 -m venv ~/Desktop/python-venvs/qwen3-tts
source ~/Desktop/python-venvs/qwen3-tts/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Inference 程式
- `infer_qwen3_tts_customvoice.py`
- `app_gradio.py`（Gradio UI，Language/Speaker 下拉選單）

### 執行方式
```bash
cd ~/Desktop/model-tester/qwen3-tts-12hz-1.7b-customvoice
source ~/Desktop/python-venvs/qwen3-tts/bin/activate

python infer_qwen3_tts_customvoice.py \
  --model-path /Users/yenhaohuang/Desktop/models/qwen3-tts \
  --text "Hello from Qwen TTS test." \
  --language English \
  --speaker Ryan \
  --instruct "Calm and clear." \
  --out outputs/test_en.wav
```

### 啟動 Gradio UI
```bash
cd ~/Desktop/model-tester/qwen3-tts-12hz-1.7b-customvoice
source ~/Desktop/python-venvs/qwen3-tts/bin/activate
python app_gradio.py
```
開啟：`http://127.0.0.1:7860`

## TTS datasets（已放到 ~/Desktop/datasets/tts/）
使用 `prepare_tts_datasets.py` 已整理並下載以下資料集資訊：
- `keithito/lj_speech`
- `openslr/librispeech_asr`
- `mozilla-foundation/common_voice_17_0`
- `google/fleurs`

輸出位置：
- `~/Desktop/datasets/tts/download_report.json`
- `~/Desktop/datasets/tts/<dataset>/README.md`
- `~/Desktop/datasets/tts/<dataset>/FILELIST.txt`

## 測試結果
### 測試 1（English）
- 指令：`python infer_qwen3_tts_customvoice.py --text 'Hello from Qwen TTS test.' --language English --speaker Ryan ...`
- 輸出：`outputs/test_en.wav`
- 取樣率：`24000`
- 狀態：成功

### 測試 2（Chinese）
- 指令：`python infer_qwen3_tts_customvoice.py --text '這是第二段測試...' --language Chinese --speaker Vivian ...`
- 輸出：`outputs/test_zh.wav`
- 取樣率：`24000`
- 狀態：成功

## 已知限制
- 目前有警告：`sox: command not found`（不影響本次 wav 生成成功）
- 目前未安裝 `flash-attn`，推論可跑但速度較慢。
