# minmax-m2.5

新增 **local inference** 能力（`huggingface transformers.from_pretrained`）。

## 1) 建立環境

```bash
python3 -m venv ~/Desktop/python-venvs/minmax-m25
source ~/Desktop/python-venvs/minmax-m25/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) 執行本地推論

> 若模型需要自訂程式碼，請加上 `--trust-remote-code`。

```bash
python run_local_inference.py \
  --model MiniMaxAI/MiniMax-M2.5 \
  --prompt "用三點整理 MiniMax-M2.5 的定位" \
  --max-new-tokens 256 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --trust-remote-code \
  --save-json outputs/run-$(date +%Y%m%d-%H%M%S).json
```

## 3) 輸出與紀錄

- 終端機會印出生成結果
- `--save-json` 會保存：時間、模型、裝置、參數、prompt、output

## 4) 已知限制

- 模型體積可能很大，首次下載時間長
- 本機記憶體/VRAM 不足時會 OOM
- 某些模型必須 `trust_remote_code=True`
- Apple Silicon（MPS）精度與速度可能與 CUDA 不同
