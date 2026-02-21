# Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-dwq4-mlx (MLX)

## 目錄結構
- `scripts/`：可執行程式
  - `run_huihui_test.py`：跑固定測試並輸出 JSON
  - `chat_cli.py`：互動式 query -> response CLI
- `outputs/`：測試輸出
  - `model_info.json`
  - `results.json`
- `logs/`：執行 log
  - `test_output.log`

## 使用方式
先啟用環境：

```bash
source /Users/yenhaohuang/Desktop/python-venvs/huihui-qwen3-30b/bin/activate
```

### 1) 固定測試
```bash
python /Users/yenhaohuang/Desktop/model-tester/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-dwq4-mlx/scripts/run_huihui_test.py
```

### 2) 互動式問答（可自行 input query）
```bash
python /Users/yenhaohuang/Desktop/model-tester/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-dwq4-mlx/scripts/chat_cli.py --max-tokens 256 --temp 0.2
```

輸入 `/exit` 可離開。
