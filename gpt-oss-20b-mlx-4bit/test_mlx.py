from datetime import datetime
import platform
from importlib.metadata import version
from mlx_lm import load, generate

MODEL = "InferenceIllusionist/gpt-oss-20b-MLX-4bit"
PROMPT = "用一句話介紹台北今天天氣不確定時該怎麼穿。"

print(f"timestamp={datetime.now().isoformat()}")
print(f"python={platform.python_version()}")
print(f"mlx={version('mlx')}")
print(f"mlx_lm={version('mlx-lm')}")
print(f"model={MODEL}")

model, tokenizer = load(MODEL)
out = generate(model, tokenizer, prompt=PROMPT, max_tokens=80)

print("\n=== output ===")
print(out)
