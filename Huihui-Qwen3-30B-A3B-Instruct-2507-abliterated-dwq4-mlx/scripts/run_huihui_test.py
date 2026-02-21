#!/usr/bin/env python3
import json
import time
from datetime import datetime
from pathlib import Path

import mlx_lm
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import model_info

MODEL_ID = "nightmedia/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-dwq4-mlx"
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    t0 = time.time()

    info = model_info(MODEL_ID)
    meta = {
        "modelId": info.id,
        "sha": info.sha,
        "lastModified": str(info.lastModified),
        "downloads": info.downloads,
        "likes": info.likes,
        "private": info.private,
        "gated": info.gated,
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "tags": info.tags,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(OUTPUTS_DIR / "model_info.json", meta)

    load_start = time.time()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    load_sec = time.time() - load_start

    tests = [
        ("English", "You are a concise assistant. In one sentence, describe Taipei in winter."),
        ("Chinese", "用一句話描述台北的冬天。"),
        ("Reasoning", "What is 7 * 13 + 29? Show your work briefly."),
    ]

    sampler = make_sampler(temp=0.2, top_p=1.0, min_p=0.0, min_tokens_to_keep=1)

    results = []
    for name, prompt in tests:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        ts = time.time()
        output = mlx_lm.generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=256,
            sampler=sampler,
            verbose=False,
        )
        sec = time.time() - ts

        results.append({
            "test": name,
            "prompt": prompt,
            "seconds": round(sec, 2),
            "output": output,
        })

    summary = {
        "model": MODEL_ID,
        "load_seconds": round(load_sec, 2),
        "total_seconds": round(time.time() - t0, 2),
        "created_at": datetime.now().isoformat(),
        "results": results,
    }

    save_json(OUTPUTS_DIR / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
