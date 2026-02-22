#!/usr/bin/env python3
"""Local inference runner for minmax-m2.5 using Hugging Face Transformers."""

import argparse
import json
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def main():
    parser = argparse.ArgumentParser(description="Run local inference with Transformers.from_pretrained")
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-M2.5", help="HF model id or local path")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True")
    parser.add_argument("--save-json", default="", help="Optional path to save full result json")
    args = parser.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "trust_remote_code": args.trust_remote_code,
        "prompt": args.prompt,
        "output": text,
    }

    print("===== OUTPUT =====")
    print(text)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved result JSON to: {args.save_json}")


if __name__ == "__main__":
    main()
