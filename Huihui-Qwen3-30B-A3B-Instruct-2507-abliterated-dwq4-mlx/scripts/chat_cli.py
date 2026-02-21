#!/usr/bin/env python3
import argparse
import time

import mlx_lm
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "nightmedia/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-dwq4-mlx"


def build_prompt(tokenizer, user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for Huihui Qwen3 MLX model")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Loading model: {MODEL_ID}")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print("Type your query and press Enter. Type /exit to quit.\n")

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            break

        if not query:
            continue
        if query.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Bye")
            break

        prompt = build_prompt(tokenizer, query)
        sampler = make_sampler(temp=args.temp, top_p=1.0, min_p=0.0, min_tokens_to_keep=1)

        ts = time.time()
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        dt = time.time() - ts

        print(f"Bot> {response.strip()}")
        print(f"(generated in {dt:.2f}s)\n")


if __name__ == "__main__":
    main()
