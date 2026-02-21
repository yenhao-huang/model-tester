import argparse
from pathlib import Path

from dotenv import load_dotenv

from .client import build_client, run_chat_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="gpt-oss-20b interactive chat")
    parser.add_argument("--prompt", default="prompts/system_v1.txt", help="system prompt file")
    args = parser.parse_args()

    load_dotenv()
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise FileNotFoundError(f"system prompt not found: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    client = build_client()

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("[chat] gpt-oss-20b interactive mode")
    print("[chat] commands: /exit, /reset, /history")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye")
            break

        if not user_input:
            continue

        if user_input in {"/exit", "/quit"}:
            print("[chat] bye")
            break

        if user_input == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("[chat] conversation reset")
            continue

        if user_input == "/history":
            turns = len([m for m in messages if m["role"] != "system"])
            print(f"[chat] turns={turns}")
            continue

        messages.append({"role": "user", "content": user_input})
        answer = run_chat_inference(client, messages)
        messages.append({"role": "assistant", "content": answer})
        print(f"assistant> {answer}\n")


if __name__ == "__main__":
    main()
