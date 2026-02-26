#!/usr/bin/env python3
from __future__ import annotations

import os
import traceback
import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = "MiniMaxAI/MiniMax-M2.5"


def _client() -> InferenceClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(model=MODEL_ID, token=token)


def chat_fn(message, history, temperature: float, max_tokens: int):
    msgs = []
    for h in history or []:
        if isinstance(h, dict):
            role = h.get("role")
            content = h.get("content", "")
            if role in {"user", "assistant", "system"}:
                msgs.append({"role": role, "content": content})
        elif isinstance(h, (list, tuple)) and len(h) >= 2:
            user_msg, assistant_msg = h[0], h[1]
            if user_msg:
                msgs.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                msgs.append({"role": "assistant", "content": str(assistant_msg)})
    msgs.append({"role": "user", "content": message})

    try:
        client = _client()
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if resp.choices and resp.choices[0].message:
            return resp.choices[0].message.content or "(empty response)"
        return str(resp)
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        return f"❌ Error: {type(e).__name__}: {e}\n{tb}"


with gr.Blocks(title=f"Model Tester - {MODEL_ID}") as demo:
    gr.Markdown(
        f"## Model Tester\n"
        f"Model: `{MODEL_ID}`\n\n"
        f"- Uses Hugging Face `InferenceClient.chat.completions`\n"
        f"- Optional token env: `HF_TOKEN`"
    )

    temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.1, label="temperature")
    max_tokens = gr.Slider(16, 1024, value=256, step=16, label="max_tokens")

    chatbot = gr.ChatInterface(
        fn=chat_fn,
        textbox=gr.Textbox(placeholder="輸入你的問題...", lines=3),
        additional_inputs=[temperature, max_tokens],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
