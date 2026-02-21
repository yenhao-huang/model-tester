import os
from typing import Any

from huggingface_hub import InferenceClient
from transformers import pipeline


class LocalPipelineClient:
    def __init__(self, model: str, device: str = "cpu") -> None:
        device_arg: Any
        if device == "cpu":
            device_arg = -1
        elif device == "mps":
            device_arg = "mps"
        else:
            # e.g. cuda:0
            device_arg = device

        self.pipe = pipeline(
            task="text-generation",
            model=model,
            device=device_arg,
        )


def build_client() -> Any:
    backend = os.getenv("INFERENCE_BACKEND", "hf_api")
    model = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")

    if backend == "local":
        device = os.getenv("DEVICE", "cpu")
        return LocalPipelineClient(model=model, device=device)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(model=model, token=token)


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join(p for p in parts if p).strip()
    return ""


def _build_local_chat_prompt(messages: list[dict[str, str]]) -> str:
    role_tokens = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }
    chunks: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        token = role_tokens.get(role, "<|user|>")
        chunks.append(f"{token}\n{m.get('content', '')}")

    chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def run_chat_inference(client: Any, messages: list[dict[str, str]]) -> str:
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    if isinstance(client, LocalPipelineClient):
        prompt = _build_local_chat_prompt(messages)
        out = client.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        if out and isinstance(out, list):
            return (out[0].get("generated_text") or "").strip()
        return ""

    resp = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _extract_content(resp.choices[0].message.content)


def run_inference(client: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return run_chat_inference(client, messages)
