import os
from typing import Any

from huggingface_hub import InferenceClient
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalPipelineClient:
    def __init__(self, model: str, device: str = "cpu") -> None:
        self.device = self._normalize_device(device)
        self.torch_dtype = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _normalize_device(device: str) -> str:
        if device.startswith("cuda") and torch.cuda.is_available():
            return device
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device == "cpu":
            return "cpu"

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _pick_dtype(device: str):
        if device.startswith("cuda"):
            return torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32


def build_client() -> Any:
    backend = os.getenv("INFERENCE_BACKEND", "hf_api")
    model = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")

    if backend == "local":
        model_path = os.path.expanduser(os.getenv("LOCAL_MODEL_PATH", "").strip())
        model_or_path = model_path or model
        device = os.getenv("DEVICE", "cpu")
        return LocalPipelineClient(model=model_or_path, device=device)

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
        inputs = client.tokenizer(prompt, return_tensors="pt").to(client.device)

        with torch.no_grad():
            output_ids = client.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=float(os.getenv("TOP_P", "0.9")),
                do_sample=temperature > 0,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return client.tokenizer.decode(generated, skip_special_tokens=True).strip()

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
