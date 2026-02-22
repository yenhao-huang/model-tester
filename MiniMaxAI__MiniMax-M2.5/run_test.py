#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.utils import HfHubHTTPError

MODEL_ID = "MiniMaxAI/MiniMax-M2.5"
PROMPT = "Write one short sentence to confirm you are working."


@dataclass
class TestResult:
    timestamp_utc: str
    model_id: str
    python_version: str
    platform: str
    huggingface_hub_version: str
    hf_token_present: bool
    model_exists: bool
    private_or_gated: bool | None
    downloads: int | None
    likes: int | None
    pipeline_tag: str | None
    inference_attempted: bool
    inference_success: bool
    inference_output: str | None
    inference_error: str | None


def main() -> int:
    import huggingface_hub

    api = HfApi()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    model_exists = False
    private_or_gated = None
    downloads = None
    likes = None
    pipeline_tag = None

    try:
        info = api.model_info(MODEL_ID, token=token)
        model_exists = True
        private_or_gated = bool(getattr(info, "gated", False) or getattr(info, "private", False))
        downloads = getattr(info, "downloads", None)
        likes = getattr(info, "likes", None)
        pipeline_tag = getattr(info, "pipeline_tag", None)
    except HfHubHTTPError as e:
        model_exists = False
        meta_error = f"model_info_failed: {type(e).__name__}: {e}"
    except Exception as e:
        model_exists = False
        meta_error = f"model_info_failed: {type(e).__name__}: {e}"
    else:
        meta_error = None

    inference_attempted = False
    inference_success = False
    inference_output = None
    inference_error = None

    if model_exists:
        inference_attempted = True
        try:
            client = InferenceClient(model=MODEL_ID, token=token)
            out = client.text_generation(PROMPT, max_new_tokens=32, temperature=0.2)
            inference_success = True
            inference_output = str(out)
        except Exception as e:
            first_err = f"{type(e).__name__}: {e}"
            try:
                chat = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": PROMPT}],
                    max_tokens=64,
                    temperature=0.2,
                )
                inference_success = True
                inference_output = chat.choices[0].message.content if chat.choices else str(chat)
            except Exception as e2:
                inference_error = f"text_generation_failed={first_err} | chat_completion_failed={type(e2).__name__}: {e2}"

    if meta_error and not inference_error:
        inference_error = meta_error

    result = TestResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model_id=MODEL_ID,
        python_version=sys.version,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        huggingface_hub_version=huggingface_hub.__version__,
        hf_token_present=bool(token),
        model_exists=model_exists,
        private_or_gated=private_or_gated,
        downloads=downloads,
        likes=likes,
        pipeline_tag=pipeline_tag,
        inference_attempted=inference_attempted,
        inference_success=inference_success,
        inference_output=inference_output,
        inference_error=inference_error,
    )

    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
