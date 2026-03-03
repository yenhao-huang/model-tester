#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import soundfile as sf

MODEL_PATH = os.getenv("QWEN_TTS_MODEL_PATH", "/Users/yenhaohuang/Desktop/models/qwen3-tts")
OUT_DIR = Path(os.getenv("QWEN_TTS_OUT_DIR", "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

SPEAKER_OPTIONS = {
    "北京人": "Dylan",
    "上海人": "Eric",
    "日本人": "Ono_Anna",
    "韓國人": "Sohee",
}

_model = None
_device_info = None


def _load_model_once():
    global _model, _device_info
    if _model is not None:
        return _model, _device_info

    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        attn_impl = None
    else:
        device = "cpu"
        dtype = torch.float32
        attn_impl = None

    kwargs = {"device_map": device, "dtype": dtype}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    _model = Qwen3TTSModel.from_pretrained(MODEL_PATH, **kwargs)
    _device_info = f"device={device}, dtype={dtype}, attn={attn_impl or 'default'}"
    return _model, _device_info


def synthesize(text: str, language: str, speaker_label: str, instruct: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("請先輸入要合成的文字")

    model, device_info = _load_model_once()

    lang = None if language == "Auto" else language
    speaker = SPEAKER_OPTIONS[speaker_label]
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=lang,
        speaker=speaker,
        instruct=instruct or "",
    )

    filename = f"tts_{speaker}_{(lang or 'Auto')}_{abs(hash(text)) % 10_000_000}.wav"
    out_path = OUT_DIR / filename
    sf.write(str(out_path), wavs[0], sr)

    return str(out_path), f"✅ 產生完成：{out_path}\n(sr={sr})\n{device_info}"


with gr.Blocks(title="Qwen3-TTS CustomVoice UI") as demo:
    gr.Markdown("## Qwen3-TTS-12Hz-1.7B-CustomVoice UI")
    gr.Markdown(f"Model path: `{MODEL_PATH}`")

    text = gr.Textbox(label="Text", lines=4, placeholder="請輸入要轉語音的文字")
    with gr.Row():
        language = gr.Dropdown(
            choices=SUPPORTED_LANGUAGES,
            value="Chinese",
            label="Language",
        )
        speaker = gr.Dropdown(
            choices=list(SPEAKER_OPTIONS.keys()),
            value="北京人",
            label="Speaker",
        )

    instruct = gr.Textbox(
        label="Instruct (optional)",
        lines=2,
        placeholder="例如：語速自然、語氣清楚",
    )

    run_btn = gr.Button("Generate Speech", variant="primary")
    audio = gr.Audio(label="Output Audio", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=synthesize,
        inputs=[text, language, speaker, instruct],
        outputs=[audio, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
