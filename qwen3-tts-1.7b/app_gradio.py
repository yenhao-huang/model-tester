#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import soundfile as sf

MODEL_PATH = os.getenv("QWEN_TTS_MODEL_PATH", "/Users/yenhaohuang/Desktop/models/qwen3-tts")
BASE_MODEL_PATH = os.getenv("QWEN_TTS_BASE_MODEL_PATH", "/Users/yenhaohuang/Desktop/models/qwen3-tts-base")
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
_base_model = None
_base_device_info = None


def _get_torch_kwargs():
    import torch
    if torch.cuda.is_available():
        device, dtype, attn_impl = "cuda:0", torch.bfloat16, "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype, attn_impl = "mps", torch.float16, None
    else:
        device, dtype, attn_impl = "cpu", torch.float32, None
    kwargs = {"device_map": device, "dtype": dtype}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    return kwargs, device, dtype, attn_impl


def _load_model_once():
    global _model, _device_info
    if _model is not None:
        return _model, _device_info

    from qwen_tts import Qwen3TTSModel
    kwargs, device, dtype, attn_impl = _get_torch_kwargs()
    _model = Qwen3TTSModel.from_pretrained(MODEL_PATH, **kwargs)
    _device_info = f"device={device}, dtype={dtype}, attn={attn_impl or 'default'}"
    return _model, _device_info


def _load_base_model_once():
    global _base_model, _base_device_info
    if _base_model is not None:
        return _base_model, _base_device_info

    from qwen_tts import Qwen3TTSModel
    kwargs, device, dtype, attn_impl = _get_torch_kwargs()
    _base_model = Qwen3TTSModel.from_pretrained(BASE_MODEL_PATH, **kwargs)
    _base_device_info = f"device={device}, dtype={dtype}, attn={attn_impl or 'default'}"
    return _base_model, _base_device_info


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


def synthesize_voice_clone(
    text: str,
    language: str,
    ref_audio_path: str,
    ref_text: str,
    x_vector_only: bool,
):
    text = (text or "").strip()
    if not text:
        raise gr.Error("請先輸入要合成的文字")
    if not ref_audio_path:
        raise gr.Error("請上傳參考音訊")
    if not x_vector_only and not (ref_text or "").strip():
        raise gr.Error("ICL 模式需要輸入參考音訊的文字稿，或勾選「僅用 Speaker Embedding」")

    model, device_info = _load_base_model_once()

    lang = None if language == "Auto" else language
    rtext = (ref_text or "").strip() or None

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=lang,
        ref_audio=ref_audio_path,
        ref_text=rtext,
        x_vector_only_mode=x_vector_only,
    )

    filename = f"clone_{abs(hash(text + ref_audio_path)) % 10_000_000}.wav"
    out_path = OUT_DIR / filename
    sf.write(str(out_path), wavs[0], sr)

    return str(out_path), f"✅ 產生完成：{out_path}\n(sr={sr})\n{device_info}"


with gr.Blocks(title="Qwen3-TTS UI") as demo:
    gr.Markdown("## Qwen3-TTS UI")
    gr.Markdown(f"Model path: `{MODEL_PATH}`")

    with gr.Tabs():
        # ── Tab 1: CustomVoice ──────────────────────────────────────
        with gr.Tab("CustomVoice（預設 Speaker）"):
            cv_text = gr.Textbox(label="Text", lines=4, placeholder="請輸入要轉語音的文字")
            with gr.Row():
                cv_language = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES,
                    value="Chinese",
                    label="Language",
                )
                cv_speaker = gr.Dropdown(
                    choices=list(SPEAKER_OPTIONS.keys()),
                    value="北京人",
                    label="Speaker",
                )
            cv_instruct = gr.Textbox(
                label="Instruct (optional)",
                lines=2,
                placeholder="例如：語速自然、語氣清楚",
            )
            cv_btn = gr.Button("Generate Speech", variant="primary")
            cv_audio = gr.Audio(label="Output Audio", type="filepath")
            cv_status = gr.Textbox(label="Status", interactive=False)

            cv_btn.click(
                fn=synthesize,
                inputs=[cv_text, cv_language, cv_speaker, cv_instruct],
                outputs=[cv_audio, cv_status],
            )

        # ── Tab 2: Voice Clone（上傳音訊） ──────────────────────────
        with gr.Tab("Voice Clone（上傳音訊）"):
            gr.Markdown(
                f"上傳一段參考音訊，模型會模仿該聲音說出你輸入的文字。\n\nBase 模型路徑：`{BASE_MODEL_PATH}`"
            )
            vc_text = gr.Textbox(label="要合成的文字", lines=4, placeholder="請輸入要轉語音的文字")
            vc_language = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="Chinese",
                label="Language",
            )
            vc_ref_audio = gr.Audio(
                label="參考音訊（上傳你的聲音）",
                type="filepath",
                sources=["upload", "microphone"],
            )
            vc_x_vector_only = gr.Checkbox(
                label="僅用 Speaker Embedding（不需文字稿，效果較弱）",
                value=False,
            )
            vc_ref_text = gr.Textbox(
                label="參考音訊的文字稿（ICL 模式必填）",
                lines=3,
                placeholder="輸入參考音訊說的內容，留空則需勾選上方選項",
            )
            vc_btn = gr.Button("Generate Voice Clone", variant="primary")
            vc_audio = gr.Audio(label="Output Audio", type="filepath")
            vc_status = gr.Textbox(label="Status", interactive=False)

            vc_x_vector_only.change(
                fn=lambda x: gr.update(interactive=not x, placeholder="勾選後不需要文字稿" if x else "輸入參考音訊說的內容，留空則需勾選上方選項"),
                inputs=[vc_x_vector_only],
                outputs=[vc_ref_text],
            )

            vc_btn.click(
                fn=synthesize_voice_clone,
                inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text, vc_x_vector_only],
                outputs=[vc_audio, vc_status],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
