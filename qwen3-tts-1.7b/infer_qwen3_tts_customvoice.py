#!/usr/bin/env python3
import argparse
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/Users/yenhaohuang/Desktop/models/qwen3-tts")
    parser.add_argument("--text", default="你好，這是一段 Qwen3 TTS CustomVoice 測試語音。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="語速自然、語氣清楚。")
    parser.add_argument("--out", default="outputs/qwen3_tts_customvoice.wav")
    args = parser.parse_args()

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

    kwargs = {
        "device_map": device,
        "dtype": dtype,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = Qwen3TTSModel.from_pretrained(args.model_path, **kwargs)

    wavs, sr = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wavs[0], sr)
    print(f"Saved: {out_path} (sr={sr}, samples={len(wavs[0])})")


if __name__ == "__main__":
    main()
