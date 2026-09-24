"""Precompute fixed CosyVoice3 voice-reference features for the 8 GiB SUT."""

import argparse
import hashlib
from functools import partial
from pathlib import Path

import torch

from app.providers.cosy import _load_reference_wav
from cosyvoice.cli import frontend as frontend_module
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from matcha.utils.audio import mel_spectrogram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--voices", default="/app/resources/voices")
    parser.add_argument("--output", default="/app/resources/voices/reference_features.pt")
    args = parser.parse_args()

    model = Path(args.model)
    voices = Path(args.voices)
    frontend_module.load_wav = _load_reference_wav
    frontend = CosyVoiceFrontEnd(
        get_tokenizer=partial(
            get_qwen_tokenizer,
            token_path=str(model / "CosyVoice-BlankEN"),
            skip_special_tokens=True,
            version="cosyvoice3",
        ),
        feat_extractor=partial(
            mel_spectrogram,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=None,
            center=False,
        ),
        campplus_model=str(model / "campplus.onnx"),
        speech_tokenizer_model=str(model / "speech_tokenizer_v3.onnx"),
    )

    records = {}
    names = [p.name for p in sorted(Path(args.voices).glob("*.wav"))]
    if not names:
        raise FileNotFoundError(f"参考音目录为空: {args.voices}")
    for name in names:
        path = voices / name
        speech_token = frontend._extract_speech_token(str(path))
        speech_feat = frontend._extract_speech_feat(str(path))
        spk_embedding = frontend._extract_spk_embedding(str(path))
        records[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "speech_token": tuple(value.detach().cpu() for value in speech_token),
            "speech_feat": tuple(value.detach().cpu() for value in speech_feat),
            "spk_embedding": spk_embedding.detach().cpu(),
        }
        print(f"precomputed {name}", flush=True)

    output = Path(args.output)
    torch.save(records, output)
    print(f"saved {output} ({output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
