# -*- coding: utf-8 -*-
"""用一男一女参考音，通过 CosyVoice3 生成固定播客试听。"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def _change_tempo(wav: np.ndarray, sample_rate: int, tempo: float) -> np.ndarray:
    """小幅调整语速而不改变音高；用原始 PCM 避免反复有损编码。"""
    if tempo == 1.0:
        return wav
    if not 0.8 <= tempo <= 1.25:
        raise ValueError("试听语速只支持 0.8–1.25 倍，避免明显失真")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sample_rate),
         "-ac", "1", "-i", "pipe:0", "-af", f"atempo={tempo:.4f}",
         "-f", "f32le", "pipe:1"],
        input=np.ascontiguousarray(wav, dtype="<f4").tobytes(),
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def _instruction(text: str, index: int) -> str:
    base = (
        "请用标准普通话，像二十多岁到三十岁的青年主播在双人商业播客中自然交谈。"
        "面向对面的搭档说话，不要新闻播音腔、广告腔或客服腔，停连和重音符合句意。"
    )
    if index == 0:
        return base + "这是开场，语气放松、有兴趣，直接进入话题。"
    if text.rstrip().endswith(("？", "?")):
        return base + "带着真实好奇发问，句尾自然，不要像照着提词器采访。"
    if len(text.strip()) <= 20:
        return base + "这是紧接对方的短回应，反应稍快，有真实交流感。"
    if re.search(r"(?:但|不过|恰恰|反过来|问题是)", text):
        return base + "这里在补充或修正对方观点，带一点思考和轻微转折，不要生硬反驳。"
    return base + "自然承接搭档上一句话，重点词适度强调。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="CosyVoice 模型目录；复用已有轮次时不需要")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--male-ref", help="男声参考 WAV；复用已有轮次时不需要")
    ap.add_argument("--female-ref", help="女声参考 WAV；复用已有轮次时不需要")
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns-limit", type=int, default=6)
    ap.add_argument("--gap-ms", type=int, default=120)
    ap.add_argument("--female-tempo", type=float, default=1.0,
                    help="女声播放语速，1.07 表示只加快 7%% 且保持音高")
    ap.add_argument("--reuse-turns-from", default="",
                    help="复用此目录的 turn_XX.wav，仅调整语速和重新拼接；无需 GPU")
    args = ap.parse_args()

    if not args.reuse_turns_from and not all((args.model_dir, args.male_ref, args.female_ref)):
        ap.error("新生成需要 --model-dir、--male-ref、--female-ref")

    with open(args.transcript, encoding="utf-8") as f:
        transcript = json.load(f)
    turns = transcript["content"]
    if args.turns_limit > 0:
        turns = turns[:args.turns_limit]

    os.makedirs(args.out, exist_ok=True)
    model = None
    if not args.reuse_turns_from:
        # 参考实验不需要在线词典。避免模型初始化触发 wetext 的远端下载。
        no_online_normalizer = types.ModuleType("wetext")
        def _offline_normalizer(*args, **kwargs):
            raise RuntimeError("offline experiment uses raw Chinese text")
        no_online_normalizer.Normalizer = _offline_normalizer
        sys.modules["wetext"] = no_online_normalizer

        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.cli import frontend as cosy_frontend

        # 新版 torchaudio.load 会强依赖与 PyTorch/CUDA 精确匹配的 torchcodec。
        # 参考音已预处理成 24kHz 单声道，直接用 soundfile 可避免无意义的 CUDA 解码依赖。
        def _load_reference_wav(path: str, target_sr: int):
            audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if source_sr != target_sr:
                divisor = int(np.gcd(source_sr, target_sr))
                audio = resample_poly(audio, target_sr // divisor, source_sr // divisor).astype(
                    np.float32, copy=False
                )
            return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        cosy_frontend.load_wav = _load_reference_wav
        t0 = time.time()
        model = AutoModel(model_dir=args.model_dir, fp16=True)
        print(f"[CosyVoice] 模型加载完成 {time.time() - t0:.1f}s", flush=True)

    pieces = []
    sample_rate = model.sample_rate if model is not None else None
    for index, turn in enumerate(turns):
        text = str(turn["text"])
        if args.reuse_turns_from:
            turn_path = os.path.join(args.reuse_turns_from, f"turn_{index + 1:02d}.wav")
            wav, turn_sr = sf.read(turn_path, dtype="float32", always_2d=False)
            if wav.ndim != 1:
                raise ValueError(f"必须为单声道轮次: {turn_path}")
            if sample_rate is None:
                sample_rate = turn_sr
            elif sample_rate != turn_sr:
                raise ValueError(f"轮次采样率不一致: {turn_path}")
        else:
            ref = args.male_ref if int(turn["speaker"]) == 1 else args.female_ref
            prompt = "You are a helpful assistant. " + _instruction(text, index) + "<|endofprompt|>"
            started = time.time()
            chunks = list(model.inference_instruct2(text, prompt, ref, stream=False,
                                                   speed=1.0, text_frontend=False))
            if not chunks:
                raise RuntimeError(f"第 {index + 1} 轮没有生成音频")
            audio = torch.cat([chunk["tts_speech"].detach().cpu() for chunk in chunks], dim=-1)
            wav = audio.squeeze(0).float().numpy()
        if int(turn["speaker"]) == 2:
            wav = _change_tempo(wav, sample_rate, args.female_tempo)
        pieces.append(wav)
        sf.write(os.path.join(args.out, f"turn_{index + 1:02d}.wav"), wav, sample_rate)
        if index < len(turns) - 1:
            pieces.append(np.zeros(int(sample_rate * args.gap_ms / 1000), dtype=np.float32))
        print(f"[CosyVoice] turn {index + 1}/{len(turns)} speaker={turn['speaker']} "
              f"{len(text)}字 {len(wav) / sample_rate:.1f}s", flush=True)

    combined = np.concatenate(pieces)
    wav_path = os.path.join(args.out, "podcast.wav")
    mp3_path = os.path.join(args.out, "podcast.mp3")
    sf.write(wav_path, combined, sample_rate)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", wav_path,
         "-c:a", "libmp3lame", "-b:a", "96k", mp3_path],
        check=True,
    )
    with open(os.path.join(args.out, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump({"title": transcript.get("title", ""), "content": turns},
                  f, ensure_ascii=False, indent=2)
    print(f"[CosyVoice] 完成 {len(combined) / sample_rate:.1f}s -> {mp3_path}", flush=True)


if __name__ == "__main__":
    main()
