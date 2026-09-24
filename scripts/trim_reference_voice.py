# -*- coding: utf-8 -*-
"""把播客候选片段修整为 CosyVoice3 参考音。

处理：24k 单声道 → 70Hz 高通去低频隆隆声 → 峰值归一到 -3dB →
首尾加 60ms 静音垫 + 20ms 淡入淡出（避免参考音边缘能量突变，
克隆出的语音首尾因此带上杂音/爆音）。
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000
PAD_S = 0.06
FADE_S = 0.02
PEAK = 0.7


def _highpass(wav: np.ndarray, sr: int, cutoff: float = 70.0) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, wav).astype(np.float32)


def trim_reference(src: str, dst: str) -> None:
    wav, sr = sf.read(src, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        sr = SR
    wav = _highpass(wav, sr)
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav * (PEAK / peak)
    # 去掉首尾原有的绝对静音，统一由 PAD_S 控制垫片长度
    th = 1e-4
    idx = np.where(np.abs(wav) > th)[0]
    if len(idx):
        wav = wav[max(idx[0] - int(0.01 * sr), 0): idx[-1] + int(0.01 * sr)]
    pad = np.zeros(int(PAD_S * sr), dtype=np.float32)
    wav = np.concatenate([pad, wav, pad])
    n_fade = int(FADE_S * sr)
    wav[:n_fade] *= np.linspace(0.0, 1.0, n_fade)
    wav[-n_fade:] *= np.linspace(1.0, 0.0, n_fade)
    sf.write(dst, wav, sr, subtype="PCM_16")
    print(f"{dst}: {len(wav) / sr:.2f}s peak={np.abs(wav).max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    trim_reference(args.src, args.dst)
