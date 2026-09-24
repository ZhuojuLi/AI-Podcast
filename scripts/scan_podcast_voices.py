# -*- coding: utf-8 -*-
"""从播客源音频中扫描干净的单人说话片段，供 CosyVoice3 参考音色使用。

用法（容器内）：
  /opt/podcast-venv/bin/python scripts/scan_podcast_voices.py \
      --episode /work/.models/ep16_16k.wav \
      --campplus /work/.models/Fun-CosyVoice3-0.5B-2512/campplus.onnx \
      --male-ref resources/voices/male2.wav \
      --female-ref resources/voices/female2.wav \
      --out-dir output/voice_rescan

对 9s 窗 / 1.5s 跳距扫描，按说话人嵌入相似度、纯度（无换人/叠话）、
能量动态（表现力）、响度打分，输出候选清单和试听 wav。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
WIN = 9.0          # 候选窗长（秒）
HOP = 1.5          # 跳距（秒）
SUB = 3.0          # 纯度校验子窗（秒）


def load_campplus(path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    return sess


def campplus_embed(sess, wav: np.ndarray) -> np.ndarray:
    """输入 16k 单通道 float32 波形，计算 kaldi fbank 后跑 campplus，输出 L2 归一化嵌入。"""
    import torch
    import torchaudio.compliance.kaldi as kaldi
    speech = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0]
    emb = emb.reshape(-1)
    n = np.linalg.norm(emb)
    return emb / (n + 1e-9)


def frame_rms(wav: np.ndarray, frame: int = 800) -> np.ndarray:
    n = len(wav) // frame
    if n == 0:
        return np.zeros(1)
    x = wav[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(x ** 2, axis=1) + 1e-12)


def f0_stats(wav: np.ndarray) -> tuple:
    import librosa
    try:
        f0 = librosa.yin(wav, fmin=60, fmax=420, sr=SR, frame_length=1024,
                         hop_length=160)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
    except Exception:
        return 0.0, 0.0, 0.0
    if len(f0) < 20:
        return 0.0, 0.0, 0.0
    voiced_ratio = len(f0) / max(len(wav) / 160, 1)
    return float(np.median(f0)), float(np.std(f0)), voiced_ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--campplus", required=True)
    ap.add_argument("--male-ref", required=True)
    ap.add_argument("--female-ref", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sess = load_campplus(args.campplus)
    ep, sr = sf.read(args.episode, dtype="float32")
    assert sr == SR, f"episode 需 {SR}Hz，实际 {sr}"
    male_emb = campplus_embed(sess, _to16k(args.male_ref))
    female_emb = campplus_embed(sess, _to16k(args.female_ref))

    win_n, hop_n = int(WIN * SR), int(HOP * SR)
    rows = []
    for start in range(0, len(ep) - win_n, hop_n):
        w = ep[start: start + win_n]
        rms_head = np.sqrt(np.mean(w[: int(0.2 * SR)] ** 2))
        rms_tail = np.sqrt(np.mean(w[-int(0.2 * SR):] ** 2))
        emb = campplus_embed(sess, w)
        subs = [campplus_embed(sess, w[int(i * SR): int((i + SUB) * SR)])
                for i in (0.0, 3.0, 6.0)]
        mean_sub = np.mean(subs, axis=0)
        mean_sub /= np.linalg.norm(mean_sub) + 1e-9
        purity = float(np.mean([np.dot(s, mean_sub) for s in subs]))
        rms_frames = frame_rms(w)
        vad = rms_frames > max(0.008, np.percentile(rms_frames, 25))
        vad_ratio = float(np.mean(vad))
        sim_m = float(np.dot(emb, male_emb))
        sim_f = float(np.dot(emb, female_emb))
        row = {
            "t": start / SR,
            "sim_male": round(sim_m, 4), "sim_female": round(sim_f, 4),
            "purity": round(purity, 4),
            "rms": round(float(np.sqrt(np.mean(w ** 2))), 4),
            "dyn": round(float(np.std(rms_frames[vad]) / (np.mean(rms_frames[vad]) + 1e-9))
                         if vad.any() else 0.0, 4),
            "voiced": round(vad_ratio, 3),
            "edge_rms": round(min(rms_head, rms_tail), 4),
        }
        # F0 只算给通过嵌入初筛的窗（yin 较慢）
        if (max(sim_m, sim_f) > 0.5 and purity > 0.7 and vad_ratio > 0.3
                and row["edge_rms"] < 0.05):
            f0m, f0s, _ = f0_stats(w)
            row["f0_med"], row["f0_std"] = float(round(f0m, 1)), float(round(f0s, 1))
        else:
            row["f0_med"] = row["f0_std"] = 0.0
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} windows...", flush=True)
    print(f"scanned {len(rows)} windows", flush=True)

    def is_male(r):
        return r["sim_male"] > 0.55 and r["sim_male"] > r["sim_female"] + 0.05

    def is_female(r):
        return r["sim_female"] > 0.55 and r["sim_female"] > r["sim_male"] + 0.05

    def clean(r):
        return (r["purity"] > 0.82 and r["voiced"] > 0.45
                and r["edge_rms"] < 0.03 and r["rms"] > 0.03)

    def male_score(r):
        # 要“更健谈”：能量动态大、整体响度足、F0 有变化
        return (2.0 * r["sim_male"] + 1.5 * min(r["dyn"], 1.2)
                + min(r["rms"], 0.25) * 4 + min(r["f0_std"], 40) / 40
                + r["purity"] + min(r["voiced"], 0.9))

    def female_score(r):
        return (2.0 * r["sim_female"] + r["purity"] + min(r["voiced"], 0.9)
                + min(r["dyn"], 1.0) + min(r["rms"], 0.2) * 3)

    report = {}
    for name, pred, score in (("male", is_male, male_score),
                              ("female", is_female, female_score)):
        cands = [r for r in rows if pred(r) and clean(r)]
        cands.sort(key=score, reverse=True)
        # 时间上去重：候选间隔 >= 20s
        picked = []
        for r in cands:
            if all(abs(r["t"] - p["t"]) > 20 for p in picked):
                picked.append(r)
            if len(picked) >= args.top:
                break
        report[name] = picked
        print(f"\n== {name} top{len(picked)} ==", flush=True)
        for r in picked[:12]:
            sim = r["sim_male"] if name == "male" else r["sim_female"]
            print(f"  t={r['t']:7.1f}s sim={sim:.3f} purity={r['purity']:.3f} "
                  f"rms={r['rms']:.3f} dyn={r['dyn']:.2f} f0={r['f0_med']:.0f}"
                  f"±{r['f0_std']:.0f} edge={r['edge_rms']:.4f}", flush=True)
        for i, r in enumerate(picked[:8]):
            s = int(r["t"] * SR)
            sf.write(out / f"{name}_cand_{i:02d}_t{int(r['t'])}s.wav",
                     ep[s: s + win_n], SR)

    (out / "scan_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\nreport: {out / 'scan_report.json'}", flush=True)


def _to16k(path: str) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


if __name__ == "__main__":
    main()
