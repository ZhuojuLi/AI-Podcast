# -*- coding: utf-8 -*-
"""从整集未裁剪播客音频自动生成 TTS 克隆参考音（冷启动，无需先验参考音）。

管线：
  1. 解码为 16k 单声道
  2. 滑窗扫描：campplus 说话人嵌入 + 纯度（无换人/叠话）+ 能量/边界启发式
  3. 嵌入球面 k-means 聚类，自动发现说话人（解决 scan_podcast_voices.py
     需要先有 male/female 参考音才能算相似度的冷启动问题）
  4. 每个说话人内按干净度 + 表现力打分，时间去重后挑 top 段
  5. 可选 DeepFilterNet 清洗（--denoise，需 pip install deepfilternet）
  6. trim_reference_voice 修整（高通/归一/淡入淡出）→ male.wav / female.wav

原则：筛选优先于清洗。句间底噪会被克隆进合成结果（实测：参考音带 BGM 时
FireRed/CosyVoice 输出字音模糊），带噪段直接丢弃，只有干净候选不足时才
用 --denoise 事后补救。

用法：
  python scripts/auto_reference.py \
      --episode .models/ep16_16k.wav \
      --campplus .models/Fun-CosyVoice3-0.5B-2512/campplus.onnx \
      --out-dir output/auto_ref

CosyVoice3 instruct2 与 FireRedTTS-2 的克隆都不需要参考音文本，故不接 ASR。
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_podcast_voices import (  # noqa: E402
    SR, WIN, HOP, SUB,
    load_campplus, campplus_embed, frame_rms, f0_stats,
)
from trim_reference_voice import trim_reference  # noqa: E402

F0_GENDER_SPLIT = 165.0   # Hz，中位数以下判男声
NOISE_FLOOR_MAX = 0.015   # 句间底噪硬门限：超过即认为带 BGM/噪声，丢弃
SNR_MIN = 6.0             # p90/p10 能量比硬门限


def load_episode(path: str, max_minutes: float = 0.0) -> np.ndarray:
    """解码为 16k 单声道 float32。wav 走 soundfile，其余格式用 ffmpeg。"""
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            g = math.gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    except Exception:
        pcm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            check=True, capture_output=True,
        ).stdout
        wav = np.frombuffer(pcm, dtype=np.float32)
    if max_minutes > 0:
        wav = wav[: int(max_minutes * 60 * SR)]
    return wav


def scan_windows(sess, ep: np.ndarray) -> list:
    """滑窗计算嵌入与启发式特征（在 scan_podcast_voices 基础上加底噪/信噪比）。"""
    win_n, hop_n = int(WIN * SR), int(HOP * SR)
    rows = []
    for start in range(0, len(ep) - win_n, hop_n):
        w = ep[start: start + win_n]
        rms_frames = frame_rms(w)
        floor = float(np.percentile(rms_frames, 10))
        p90 = float(np.percentile(rms_frames, 90))
        vad = rms_frames > max(0.008, np.percentile(rms_frames, 25))
        emb = campplus_embed(sess, w)
        subs = [campplus_embed(sess, w[int(i * SR): int((i + SUB) * SR)])
                for i in (0.0, 3.0, 6.0)]
        mean_sub = np.mean(subs, axis=0)
        mean_sub /= np.linalg.norm(mean_sub) + 1e-9
        rows.append({
            "t": start / SR,
            "emb": emb,
            "purity": float(np.mean([np.dot(s, mean_sub) for s in subs])),
            "rms": float(np.sqrt(np.mean(w ** 2))),
            "dyn": float(np.std(rms_frames[vad]) / (np.mean(rms_frames[vad]) + 1e-9))
                   if vad.any() else 0.0,
            "voiced": float(np.mean(vad)),
            "edge_rms": float(min(
                np.sqrt(np.mean(w[: int(0.2 * SR)] ** 2)),
                np.sqrt(np.mean(w[-int(0.2 * SR):] ** 2)),
            )),
            "noise_floor": floor,
            "snr": p90 / (floor + 1e-9),
        })
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} windows...", flush=True)
    print(f"scanned {len(rows)} windows", flush=True)
    return rows


def spherical_kmeans(embs: np.ndarray, k: int, iters: int = 50,
                     restarts: int = 16, seed: int = 0):
    """L2 归一化嵌入上的 k-means（余弦），多次随机重启取簇内相似度最高者。"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = embs[rng.choice(len(embs), k, replace=False)].copy()
        labels = np.zeros(len(embs), dtype=int)
        for _ in range(iters):
            sims = embs @ centers.T
            labels = sims.argmax(axis=1)
            new = centers.copy()
            for i in range(k):
                if (labels == i).any():
                    new[i] = embs[labels == i].mean(axis=0)
            new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
            if np.allclose(new, centers, atol=1e-6):
                centers = new
                break
            centers = new
        cohesion = float(np.mean(np.max(embs @ centers.T, axis=1)))
        if best is None or cohesion > best[0]:
            best = (cohesion, labels, centers)
    return best[1], best[2]


def clean_enough(r: dict) -> bool:
    return (r["purity"] > 0.82 and r["voiced"] > 0.45
            and r["edge_rms"] < 0.03 and r["rms"] > 0.03
            and r["noise_floor"] < NOISE_FLOOR_MAX and r["snr"] > SNR_MIN)


def expressiveness(r: dict) -> float:
    """干净度已过硬门限后，按表现力排序：能量动态 + 响度 + F0 变化。"""
    return (r["purity"] + min(r["voiced"], 0.9)
            + 1.5 * min(r["dyn"], 1.2) + min(r["rms"], 0.2) * 4
            + min(r["f0_std"], 40.0) / 40.0)


def denoise_if_asked(path: Path, enabled: bool) -> Path:
    if not enabled:
        return path
    try:
        import torch  # noqa: F401
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        sys.exit("--denoise 需要 deepfilternet：pip install deepfilternet")
    model, df_state, _ = init_df()
    audio, _ = load_audio(str(path), sr=df_state.sr())
    out = enhance(model, df_state, audio)
    dst = path.with_name(path.stem + "_denoised.wav")
    save_audio(str(dst), out, df_state.sr())
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="整集播客音频（wav/mp3，未裁剪）")
    ap.add_argument("--campplus", required=True, help="campplus.onnx 路径")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speakers", type=int, default=2, help="说话人数量（聚类 k）")
    ap.add_argument("--top", type=int, default=8, help="每个说话人导出的候选数")
    ap.add_argument("--denoise", action="store_true",
                    help="对入选候选做 DeepFilterNet 清洗（默认关，筛选优先）")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="只扫描前 N 分钟（调试用，0=全片）")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 解码 {args.episode}", flush=True)
    ep = load_episode(args.episode, args.max_minutes)
    print(f"  {len(ep) / SR / 60:.1f} min @ {SR}Hz", flush=True)

    print("[2/4] 滑窗扫描 + 嵌入提取", flush=True)
    sess = load_campplus(args.campplus)
    rows = scan_windows(sess, ep)

    speechy = [r for r in rows
               if r["purity"] > 0.75 and r["voiced"] > 0.4 and r["rms"] > 0.02]
    if len(speechy) < args.speakers * 10:
        sys.exit(f"可用语音窗太少（{len(speechy)}），无法聚类：检查音频内容")
    print(f"[3/4] 对 {len(speechy)} 个语音窗做 {args.speakers} 类球面 k-means",
          flush=True)
    embs = np.stack([r["emb"] for r in speechy])
    labels, centers = spherical_kmeans(embs, args.speakers)

    print("[4/4] 逐说话人打分挑选", flush=True)
    report = {}
    for cid in range(args.speakers):
        members = [r for r, lab in zip(speechy, labels)
                   if lab == cid and clean_enough(r)]
        if len(members) < 3:
            print(f"  cluster{cid}: 干净候选不足（{len(members)}），跳过", flush=True)
            continue
        for r in members:
            f0m, f0s, _ = f0_stats(
                ep[int(r["t"] * SR): int((r["t"] + WIN) * SR)])
            r["f0_med"], r["f0_std"] = f0m, f0s
        gender = "male" if np.median([r["f0_med"] for r in members]) < F0_GENDER_SPLIT \
            else "female"
        members.sort(key=expressiveness, reverse=True)
        picked = []
        for r in members:
            if all(abs(r["t"] - p["t"]) > 20 for p in picked):
                picked.append(r)
            if len(picked) >= args.top:
                break
        print(f"  cluster{cid} -> {gender}: {len(members)} 干净候选", flush=True)
        for r in picked:
            print(f"    t={r['t']:7.1f}s f0={r['f0_med']:.0f}±{r['f0_std']:.0f} "
                  f"purity={r['purity']:.3f} floor={r['noise_floor']:.4f} "
                  f"snr={r['snr']:.1f} dyn={r['dyn']:.2f}", flush=True)

        cand_paths = []
        for i, r in enumerate(picked):
            s = int(r["t"] * SR)
            raw = out / f"{gender}_cand_{i:02d}_t{int(r['t'])}s.wav"
            sf.write(raw, ep[s: s + int(WIN * SR)], SR)
            cand_paths.append(denoise_if_asked(raw, args.denoise))
        final = out / f"{gender}.wav"
        trim_reference(str(cand_paths[0]), str(final))
        report[gender] = {
            "cluster": cid, "members": len(members),
            "chosen_t": picked[0]["t"],
            "scores": [{k: r[k] for k in
                        ("t", "f0_med", "f0_std", "purity", "noise_floor",
                         "snr", "dyn", "rms")} for r in picked],
        }

    (out / "auto_ref_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\nreport: {out / 'auto_ref_report.json'}", flush=True)
    print("试听 *_cand_*.wav 后若第一名不是最佳，把更好的候选 trim 后覆盖 "
          "male.wav/female.wav 即可。", flush=True)


if __name__ == "__main__":
    main()
