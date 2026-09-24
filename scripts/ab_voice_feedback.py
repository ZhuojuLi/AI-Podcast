# -*- coding: utf-8 -*-
"""反馈修复的音色/语气 AB 试听：用 CosyVoice3 合成多组例句，输出 mp3 供人工试听，
并统计每轮的时长与首尾噪声电平（验证女声2杂音修复效果）。

容器内运行（GPU）：
  PYTHONPATH=/work:/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
  /opt/podcast-venv/bin/python scripts/ab_voice_feedback.py \
      --model /work/.models/Fun-CosyVoice3-0.5B-2512 --out output/voice_ab
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

# 与 cosy.py 保持同一套提示策略
from app.providers import cosy as cosy_mod

SAMPLE_TURNS = [
    "大家好，欢迎收听今天的节目，我们要聊的是一个让很多人意外的商业故事。",
    "说到这我想问一句，当时真的没有人看好这个项目吗？",
    "问题恰恰出在成本端。",
    "但你注意到没有，他们竟然在一年内把门店翻了三倍，这个速度谁见过？",
    "数据显示，上线第一个月就卖出了一千万杯，复购率还远远高于店里其他单品。",
    "说到底，这一仗赢就赢在了把补贴烧在了用户真正在意的地方。",
]

MALE2_OLD = "resources/voices/male2.wav"
FEMALE2_OLD = "resources/voices/female2.wav"


def synth_turns(model, ref, turns, voice_style, tempo):
    results = []
    for i, text in enumerate(turns):
        prompt = ("You are a helpful assistant. "
                  + cosy_mod._instruction(text, i, voice_style) + "<|endofprompt|>")
        with cosy_mod._INFER_LOCK:
            chunks = list(model.inference_instruct2(
                text, prompt, ref, stream=False, speed=1.0, text_frontend=False))
        wav = np.concatenate(
            [c["tts_speech"].detach().cpu().numpy().squeeze(0) for c in chunks])
        if tempo != 1.0:
            wav = cosy_mod._change_tempo(wav.astype(np.float32), model.sample_rate, tempo)
        results.append(wav.astype(np.float32))
    return results


def edge_noise(wav, sr):
    """首尾 80ms RMS（dB），用于检验参考音边缘杂音是否被克隆出来。"""
    n = int(0.08 * sr)
    def db(x):
        r = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        return round(20 * np.log10(r + 1e-9), 1)
    return db(wav[:n]), db(wav[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--male2-new", action="append", default=[])
    ap.add_argument("--female2-new", action="append", default=[])
    ap.add_argument("--tempo", type=float, default=0.97)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    offline_wetext = types.ModuleType("wetext")
    offline_wetext.Normalizer = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("offline only"))
    sys.modules["wetext"] = offline_wetext
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.cli import frontend as cosy_frontend
    cosy_frontend.load_wav = cosy_mod._load_reference_wav
    import torch
    prev = torch.get_default_device()
    try:
        torch.set_default_device("cuda")
        model = AutoModel(model_dir=args.model, fp16=True)
    finally:
        torch.set_default_device(prev)

    groups = [("male2_old", MALE2_OLD, cosy_mod._SECONDARY_MALE_STYLE)]
    for p in args.male2_new:
        groups.append((f"male2_new_{Path(p).stem}", p, cosy_mod._SECONDARY_MALE_STYLE))
    groups.append(("female2_old", FEMALE2_OLD, cosy_mod._SECONDARY_FEMALE_STYLE))
    for p in args.female2_new:
        groups.append((f"female2_new_{Path(p).stem}", p, cosy_mod._SECONDARY_FEMALE_STYLE))

    stats = {}
    for name, ref, style in groups:
        started = time.time()
        wavs = synth_turns(model, ref, SAMPLE_TURNS, style, args.tempo)
        # 连成一段（带 200ms 间隔）便于试听
        gap = np.zeros(int(model.sample_rate * 0.2), dtype=np.float32)
        merged = np.concatenate([x for w in wavs for x in (w, gap)])
        mp3 = cosy_mod._encode_mp3(merged, model.sample_rate, with_id3=True)
        (out / f"{name}.mp3").write_bytes(mp3)
        stats[name] = {
            "ref": ref,
            "turn_dur_s": [round(len(w) / model.sample_rate, 2) for w in wavs],
            "edge_db": [edge_noise(w, model.sample_rate) for w in wavs],
            "total_s": round(len(merged) / model.sample_rate, 2),
            "cost_s": round(time.time() - started, 1),
        }
        print(f"[AB] {name}: {stats[name]['total_s']}s "
              f"edges={stats[name]['edge_db']}", flush=True)
    (out / "ab_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
