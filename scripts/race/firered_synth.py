#!/usr/bin/env python3
"""FireRedTTS-2 赛马适配器：bank JSON -> mp3。

用法：
  python scripts/race/firered_synth.py --script <bank.json> --out <out.mp3> \
      --repo <FireRedTTS-2 源码目录> --model-dir <FireRedTTS-2 权重目录> \
      --prompt-dir <包含 prompt_S1.wav/prompt_S2.wav[/prompts.json] 的目录>

环境变量 FIRERED_REPO / FIRERED_MODEL_DIR / FIRERED_PROMPT_DIR 可替代命令行参数。
FireRed 以整段对话方式生成（轮替/韵律由模型内部决定），不再额外插静音。
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
import soundfile as sf


def _sf_load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def _sf_save(path, tensor, sr):
    sf.write(path, tensor.squeeze(0).cpu().numpy(), sr, subtype="PCM_16")


import torchaudio
torchaudio.load = _sf_load
torchaudio.save = _sf_save

REPO = os.environ.get("FIRERED_REPO", "")
MODEL_DIR = os.environ.get("FIRERED_MODEL_DIR", "models/FireRedTTS2")
DEFAULT_PROMPT_DIR = os.environ.get("FIRERED_PROMPT_DIR", "resources/voices")
PROMPT_TEXTS_DEFAULT = [
    "[S1]啊，可能说更适合美国市场应该是什么样子。那这这个可能说当然如果说有有机会能亲身的去考察去了解一下，那当然是有更好的帮助。",
    "[S2]比如具体一点的，他觉得最大的一个跟他预想的不一样的是在什么地方。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO,
                    help="FireRedTTS-2 源码目录（fireredtts2 包的父目录），"
                         "也可用 FIRERED_REPO 环境变量")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="FireRedTTS-2 权重目录（FIRERED_MODEL_DIR）")
    ap.add_argument("--prompt-dir", default=DEFAULT_PROMPT_DIR,
                    help="包含 prompt_S1.wav/prompt_S2.wav[/prompts.json] 的目录；"
                         "prompts.json 存在时优先用其中的 text 字段（干声 prompt 场景）")
    args = ap.parse_args()

    if args.repo:
        sys.path.insert(0, args.repo)
    from fireredtts2.fireredtts2 import FireRedTTS2

    with open(args.script, encoding="utf-8") as f:
        script = json.load(f)

    prompt_wavs = [f"{args.prompt_dir}/prompt_S1.wav",
                   f"{args.prompt_dir}/prompt_S2.wav"]
    prompt_texts = list(PROMPT_TEXTS_DEFAULT)
    prompts_json = f"{args.prompt_dir}/prompts.json"
    if os.path.isfile(prompts_json):
        with open(prompts_json, encoding="utf-8") as f:
            meta = json.load(f)
        for i, name in enumerate(("S1", "S2")):
            prompt_texts[i] = f"[{name}]{meta[name]['text']}"
            prompt_wavs[i] = meta[name].get("wav", prompt_wavs[i])

    model = FireRedTTS2(pretrained_dir=args.model_dir, gen_type="dialogue",
                        device="cuda", use_bf16=True)
    text_list = [f"[S{t['speaker']}]{t['text']}" for t in script["content"]]
    t0 = time.perf_counter()
    audio = model.generate_dialogue(
        text_list=text_list,
        prompt_wav_list=prompt_wavs,
        prompt_text_list=prompt_texts,
        temperature=0.9,
        topk=30,
    )
    dt = time.perf_counter() - t0

    wav_path = args.out + ".wav"
    sf.write(wav_path, audio.squeeze(0).cpu().numpy(), 24000,
             subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                    "-ac", "1", "-ar", "24000", "-b:a", "96k", args.out],
                   check=True)
    os.remove(wav_path)
    dur = audio.shape[1] / 24000
    print(f"[race-firered] {dur:.1f}s audio, {dt:.1f}s synth, "
          f"rtf={dt / dur:.2f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
