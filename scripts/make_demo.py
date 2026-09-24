# -*- coding: utf-8 -*-
"""本地端到端 Demo：主题 → LLM 文稿 → 双人 TTS → 封面。

用法（在项目根目录）：
  # 1) 用本地 vLLM 写稿（需先起 scripts/start_llm_dev.sh）
  CUDA_VISIBLE_DEVICES=0 <venv>/bin/python scripts/make_demo.py \
      --topic "瑞幸如何靠生椰拿铁翻盘" --g1 男 --g2 女 --target-seconds 120

  # 2) 跳过 LLM，直接用现成文稿
  ... scripts/make_demo.py --transcript-json output/demo/transcript.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TTS_MODEL = os.path.join(
    os.getenv("MODELS_ROOT", "models"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)


def build_transcript(args):
    from app.schemas import Transcript, Turn

    if args.transcript_json:
        with open(args.transcript_json, encoding="utf-8") as f:
            obj = json.load(f)
        turns = [Turn(speaker=int(t["speaker"]), text=str(t["text"]))
                 for t in obj.get("content", [])]
        return Transcript(title=obj.get("title", ""), turns=turns)

    from app.providers.llm import QwenScriptProvider

    t0 = time.time()
    provider = QwenScriptProvider(base_url=args.llm_base_url, model=args.llm_model)
    transcript = provider.generate(
        item_id=args.item_id, topic=args.topic,
        speaker_gender1=args.g1, speaker_gender2=args.g2,
        target_seconds=args.target_seconds,
    )
    print(f"[LLM] {time.time() - t0:.1f}s, {len(transcript.turns)} 轮", flush=True)
    return transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="瑞幸如何靠生椰拿铁翻盘")
    ap.add_argument("--item-id", default="1")
    ap.add_argument("--g1", default="男")
    ap.add_argument("--g2", default="女")
    ap.add_argument("--target-seconds", type=int, default=120)
    ap.add_argument("--out", default="./output/demo")
    ap.add_argument("--transcript-json", default="")
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "qwen2.5-7b-awq"))
    ap.add_argument("--tts-model", default=os.getenv("TTS_MODEL_PATH", DEFAULT_TTS_MODEL))
    ap.add_argument("--turns-limit", type=int, default=0, help="只合成前 N 轮（0=全部）")
    ap.add_argument("--script-only", action="store_true", help="只生成并保存文稿，不加载 TTS")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) 文稿
    transcript = build_transcript(args)
    if args.turns_limit > 0:
        transcript.turns = transcript.turns[:args.turns_limit]
    transcript_path = os.path.join(args.out, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[文稿] {transcript.title} | {len(transcript.turns)} 轮 -> {transcript_path}")

    if args.script_only:
        return

    # 2) 音频
    from app.providers.tts import QwenTTSProvider
    tts = QwenTTSProvider(model_path=args.tts_model)
    mp3_path = os.path.join(args.out, f"{args.item_id}_audio.mp3")
    dur = tts.synthesize_mp3(transcript, args.g1, args.g2, mp3_path)
    print(f"[音频] {dur:.1f}s -> {mp3_path}")

    # 3) 封面
    from app.providers.image import PillowCoverProvider
    cover_path = os.path.join(args.out, f"{args.item_id}_cover.png")
    PillowCoverProvider().generate(args.item_id, transcript.title, args.topic, cover_path)
    print(f"[封面] -> {cover_path}")

    print("\n=== DEMO 产物 ===")
    for name in (f"{args.item_id}_audio.mp3", f"{args.item_id}_cover.png", "transcript.json"):
        p = os.path.join(args.out, name)
        print(f"  {p}  ({os.path.getsize(p) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
