# -*- coding: utf-8 -*-
"""CosyVoice 端到端验证：现成文稿 → CosyTTSProvider.stream_mp3 → mp3。

覆盖：男男（走 male2 新参考音）与男女组合、问句/短回应/长段/数字/收束等
不同语气分支、动态轮间停顿、0.97 全局减速。

容器内运行：
  PYTHONPATH=/work /opt/podcast-venv/bin/python scripts/make_cosy_demo.py \
      --g1 男 --g2 男 --out output/feedback_091/male_male
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMO_TURNS = [
    (1, "大家好，欢迎收听今天的节目，我们要聊的是一个让很多人意外的商业故事。"),
    (2, "没错，这家公司最惨的时候，账上的钱只够撑三个月。"),
    (1, "说到这我想问一句，当时真的没有人看好这个项目吗？"),
    (2, "问题恰恰出在成本端。"),
    (1, "但你注意到没有，他们竟然在一年内把门店翻了三倍，这个速度谁见过？"),
    (2, "数据显示，上线第一个月就卖出了一千万杯，复购率还远远高于店里其他单品，这说明用户不是图便宜，而是真的认可这个产品。"),
    (1, "话又说回来，这种增长能持续吗？我觉得得看供应链跟不跟得上。"),
    (2, "这个判断我同意，规模一旦上去，品控就是生死线。"),
    (1, "说到底，这一仗赢就赢在了把资源压在了用户真正在意的地方。"),
    (2, "聊到这里，今天的分享差不多了，我们下期接着聊，再见。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", default="男")
    ap.add_argument("--g2", default="男")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcript-json", default="",
                    help="跳过 LLM，直接用现成文稿（content/speaker 结构）")
    ap.add_argument("--model", default=os.getenv(
        "TTS_MODEL_PATH", "/work/.models/Fun-CosyVoice3-0.5B-2512"))
    args = ap.parse_args()

    os.environ["TTS_MODEL_PATH"] = args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from app.providers.cosy import CosyTTSProvider
    from app.schemas import Transcript, Turn

    if args.transcript_json:
        obj = json.loads(Path(args.transcript_json).read_text(encoding="utf-8"))
        transcript = Transcript(
            title=obj.get("title", ""),
            turns=[Turn(speaker=int(t["speaker"]), text=str(t["text"]))
                   for t in obj.get("content", [])],
        )
    else:
        transcript = Transcript(
            title="反馈修复验证 Demo",
            turns=[Turn(speaker=s, text=t) for s, t in DEMO_TURNS],
        )
    (out / "transcript.json").write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2))

    provider = CosyTTSProvider()
    provider.preload()

    t0 = time.time()
    chunks = []
    total = 0
    for chunk in provider.stream_mp3(transcript, args.g1, args.g2):
        chunks.append(chunk)
        total += len(chunk)
    audio = b"".join(chunks)
    (out / "1_audio.mp3").write_bytes(audio)
    print(f"[demo] mp3 {total / 1024:.0f}KiB, {time.time() - t0:.1f}s -> {out / '1_audio.mp3'}")


if __name__ == "__main__":
    main()
