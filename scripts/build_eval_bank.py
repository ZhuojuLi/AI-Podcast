# -*- coding: utf-8 -*-
"""生成评测题库（eval bank）：固定题材 × 固定轮数，带 Performance Script 表演标签。

题库是模型赛马的"同稿对比"基准：所有 TTS provider 必须跑同一批稿子。
题材覆盖对话难点：知识讲解/强观点/轻松聊天/情绪故事/数字密集/快速问答/
附和接话/争论打断/长回答/幽默。

用法：
  LLM_BASE_URL=http://127.0.0.1:8100/v1 LLM_MODEL=qwen3-14b-awq \
  python3 scripts/build_eval_bank.py --out output/eval_bank --per-topic-turns 12
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOPICS = [
    ("知识讲解", " DeepSeek 为什么能把大模型训练成本打下来"),
    ("强观点", " 实体店不会因为电商而消失"),
    ("轻松聊天", " 咖啡和茶，哪个更适合当打工人的续命水"),
    ("情绪故事", " 一位连续创业者的第五次失败"),
    ("数字密集", " 中国新能源汽车出口的增长密码"),
    ("快速问答", " 关于睡眠的十个常见误区"),
    ("附和接话", " 两个人复盘一次失败的露营经历"),
    ("争论打断", " 预制菜到底该不该进校园"),
    ("长回答", " 讲讲敦煌壁画修复师的一天"),
    ("幽默", " 年轻人为什么越来越爱养宠物"),
]

_GENRE_REQUIREMENT = {
    "知识讲解": "一方负责讲清楚原理，另一方不断追问「为什么」和「所以呢」，有恍然大悟的时刻",
    "强观点": "一方立场鲜明地抛观点，另一方真质疑、真反驳，有来回拉锯",
    "轻松聊天": "氛围松弛，有玩笑、互怼和笑声，像老朋友闲聊",
    "情绪故事": "有情感起伏：惋惜、欣慰、感慨，允许叹气",
    "数字密集": "每轮都带具体数字或比例，重音要落在数字上",
    "快速问答": "节奏快，短问短答，大量 quick_response 式接话",
    "附和接话": "一方讲经历，另一方不断用短句附和、追问、补充",
    "争论打断": "有观点冲突，允许 interrupt 式抢话，语速偏快",
    "长回答": "一方多次给出 60 字以上的完整长段讲述，另一方倾听后总结",
    "幽默": "有自嘲、反差和笑点，至少一处 laughter",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/eval_bank")
    ap.add_argument("--per-topic-turns", type=int, default=12)
    ap.add_argument("--topics", default="", help="只生成指定题材（逗号分隔）")
    args = ap.parse_args()

    from app.providers.llm import QwenScriptProvider

    provider = QwenScriptProvider()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    only = set(args.topics.split(",")) if args.topics else None
    for genre, topic in TOPICS:
        if only and genre not in only:
            continue
        req = _GENRE_REQUIREMENT.get(genre, "")
        # 用完整生成链路（含搜索/表演标签），体裁要求追加在主题里
        t0 = time.time()
        topic_full = f"{topic.strip()}（这期播客的体裁要求：{req}）" if req else topic.strip()
        transcript = provider.generate(
            item_id=f"bank-{genre}", topic=topic_full,
            speaker_gender1="男", speaker_gender2="女",
            target_seconds=int(args.per_topic_turns * 7),
        )
        transcript = provider.annotate_delivery(transcript)
        path = out / f"{genre}.json"
        data = transcript.to_dict()
        data["genre"] = genre
        data["requirement"] = req
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        labeled = sum(1 for t in data["content"] if t.get("delivery"))
        print(f"[bank] {genre}: {len(data['content'])} 轮, "
              f"{labeled} 轮带表演标签, {time.time() - t0:.0f}s -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
