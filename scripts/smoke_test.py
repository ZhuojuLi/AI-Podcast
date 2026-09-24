# -*- coding: utf-8 -*-
"""契约冒烟测试：复刻评测端 run_for_darvin 的核心校验流程。

延迟口径（三者分开报告，勿混用）：
  first_byte   首个响应字节 —— 服务端开了 TTS_EARLY_ID3_KIB 时，
               这是提前发送的 ID3 元数据到达时间，不代表能听到声音
  first_audio  首段可播放语音到达 —— 跳过前导 ID3 标签后，
               第一个 MP3 音频帧的到达时间（用户感知口径）
  elapsed/rtf  整单完成时间与实时率

用法：
    python scripts/smoke_test.py --base-url http://localhost:8086
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import requests


def first_audio_offset(buf: bytearray):
    """跳过前导 ID3v2 标签，返回首个 MP3 音频帧偏移；数据不足返回 None。"""
    offset = 0
    while len(buf) >= offset + 10 and buf[offset:offset + 3] == b"ID3":
        size = 0
        for byte in buf[offset + 6: offset + 10]:   # syncsafe 整数
            size = (size << 7) | (byte & 0x7F)
        offset += 10 + size
    if len(buf) > offset + 1 and buf[offset] == 0xFF \
            and (buf[offset + 1] & 0xE0) == 0xE0:
        return offset
    return None


def parse_sse(lines):
    """解析 SSE 文本行，返回 (events, transcript)。"""
    events = []
    event, buf = "", ""
    title = ""
    content = []
    for line in lines:
        if line is None:
            continue
        if line == "":
            if event and buf:
                events.append(event)
                try:
                    payload = json.loads(buf)
                except json.JSONDecodeError:
                    payload = {}
                if event == "content_start":
                    title = payload.get("title", "")
                    content.extend(json.loads(payload.get("content", "[]")))
                elif event == "content":
                    content.extend(json.loads(payload.get("append", "[]")))
            event, buf = "", ""
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            buf += line[len("data:"):].strip()
    return events, {"title": title, "content": content}


def run_case(base_url, item_id, topic, g1, g2, out_dir):
    results = {}

    def _audio():
        path = os.path.join(out_dir, f"{item_id}_audio.mp3")
        first_chunk = None
        first_audio = None
        buf = bytearray()
        start = time.time()
        with requests.post(
            f"{base_url}/generate_audio",
            data={"item_id": str(item_id), "topic": topic,
                  "speaker_gender1": g1, "speaker_gender2": g2},
            stream=True, timeout=600,
        ) as resp:
            if resp.status_code != 200:
                results["audio_error"] = f"HTTP {resp.status_code}"
                return
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        now = time.time() - start
                        if first_chunk is None:
                            first_chunk = now
                        if first_audio is None:
                            buf += chunk
                            if first_audio_offset(buf) is not None:
                                first_audio = now
                        f.write(chunk)
        results["audio_path"] = path
        results["first_chunk"] = first_chunk or 0.0
        results["first_audio"] = first_audio if first_audio is not None else -1.0

    def _content():
        with requests.post(
            f"{base_url}/generate_content",
            data={"item_id": str(item_id)},
            stream=True, timeout=600,
        ) as resp:
            if resp.status_code != 200:
                results["content_error"] = f"HTTP {resp.status_code}"
                return
            events, transcript = parse_sse(resp.iter_lines(decode_unicode=True))
        results["events"] = events
        results["transcript"] = transcript

    ta = threading.Thread(target=_audio)
    tc = threading.Thread(target=_content)
    ta.start(); tc.start()
    ta.join(); tc.join()

    problems = []
    if "audio_error" in results:
        problems.append(results["audio_error"])
    if "content_error" in results:
        problems.append(results["content_error"])

    # 1) SSE 事件序列
    events = results.get("events", [])
    if not events or events[0] != "content_start" or events[-1] != "done":
        problems.append(f"SSE 事件序列异常: {events}")
    turns = results.get("transcript", {}).get("content", [])
    if not turns:
        problems.append("文稿为空")
    else:
        for t in turns:
            if t.get("speaker") not in (1, 2) or not isinstance(t.get("text"), str):
                problems.append(f"轮次结构异常: {t}")
                break

    # 2) 音频时长
    audio_path = results.get("audio_path")
    duration = 0.0
    if audio_path and os.path.exists(audio_path):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        try:
            duration = float(out.stdout.strip())
        except ValueError:
            problems.append("ffprobe 解析时长失败")
    else:
        problems.append("音频文件缺失")
    if duration and not (300 <= duration <= 900):
        problems.append(f"音频时长越界: {duration:.1f}s（需 300-900s）")
    if results.get("first_chunk", 0) > 30:
        problems.append(f"首 chunk 超时: {results['first_chunk']:.2f}s")
    # 3) 封面
    cover_path = os.path.join(out_dir, f"{item_id}_cover.png")
    r = requests.get(f"{base_url}/download/image/{item_id}_cover", timeout=120)
    if r.status_code != 200:
        problems.append(f"封面下载失败 HTTP {r.status_code}: {r.text[:120]}")
    else:
        with open(cover_path, "wb") as f:
            f.write(r.content)
        try:
            from PIL import Image
            with Image.open(cover_path) as img:
                if img.size != (1024, 1024) or img.format != "PNG":
                    problems.append(f"封面规格错误: {img.size} {img.format}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"封面校验异常: {exc}")

    return results, problems, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8086")
    parser.add_argument("--item-id", default="1")
    parser.add_argument("--topic", default="京东自建物流如何从烧钱变成护城河")
    parser.add_argument("--g1", default="男")
    parser.add_argument("--g2", default="女")
    parser.add_argument("--out", default="./output/smoke")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # /ready
    ready = requests.get(f"{args.base_url}/ready", timeout=30)
    print(f"/ready -> {ready.status_code} {ready.text!r}")
    if ready.status_code != 200:
        print("FAIL: /ready 未就绪")
        sys.exit(1)

    start = time.time()
    results, problems, duration = run_case(
        args.base_url, args.item_id, args.topic, args.g1, args.g2, args.out
    )
    elapsed = time.time() - start
    rtf = elapsed / duration if duration else 0

    print(f"events       : {results.get('events')}")
    print(f"turns        : {len(results.get('transcript', {}).get('content', []))}")
    print(f"title        : {results.get('transcript', {}).get('title')}")
    print(f"duration     : {duration:.1f}s")
    print(f"first_byte   : {results.get('first_chunk', 0):.2f}s"
          "  (首个响应字节，含提前发送的 ID3 元数据)")
    first_audio = results.get("first_audio", -1.0)
    if first_audio >= 0:
        print(f"first_audio  : {first_audio:.2f}s"
              "  (首段可播放语音到达，用户感知口径)")
        if first_audio > 30:
            print("WARN: 首段可播放语音超过 30s，用户感知偏慢（不影响评测门禁）")
    else:
        print("first_audio  : 未检测到有效 MP3 帧（请检查流内容）")
    print(f"elapsed      : {elapsed:.1f}s   rtf={rtf:.3f}")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nPASS: 契约与机器校验全部通过")


if __name__ == "__main__":
    main()
