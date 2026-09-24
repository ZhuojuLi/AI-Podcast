# -*- coding: utf-8 -*-
"""生成流水线：把可插拔 provider 组合成契约层需要的两条流。

一次 /generate_audio 请求会：
  1. 创建会话状态（文稿内容队列 + 文稿分段队列）
  2. 后台线程**流式**生成文稿：每段完成即推入内容队列（SSE）和分段队列（TTS）
  3. 后台线程等首段标题就绪后生成封面，供 /download/image/<id> 下载
  4. 前台以生成器形式消费分段队列，**拿到第一批轮次就开始合成并吐出 MP3**

关键：首字节 = 首段文稿时间 + 第一句合成时间，而非整篇文稿时间。
"""
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

from app.config import config
from app.providers.registry import get_cover_provider, get_script_provider, get_tts_provider
from app.state import STORE, ItemState


def _content_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_script(state: ItemState) -> None:
    """后台：流式生成文稿，逐段推入内容队列与分段队列。"""
    queue = state.content_queue
    first = True
    try:
        provider = get_script_provider()
        for seg in provider.iter_segments(
            state.item_id, state.topic, state.speaker_gender1, state.speaker_gender2
        ):
            if seg.title:
                state.set_title(seg.title)
            if not seg.turns:
                continue
            payload = json.dumps([t.to_dict() for t in seg.turns], ensure_ascii=False)
            if first:
                queue.put(("content_start", {
                    "title": state.title or f"AI 播客：{state.topic}",
                    "content": payload,
                }))
                first = False
            else:
                queue.put(("content", {"append": payload}))
            state.turn_queue.put(seg.turns)
            if config.CONTENT_CHUNK_DELAY:
                time.sleep(config.CONTENT_CHUNK_DELAY)
        if first:  # 无任何轮次
            queue.put(("content_start", {
                "title": state.title or f"AI 播客：{state.topic}", "content": "[]"}))
        queue.put(("done", None))
    except Exception as exc:  # noqa: BLE001
        state.mark_failed(f"文稿生成失败: {exc}")
        queue.put(("error", str(exc)))
    finally:
        state.set_title(state.title or f"AI 播客：{state.topic}")
        state.turn_queue.put(None)   # 结束哨兵


def _run_cover(state: ItemState) -> None:
    """后台：等标题就绪后生成封面图。"""
    Path(config.IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(config.IMAGE_DIR, f"{state.cover_id}.png")
    try:
        state.title_event.wait(config.TRANSCRIPT_WAIT_TIMEOUT)
        title = state.title or state.topic or state.item_id
        get_cover_provider().generate(state.item_id, title, state.topic, out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[封面] item_id={state.item_id} 生成失败: {exc}")


def start_session(
    item_id: str,
    topic: str,
    speaker_gender1: str,
    speaker_gender2: str,
) -> ItemState:
    """创建会话并启动后台文稿 / 封面线程，返回会话状态。"""
    Path(config.AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    state = STORE.create(
        item_id=item_id,
        topic=topic,
        speaker_gender1=speaker_gender1,
        speaker_gender2=speaker_gender2,
        audio_id=f"{item_id}_audio",
        cover_id=f"{item_id}_cover",
    )
    state.content_thread = threading.Thread(target=_run_script, args=(state,), daemon=True)
    state.cover_thread = threading.Thread(target=_run_cover, args=(state,), daemon=True)
    state.content_thread.start()
    state.cover_thread.start()
    return state


def audio_stream(state: ItemState) -> Iterator[bytes]:
    """前台生成器：逐轮输出 MP3，同时保存完整文件供下载接口使用。"""
    temp_path = None
    try:
        tts = get_tts_provider()
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".podcast-", suffix=".mp3", dir=config.AUDIO_DIR,
            delete=False,
        ) as audio_file:
            temp_path = audio_file.name
            for chunk in tts.stream_turns(
                state.iter_segments(), state.speaker_gender1, state.speaker_gender2
            ):
                audio_file.write(chunk)
                yield chunk
        os.replace(temp_path, os.path.join(config.AUDIO_DIR, f"{state.audio_id}.mp3"))
        temp_path = None
    except GeneratorExit:
        # 客户端中断（连接关闭）：不算服务故障，但会话不能停在 processing
        state.mark_failed("客户端中断，生成取消")
        raise
    except Exception as exc:  # noqa: BLE001
        state.mark_failed(f"音频生成失败: {exc}")
        state.content_queue.put(("error", f"音频生成失败: {exc}"))
        raise
    else:
        state.mark_completed()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if state.cover_thread is not None:
            state.cover_thread.join(timeout=config.TRANSCRIPT_WAIT_TIMEOUT)


def content_events(state: ItemState) -> Iterator[str]:
    """前台生成器：从内容队列消费并产出 SSE 文本。"""
    queue = state.content_queue
    while True:
        try:
            msg_type, msg_data = queue.get(timeout=config.CONTENT_QUEUE_TIMEOUT)
        except Exception:  # queue.Empty
            yield _content_event("error", {"item_id": state.item_id, "message": "等待文稿数据超时"})
            break

        if msg_type == "content_start":
            payload = {"item_id": state.item_id, "title": msg_data["title"], "content": msg_data["content"]}
            yield _content_event("content_start", payload)
        elif msg_type == "content":
            yield _content_event("content", {"item_id": state.item_id, "append": msg_data["append"]})
        elif msg_type == "done":
            yield _content_event("done", {})
            break
        elif msg_type == "error":
            yield _content_event("error", {"item_id": state.item_id, "message": str(msg_data)})
            break
