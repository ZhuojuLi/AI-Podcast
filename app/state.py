# -*- coding: utf-8 -*-
"""按 item_id 维护的生成会话状态。

每个 case 由 /generate_audio 创建一份会话，包含：
  - 文稿共享队列（供 /generate_content 消费）
  - 文稿对象与就绪事件（供 TTS 后端等待真实文稿）
  - 音频 / 封面文件 id
"""
import queue
import threading
import time
from typing import Iterator, List, Optional

from app.schemas import Transcript, Turn


class ItemState:
    def __init__(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str,
        speaker_gender2: str,
        audio_id: str,
        cover_id: str,
    ) -> None:
        self.item_id = item_id
        self.topic = topic
        self.speaker_gender1 = speaker_gender1
        self.speaker_gender2 = speaker_gender2
        self.audio_id = audio_id
        self.cover_id = cover_id

        self.content_queue: "queue.Queue" = queue.Queue()
        # 文稿分段队列：给 TTS 按段消费（段为 List[Turn]，末尾 None 为结束哨兵）
        self.turn_queue: "queue.Queue" = queue.Queue()
        self.title = ""
        self.title_event = threading.Event()
        self.status = "processing"
        self.finished = False
        self.created_at = time.time()
        self.cover_thread = None
        self.content_thread = None

    def set_title(self, title: str) -> None:
        if title and not self.title:
            self.title = title
            self.title_event.set()

    def iter_segments(self) -> Iterator[List[Turn]]:
        """从队列消费文稿分段，直到 None 哨兵。"""
        while True:
            seg = self.turn_queue.get()
            if seg is None:
                break
            yield seg

        self._transcript: Optional[Transcript] = None
        self._transcript_event = threading.Event()
        self._lock = threading.Lock()

    # ---- 文稿就绪同步 ----
    def set_transcript(self, transcript: Optional[Transcript]) -> None:
        with self._lock:
            self._transcript = transcript
        self._transcript_event.set()

    def wait_transcript(self, timeout: float) -> Optional[Transcript]:
        self._transcript_event.wait(timeout)
        with self._lock:
            return self._transcript

    # ---- 生命周期 ----
    def mark_completed(self) -> None:
        self.status = "completed"
        self.finished = True


class ItemStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items = {}

    def create(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str,
        speaker_gender2: str,
        audio_id: str,
        cover_id: str,
    ) -> ItemState:
        state = ItemState(item_id, topic, speaker_gender1, speaker_gender2, audio_id, cover_id)
        with self._lock:
            self._items[item_id] = state
        return state

    def get(self, item_id: str) -> Optional[ItemState]:
        with self._lock:
            return self._items.get(item_id)

    def wait_for(self, item_id: str, timeout: float):
        """等待 /generate_audio 创建会话（并发请求时 /generate_content 可能先到）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get(item_id)
            if state is not None:
                return state
            time.sleep(0.2)
        return None


STORE = ItemStore()
