# -*- coding: utf-8 -*-
"""生成能力抽象接口。

真实后端只需实现这三个接口之一，即可通过环境变量接入，无需改动契约层：

  - SCRIPT_PROVIDER : 文稿生成（LLM）
  - TTS_PROVIDER    : 音频合成（TTS），必须输出 MP3 字节流
  - COVER_PROVIDER  : 封面图生成

选择方式：``<provider_name>`` 或 ``<python.module.path>:<ClassName>``，
例如 ``SCRIPT_PROVIDER=app.providers.my_llm:MyLLMScriptProvider``。
"""
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from app.schemas import ScriptSegment, Transcript, Turn


class ScriptProvider(ABC):
    """文稿生成后端。"""

    name = "base"

    @abstractmethod
    def generate(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Transcript:
        """根据主题生成一份双人播客文稿。"""
        raise NotImplementedError

    def iter_segments(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str = "",
        speaker_gender2: str = "",
        target_seconds: int = 480,
    ) -> Iterator[ScriptSegment]:
        """流式产出文稿片段；默认实现一次性产出全部（子类可覆盖为真流式）。"""
        transcript = self.generate(item_id, topic, speaker_gender1, speaker_gender2)
        yield ScriptSegment(turns=transcript.turns, title=transcript.title)


class TTSProvider(ABC):
    """音频合成后端。必须产出可直接拼接的 MP3 字节流。"""

    name = "base"

    @abstractmethod
    def stream_mp3(
        self,
        transcript: Optional[Transcript],
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        """流式产出 MP3 字节。"""
        raise NotImplementedError

    def stream_turns(
        self,
        segments: Iterator[List[Turn]],
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        """按段流式合成：默认先把所有轮次收集起来再调用 stream_mp3。"""
        turns: List[Turn] = []
        for seg in segments:
            turns.extend(seg)
        transcript = Transcript(title="", turns=turns)
        yield from self.stream_mp3(transcript, speaker_gender1, speaker_gender2)


class CoverProvider(ABC):
    """封面图生成后端。输出必须为 PNG，且分辨率符合评测要求。"""

    name = "base"

    @abstractmethod
    def generate(
        self,
        item_id: str,
        title: str,
        topic: str,
        out_path: str,
    ) -> bool:
        """在 out_path 生成封面图，成功返回 True。"""
        raise NotImplementedError
