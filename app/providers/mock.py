# -*- coding: utf-8 -*-
"""默认 mock 生成后端。

用于在接入真实模型前跑通整条契约链路与机器校验：
  - 文稿：按主题拼接固定话术
  - 音频：ffmpeg 生成指定时长的静音 MP3（16kHz 单声道）
  - 封面：Pillow 生成 1024x1024 PNG
"""
import random
import subprocess
import time
from typing import Iterator, Optional

from app.config import config
from app.schemas import Transcript, Turn
from app.providers.base import CoverProvider, ScriptProvider, TTSProvider


class MockScriptProvider(ScriptProvider):
    name = "mock"

    def generate(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Transcript:
        subject = topic or "当前话题"
        title = f"AI 播客：{topic}" if topic else f"AI 播客 Episode {item_id}"
        turns = [
            Turn(1, f"大家好，欢迎收听本期 AI 播客。今天我们聊的主题是{subject}。"),
            Turn(2, "开场我们先简单介绍一下背景。这是一个值得深入探讨的话题，背后有很多有趣的故事值得我们去挖掘。"),
            Turn(1, "你觉得这个话题最值得关注的是哪一点？"),
            Turn(2, "我认为最重要的是它的实践意义，很多理念听起来很好，但真正落地却面临不少挑战。"),
            Turn(1, "我们接下来会围绕几个关键问题展开讨论，包括它的背景、发展、现状以及未来的趋势。希望对大家有所帮助。"),
            Turn(2, "感谢收听，我们下期再见！"),
        ]
        return Transcript(title=title, turns=turns)


class MockTTSProvider(TTSProvider):
    name = "mock"

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        bitrate: str = None,
        min_duration: int = None,
        max_duration: int = None,
        chunk_size: int = None,
        chunk_delay: float = None,
    ) -> None:
        self.sample_rate = sample_rate or config.AUDIO_SAMPLE_RATE
        self.channels = channels or config.AUDIO_CHANNELS
        self.bitrate = bitrate or config.AUDIO_BITRATE
        self.min_duration = min_duration or config.AUDIO_MIN_DURATION
        self.max_duration = max_duration or config.AUDIO_MAX_DURATION
        self.chunk_size = chunk_size or config.AUDIO_CHUNK_SIZE
        self.chunk_delay = config.AUDIO_CHUNK_DELAY if chunk_delay is None else chunk_delay

    def stream_mp3(
        self,
        transcript: Optional[Transcript],
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        duration = random.randint(self.min_duration, self.max_duration)
        layout = "mono" if self.channels == 1 else "stereo"
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", f"anullsrc=r={self.sample_rate}:cl={layout}",
                "-t", str(duration),
                "-c:a", "libmp3lame", "-b:a", self.bitrate,
                "-f", "mp3", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            while True:
                chunk = proc.stdout.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
                if self.chunk_delay:
                    time.sleep(self.chunk_delay)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            _, err = proc.communicate()
            if proc.returncode not in (0, None):
                message = (err or b"").decode("utf-8", "ignore")
                raise RuntimeError(f"ffmpeg 生成音频失败: {message[:300]}")


class MockCoverProvider(CoverProvider):
    name = "mock"

    def __init__(self, width: int = None, height: int = None) -> None:
        self.width = width or config.COVER_REQUIRED_WIDTH
        self.height = height or config.COVER_REQUIRED_HEIGHT

    def generate(
        self,
        item_id: str,
        title: str,
        topic: str,
        out_path: str,
    ) -> bool:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new(
            "RGB",
            (self.width, self.height),
            color=(random.randint(24, 200), random.randint(24, 200), random.randint(24, 200)),
        )
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except Exception:
            font = ImageFont.load_default()
        draw.text((self.width // 8, self.height // 2), "AI Podcast Cover",
                  fill=(255, 255, 255), font=font)
        img.save(out_path, "PNG")
        return True
