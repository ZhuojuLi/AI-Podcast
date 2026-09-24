# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from unittest.mock import patch

from app.config import config
from app.pipeline import audio_stream
from app.state import ItemState


class AudioPersistenceTest(unittest.TestCase):
    def test_stream_is_saved_for_download(self):
        class FakeTTS:
            def stream_turns(self, segments, gender1, gender2):
                list(segments)
                yield b"ID3"
                yield b"FRAME"

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, "AUDIO_DIR", directory), \
             patch("app.pipeline.get_tts_provider", return_value=FakeTTS()):
            state = ItemState("test", "topic", "男", "女", "test_audio", "test_cover")
            state.turn_queue.put([])
            state.turn_queue.put(None)
            self.assertEqual(list(audio_stream(state)), [b"ID3", b"FRAME"])
            with open(os.path.join(directory, "test_audio.mp3"), "rb") as output:
                self.assertEqual(output.read(), b"ID3FRAME")
            self.assertEqual(state.status, "completed")

    def test_tts_failure_marks_failed_not_completed(self):
        """合成中途抛错：状态必须是 failed，不能落盘完整文件，错误进内容队列。"""

        class BrokenTTS:
            def stream_turns(self, segments, gender1, gender2):
                list(segments)
                yield b"ID3"
                raise RuntimeError("合成爆炸")

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, "AUDIO_DIR", directory), \
             patch("app.pipeline.get_tts_provider", return_value=BrokenTTS()):
            state = ItemState("t2", "topic", "男", "女", "t2_audio", "t2_cover")
            state.turn_queue.put(None)
            stream = audio_stream(state)
            self.assertEqual(next(stream), b"ID3")
            with self.assertRaises(RuntimeError):
                list(stream)
            self.assertEqual(state.status, "failed")
            self.assertIn("合成爆炸", state.error)
            self.assertTrue(state.finished)
            self.assertFalse(os.path.exists(os.path.join(directory, "t2_audio.mp3")))
            self.assertEqual(state.content_queue.get_nowait()[0], "error")

    def test_client_disconnect_marks_failed(self):
        """客户端中断（GeneratorExit）：会话不能停在 processing。"""

        class SlowTTS:
            def stream_turns(self, segments, gender1, gender2):
                list(segments)
                yield b"ID3"
                yield b"FRAME"

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, "AUDIO_DIR", directory), \
             patch("app.pipeline.get_tts_provider", return_value=SlowTTS()):
            state = ItemState("t3", "topic", "男", "女", "t3_audio", "t3_cover")
            state.turn_queue.put(None)
            stream = audio_stream(state)
            self.assertEqual(next(stream), b"ID3")
            stream.close()   # 模拟客户端断开
            self.assertEqual(state.status, "failed")
            self.assertIn("客户端中断", state.error)
            self.assertFalse(os.path.exists(os.path.join(directory, "t3_audio.mp3")))


if __name__ == "__main__":
    unittest.main()
