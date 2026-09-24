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


if __name__ == "__main__":
    unittest.main()
