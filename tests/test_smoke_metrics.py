# -*- coding: utf-8 -*-
"""smoke_test 延迟口径解析的回归测试：ID3 填充不得计入首段可播放语音。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from smoke_test import first_audio_offset  # noqa: E402


def make_id3(payload_size: int) -> bytes:
    """与 app.providers.cosy._id3_padding 相同的 ID3v2.3 空白标签。"""
    syncsafe = bytes((payload_size >> shift) & 0x7F for shift in (21, 14, 7, 0))
    return b"ID3\x03\x00\x00" + syncsafe + bytes(payload_size)


FRAME = b"\xff\xfb" + bytes(100)   # MP3 帧同步头 + 帧体


class FirstAudioOffsetTest(unittest.TestCase):
    def test_no_id3(self):
        self.assertEqual(first_audio_offset(bytearray(FRAME)), 0)

    def test_padding_then_frame(self):
        data = make_id3(1024) + FRAME
        self.assertEqual(first_audio_offset(bytearray(data)), 1034)

    def test_two_tags_then_frame(self):
        data = make_id3(100) + make_id3(50) + FRAME
        self.assertEqual(first_audio_offset(bytearray(data)), 110 + 60)

    def test_partial_tag_body_waits(self):
        data = make_id3(1024)[:512]   # 标签体未收完
        self.assertIsNone(first_audio_offset(bytearray(data)))

    def test_partial_frame_waits(self):
        data = make_id3(16) + b"\xff"   # 帧头只来了一半
        self.assertIsNone(first_audio_offset(bytearray(data)))

    def test_garbage_is_not_audio(self):
        self.assertIsNone(first_audio_offset(bytearray(b"\x00\x01\x02\x03")))


if __name__ == "__main__":
    unittest.main()
