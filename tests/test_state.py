# -*- coding: utf-8 -*-
"""ItemState 生命周期与文稿同步的回归测试。"""
import unittest

from app.schemas import Transcript, Turn
from app.state import ItemState


def make_state() -> ItemState:
    return ItemState("s1", "topic", "男", "女", "s1_audio", "s1_cover")


class TranscriptSyncTest(unittest.TestCase):
    def test_fresh_state_wait_transcript_returns_none(self):
        """回归：同步字段曾延迟到 iter_segments() 结束才初始化，
        新建对象直接 wait_transcript 会 AttributeError。"""
        state = make_state()
        self.assertIsNone(state.wait_transcript(0.01))

    def test_set_transcript_before_iter_segments(self):
        state = make_state()
        transcript = Transcript(title="标题", turns=[Turn(1, "你好")])
        state.set_transcript(transcript)
        self.assertIs(state.wait_transcript(0.01), transcript)

    def test_wait_transcript_after_iter_segments(self):
        state = make_state()
        state.turn_queue.put([Turn(1, "你好")])
        state.turn_queue.put(None)
        self.assertEqual(len(list(state.iter_segments())), 1)
        state.set_transcript(None)
        self.assertIsNone(state.wait_transcript(0.01))


class LifecycleTest(unittest.TestCase):
    def test_completed_not_overwritten_by_failed(self):
        state = make_state()
        state.mark_completed()
        state.mark_failed("晚到的错误")
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.error, "")

    def test_failed_not_overwritten_by_completed(self):
        state = make_state()
        state.mark_failed("文稿生成失败: boom")
        state.mark_completed()
        self.assertEqual(state.status, "failed")
        self.assertIn("boom", state.error)
        self.assertTrue(state.finished)


if __name__ == "__main__":
    unittest.main()
