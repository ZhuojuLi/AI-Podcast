# -*- coding: utf-8 -*-
"""Performance Script 协议测试：行尾 JSON 表演指示的解析与 instruct 映射。"""
import unittest

from app.providers import cosy, llm
from app.providers.llm import _parse_dialogue
from app.schemas import Delivery, Transcript, Turn


class ParseDeliveryTest(unittest.TestCase):
    def test_line_with_delivery(self):
        t = _parse_dialogue(
            "标题：测试\n"
            '1|等等，这个数字真的假的？|{"emotion":"surprised","emphasis":["真的"]}\n'
            '2|千真万确。|\u007b"transition":"quick_response"\u007d\n'
        )
        self.assertEqual(len(t.turns), 2)
        d0 = t.turns[0].delivery
        self.assertIsNotNone(d0)
        self.assertEqual(d0.emotion, "surprised")
        self.assertEqual(d0.emphasis, ["真的"])
        d1 = t.turns[1].delivery
        self.assertEqual(d1.transition, "quick_response")

    def test_line_without_delivery_and_garbage_tail(self):
        t = _parse_dialogue("1|普通台词没有标签\n2|台词|{不是合法json\n1|台词|{\"unknown\":1}\n")
        self.assertEqual(len(t.turns), 3)
        self.assertIsNone(t.turns[0].delivery)
        # 尾缀无法解析时保留原文（台词里含 "|{..."）
        self.assertIn("{", t.turns[1].text)
        # 不含任何已知键的 JSON 视为无 delivery
        self.assertIsNone(t.turns[2].delivery)

    def test_delivery_coerce_filters_unknown_values(self):
        d = Delivery.coerce({"emotion": "angry", "paralinguistic": "scream",
                             "transition": "teleport", "emphasis": ["a", "", 1]})
        self.assertEqual(d.emotion, "calm")
        self.assertIsNone(d.paralinguistic)
        self.assertEqual(d.transition, "normal")
        self.assertEqual(d.emphasis, ["a", "1"])

    def test_json_fallback_with_delivery(self):
        t = _parse_dialogue(
            '{"title": "T", "content": [{"speaker": 1, "text": "你好",'
            ' "delivery": {"emotion": "amused", "paralinguistic": "laugh"}}]}')
        self.assertEqual(t.turns[0].delivery.emotion, "amused")
        self.assertEqual(t.turns[0].delivery.paralinguistic, "laugh")


class DeliveryInstructTest(unittest.TestCase):
    def _prompt(self, delivery):
        return cosy._instruction("测试台词。", 1, delivery=delivery)

    def test_explicit_delivery_overrides_heuristic(self):
        p = self._prompt(Delivery.coerce({"emotion": "surprised",
                                          "emphasis": ["三倍"],
                                          "paralinguistic": "laugh",
                                          "transition": "quick_response"}))
        self.assertIn("意外", p)
        self.assertIn("重点重读：三倍", p)
        self.assertIn("轻笑", p)
        self.assertIn("快速回应", p)
        # 启发式猜测不应再出现
        self.assertNotIn("自然承接搭档上一句话，重点词适度强调，长短句", p)

    def test_default_delivery_skips_heuristic(self):
        p = self._prompt(Delivery.coerce({}))
        self.assertIn("自然承接搭档上一句话", p)
        self.assertNotIn("这是紧接对方的短回应", p)

    def test_none_delivery_uses_heuristic(self):
        p = cosy._instruction("但这背后的成本压力，其实一直都没有人真正去管过。", 1, delivery=None)
        self.assertIn("补充或修正对方观点", p)

    def test_transition_gap_table(self):
        self.assertEqual(cosy._TRANSITION_GAP_MS["backchannel"], 60)
        self.assertEqual(cosy._TRANSITION_GAP_MS["pause"], 380)
        self.assertNotIn("normal", cosy._TRANSITION_GAP_MS)


class AnnotateDeliveryTest(unittest.TestCase):
    def test_annotator_labels_turns(self):
        from unittest.mock import patch
        provider = llm.QwenScriptProvider.__new__(llm.QwenScriptProvider)
        fake_reply = ('[{"i": 0, "emotion": "surprised", "emphasis": ["三倍"],'
                      ' "transition": "quick_response"},'
                      '{"i": 1, "emotion": "calm", "paralinguistic": "laugh"},'
                      '{"i": 99, "emotion": "surprised"},'
                      '{"i": "x"}] 多余文字')
        with patch.object(provider, "_chat", return_value=fake_reply):
            t = provider.annotate_delivery(Transcript(title="t", turns=[
                Turn(1, "轮零"), Turn(2, "轮一"), Turn(1, "轮二"),
            ]))
        self.assertEqual(t.turns[0].delivery.emotion, "surprised")
        self.assertEqual(t.turns[0].delivery.transition, "quick_response")
        self.assertEqual(t.turns[1].delivery.paralinguistic, "laugh")
        self.assertIsNone(t.turns[2].delivery)  # calm+normal 不标

    def test_annotator_failure_keeps_transcript(self):
        from unittest.mock import patch
        provider = llm.QwenScriptProvider.__new__(llm.QwenScriptProvider)
        with patch.object(provider, "_chat", side_effect=RuntimeError("down")):
            t = provider.annotate_delivery(Transcript(title="t", turns=[Turn(1, "x")]))
        self.assertIsNone(t.turns[0].delivery)


if __name__ == "__main__":
    unittest.main()
