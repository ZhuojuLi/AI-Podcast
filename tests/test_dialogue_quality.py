# -*- coding: utf-8 -*-
import unittest

from unittest.mock import patch

from app.providers.llm import QwenScriptProvider, _parse_dialogue
from app.providers.tts import VOICE_BANK, build_turn_instruct, pick_voices
from app.schemas import Turn


class VoicePolicyTest(unittest.TestCase):
    def test_sichuan_voice_is_not_in_default_bank(self):
        self.assertNotIn("eric", VOICE_BANK["male"])
        self.assertEqual("dylan", VOICE_BANK["male"][0])
        self.assertEqual("vivian", VOICE_BANK["female"][0])
        self.assertNotIn("uncle_fu", VOICE_BANK["male"])
        self.assertNotIn("serena", VOICE_BANK["female"])

    def test_mixed_gender_uses_young_voices(self):
        self.assertEqual(("dylan", "vivian"), pick_voices("男", "女"))
        self.assertEqual(("vivian", "dylan"), pick_voices("女", "男"))

    def test_every_style_requests_standard_conversational_mandarin(self):
        cases = [
            Turn(1, "这件事真正反常的地方在哪儿？"),
            Turn(2, "问题恰恰出在成本端。"),
            Turn(1, "不过换个角度看，这也是它后来翻盘的起点。"),
        ]
        previous = ""
        for index, turn in enumerate(cases):
            instruct = build_turn_instruct(turn, index, previous)
            self.assertIn("标准普通话", instruct)
            self.assertIn("自然", instruct)
            self.assertNotIn("播音腔。", instruct.replace("不要播音腔。", ""))
            previous = turn.text


class ScriptPolicyTest(unittest.TestCase):
    def test_empty_acknowledgement_prefix_is_removed(self):
        transcript = _parse_dialogue(
            "标题：测试\n"
            "1|好的，我们先看成本端。\n"
            "2|没错。问题真正出在供应链。\n"
            "1|同意\n"
            "2|对，这不是单纯的配送速度问题。\n"
        )
        self.assertEqual(
            ["我们先看成本端。", "问题真正出在供应链。", "这不是单纯的配送速度问题。"],
            [turn.text for turn in transcript.turns],
        )

    def test_question_gets_question_prosody(self):
        instruct = build_turn_instruct(Turn(1, "那它为什么偏偏在这时降价？"), 3, "上一句")
        self.assertIn("真实好奇", instruct)

    def test_continuation_starts_with_the_other_host(self):
        provider = QwenScriptProvider()
        provider.max_attempts = 2
        outputs = [
            "标题：测试\n1|第一轮。\n2|第二轮。",
            "2|第三轮被错误标成了二号。\n1|第四轮。",
        ]
        with patch.object(provider, "_chat", side_effect=outputs):
            segments = list(provider.iter_segments("test", "商业案例", "男", "女", 300))
        self.assertEqual([t.speaker for s in segments for t in s.turns], [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
