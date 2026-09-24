# -*- coding: utf-8 -*-
"""联网搜索资料注入写稿 prompt 的行为测试。"""
import os
import unittest
from unittest.mock import patch

from app.providers import llm
from app.providers.search import SearchProvider, _strip_html


class _FakeSearch(SearchProvider):
    name = "fake"
    enabled = True

    def __init__(self, digest):
        self._digest = digest

    def gather(self, topic):
        return self._digest


class _FakeChatSearch(_FakeSearch):
    """同时记录 chat 入参，按段数返回固定行式文本。"""

    def __init__(self, digest):
        super().__init__(digest)
        self.messages = []

    def attach(self, provider):
        provider.search = self

    def chat_side_effect(self, messages, max_tokens=None):
        self.messages.append(messages)
        n = len(self.messages)
        if n == 1:
            return "标题：测试播客\n1|大家好，今天聊一个有意思的话题。\n2|这个话题最近特别火。"
        if n == 2:
            return ("1|那个数据确实很惊人，和公开报道完全对得上。\n"
                    "2|没错，我当时看到这个数字也愣了一下。")
        return "1|今天就聊到这里。\n2|我们下期再见。"


class SearchInjectionTest(unittest.TestCase):
    def _provider(self, digest, target=60):
        provider = llm.QwenScriptProvider.__new__(llm.QwenScriptProvider)
        provider.base_url = "http://unused"
        provider.model = "fake-model"
        provider.api_key = "EMPTY"
        provider.target_seconds = target
        provider.max_tokens = 512
        provider.temperature = 0.7
        provider.max_attempts = 3
        provider.search_first_wait = 1
        provider.search_wait = 1
        fake = _FakeChatSearch(digest)
        provider.search = fake
        self.fake = fake
        return provider

    def test_digest_injected_into_followup_segments(self):
        provider = self._provider("[1] 生椰拿铁五年卖出20亿杯。\n瑞幸门店超过两万家。")
        with patch.object(provider, "_chat",
                          side_effect=lambda m, max_tokens=None:
                          self.fake.chat_side_effect(m, max_tokens)):
            segs = list(provider.iter_segments(
                item_id="t1", topic="瑞幸如何靠生椰拿铁翻盘"))
        self.assertGreaterEqual(len(self.fake.messages), 2)
        first_user = self.fake.messages[0][1]["content"]
        second_user = self.fake.messages[1][1]["content"]
        # 首段可能等不到搜索，但至少续写段必须带资料
        self.assertIn("参考资料", second_user)
        self.assertIn("20亿杯", second_user)
        self.assertIn("已有对话", second_user)
        self.assertIn("以提供的参考资料为准",
                      self.fake.messages[1][0]["content"])
        # 搜索结果不得影响行式协议解析
        total_turns = sum(len(s.turns) for s in segs)
        self.assertGreaterEqual(total_turns, 4)

    def test_empty_digest_falls_back_cleanly(self):
        provider = self._provider("")
        with patch.object(provider, "_chat",
                          side_effect=lambda m, max_tokens=None:
                          self.fake.chat_side_effect(m, max_tokens)):
            list(provider.iter_segments(
                item_id="t2", topic="AI技术如何改变医疗诊断"))
        for messages in self.fake.messages:
            self.assertNotIn("参考资料", messages[1]["content"])
            self.assertNotIn("以提供的参考资料为准", messages[0]["content"])

    def test_search_failure_degrades_to_empty(self):
        class _BoomSearch(SearchProvider):
            name = "boom"
            enabled = True

            def gather(self, topic):
                raise RuntimeError("network down")

        with patch.dict(os.environ, {"SEARCH_PROVIDER": "boom"}):
            pass  # build_search_provider 不支持 boom；直接验证 gather_async 兜底
        from app.providers.search import gather_async
        state, done = gather_async(_BoomSearch(), "任意主题")
        self.assertTrue(done.wait(timeout=5))
        self.assertEqual(state["digest"], "")


class StripHtmlTest(unittest.TestCase):
    def test_strip_removes_scripts_and_tags(self):
        html = ("<html><head><style>.a{color:red}</style>"
                "<script>var x=1;</script></head>"
                "<body><p>瑞幸生椰拿铁&nbsp;销量</p><p>第二段文字</p></body></html>")
        text = _strip_html(html)
        self.assertNotIn("var x", text)
        self.assertNotIn("color", text)
        self.assertIn("瑞幸生椰拿铁", text)
        self.assertIn("第二段文字", text)


if __name__ == "__main__":
    unittest.main()
