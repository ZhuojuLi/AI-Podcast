# -*- coding: utf-8 -*-
"""联网资料搜索：为写稿 LLM 提供事实核查素材，降低幻觉、让内容更具体。

链路：query（主题 + 补充词）→ ddgs（免 key DuckDuckGo 聚合搜索）→
取 top 结果抓取正文（stdlib HTMLParser 提文本，不引重依赖）→
裁剪成预算内的资料摘要，注入写稿 prompt。

任何环节失败都静默降级为空摘要（退回原有无搜索行为）。
环境变量：SEARCH_PROVIDER=ddgs/none/module:Class（默认 ddgs）
          SEARCH_MAX_RESULTS / SEARCH_FETCH_PAGES / SEARCH_TIMEOUT
          SEARCH_BUDGET_CHARS
"""
import os
import re
import threading
from html.parser import HTMLParser
from typing import List

_QUERY_SUFFIXES = ("", " 现状 分析", " 数据")
_SKIP_DOMAINS = (
    "youtube.com", "youtu.be", "bilibili.com", "tiktok.com", "douyin.com",
    "instagram.com", "twitter.com", "x.com", "facebook.com",
)


def _strip_html(html: str, limit: int = 1200) -> str:
    """stdlib 提取网页可见文本（去 script/style），压缩空白。"""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    chunks: List[str] = []

    class _Text(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript"):
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript") and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                chunks.append(data)

    try:
        _Text().feed(html)
    except Exception:
        pass
    text = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    return text[:limit]


class SearchProvider:
    name = "none"
    enabled = False

    def gather(self, topic: str) -> str:
        return ""


class DDGSearchProvider(SearchProvider):
    """DuckDuckGo 聚合搜索（ddgs 库，免 API key）。"""
    name = "ddgs"
    enabled = True

    def __init__(self) -> None:
        self.max_results = int(os.getenv("SEARCH_MAX_RESULTS", "6"))
        self.fetch_pages = int(os.getenv("SEARCH_FETCH_PAGES", "3"))
        self.timeout = float(os.getenv("SEARCH_TIMEOUT", "8"))
        self.budget = int(os.getenv("SEARCH_BUDGET_CHARS", "1800"))

    def _query(self, ddgs, query: str) -> List[dict]:
        try:
            return list(ddgs.text(query, max_results=self.max_results))
        except Exception as exc:
            print(f"[search] 查询失败 {query!r}: {exc}", flush=True)
            return []

    def gather(self, topic: str) -> str:
        import requests
        from ddgs import DDGS

        topic = (topic or "").strip()
        if not topic:
            return ""
        results: List[dict] = []
        with DDGS(timeout=self.timeout) as ddgs:
            for suffix in _QUERY_SUFFIXES:
                results.extend(self._query(ddgs, topic + suffix))
                if len(results) >= self.max_results:
                    break

        seen, entries = set(), []
        for r in results:
            href = (r.get("href") or "").strip()
            title = re.sub(r"\s+", " ", (r.get("title") or "")).strip()
            body = re.sub(r"\s+", " ", (r.get("body") or "")).strip()
            if not href or href in seen or not title:
                continue
            host = re.sub(r"^https?://", "", href).split("/")[0]
            if any(d in host for d in _SKIP_DOMAINS):
                continue
            seen.add(href)
            entries.append({"href": href, "title": title, "body": body})
            if len(entries) >= self.max_results:
                break
        if not entries:
            return ""

        # 只对最靠前的几条抓正文，控制耗时
        session = requests.Session()
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
        for i, e in enumerate(entries[: self.fetch_pages]):
            try:
                resp = session.get(e["href"], timeout=self.timeout)
                if resp.status_code == 200 and "text/html" in resp.headers.get(
                        "Content-Type", "text/html"):
                    e["text"] = _strip_html(resp.text)
            except Exception:
                pass

        parts, used = [], 0
        for i, e in enumerate(entries, 1):
            content = e.get("text") or e["body"]
            content = content[:500]
            if len(content) < 40:
                continue
            block = f"[{i}] {e['title']}\n{content}"
            if used + len(block) > self.budget:
                break
            parts.append(block)
            used += len(block)
        if not parts:
            return ""
        print(f"[search] {topic!r} -> {len(parts)} 条资料 "
              f"({used} 字)", flush=True)
        return "\n\n".join(parts)


def build_search_provider() -> SearchProvider:
    kind = os.getenv("SEARCH_PROVIDER", "ddgs").strip()
    if not kind or kind == "none":
        return SearchProvider()
    if kind == "ddgs":
        return DDGSearchProvider()
    if ":" in kind:
        import importlib
        module, cls = kind.split(":", 1)
        provider = getattr(importlib.import_module(module), cls)()
        provider.enabled = True
        return provider
    raise ValueError(f"未知 SEARCH_PROVIDER: {kind}")


def gather_async(provider: SearchProvider, topic: str):
    """后台线程搜索；返回 (digest, done_event)。"""
    state = {"digest": ""}
    done = threading.Event()

    def _run():
        try:
            state["digest"] = provider.gather(topic)
        except Exception as exc:
            print(f"[search] 搜索异常，降级为空摘要: {exc}", flush=True)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    return state, done
