# -*- coding: utf-8 -*-
"""文稿生成后端：本地/远端 OpenAI 兼容 LLM（当前提交用 Qwen3-14B-AWQ）。

流程：
  1. 用**行式协议**生成对话（对模型比 JSON 稳）：标题行 + `序号|台词` 行
  2. 解析并**丢弃空轮次**
  3. 按目标时长估算（rate 字/秒）；不足则**续写**，最多 3 段，
     直到预计时长进入目标区间 —— 保证 5-15min 的硬门禁

环境变量：LLM_BASE_URL / LLM_MODEL / LLM_API_KEY / LLM_TARGET_SECONDS
          LLM_MAX_TOKENS / LLM_TEMPERATURE / LLM_SPEAK_RATE / LLM_MAX_ATTEMPTS
"""
import json
import os
import re
from typing import List, Optional

import requests

from app.providers.base import ScriptProvider
from app.providers.search import build_search_provider, gather_async
from app.schemas import Delivery, ScriptSegment, Transcript, Turn

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-awq")
SPEAK_RATE = float(os.getenv("LLM_SPEAK_RATE", "4.2"))   # 中文约 4.2 字/秒（保守，宁长勿短）

_SYSTEM = """你是资深中文商业播客编剧。请围绕给定主题，写一段两位搭档共同完成的商业播客。

角色：
- 主播1（{g1}）：擅长讲故事和抓矛盾，但也会表达判断、回答问题。
- 主播2（{g2}）：擅长拆商业逻辑和举例，但也会主动追问、质疑和推进话题。

写作要求：
- 像两位熟悉彼此的真人搭档聊天，不是采访，也不是轮流朗读两篇文章。
- 每一轮都要接住上一轮的**具体信息**：可追问其中一个词、补充例子、纠正前提、
  表达保留意见或把对方的话推进一步。不能只加一句空泛认同再念自己的内容。
- 两个人都可以提问、回答和表达观点，避免固定成“一问一答”；允许偶尔出现 8~20 字的
  短回应，但短回应必须有信息量，例如“问题恰恰出在成本端”，不能只是表示同意。
- 对话要有真实的互动火花：至少 2~3 处真实的惊讶、质疑或恍然，自然地用
  “等等”“不对”“你这么一说我想起来了”这类插话，可以有笑场和被说服的瞬间。
- 情绪要有轻重缓急：开场抓人（抛悬念或反常识判断），发现反常时好奇，谈冲突时有分歧，
  关键数据处笃定放慢，高潮处放开，结尾放松收束。不要全程同一种语气。
- 适当使用省略号“……”和破折号“——”表现思考、停顿和被打断，每篇 3~5 处即可。
- 内容要具体：讲清楚"原来怎样 → 出了什么问题 → 谁做了什么 → 为什么有效 → 后来怎样"，
  覆盖关键事件、核心人物、行业背景与实质讨论，不要泛泛而谈。
- **不要编造具体的数字、年份、金额、份额**；不确定处用"当时""大幅""很多"这类描述性表达。
- 禁止使用空泛衔接词：好的、是的、对、没错、同意、确实如此、你说得对、值得注意的是、
  此外、综上所述、我们不难发现。
- 全中文；不要 Markdown、emoji、括号舞台说明。
- 长短句交替；大部分台词 25~65 字，少量 8~20 字的有效短句，每行最多 80 字，共约 {turns} 行。

输出格式（严格遵守，不要 JSON，不要任何解释）：
标题：<节目标题>
1|<主播1说的一句话>
2|<主播2回应的一句话>
1|<主播1说的一句话>|{{"emotion":"surprised","emphasis":["三倍"]}}
2|……

可选：台词后可以加第三段 `|{{表演指示}}`（必须是 JSON），给 TTS 明确的表演指令：
- emotion 词表（选其一）：calm / curious / surprised / amused / doubtful /
  emphatic / thoughtful / warm，只在情绪明确时使用，不要每行都标。
- emphasis：0~2 个需要重读的词，例如 ["三倍"]。
- paralinguistic（全场 3~5 次即可）：laugh / sigh / breath / hmm / oh。
- transition：quick_response（紧接对方的快回应）/ backchannel（附和）/
  pause（说完留余韵）；默认省略。
- 拿不准就省略第三段，宁缺毋滥。

注意：
- 每行 `数字|` 之后**只写要说出口的对话内容**（表演指示除外）。
- 对话中**绝对不要出现** Story Host、Analyst、主播1、主播2 这类角色称呼（不管句首还是句中）。
- **不要写空行、不要留空台词**。"""

_CONTINUE_SYSTEM = """你是资深中文商业播客编剧。下面是一段两位搭档的商业播客已有内容，
请**接着往下写**，保持同样的话题、人物与风格。

要求：
- 只输出**新增**的对话行，格式为 `序号|台词`（序号只用 1 和 2）。
- 第一行必须由主播 {next_speaker} 说，并具体回应已有内容的最后一句。
- 继续深化讨论（补充案例、争议、影响、总结），不要重复已说过的内容。
- 两个人都能提问、回答、质疑和推进；不要写成机械的一问一答。
- 每轮要承接上一轮的具体信息，禁止用“好的、是的、对、没错、同意、确实如此、你说得对”
  充当衔接。保持真实的互动感：可以有追问、质疑、恍然和插话，情绪随内容起伏。
- 适当使用省略号“……”和破折号“——”表现停顿和思考。
- 台词后可加第三段 `|{{表演指示}}`（JSON，可选）：emotion（calm/curious/surprised/amused/
  doubtful/emphatic/thoughtful/warm）、emphasis（0~2 个重读词）、
  paralinguistic（laugh/sigh/breath/hmm/oh，克制使用）、
  transition（quick_response/backchannel/pause）。拿不准就省略。
- 大部分台词 25~65 字，少量 8~20 字的有效短句，每行最多 80 字。
- {ending_instruction}
- 全中文；不要 Markdown、emoji、角色称呼；不要空行、不要空台词。
- 再写约 {turns} 行。"""


_FAREWELL_SYSTEM = """你是资深中文商业播客编剧。为下面的播客写一个自然的结尾告别。
只输出 1~2 行 `序号|台词`（序号只用 1 和 2）：先由一位主播用一句话收住本期观点，
再由另一位道别（可以约下期或一句轻松的结束语）。简短口语、像真人聊天收尾，
不要客套、不要 Markdown、不要重复已有内容。"""

_DELIVERY_ANNOTATOR_SYSTEM = """你是中文播客导演。下面是一场双人播客的逐轮台词（格式 `序号|台词`，
序号从 0 开始）。请为每一轮标注表演指示，帮助 TTS 演员演绎。

只输出一个 JSON 数组，每项格式：
{"i": 轮次序号, "emotion": "...", "emphasis": ["..."], "paralinguistic": "...", "transition": "..."}

词表（超出词表的值会被丢弃）：
- emotion: calm / curious / surprised / amused / doubtful / emphatic / thoughtful / warm
- emphasis: 该轮需要重读的 0~2 个词（数组）
- paralinguistic: laugh / sigh / breath / hmm / oh（全篇合计 3~5 次，留给最有需要的轮次）
- transition: normal / quick_response / backchannel / pause

判断要点：
- 紧接对方的短促回应用 quick_response；纯附和用 backchannel
- 总结、道别、金句之后用 pause（说完留余韵）
- 大部分轮次保持 calm + normal，不要每轮都标情绪
- 没有需要重读的词时 emphasis 给空数组
不要输出任何解释，只要 JSON 数组。"""


def _clean(text: str) -> str:
    text = re.sub(r"\*\*|\*|`|#+", "", text)
    return text.strip().strip('"').strip("“”").strip()


_ROLE_PREFIX = re.compile(
    r"^(?:story\s*host|analyst|host|guest|moderator|主播\s*[12]|speaker\s*[12]|s[12]"
    r"|分析师(?:认为)?|主持人|嘉宾)\s*[，,：:]?\s*",
    re.IGNORECASE,
)
_ROLE_INLINE = re.compile(
    r"(?i)\b(?:story\s*host|analyst|host|guest|moderator)\b|主播\s*[12]|speaker\s*[12]",
)


def _strip_role(text: str) -> str:
    """去掉模型误写入正文的角色称呼，并清理残留标点。"""
    prev = None
    while prev != text:
        prev = text
        text = _ROLE_PREFIX.sub("", text)
        text = _ROLE_INLINE.sub("", text)
    text = re.sub(r"[，,、]\s*(?=[。.！!？?]|$)", "", text)
    text = re.sub(r"^[\s，,、。.！!？?：:]+", "", text)
    text = re.sub(r"[\s，,、]+$", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


_EMPTY_ACK_PREFIX = re.compile(
    r"^(?:(?:好的|是的|对(?=[，,。；;：:])|没错|同意|确实如此|你说得对|非常赞同|完全同意)"
    r"[，,。；;：:]?\s*)+"
)


def _strip_empty_ack(text: str) -> str:
    """移除“先假装回应、再念正文”的空泛认同前缀。"""
    cleaned = _EMPTY_ACK_PREFIX.sub("", text).strip()
    # 若整轮只有一句空泛认同，宁可丢弃，避免把产品明确反感的口头禅送进 TTS。
    return cleaned


def _clean_turn_text(text: str) -> str:
    return _strip_empty_ack(_strip_role(text))


def _split_delivery(rest: str):
    """从行尾拆出可选的 `|{JSON表演指示}`，返回 (台词, Delivery|None)。"""
    rest = rest.strip()
    if rest.endswith("}") and "|" in rest:
        head, _, tail = rest.rpartition("|")
        tail = tail.strip()
        if tail.startswith("{"):
            try:
                obj = json.loads(tail)
            except json.JSONDecodeError:
                return rest, None
            if isinstance(obj, dict) and (set(obj) & {"emotion", "emphasis",
                                                      "paralinguistic", "transition"}):
                return head.strip(), Delivery.coerce(obj)
    return rest, None


def _parse_dialogue(text: str) -> Transcript:
    """解析行式协议；失败回退 JSON；丢弃空轮次。"""
    title = ""
    turns: List[Turn] = []
    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        m = re.match(r"^(?:标题|title)\s*[:：]\s*(.+)$", line, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            continue
        m = re.match(r"^speaker\s*([12])\s*[:：]\s*(.*)$", line, re.IGNORECASE)
        if m:
            body, delivery = _split_delivery(m.group(2).strip())
            txt = _clean_turn_text(body)
            if txt:
                turns.append(Turn(speaker=int(m.group(1)), text=txt,
                                  delivery=delivery))
            continue
        # 行式协议：`序号|台词`。模型有时 1/2 交替、有时从 3/4/5 连续编号，
        # 统一按奇偶映射 speaker（奇→1，偶→2），两种编号都正确。
        m = re.match(r"^(?:主播)?\s*(\d{1,2})\s*[|｜:：.、,，\-]\s*(.*)$", line)
        if m:
            speaker = 1 if int(m.group(1)) % 2 == 1 else 2
            body, delivery = _split_delivery(m.group(2).strip())
            txt = _clean_turn_text(body)
            if txt:                       # 丢弃空台词
                turns.append(Turn(speaker=speaker, text=txt, delivery=delivery))
    if turns:
        return Transcript(title=title, turns=turns)
    return _parse_json(text, title)


def _parse_json(text: str, title: str = "") -> Transcript:
    """回退解析：从模型输出里提取 JSON 数组/对象里的对话轮次。"""
    turns: List[Turn] = []
    for snippet in _json_candidates(_clean(text)):
        try:
            obj = json.loads(snippet, strict=False)
        except json.JSONDecodeError:
            continue
        items = obj.get("content") if isinstance(obj, dict) else obj
        if isinstance(obj, dict) and obj.get("title") and not title:
            title = str(obj["title"])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                speaker = int(item.get("speaker", 1))
            except (TypeError, ValueError):
                speaker = 1
            speaker = 1 if speaker not in (1, 2) else speaker
            body, embedded = _split_delivery(str(item.get("text", "")))
            item_delivery = item.get("delivery")
            if embedded is None and isinstance(item_delivery, dict):
                embedded = Delivery.coerce(item_delivery)
            txt = _clean_turn_text(_clean(body))
            if txt:
                turns.append(Turn(speaker=speaker, text=txt, delivery=embedded))
        if turns:
            break
    return Transcript(title=title, turns=turns)


def _json_candidates(text: str):
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            snippet = text[start:end + 1]
            snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
            yield snippet
    yield text.replace("'", '"')


class QwenScriptProvider(ScriptProvider):
    name = "qwen"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.target_seconds = int(os.getenv("LLM_TARGET_SECONDS", "480"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "6"))
        # 联网资料搜索（降低幻觉；搜索失败/超时自动降级为不注入）
        self.search = build_search_provider()
        self.search_first_wait = float(os.getenv("SEARCH_FIRST_WAIT", "6"))
        self.search_wait = float(os.getenv("SEARCH_WAIT", "15"))

    def _chat(self, messages: List[dict], max_tokens: Optional[int] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        # 关键：关闭思维链（Qwen3.5 默认 thinking 会输出 reasoning 文本，破坏行式格式）
        if os.getenv("LLM_DISABLE_THINKING", "1").strip() not in ("0", "false", "False"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=900,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def annotate_delivery(self, transcript: Transcript) -> Transcript:
        """导演视角二次标注：为每轮台词补 Performance Script 表演指示。

        实测写稿模型对行内第三段格式遵从度低，独立标注 pass 格式可控
        （只要 JSON 数组），且能看到全篇上下文做 transition 判断。
        标注失败时原样返回（delivery 保持 None，TTS 走启发式回退）。
        """
        if not transcript.turns:
            return transcript
        lines = [f"{i}|{t.text}" for i, t in enumerate(transcript.turns)]
        try:
            reply = self._chat(
                [
                    {"role": "system", "content": _DELIVERY_ANNOTATOR_SYSTEM},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                max_tokens=max(600, 120 * len(transcript.turns)),
            )
        except Exception as exc:
            print(f"[LLM] 表演标注失败，跳过：{exc}", flush=True)
            return transcript
        start, end = reply.find("["), reply.rfind("]")
        if start == -1 or end <= start:
            print(f"[LLM] 表演标注输出无法解析，跳过", flush=True)
            return transcript
        try:
            items = json.loads(reply[start:end + 1])
        except json.JSONDecodeError:
            print(f"[LLM] 表演标注 JSON 解析失败，跳过", flush=True)
            return transcript
        labeled = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(transcript.turns):
                d = Delivery.coerce(item)
                if (d.emotion != "calm" or d.emphasis or d.paralinguistic
                        or d.transition != "normal"):
                    transcript.turns[idx].delivery = d
                    labeled += 1
        print(f"[LLM] 表演标注完成：{labeled}/{len(transcript.turns)} 轮",
              flush=True)
        return transcript

    # ---------- 流式：按段产出（首段短，尽快开唱） ----------
    def iter_segments(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str = "",
        speaker_gender2: str = "",
        target_seconds: Optional[int] = None,
    ):
        target = target_seconds or self.target_seconds
        max_chars = int(min(target * 1.25, target + 90) * SPEAK_RATE)
        g1, g2 = speaker_gender1 or "男", speaker_gender2 or "女"

        turns: List[Turn] = []
        title = ""
        yielded = False

        # 联网搜索与写稿并发：首段不等搜索结果（保首字节），续写段注入资料摘要。
        digest_state, digest_done = ({"digest": ""}, None)
        if self.search.enabled:
            digest_state, digest_done = gather_async(self.search, topic)

        def _digest(first: bool) -> str:
            if digest_done is None:
                return ""
            digest_done.wait(timeout=self.search_first_wait if first else self.search_wait)
            return (digest_state["digest"] or "").strip()

        facts_rule = ("涉及具体数据、年份、事件时以提供的参考资料为准；"
                      "资料里没有的具体数字不要编造，用模糊表达。")
        for attempt in range(1, self.max_attempts + 1):
            est = sum(len(t.text) for t in turns) / SPEAK_RATE
            ends_with_question = bool(turns and turns[-1].text.rstrip().endswith(("？", "?")))
            if turns and est >= target and not ends_with_question:
                break
            remaining = max(target - est, 20 if ends_with_question else 0)
            if not turns:
                remaining = target
            if not turns:
                # 首段只要 2 行短句：让 TTS 尽快出声，确保首字节 <30s
                need_lines = 2
                digest = _digest(first=True)
                system = _SYSTEM.format(g1=g1, g2=g2, turns=need_lines) + (
                    "\n本次是**开场**，只写 2 行，每行 15~35 字。第一句要抓人——抛悬念或反常识判断；"
                    "第二句接住第一句的具体悬念，简短自然地进入主题。")
                if digest:
                    system += "\n- " + facts_rule
                user = f"主题：{topic}"
                if digest:
                    user += ("\n\n参考资料（联网搜索摘要，供核对事实，"
                             "不要在节目里提及来源）：\n" + digest)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            else:
                need_lines = (
                    2 if ends_with_question and est >= target
                    else max(12, int(remaining * SPEAK_RATE / 70))
                )
                prev = "\n".join(f"{t.speaker}|{t.text}" for t in turns[-24:])
                digest = _digest(first=False)
                ending_instruction = (
                    "这是预计的最后一段，也是全篇结尾：**必须**以告别收尾——"
                    "倒数第二轮自然收住观点，最后一轮道别（可以约下期或一句轻松的结束语），"
                    "不要突然喊口号、重复感谢收听，也不要在道别之后再加新观点。"
                    if remaining <= 180 else
                    "继续推进故事，不要提前总结或说感谢收听。"
                )
                continue_system = _CONTINUE_SYSTEM.format(
                    turns=need_lines,
                    next_speaker=2 if turns[-1].speaker == 1 else 1,
                    ending_instruction=ending_instruction,
                )
                user = f"主题：{topic}"
                if digest:
                    continue_system += "\n- " + facts_rule
                    user += ("\n\n参考资料（联网搜索摘要，供核对事实，"
                             "不要在节目里提及来源）：\n" + digest)
                messages = [
                    {"role": "system", "content": continue_system},
                    {"role": "user",
                     "content": user + f"\n\n已有对话：\n{prev}"},
                ]
            content = self._chat(messages)
            parsed = _parse_dialogue(content)
            if parsed.title and not title:
                title = parsed.title
            if not parsed.turns:
                print(f"[LLM] 第 {attempt} 段未解析出轮次（len={len(content)}）"
                      f"raw_head={content[:160]!r}", flush=True)
                continue

            seg_turns = parsed.turns
            # 续写段有时重新从 1/2 编号，有时延续全局序号；模型还可能
            # 从错误一侧开头。上一段结尾与本段开头若是同一人，翻转本段
            # 的角色标签，保持两位固定主播自然接话，而不改动台词内容。
            if turns and seg_turns and seg_turns[0].speaker == turns[-1].speaker:
                seg_turns = [Turn(speaker=3 - t.speaker, text=t.text) for t in seg_turns]
            used = sum(len(t.text) for t in turns)
            if used + sum(len(t.text) for t in seg_turns) > max_chars:
                keep, acc = [], 0
                for t in seg_turns:
                    if acc + len(t.text) > max_chars - used and keep:
                        break
                    keep.append(t)
                    acc += len(t.text)
                seg_turns = keep or seg_turns[:2]

            turns.extend(seg_turns)
            est = sum(len(t.text) for t in turns) / SPEAK_RATE
            print(f"[LLM] 第 {attempt} 段 +{len(seg_turns)} 轮，累计 {len(turns)} 轮 "
                  f"/ {sum(len(t.text) for t in turns)} 字 / 预计 {est:.0f}s", flush=True)
            yield ScriptSegment(turns=seg_turns, title=title if not yielded else "")
            yielded = True

        if not yielded:
            raise ValueError("模型未产出有效对话轮次")

        # 兜底结尾：模型常把全篇收在一个漂亮句子上却忘了道别，加一个短告别。
        _FAREWELL_HINT = re.compile(
            r"(?:再见|下期|聊到这里|聊到这|今天就|就到这里|收工|拜拜|下次见)")
        if (turns and len(turns) >= 8
                and not _FAREWELL_HINT.search(turns[-1].text)):
            tail = "\n".join(f"{t.speaker}|{t.text}" for t in turns[-6:])
            try:
                reply = self._chat(
                    [
                        {"role": "system", "content": _FAREWELL_SYSTEM},
                        {"role": "user",
                         "content": f"节目主题：{topic}\n\n已有对话的最后几轮：\n{tail}"},
                    ],
                    max_tokens=180,
                )
            except Exception as exc:  # 告别兜底失败不影响主体内容
                print(f"[LLM] 结尾告别生成失败，跳过：{exc}", flush=True)
            else:
                parsed = _parse_dialogue(reply)
                farewell = parsed.turns[:2]
                if farewell:
                    if farewell[0].speaker == turns[-1].speaker:
                        farewell = [Turn(speaker=3 - t.speaker, text=t.text)
                                    for t in farewell]
                    turns.extend(farewell)
                    print(f"[LLM] 补结尾告别 +{len(farewell)} 轮", flush=True)
                    yield ScriptSegment(turns=farewell, title="")

    def generate(
        self,
        item_id: str,
        topic: str,
        speaker_gender1: str = "",
        speaker_gender2: str = "",
        target_seconds: Optional[int] = None,
    ) -> Transcript:
        turns: List[Turn] = []
        title = ""
        for seg in self.iter_segments(item_id, topic, speaker_gender1, speaker_gender2,
                                      target_seconds):
            if seg.title and not title:
                title = seg.title
            turns.extend(seg.turns)
        if not turns:
            raise ValueError("模型未产出有效对话轮次")
        if not title:
            title = f"AI 播客：{topic}"
        total_chars = sum(len(t.text) for t in turns)
        print(f"[LLM] 完成：{len(turns)} 轮 / {total_chars} 字 / "
              f"预计 {total_chars / SPEAK_RATE:.0f}s", flush=True)
        return Transcript(title=title, turns=turns)
