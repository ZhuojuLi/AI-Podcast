# -*- coding: utf-8 -*-
"""跨模块共享的数据结构。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Performance Script 的受控词表（Script-to-Performance 协议 V0）。
# 字段刻意保持最小集：emotion / emphasis / paralinguistic / transition。
EMOTIONS = ("calm", "curious", "surprised", "amused", "doubtful",
            "emphatic", "thoughtful", "warm")
PARALINGUISTICS = (None, "laugh", "sigh", "breath", "hmm", "oh")
TRANSITIONS = ("normal", "quick_response", "backchannel", "interrupt",
               "overlap", "pause")


@dataclass
class Delivery:
    """一轮台词的表演指示（可选，全部缺省时 TTS 走启发式回退）。"""

    emotion: str = "calm"
    emphasis: List[str] = field(default_factory=list)
    paralinguistic: Optional[str] = None
    transition: str = "normal"

    @classmethod
    def coerce(cls, data: Any) -> "Delivery":
        if not isinstance(data, dict):
            return cls()
        emotion = str(data.get("emotion") or "calm")
        if emotion not in EMOTIONS:
            emotion = "calm"
        paralinguistic = data.get("paralinguistic")
        if paralinguistic not in PARALINGUISTICS:
            paralinguistic = None
        transition = str(data.get("transition") or "normal")
        if transition not in TRANSITIONS:
            transition = "normal"
        emphasis = [str(e) for e in (data.get("emphasis") or []) if str(e).strip()][:3]
        return cls(emotion=emotion, emphasis=emphasis,
                   paralinguistic=paralinguistic, transition=transition)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "emotion": self.emotion,
            "transition": self.transition,
        }
        if self.emphasis:
            out["emphasis"] = list(self.emphasis)
        if self.paralinguistic:
            out["paralinguistic"] = self.paralinguistic
        return out


@dataclass
class Turn:
    """一轮播客对话。speaker 取 1 或 2；delivery 为可选表演指示。"""

    speaker: int
    text: str
    delivery: Optional[Delivery] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"speaker": int(self.speaker), "text": str(self.text)}
        if self.delivery is not None:
            out["delivery"] = self.delivery.to_dict()
        return out


@dataclass
class Transcript:
    """播客文稿：标题 + 对话轮次列表。"""

    title: str
    turns: List[Turn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": [t.to_dict() for t in self.turns]}

    def to_plain_text(self) -> str:
        lines = []
        for turn in self.turns:
            prefix = f"主播{turn.speaker}：" if turn.speaker else ""
            lines.append(f"{prefix}{turn.text}")
        return "\n".join(lines)


@dataclass
class ScriptSegment:
    """文稿的一个增量片段（流式产出用）。title 仅首段非空。"""

    turns: List[Turn] = field(default_factory=list)
    title: str = ""

    @property
    def chars(self) -> int:
        return sum(len(t.text) for t in self.turns)
