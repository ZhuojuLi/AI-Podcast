# -*- coding: utf-8 -*-
"""ASR 回听后端：Qwen3-ASR-0.6B（+ Qwen3-ForcedAligner-0.6B）。

用途：TTS 后回听，计算 CER，重点检查公司名/英文品牌/人名/数字/金额/年份。
CER > cer_retry 触发单句重新合成；> cer_pass 但 < cer_retry 可接受或重试。
"""
from typing import Tuple


class QwenASRProvider:
    name = "qwen"

    def __init__(self, model_path: str = "", device: str = "cuda:0") -> None:
        self.model_path = model_path
        self.device = device

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError("QwenASRProvider 待实现（D5）")

    def cer(self, ref_text: str, hyp_text: str) -> float:
        raise NotImplementedError("QwenASRProvider 待实现（D5）")

    def verify(self, audio_path: str, ref_text: str, threshold: float) -> Tuple[bool, float, str]:
        raise NotImplementedError("QwenASRProvider 待实现（D5）")
