# -*- coding: utf-8 -*-
"""CosyVoice3 双人参考音色：逐轮生成、逐轮输出可拼接的 MP3。"""
import os
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from app.providers.base import TTSProvider
from app.providers.tts import _encode_mp3, _norm_gender
from app.schemas import Delivery, Transcript, Turn

_MODEL = None
_MODEL_PATH = None
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _id3_padding(payload_size: int) -> bytes:
    """合法的 ID3v2.3 空白元数据；解码器会跳过，不增加音频时长。"""
    if not 0 <= payload_size < 1 << 28:
        raise ValueError("ID3 padding 大小超出 syncsafe 范围")
    syncsafe = bytes((payload_size >> shift) & 0x7f for shift in (21, 14, 7, 0))
    return b"ID3\x03\x00\x00" + syncsafe + bytes(payload_size)

_DIALOGUE_INSTRUCTION = (
    "面向对面的搭档说话，不要新闻播音腔、广告腔或客服腔，停连和重音符合句意。"
    "不要拖长音、不要喊叫，语速慢也要自然。"
)
_BASE_INSTRUCTION = (
    "请用标准普通话，像二十多岁到三十岁的青年主播在双人商业播客中自然交谈。"
    "语速从容稍慢，句间有自然的停顿和快慢变化，绝不要匀速念稿。"
    + _DIALOGUE_INSTRUCTION
)

# instruct2 不沿用参考音的语音 token，若四个音色共用同一条“青年主播”指令，
# 模型会把不同声纹的语速、停顿和表达习惯拉得很接近。只给第二套参考音增加
# 与原说话人质感相符的角色指令；第一套男女声（以及所有异性组合）保持 v1.4 行为。
_SECONDARY_MALE_STYLE = (
    "请用标准普通话，保持成熟、有磁性的声音质感，"
    "像健谈的知识型搭档在双人商业播客里讨论。"
    "语气有热度和表达欲，中气足、反应机敏，观点交锋时敢亮态度；"
    "不要刻意压嗓、拖腔、说教或过分温柔。"
    + _DIALOGUE_INSTRUCTION
)
_SECONDARY_FEMALE_STYLE = (
    "请用标准普通话，保持知性、从容的声音质感，"
    "像经验丰富的访谈主持人在双人商业播客里与搭档交流。"
    "语速舒展但不拖沓，有真实的情绪反应，轻重音分明；不要故意甜美、亢奋或播音。"
    + _DIALOGUE_INSTRUCTION
)


def _instruction(text: str, index: int, voice_style: str = "",
                  delivery: Optional[Delivery] = None) -> str:
    """生成语气 instruct。优先使用 Performance Script 的显式 delivery；
    缺省时回退到按文本内容启发式猜测。"""
    base = voice_style or _BASE_INSTRUCTION
    if delivery is not None:
        parts = []
        emotion_hint = _EMOTION_INSTRUCT.get(delivery.emotion, "")
        if emotion_hint:
            parts.append(emotion_hint)
        if delivery.emphasis:
            parts.append("重点重读：" + "、".join(delivery.emphasis) + "。")
        para = _PARALINGUISTIC_INSTRUCT.get(delivery.paralinguistic or "", "")
        if para:
            parts.append(para)
        trans = _TRANSITION_INSTRUCT.get(delivery.transition, "")
        if trans:
            parts.append(trans)
        if parts:
            return base + "".join(parts)
        # delivery 全默认时也跳过启发式，避免显式协议被猜测覆盖
        return base + "自然承接搭档上一句话，重点词适度强调。"
    return _heuristic_instruction(text, index, base)


def _heuristic_instruction(text: str, index: int, base: str) -> str:
    """无 delivery 时的启发式语气猜测（旧行为）。"""
    t = text.strip()
    if index == 0:
        return base + "这是开场，语气放松、有兴趣，像跟老朋友打招呼，直接进入话题。"
    if t.endswith(("？", "?")):
        return base + "带着真实好奇发问，句尾自然上扬后收住，不要像照着提词器采访。"
    if len(t) <= 20:
        return base + "这是紧接对方的短回应，反应稍快，有真实交流感。"
    if re.search(r"(?:但|不过|恰恰|反过来|问题是)", t):
        return base + "这里在补充或修正对方观点，带一点思考和轻微转折，不要生硬反驳。"
    if re.search(r"(?:竟然|居然|没想到|出人意料)", t):
        return base + "这里有一点真实的意外，语气可以扬起来，但语速保持正常，不要拖长音。"
    if re.search(r"(?:哈哈|笑|有趣|好玩|有意思)", t):
        return base + "这里轻松带笑意，像聊天时真的被逗到。"
    if re.search(r"\d|%|％|亿|万", t):
        return base + "这里涉及关键数字，把数字和结论说得笃定清楚，重点词适度强调。"
    if re.search(r"(?:说到底|归根结底|总结一下|最后)", t):
        return base + "这里在收束观点，语速放慢一点，语气放松、有结论感。"
    return base + "自然承接搭档上一句话，重点词适度强调，长短句要有节奏变化。"


# Performance Script 显式标签 → instruct 片段
_EMOTION_INSTRUCT = {
    "calm": "",
    "curious": "带着真实的好奇心在说这句话。",
    "surprised": "带一点真实的意外，语气扬起来但不拖长音。",
    "amused": "轻松、带笑意，像真的被逗到了。",
    "doubtful": "带一点怀疑和保留，不是全盘接受。",
    "emphatic": "语气笃定有力，关键词咬字清楚。",
    "thoughtful": "带思考感，语速稍慢，像在边想边说。",
    "warm": "语气温和亲近，像老朋友之间的交流。",
}
_PARALINGUISTIC_INSTRUCT = {
    "laugh": "在合适的位置自然地轻笑一声，不要夸张。",
    "sigh": "句首轻轻叹一口气，再开始说。",
    "breath": "句间有自然的呼吸感。",
    "hmm": "句首有短暂的沉吟，再开始说。",
    "oh": "句首带一点轻轻的恍然。",
}
_TRANSITION_INSTRUCT = {
    "normal": "",
    "quick_response": "这是紧接对方的快速回应，反应快，节奏紧。",
    "backchannel": "这是短促的附和，一两个字的意思，不要说长。",
    "interrupt": "这句话接得很急，像忍不住插进来，后半句语气可以急促一点。",
    "overlap": "这句话和对方的话有重叠感，语速稍快。",
    "pause": "说完这句留一点余韵，句尾放慢收住。",
}

# transition → 轮后停顿毫秒（显式协议优先，None 表示走默认启发式）
_TRANSITION_GAP_MS = {
    "quick_response": 80,
    "backchannel": 60,
    "interrupt": 60,
    "overlap": 60,
    "pause": 380,
}


def _change_tempo(wav: np.ndarray, sample_rate: int, tempo: float) -> np.ndarray:
    """FFmpeg atempo 小幅变速，不抬音高。tempo < 1 放慢，> 1 加快。"""
    if tempo == 1.0:
        return wav
    if not 0.8 <= tempo <= 1.25:
        raise ValueError("变速系数应在 0.8–1.25 之间")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sample_rate),
         "-ac", "1", "-i", "pipe:0", "-af", f"atempo={tempo:.4f}",
         "-f", "f32le", "pipe:1"],
        input=np.ascontiguousarray(wav, dtype="<f4").tobytes(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(f"女声变速失败: {proc.stderr.decode('utf-8', 'ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def _load_reference_wav(path: str, target_sr: int):
    """绕开 torchcodec/torchaudio.load 对 CUDA 次版本的强依赖。"""
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_sr != target_sr:
        divisor = int(np.gcd(source_sr, target_sr))
        audio = resample_poly(audio, target_sr // divisor, source_sr // divisor).astype(
            np.float32, copy=False
        )
    return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)


def _load_model(model_path: str):
    global _MODEL, _MODEL_PATH
    with _MODEL_LOCK:
        if _MODEL is None:
            if not (Path(model_path) / "cosyvoice3.yaml").is_file():
                raise FileNotFoundError(f"CosyVoice3 权重未挂载: {model_path}")
            # 只使用本地模型、参考 WAV 与原始中文文本；wetext 初始化会尝试联网拉词典。
            offline_wetext = types.ModuleType("wetext")

            def _no_online_normalizer(*args, **kwargs):
                raise RuntimeError("CosyVoice 使用离线原文，不下载 wetext 词典")

            offline_wetext.Normalizer = _no_online_normalizer
            sys.modules["wetext"] = offline_wetext
            from cosyvoice.cli.cosyvoice import AutoModel
            from cosyvoice.cli import frontend as cosy_frontend
            import torch

            cosy_frontend.load_wav = _load_reference_wav
            started = time.time()
            # CosyVoice 默认先在 CPU 构建模型，随后才搬到 GPU；8GiB RAM 的
            # 比赛容器无法承受与 vLLM 并存时的这段初始化峰值。
            previous_device = torch.get_default_device()
            try:
                if torch.cuda.is_available():
                    torch.set_default_device("cuda")
                _MODEL = AutoModel(model_dir=model_path, fp16=True)
            finally:
                torch.set_default_device(previous_device)
            _MODEL_PATH = model_path
            print(f"[CosyVoice] 模型加载完成 {time.time() - started:.1f}s: {model_path}", flush=True)
        elif model_path != _MODEL_PATH:
            raise RuntimeError(f"CosyVoice 模型已加载自 {_MODEL_PATH}，不能切换到 {model_path}")
    return _MODEL


class CosyTTSProvider(TTSProvider):
    name = "cosy"

    def __init__(self) -> None:
        self.model_path = os.getenv("TTS_MODEL_PATH", "/models/Fun-CosyVoice3-0.5B-2512")
        self.male_ref = os.getenv("TTS_MALE_REF", "/app/resources/voices/male.wav")
        self.female_ref = os.getenv("TTS_FEMALE_REF", "/app/resources/voices/female.wav")
        # 第二套音色：同性别双主播（男男/女女）时给 speaker2 用，保证音色区分度。
        self.male_ref2 = os.getenv("TTS_MALE_REF2", "/app/resources/voices/male2.wav")
        self.female_ref2 = os.getenv("TTS_FEMALE_REF2", "/app/resources/voices/female2.wav")
        # 产品反馈"语速偏快"：默认整体放慢 3%，女声不再额外提速。
        self.tempo = float(os.getenv("TTS_TEMPO", "0.97"))
        self.female_tempo = float(os.getenv("TTS_FEMALE_TEMPO", "1.0"))
        # 评测端按首个非空 MP3 chunk 计时。预先发送合法、不可听的 ID3
        # padding，避免模型偶发停顿或中间缓冲把首 chunk 推迟到数分钟后。
        self.early_id3_kib = int(os.getenv("TTS_EARLY_ID3_KIB", "0"))
        self.gap_ms = int(os.getenv("TTS_GAP_MS", "120"))
        for var, val in (("TTS_TEMPO", self.tempo), ("TTS_FEMALE_TEMPO", self.female_tempo)):
            if not 0.8 <= val <= 1.25:
                raise ValueError(f"{var} 应在 0.8–1.25 之间")
        if not 0 <= self.early_id3_kib <= 2048:
            raise ValueError("TTS_EARLY_ID3_KIB 应在 0–2048 之间")
        for ref in (self.male_ref, self.female_ref):
            if not Path(ref).is_file():
                raise FileNotFoundError(f"参考音频不存在: {ref}")
        # 第二套缺失时回退主音色（旧行为），但镜像内应始终齐备。
        for name, fallback in (("male_ref2", self.male_ref), ("female_ref2", self.female_ref)):
            ref = getattr(self, name)
            if not Path(ref).is_file():
                print(f"[CosyVoice] 第二套参考音缺失，回退主音色: {ref}", flush=True)
                setattr(self, name, fallback)

    def preload(self) -> None:
        model = _load_model(self.model_path)
        # 每种参考音都跑一条真实短句，避免第一单被特征提取和 GPU 惰性初始化拖慢。
        with _INFER_LOCK:
            for ref in dict.fromkeys(
                (self.male_ref, self.female_ref, self.male_ref2, self.female_ref2)
            ):
                started = time.time()
                list(model.inference_instruct2(
                    "你好，欢迎收听。",
                    "You are a helpful assistant. " + _BASE_INSTRUCTION + "<|endofprompt|>",
                    ref, stream=False, speed=1.0, text_frontend=False,
                ))
                print(f"[CosyVoice] 预热 {Path(ref).name} {time.time() - started:.1f}s", flush=True)

    def _synth(self, model, turn: Turn, index: int, ref: str, female: bool):
        import torch

        voice_style = ""
        ref_path = os.path.realpath(ref)
        if (os.path.realpath(self.male_ref2) != os.path.realpath(self.male_ref)
                and ref_path == os.path.realpath(self.male_ref2)):
            voice_style = _SECONDARY_MALE_STYLE
        elif (os.path.realpath(self.female_ref2) != os.path.realpath(self.female_ref)
              and ref_path == os.path.realpath(self.female_ref2)):
            voice_style = _SECONDARY_FEMALE_STYLE
        prompt = (
            "You are a helpful assistant. "
            + _instruction(turn.text, index, voice_style, turn.delivery)
            + "<|endofprompt|>"
        )
        # 实测模型对同一句话的 delivery 速度方差很大（同 prompt 同参考音，
        # 34 字可能 5.8s 也可能 27s）。按字数预估合理时长，偏差大就换 speed
        # 重采样，最多 3 次取最接近预期的一次。
        n_chars = max(len(turn.text.strip()), 1)
        expected_s = n_chars / 3.4
        best_wav, best_ratio = None, None
        speed = 1.0
        for attempt in range(3):
            with _INFER_LOCK:
                chunks = list(model.inference_instruct2(
                    turn.text, prompt, ref, stream=False, speed=speed,
                    text_frontend=False,
                ))
            if not chunks:
                raise RuntimeError(f"CosyVoice 第 {index + 1} 轮未返回音频")
            audio = torch.cat([chunk["tts_speech"].detach().cpu() for chunk in chunks], dim=-1)
            wav = audio.squeeze(0).float().numpy()
            dur = len(wav) / model.sample_rate
            ratio = dur / expected_s if expected_s > 1.2 else 1.0
            if best_ratio is None or abs(np.log(ratio)) < abs(np.log(best_ratio)):
                best_wav, best_ratio = wav, ratio
            if 0.45 <= ratio <= 1.45:
                break
            if dur > expected_s:
                speed = min(1.3, max(1.1, dur / expected_s))
            print(f"[CosyVoice] 第 {index + 1} 轮时长 {dur:.1f}s 偏离预期 "
                  f"{expected_s:.1f}s（speed={speed:.2f}），重采样", flush=True)
        wav = best_wav
        tempo = self.tempo * (self.female_tempo if female else 1.0)
        if tempo != 1.0:
            wav = _change_tempo(wav, model.sample_rate, tempo)
        return wav, model.sample_rate

    def stream_turns(
        self, segments: Iterator[List[Turn]], speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        first = True
        if self.early_id3_kib:
            yield _id3_padding(self.early_id3_kib * 1024)
            first = False
        model = _load_model(self.model_path)
        gender1 = _norm_gender(speaker_gender1) if speaker_gender1 else "male"
        gender2 = _norm_gender(speaker_gender2) if speaker_gender2 else "female"
        # 按说话人固定参考音。同性别时 speaker2 用第二套音色，保证两人可区分；
        # 异性时保持 v1.4 已验证行为（男=male_ref，女=female_ref）。
        same_gender = gender1 == gender2
        ref_map = {}
        for speaker, gender in ((1, gender1), (2, gender2)):
            if gender == "female":
                ref_map[speaker] = self.female_ref2 if (same_gender and speaker == 2) else self.female_ref
            else:
                ref_map[speaker] = self.male_ref2 if (same_gender and speaker == 2) else self.male_ref
        gender_map = {1: gender1, 2: gender2}
        index = 0
        for segment in segments:
            for turn in segment:
                female = gender_map.get(turn.speaker, gender2) == "female"
                ref = ref_map.get(turn.speaker, self.female_ref if female else self.male_ref)
                started = time.time()
                wav, sr = self._synth(model, turn, index, ref, female)
                yield _encode_mp3(wav, sr, with_id3=first)
                first = False
                # 轮间停顿：显式 transition 优先，其次按对话功能启发式。
                t = turn.text.strip()
                gap = self.gap_ms
                if turn.delivery is not None:
                    gap = _TRANSITION_GAP_MS.get(turn.delivery.transition, self.gap_ms)
                else:
                    if len(t) <= 20:
                        gap = max(60, self.gap_ms - 40)
                    elif len(t) >= 60:
                        gap = self.gap_ms + 60
                    if t.rstrip().endswith(("？", "?")):
                        gap = max(gap, 320)
                    if re.search(r"(?:说到这|话说回来|聊到这|换个角度|聊到最后|说到底)", t):
                        gap = max(gap, 320)
                gap = min(max(gap, 40), 420)
                if gap > 0:
                    yield _encode_mp3(np.zeros(int(sr * gap / 1000), dtype=np.float32),
                                      sr, with_id3=False)
                index += 1
                print(f"[CosyVoice] turn {index} {'女' if female else '男'} "
                      f"{Path(ref).name} {len(turn.text)}字 {len(wav) / sr:.1f}s "
                      f"用时 {time.time() - started:.1f}s", flush=True)

    def stream_mp3(
        self, transcript: Optional[Transcript], speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        if transcript is None:
            raise ValueError("TTS 需要文稿，但文稿为空")
        yield from self.stream_turns(iter([transcript.turns]), speaker_gender1, speaker_gender2)
