# -*- coding: utf-8 -*-
"""音频合成后端：Qwen3-TTS-12Hz-CustomVoice（内置音色），按轮流式输出 MP3。

设计要点：
  - 按 turn 合成（每个说话人固定内置音色），合成完一个 turn 立即编码为 MP3 并 yield，
    因此首字节 ≈ 第一个 turn 的合成时间（满足"音频首字 <30s"）。
  - MP3 以帧流拼接：首个块带 ID3，后续块 -id3v2_version 0，拼起来是合法 MP3 流。
  - 采样参数默认比模型默认更稳（temp0.6/top_p0.9），减少音色漂移。

环境变量：TTS_MODEL_PATH / TTS_DEVICE / TTS_GAP_MS / TTS_INSTRUCT
          TTS_MALES / TTS_FEMALES / TTS_TEMPERATURE / TTS_TOP_P / TTS_TOP_K / TTS_SEED
"""
import os
import subprocess
import threading
import re
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

from app.config import config
from app.providers.base import TTSProvider
from app.schemas import Transcript, Turn

_DEV_DEFAULT_MODEL = os.path.join(
    os.getenv("MODELS_ROOT", "models"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)
DEFAULT_MODEL_PATH = os.getenv("TTS_MODEL_PATH") or (
    _DEV_DEFAULT_MODEL if os.path.isdir(_DEV_DEFAULT_MODEL) else ""
)

_DIALECT_VOICES = {"eric"}  # 官方定义为 Chengdu/Sichuan voice，商业播客默认禁用
_ALLOW_DIALECT = os.getenv("TTS_ALLOW_DIALECT_VOICES", "0").strip().lower() in (
    "1", "true", "yes",
)


def _voice_list(env_name: str, defaults: str) -> List[str]:
    voices = [s.strip().lower() for s in os.getenv(env_name, defaults).split(",") if s.strip()]
    if not _ALLOW_DIALECT:
        voices = [s for s in voices if s not in _DIALECT_VOICES]
    return voices


# 商业播客默认只使用青年音色：dylan（男）+ vivian（女）。
# uncle_fu 年龄感偏大、eric 是四川话、serena 实听有口音，均不进入默认音色池。
VOICE_BANK = {
    "male": _voice_list("TTS_MALES", "dylan") or ["dylan"],
    "female": _voice_list("TTS_FEMALES", "vivian") or ["vivian"],
}

GEN_PARAMS = {
    "temperature": float(os.getenv("TTS_TEMPERATURE", "0.6")),
    "top_p": float(os.getenv("TTS_TOP_P", "0.9")),
    "top_k": int(os.getenv("TTS_TOP_K", "40")),
    "repetition_penalty": float(os.getenv("TTS_REPETITION_PENALTY", "1.05")),
}
TTS_SEED = os.getenv("TTS_SEED", "").strip()

DEFAULT_DIALOGUE_INSTRUCT = (
    "使用自然、清晰的标准普通话，不带四川话、北京腔、儿化音或其他地方口音。"
    "像在录制轻松但专业的双人商业播客，面向对面的搭档自然交谈，"
    "不要播音腔，不要朗读稿件，停连和重音要符合句意。"
)

_VOICE_STYLE_INSTRUCT = {
    "dylan": "保持二十多岁青年男声，清爽、有活力；严格说标准普通话，不要北京腔和儿化音。",
    "vivian": "保持二十多岁青年女声，明亮、自然；严格说标准普通话，不带港台腔或南方口音。",
}

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _norm_gender(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("男", "male", "m", "man", "boy", "1"):
        return "male"
    if v in ("女", "female", "f", "woman", "girl", "2"):
        return "female"
    return "female"


def pick_voices(gender1: str, gender2: str) -> Tuple[str, str]:
    """按性别给两个主播分配不同内置音色。"""
    a, b = _norm_gender(gender1), _norm_gender(gender2)
    bank1 = VOICE_BANK[a]
    bank2 = VOICE_BANK[b]
    s1 = bank1[0]
    # 同性组合优先使用第二音色；配置只有一个音色时安全回退，不因 IndexError 整单失败。
    s2 = bank2[1] if a == b and len(bank2) > 1 else bank2[0]
    return s1, s2


_CONTRAST_RE = re.compile(r"(?:但|不过|反过来|未必|问题是|恰恰|别急|换个角度)")
_WRAP_RE = re.compile(r"(?:回到最初|说到底|归根结底|最后|收个尾|下期|感谢收听)")


def build_turn_instruct(turn: Turn, index: int, previous_text: str = "",
                        base: str = DEFAULT_DIALOGUE_INSTRUCT) -> str:
    """按对话功能给每一轮分配轻量语气，避免所有台词用同一种播报腔。"""
    text = turn.text.strip()
    cues = [base.strip()] if base.strip() else []
    if index == 0:
        cues.append("这是开场，语气松弛、有兴趣，直接把听众带进话题，不要喊口号。")
    elif text.endswith(("？", "?")):
        cues.append("带着真实好奇发问，句尾自然上扬，但不要像采访提词。")
    elif len(text) <= 20:
        cues.append("这是紧接对方的短回应，反应稍快，语气真实，有交流感，不要刻意拖长。")
    elif _CONTRAST_RE.search(text):
        cues.append("这里在补充或修正对方观点，语气有轻微转折和思考感，不要生硬反驳。")
    elif _WRAP_RE.search(text):
        cues.append("语气放松下来，像两位搭档自然收束讨论，不要念总结报告。")
    elif previous_text:
        cues.append("这是对搭档上一句话的具体承接，开头连贯，重点词适度强调。")
    return "".join(cues)


def _load_model(model_path: str, device: str):
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            t0 = time.time()
            kwargs = dict(
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            # 低内存加载：8G RAM 环境下避免把权重整体读进内存
            try:
                _MODEL = Qwen3TTSModel.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            except TypeError:
                _MODEL = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
            print(f"[TTS] 模型加载完成 {time.time() - t0:.1f}s: {model_path}", flush=True)
    return _MODEL


def _encode_mp3(wav: np.ndarray, sr: int, with_id3: bool, bitrate: str = "96k") -> bytes:
    """把单声道 float32 波形编码为 MP3 字节（raw f32le 经 stdin 送入 ffmpeg）。"""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libmp3lame", "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-b:a", bitrate, "-f", "mp3",
    ]
    if not with_id3:
        cmd += ["-id3v2_version", "0", "-write_xing", "0"]
    cmd.append("-")  # 输出到 stdout
    data = np.ascontiguousarray(wav, dtype="<f4").tobytes()
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 编码失败: {proc.stderr.decode('utf-8', 'ignore')[:300]}")
    return proc.stdout


class QwenTTSProvider(TTSProvider):
    name = "qwen"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        gap_ms: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device or os.getenv("TTS_DEVICE", "cuda:0")
        self.gap_ms = int(gap_ms if gap_ms is not None else os.getenv("TTS_GAP_MS", "120"))
        self.instruct = instruct if instruct is not None else os.getenv(
            "TTS_INSTRUCT", DEFAULT_DIALOGUE_INSTRUCT
        )
        self._silence_cache = {}
        if not self.model_path:
            raise ValueError("TTS_MODEL_PATH 未设置")

    # ---------- 预热 ----------
    def preload(self) -> None:
        model = _load_model(self.model_path, self.device)
        # 跑一次极短推理，触发 kernel 调优/惰性编译；否则首个真实请求的第一句
        # 会慢 20~35s，导致"音频首字节 > 30s"直接判失败。
        try:
            t0 = time.time()
            speaker = VOICE_BANK["male"][0] if VOICE_BANK["male"] else VOICE_BANK["female"][0]
            model.generate_custom_voice(text="你好，欢迎收听。", language="Chinese",
                                        speaker=speaker)
            print(f"[TTS] 预热推理完成 {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[TTS] 预热推理失败（忽略）: {exc}", flush=True)

    # ---------- 单轮合成 ----------
    def _synth_turn(self, model, turn: Turn, speaker: str, index: int,
                    previous_text: str = ""):
        import torch

        kwargs = {"text": turn.text, "language": "Chinese", "speaker": speaker, **GEN_PARAMS}
        voice_instruct = self.instruct + _VOICE_STYLE_INSTRUCT.get(speaker.lower(), "")
        turn_instruct = build_turn_instruct(turn, index, previous_text, voice_instruct)
        if turn_instruct:
            kwargs["instruct"] = turn_instruct
        if TTS_SEED:
            torch.manual_seed(int(TTS_SEED) + index * 2 + (0 if turn.speaker == 1 else 1))
        wavs, sr = model.generate_custom_voice(**kwargs)
        wav = np.asarray(wavs[0], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, sr

    def _iter_turn_audio(self, transcript: Transcript, gender1: str, gender2: str):
        model = _load_model(self.model_path, self.device)
        s1, s2 = pick_voices(gender1, gender2)
        print(f"[TTS] 音色 speaker1={s1} speaker2={s2}，共 {len(transcript.turns)} 轮", flush=True)
        previous_text = ""
        for i, turn in enumerate(transcript.turns):
            speaker = s1 if turn.speaker == 1 else s2
            t0 = time.time()
            wav, sr = self._synth_turn(model, turn, speaker, i, previous_text)
            print(f"[TTS] turn {i + 1}/{len(transcript.turns)} spk={speaker} "
                  f"{len(turn.text)}字 {len(wav) / sr:.1f}s 用时{time.time() - t0:.1f}s", flush=True)
            previous_text = turn.text
            yield wav, sr

    def _silence(self, sr: int, gap_ms: Optional[int] = None) -> np.ndarray:
        gap_ms = self.gap_ms if gap_ms is None else gap_ms
        key = (sr, gap_ms)
        if key not in self._silence_cache:
            self._silence_cache[key] = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
        return self._silence_cache[key]

    def _turn_gap_ms(self, turn: Turn) -> int:
        """短回应更快接话，问句给对方少量思考空间。"""
        if len(turn.text.strip()) <= 20:
            return max(60, self.gap_ms - 40)
        if turn.text.rstrip().endswith(("？", "?")):
            return self.gap_ms + 30
        return self.gap_ms

    # ---------- 契约层接口：按轮流式 MP3 ----------
    def stream_mp3(
        self,
        transcript: Optional[Transcript],
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        if transcript is None:
            raise ValueError("TTS 需要文稿，但文稿为空")
        first = True
        n = len(transcript.turns)
        for i, (wav, sr) in enumerate(self._iter_turn_audio(transcript, speaker_gender1, speaker_gender2)):
            yield _encode_mp3(wav, sr, with_id3=first)
            first = False
            if i < n - 1:
                yield _encode_mp3(self._silence(sr, self._turn_gap_ms(transcript.turns[i])),
                                  sr, with_id3=False)

    # ---------- 按段流式：拿到第一批轮次就出声 ----------
    def stream_turns(
        self,
        segments: Iterator[List[Turn]],
        speaker_gender1: str = "",
        speaker_gender2: str = "",
    ) -> Iterator[bytes]:
        model = _load_model(self.model_path, self.device)
        s1, s2 = pick_voices(speaker_gender1, speaker_gender2)
        first = True
        idx = 0
        previous_text = ""
        for seg in segments:
            for turn in seg:
                speaker = s1 if turn.speaker == 1 else s2
                t0 = time.time()
                wav, sr = self._synth_turn(model, turn, speaker, idx, previous_text)
                if first:
                    print(f"[TTS] 首字节即将产出：turn1 spk={speaker} "
                          f"用时{time.time() - t0:.1f}s", flush=True)
                yield _encode_mp3(wav, sr, with_id3=first)
                first = False
                yield _encode_mp3(self._silence(sr, self._turn_gap_ms(turn)),
                                  sr, with_id3=False)
                idx += 1
                previous_text = turn.text
                print(f"[TTS] turn {idx} spk={speaker} {len(turn.text)}字 "
                      f"{len(wav) / sr:.1f}s 用时{time.time() - t0:.1f}s", flush=True)

    # ---------- 落盘整段 MP3（demo / 调试用） ----------
    def synthesize_mp3(self, transcript: Transcript, gender1: str, gender2: str,
                       out_mp3: str) -> float:
        dur = 0.0
        with open(out_mp3, "wb") as f:
            for chunk in self.stream_mp3(transcript, gender1, gender2):
                f.write(chunk)
        total = sum(len(t.text) for t in transcript.turns)
        # 用 ffprobe 取实际时长
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", out_mp3],
                capture_output=True, text=True,
            )
            dur = float(out.stdout.strip())
        except Exception:
            pass
        print(f"[TTS] 合成完成 {dur:.1f}s（{total} 字）-> {out_mp3}", flush=True)
        return dur
