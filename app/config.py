# -*- coding: utf-8 -*-
"""全局配置。

所有参数均可通过环境变量覆盖；默认值与 starting-kit 契约保持一致。
生成能力通过 *_PROVIDER 三个开关选择，支持 ``mock`` 或
``module.path:ClassName`` 形式的自定义实现（见 app/providers/registry.py）。
"""
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Config:
    # ---------- 服务 ----------
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = _env_int("MOCK_SERVER_PORT", _env_int("PORT", 80))

    # ---------- 产物目录 ----------
    # 必须绝对路径：werkzeug.send_file 会把相对路径按 Flask root_path 解析
    AUDIO_DIR = os.path.abspath(os.getenv("AUDIO_DIR", "./mock_audios"))
    IMAGE_DIR = os.path.abspath(os.getenv("IMAGE_DIR", "./mock_images"))

    # ---------- 音频规格 ----------
    AUDIO_SAMPLE_RATE = _env_int("AUDIO_SAMPLE_RATE", 16000)
    AUDIO_CHANNELS = _env_int("AUDIO_CHANNELS", 1)
    AUDIO_MIN_DURATION = _env_int("AUDIO_MIN_DURATION", 420)  # 7 分钟（评测要求 5-15 分钟）
    AUDIO_MAX_DURATION = _env_int("AUDIO_MAX_DURATION", 600)  # 10 分钟
    AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")
    AUDIO_CHUNK_SIZE = _env_int("AUDIO_CHUNK_SIZE", 8192)
    AUDIO_CHUNK_DELAY = _env_float("AUDIO_CHUNK_DELAY", 0.01)

    # ---------- 封面规格 ----------
    COVER_REQUIRED_WIDTH = _env_int("COVER_REQUIRED_WIDTH", 1024)
    COVER_REQUIRED_HEIGHT = _env_int("COVER_REQUIRED_HEIGHT", 1024)

    # ---------- 文稿流 ----------
    CONTENT_CHUNK_DELAY = _env_float("CONTENT_CHUNK_DELAY", 0.2)
    CONTENT_QUEUE_WAIT_TIMEOUT = _env_int("CONTENT_QUEUE_WAIT_TIMEOUT", 30)
    # 队列无数据时的最大等待：必须 > LLM 写稿耗时（27B 可达 1-3min）
    CONTENT_QUEUE_TIMEOUT = _env_int("CONTENT_QUEUE_TIMEOUT", 900)
    # TTS 等待文稿就绪的最大时间
    TRANSCRIPT_WAIT_TIMEOUT = _env_int("TRANSCRIPT_WAIT_TIMEOUT", 900)

    # ---------- 生成后端（可插拔） ----------
    SCRIPT_PROVIDER = os.getenv("SCRIPT_PROVIDER", "mock")
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "mock")
    COVER_PROVIDER = os.getenv("COVER_PROVIDER", "mock")

    # ---------- 就绪 / 预热 ----------
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v1")
    # /ready 是否要求 LLM 就绪（未就绪返回 503，评测端会重试）
    READY_REQUIRE_LLM = os.getenv("READY_REQUIRE_LLM", "1").strip() not in ("0", "false", "False", "")
    # 启动时后台预热 TTS 模型
    TTS_PRELOAD = os.getenv("TTS_PRELOAD", "0").strip() in ("1", "true", "True")
    # /ready 是否要求 TTS 也预热完成（避免首个 case 首字节被冷加载拖慢）
    READY_REQUIRE_TTS = os.getenv("READY_REQUIRE_TTS", "1").strip() not in ("0", "false", "False", "")


config = Config()
