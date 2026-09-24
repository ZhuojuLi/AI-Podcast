# -*- coding: utf-8 -*-
"""生成后端注册与加载。

支持三种写法：
  - ``mock``：内置默认实现（app/providers/mock.py）
  - 短名：``qwen``（写稿 LLM）/ ``qwen``（TTS）/ ``sd``（封面）
  - ``<python.module.path>:<ClassName>``：动态导入自定义实现

示例：
  SCRIPT_PROVIDER=qwen
  TTS_PROVIDER=qwen
  COVER_PROVIDER=sd
  SCRIPT_PROVIDER=my_pkg.my_llm:MyLLMScriptProvider
"""
import importlib
from typing import Optional

from app.config import config
from app.providers.base import CoverProvider, ScriptProvider, TTSProvider

# 短名 -> "module:ClassName"
_SHORT_NAMES = {
    "script": {
        "mock": "app.providers.mock:MockScriptProvider",
        "qwen": "app.providers.llm:QwenScriptProvider",
    },
    "tts": {
        "mock": "app.providers.mock:MockTTSProvider",
        "qwen": "app.providers.tts:QwenTTSProvider",
        "cosy": "app.providers.cosy:CosyTTSProvider",
    },
    "cover": {
        "mock": "app.providers.mock:MockCoverProvider",
        "pil": "app.providers.image:PillowCoverProvider",
        "sd": "app.providers.image:SDXLCoverProvider",
    },
}


def _load(spec: str, kind: str, short_map: dict) -> object:
    spec = (spec or "mock").strip()
    if spec in short_map:
        spec = short_map[spec]
    if ":" not in spec:
        raise ValueError(
            f"未知 {kind} 配置: {spec!r}；应为短名 {list(short_map)} 或 'module.path:ClassName'"
        )
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


def get_script_provider() -> ScriptProvider:
    return _load(config.SCRIPT_PROVIDER, "SCRIPT_PROVIDER", _SHORT_NAMES["script"])  # type: ignore


def get_tts_provider() -> TTSProvider:
    return _load(config.TTS_PROVIDER, "TTS_PROVIDER", _SHORT_NAMES["tts"])  # type: ignore


def get_cover_provider() -> CoverProvider:
    return _load(config.COVER_PROVIDER, "COVER_PROVIDER", _SHORT_NAMES["cover"])  # type: ignore
