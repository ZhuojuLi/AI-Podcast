#!/bin/bash
# 本地开发：启动契约服务（真实 provider）
# 依赖：LLM 在 127.0.0.1:8100（scripts/start_llm_dev.sh）
set -e
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="${SUT_GPU:-0}"
export SCRIPT_PROVIDER="${SCRIPT_PROVIDER:-qwen}"
export TTS_PROVIDER="${TTS_PROVIDER:-qwen}"
export COVER_PROVIDER="${COVER_PROVIDER:-pil}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8100/v1}"
export LLM_MODEL="${LLM_MODEL:-qwen-awq}"
export LLM_TARGET_SECONDS="${LLM_TARGET_SECONDS:-120}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"
export MOCK_SERVER_PORT="${MOCK_SERVER_PORT:-8086}"
# TTS_MODEL_PATH 需指向本地 TTS 模型目录（如 Qwen3-TTS-12Hz-1.7B-CustomVoice）
export TTS_MODEL_PATH="${TTS_MODEL_PATH:-}"
exec "${VENV_PY:-.venv-tts/bin/python}" -m app.main
