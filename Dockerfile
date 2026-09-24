# =========================================================
# AI 播客生成服务镜像
# 基座：任意 vLLM 0.23 + cu129 基座镜像（Python3.12 / torch2.11 / vLLM0.23 /
# 自带 ffmpeg+curl 即可；如 docker.io/vllm/vllm-openai 对应版本）
# 模型权重不进镜像，运行时挂载到 /models
# =========================================================
ARG BASE_IMAGE=vllm/vllm-openai:v0.23.0-cu129
FROM ${BASE_IMAGE}

WORKDIR /app

# ---------- 系统依赖：中文封面字体 ----------
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
        if [ -f "$f" ]; then \
            sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.aliyun.com/ubuntu@g; s@http://security.ubuntu.com/ubuntu@https://mirrors.aliyun.com/ubuntu@g' "$f"; \
        fi; \
    done; \
    apt-get update && apt-get install -y --no-install-recommends \
        fonts-wqy-zenhei fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# ---------- 契约服务依赖（系统 python，与 vLLM 同环境） ----------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --ignore-installed blinker \
        -r /app/requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
    && python3 -c "import flask, PIL, requests, yaml; print('contract deps OK')"

# ---------- CosyVoice3 推理环境（隔离依赖，共享基座 torch/torchaudio） ----------
COPY requirements-cosy.txt /app/requirements-cosy.txt
COPY third_party/CosyVoice /opt/CosyVoice
RUN python3 -m venv --system-site-packages /opt/podcast-venv \
    && /opt/podcast-venv/bin/python -m pip install -U "setuptools<81" wheel \
        -i https://mirrors.aliyun.com/pypi/simple/
# Whisper 2023 的 setup.py 依赖 pkg_resources；避免隔离构建时拿到移除此模块的 setuptools。
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/podcast-venv/bin/python -m pip install --no-build-isolation --no-deps \
        "openai-whisper==20231117" -i https://mirrors.aliyun.com/pypi/simple/
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/podcast-venv/bin/python -m pip install \
        -r /app/requirements-cosy.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    && PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
        /opt/podcast-venv/bin/python -c "import torch, transformers; from cosyvoice.cli.cosyvoice import AutoModel; print('CosyVoice env OK', torch.__version__, transformers.__version__)"

# ---------- 应用代码 ----------
COPY app /app/app
COPY config /app/config
COPY scripts /app/scripts
COPY resources /app/resources
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ---------- 运行配置（模型路径均在 /models 下，由 submit.yaml 挂载） ----------
ENV PORT=80 \
    HOST=0.0.0.0 \
    VLLM_PORT=8100 \
    MODELS_ROOT=/models \
    MODEL_PATH=/models/Qwen3-14B-AWQ \
    MODEL_NAME=qwen3-14b-awq \
    MAX_MODEL_LEN=8192 \
    LLM_MEM_GB=24 \
    LLM_MEM_GB_FALLBACK=26 \
    LLM_BASE_URL=http://127.0.0.1:8100/v1 \
    LLM_MODEL=qwen3-14b-awq \
    LLM_TARGET_SECONDS=480 \
    LLM_MAX_TOKENS=3072 \
    TTS_MODEL_PATH=/models/Fun-CosyVoice3-0.5B-2512 \
    COSYVOICE_PRECOMPUTED_REFS=/app/resources/voices/reference_features.pt \
    TTS_MALE_REF=/app/resources/voices/male.wav \
    TTS_FEMALE_REF=/app/resources/voices/female.wav \
    TTS_MALE_REF2=/app/resources/voices/male2.wav \
    TTS_FEMALE_REF2=/app/resources/voices/female2.wav \
    TTS_TEMPO=0.97 \
    TTS_FEMALE_TEMPO=1.0 \
    TTS_EARLY_ID3_KIB=1024 \
    TTS_GAP_MS=120 \
    SCRIPT_PROVIDER=qwen \
    TTS_PROVIDER=cosy \
    COVER_PROVIDER=pil \
    SEARCH_PROVIDER=ddgs \
    TTS_PRELOAD=1 \
    READY_REQUIRE_LLM=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app:/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    PYTHONUNBUFFERED=1

EXPOSE 80

# 基座 ENTRYPOINT 是 `vllm serve`，必须显式覆盖
ENTRYPOINT ["/app/start.sh"]
