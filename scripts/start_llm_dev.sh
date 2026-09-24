#!/bin/bash
# 本地开发：启动写稿 LLM（vLLM OpenAI 兼容）。
# 需要：装了 vLLM 的 python 环境（VLLM_PY）和一个本地量化模型目录（LLM_MODEL_DIR）。
set -e
VLLM_PY="${VLLM_PY:-python3}"
export CUDA_VISIBLE_DEVICES="${LLM_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
: "${LLM_MODEL_DIR:?请设置 LLM_MODEL_DIR 指向本地模型目录（如 Qwen2.5-7B-Instruct-AWQ）}"
exec "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL_DIR" \
    --served-model-name "${LLM_MODEL_NAME:-qwen-awq}" \
    --host 127.0.0.1 --port "${LLM_PORT:-8100}" \
    --gpu-memory-utilization "${LLM_GPU_MEM:-0.09}" \
    --max-model-len "${LLM_MAX_LEN:-4096}" \
    --enforce-eager --trust-remote-code --no-enable-prefix-caching
