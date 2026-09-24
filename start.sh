#!/bin/bash
# =========================================================
# AI 播客生成 SUT 启动：vLLM(写稿) + 契约服务
# 就绪语义：/ready 在 vLLM 就绪前返回 503，评测端 readinessProbe 持续重试
#
# 注意：Qwen3.5 是混合 Mamba 架构，vLLM 下**不能**用默认的 'all' mamba 缓存
#      （开 --enable-prefix-caching 会把 mamba_cache_mode 设为 'all' 并抛
#       NotImplementedError）。因此这里不开 prefix caching。
# =========================================================
set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"; }

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-14B-AWQ}"
MODEL_NAME="${MODEL_NAME:-qwen3-14b-awq}"
VLLM_PORT="${VLLM_PORT:-8100}"
PORT="${PORT:-80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LLM_MEM_GB="${LLM_MEM_GB:-24}"            # LLM(vLLM) 目标显存（GB，绝对值）
LLM_MEM_GB_FALLBACK="${LLM_MEM_GB_FALLBACK:-26}"
LLM_GPU="${LLM_GPU:-0}"
TTS_GPU="${TTS_GPU:-0}"
[ -z "$MODEL_PATH" ] && MODEL_PATH="/models/Qwen3-14B-AWQ"
[ -z "$MODEL_NAME" ] && MODEL_NAME="qwen3-14b-awq"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

VLLM_LOG=/var/log/vllm.log

# ---------- 计算 GPU_MEM_UTIL：把「绝对 GB 预算」换算成 vLLM 需要的比例 ----------
# 这样无论平台给 40G 还是 80G，LLM 都只占用目标 GB。
# 显式设置 GPU_MEM_UTIL 时以其为准（本地小卡调试用）。
TOTAL_MEM_GB=$(python3 - <<'PY' 2>/dev/null || echo "0"
import torch, os
idx = int(os.environ.get("LLM_GPU", "0"))
print(f"{torch.cuda.get_device_properties(idx).total_memory/1024**3:.1f}")
PY
)
calc_util() {
    python3 - "$1" "$TOTAL_MEM_GB" <<'PY'
import sys
target_gb, total_gb = float(sys.argv[1]), float(sys.argv[2])
print("0.7" if total_gb <= 0 else f"{min(0.92, target_gb/total_gb):.3f}")
PY
}
if [ -n "${GPU_MEM_UTIL:-}" ]; then
    UTIL_A="$GPU_MEM_UTIL"
else
    UTIL_A="$(calc_util "$LLM_MEM_GB")"
fi
UTIL_B="$(calc_util "$LLM_MEM_GB_FALLBACK")"

log "===== AI 播客生成 SUT 启动 ====="
log "MODEL_PATH=${MODEL_PATH} MODEL_NAME=${MODEL_NAME} PORT=${PORT}"
log "GPU 总显存≈${TOTAL_MEM_GB}G；LLM 目标 ${LLM_MEM_GB}G → util_A=${UTIL_A}；fallback ${LLM_MEM_GB_FALLBACK}G → util_B=${UTIL_B}"
log "MAX_MODEL_LEN=${MAX_MODEL_LEN} LLM_GPU=${LLM_GPU} TTS_GPU=${TTS_GPU}"
log "PROVIDERS: script=${SCRIPT_PROVIDER} tts=${TTS_PROVIDER} cover=${COVER_PROVIDER} TTS_MODEL_PATH=${TTS_MODEL_PATH}"
log "---- nvidia-smi ----"; nvidia-smi 2>&1 | head -n 18 || true
log "---- free -m ----"; free -m 2>&1 || true

dump_vllm() {
    log "==== vLLM 失败诊断（关键行） ===="
    grep -nE "ERROR|Traceback|NotImplementedError|OutOfMemory|CUDA out of memory|Killed|RuntimeError|ValueError|not support|Free memory|less than desired" "$VLLM_LOG" 2>/dev/null | tail -n 40 || true
    log "---- vLLM 日志尾部 120 行 ----"
    tail -n 120 "$VLLM_LOG" 2>/dev/null || true
}

start_vllm() {
    log "启动 vLLM: $*"
    CUDA_VISIBLE_DEVICES="$LLM_GPU" python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --trust-remote-code "$@" >> "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    log "vLLM pid=${VLLM_PID}"
}

wait_ready() {
    while :; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then return 1; fi
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
        sleep 3
    done
}

# ---------- 尝试 A：省显存（关视觉塔 + 限制并发） ----------
: > "$VLLM_LOG"
start_vllm \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$UTIL_A" \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --language-model-only

if wait_ready; then
    log "vLLM 就绪（配置 A）"
else
    dump_vllm
    kill -9 "$VLLM_PID" 2>/dev/null || true
    sleep 2
    log "配置 A 失败，改用配置 B 重试"
    : > "$VLLM_LOG"
    start_vllm \
        --max-model-len 4096 \
        --gpu-memory-utilization "$UTIL_B" \
        --max-num-seqs 1
    if wait_ready; then
        log "vLLM 就绪（配置 B）"
    else
        dump_vllm
        kill -9 "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
fi

# ---------- 契约服务（含 TTS 预热，等 LLM 就绪后加载） ----------
CUDA_VISIBLE_DEVICES="$TTS_GPU" TTS_DEVICE="cuda:0" \
/opt/podcast-venv/bin/python -m app.main &
APP_PID=$!
log "契约服务 pid=${APP_PID}"

# ---------- 任一进程退出则整体退出（交给平台重启） ----------
while kill -0 "${VLLM_PID}" 2>/dev/null && kill -0 "${APP_PID}" 2>/dev/null; do
    sleep 5
done
log "有进程退出，关闭服务"
kill "${VLLM_PID}" "${APP_PID}" 2>/dev/null || true
sleep 2
dump_vllm
exit 1
