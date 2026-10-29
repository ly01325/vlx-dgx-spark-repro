#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OMCHAT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-vlx-seek-1.5-3b-sanitized}"
MODEL_PATH="${MODEL_PATH:-${OMCHAT_ROOT}/resources/${MODEL_NAME}}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_LEN:-16384}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${MAX_SEQ:-256}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_BATCHED_TOKENS:-4096}}"
MAX_IMAGES="${MAX_IMAGES:-${MAX_IMG:-1}}"

export OMCHAT_PROFILE_MODEL_PHASES="${OMCHAT_PROFILE_MODEL_PHASES:-1}"
export OMCHAT_PROFILE_OBJECT_FEATURES="${OMCHAT_PROFILE_OBJECT_FEATURES:-1}"
export VLLM_USE_FLASHRT_OPS="${VLLM_USE_FLASHRT_OPS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${OMCHAT_CPU_THREADS:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMCHAT_CPU_THREADS:-16}}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to the vlx-seek-1.5-3b-sanitized directory." >&2
    exit 1
fi

python - <<'PY'
from vllm.model_executor.models import ModelRegistry

architecture = "OmChatQwen3_5ForCausalLM"
if architecture not in ModelRegistry.get_supported_archs():
    raise SystemExit(
        "The installed vLLM does not contain OmChatQwen3_5ForCausalLM. "
        "Copy the supplied vllm_ directory into the installed vllm package first."
    )
PY

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "No CUDA device is visible to PyTorch." >&2
    exit 1
fi

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"

if [[ -z "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    GPU_MEMORY_UTILIZATION="$(python - <<'PY'
import torch

count = torch.cuda.device_count()
total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3) * count
cap = {1: 0.93, 2: 0.90, 4: 0.88, 8: 0.85}.get(count, 0.90)
print(f"{min(8.0 / total_gib + 0.25, cap):.4f}")
PY
)"
fi

args=(
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}"
    --allowed-local-media-path /
    --trust-remote-code
)

if [[ "${NO_PREFIX_CACHE:-1}" == "1" ]]; then
    args+=(--no-enable-prefix-caching)
else
    args+=(--enable-prefix-caching)
fi

if [[ "${NO_MM_CACHE:-1}" == "1" ]]; then
    args+=(--mm-processor-cache-gb 0)
else
    args+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB:-4}")
fi

if [[ "${OMCHAT_PROFILE_MODEL_PHASES}" != "1" ]]; then
    args+=(--disable-log-stats)
fi

if [[ -n "${APIKEY:-}" ]]; then
    args+=(--api-key "${APIKEY}")
fi

echo "Starting raw vLLM server"
echo "  model: ${MODEL_PATH}"
echo "  served name: ${MODEL_NAME}"
echo "  endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo "  tensor parallel: ${TENSOR_PARALLEL_SIZE}"
echo "  GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"

exec python "${SCRIPT_DIR}/launch_vllm.py" "${args[@]}" "$@"
