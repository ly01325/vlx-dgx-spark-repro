#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8101}"
GEMM_BACKEND="${2:-humming}"
LM_HEAD_W4A16="${3:-1}"
PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
CONTAINER="vlx-humming-${GEMM_BACKEND}-lm${LM_HEAD_W4A16}-${PORT}"
RESULT_ROOT="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/humming-validation-${PORT}}"
RESULT_DIR="${RESULT_ROOT}/${GEMM_BACKEND}-lm-w4a16-${LM_HEAD_W4A16}"

mkdir -p "${RESULT_DIR}"
cleanup() { docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --network host \
    -v "${PROJECT_DIR}:/workspace/to_nvidia" \
    -e "VLLM_NVFP4_GEMM_BACKEND=${GEMM_BACKEND}" \
    -e "VLLM_HUMMING_LM_HEAD_W4A16=${LM_HEAD_W4A16}" \
    -e VLLM_USE_FLASHRT_OPS=0 \
    -e ATTENTION_BACKEND=FLASH_ATTN \
    -e "MODEL_NAME=${MODEL_NAME}" \
    -e "MODEL_PATH=/workspace/to_nvidia/resources/${MODEL_NAME}" \
    -e "PORT=${PORT}" \
    -e HOST=0.0.0.0 \
    -e GPU_MEMORY_UTILIZATION=0.10 \
    -e MAX_MODEL_LEN=4096 \
    -e MAX_NUM_SEQS=16 \
    -e MAX_NUM_BATCHED_TOKENS=4096 \
    -e NO_PREFIX_CACHE=1 \
    -e NO_MM_CACHE=1 \
    --entrypoint bash \
    "${IMAGE}" -lc '
        python3 -m pip install --no-deps --no-build-isolation \
            /workspace/to_nvidia/third_party/sources/humming-a74973b.tar.gz
        cp -a /workspace/to_nvidia/vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
        cd /workspace/to_nvidia
        exec bash raw_vllm_server/start_vllm_server.sh \
            --quantization modelopt_fp4 --attention-backend FLASH_ATTN
    ' >/dev/null

ready=0
for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
        break
    fi
    sleep 1
done

if (( ! ready )); then
    docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true
    tail -120 "${RESULT_DIR}/server.log" >&2
    exit 1
fi

python3 "${PROJECT_DIR}/benchmarks/bench_qwen35.py" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --input-tokens 2200 --output-tokens 27 \
    --warmup 3 --latency-runs 10 \
    --concurrency 1,8 --requests-per-concurrency 10 \
    --out "${RESULT_DIR}/benchmark.json"

python3 "${PROJECT_DIR}/benchmarks/compare_qwen35_accuracy.py" \
    --baseline-url http://127.0.0.1:8000/v1/completions \
    --candidate-url "http://127.0.0.1:${PORT}/v1/completions" \
    --model "${MODEL_NAME}" --max-tokens 27 \
    --out "${RESULT_DIR}/accuracy.json"

docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true
grep -E "Humming|humming|NvFp4LinearBackend|FlashAttention" \
    "${RESULT_DIR}/server.log" | tail -60 || true
