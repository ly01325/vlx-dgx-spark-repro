#!/usr/bin/env bash
set -euo pipefail

# Run one isolated Qwen3.5 kernel candidate on DGX Spark. This script is meant
# to execute on the Spark host, not on the Mac.

GEMM_BACKEND="${1:?usage: run-qwen35-kernel-candidate.sh GEMM_BACKEND ATTENTION_BACKEND [PORT]}"
ATTENTION_BACKEND="${2:?usage: run-qwen35-kernel-candidate.sh GEMM_BACKEND ATTENTION_BACKEND [PORT]}"
PORT="${3:-8101}"

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
CONTAINER="vlx-kernel-test-${PORT}"
RESULT_DIR="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/kernel-test-${PORT}}"
OUTPUT_LENGTHS="${OUTPUT_LENGTHS:-27}"
CONCURRENCIES="${CONCURRENCIES:-1,8}"
REQUESTS_PER_CONCURRENCY="${REQUESTS_PER_CONCURRENCY:-8}"
LATENCY_RUNS="${LATENCY_RUNS:-6}"

mkdir -p "${RESULT_DIR}"

cleanup() {
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

echo "Starting candidate GEMM=${GEMM_BACKEND} attention=${ATTENTION_BACKEND}"
docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --network host \
    -v "${PROJECT_DIR}:/workspace/to_nvidia" \
    -v "${PROJECT_DIR}/resources/demo_1920_1080.jpeg:/workspace/omchat/demo_1920_1080.jpeg:ro" \
    -e "VLLM_NVFP4_GEMM_BACKEND=${GEMM_BACKEND}" \
    -e VLLM_USE_FLASHRT_OPS=0 \
    -e VLLM_HUMMING_LM_HEAD_W4A16=0 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e GDN_PREFILL_BACKEND=flashinfer \
    -e "ATTENTION_BACKEND=${ATTENTION_BACKEND}" \
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
        set -euo pipefail
        PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
            /workspace/to_nvidia/third_party/wheels/apache_tvm_ffi-0.1.13.post3-*.whl
        PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
            /workspace/to_nvidia/third_party/wheels/nvidia_cuda_nvdisasm-13.4.49-*.whl \
            /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl_libs_core-4.7.1-*.whl \
            /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl_libs_base-4.7.1-*.whl \
            /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl_libs_cu13-4.7.1-*.whl \
            /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl-4.7.1-*.whl
        PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
            /workspace/to_nvidia/third_party/wheels/flashinfer_python-0.6.18-py3-none-any.whl
        PIP_CONSTRAINT= python3 -m pip install --no-deps --no-build-isolation \
            /workspace/to_nvidia/third_party/sources/humming-a74973b.tar.gz
        cp -a /workspace/to_nvidia/vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
        cd /workspace/to_nvidia
        exec bash raw_vllm_server/start_vllm_server.sh \
            --quantization modelopt_fp4 \
            --attention-backend "${ATTENTION_BACKEND}" \
            --gdn-prefill-backend flashinfer
    ' >/dev/null

ready=0
for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
        break
    fi
    sleep 1
done

if (( ! ready )); then
    docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true
    echo "Candidate server failed to become healthy" >&2
    tail -80 "${RESULT_DIR}/server.log" >&2
    exit 1
fi

docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true
IFS=, read -ra output_lengths <<< "${OUTPUT_LENGTHS}"
for output_tokens in "${output_lengths[@]}"; do
    python3 "${PROJECT_DIR}/benchmarks/bench_qwen35.py" \
        --url "http://127.0.0.1:${PORT}/v1/completions" \
        --input-tokens 2200 \
        --output-tokens "${output_tokens}" \
        --warmup 2 \
        --latency-runs "${LATENCY_RUNS}" \
        --concurrency "${CONCURRENCIES}" \
        --requests-per-concurrency "${REQUESTS_PER_CONCURRENCY}" \
        --out "${RESULT_DIR}/benchmark-${output_tokens}.json"
done
docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true

grep -E "NvFp4LinearBackend|attention backend|FlashAttention|FlashInfer" \
    "${RESULT_DIR}/server.log" | tail -30 || true
