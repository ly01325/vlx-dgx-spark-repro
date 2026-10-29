#!/usr/bin/env bash
set -euo pipefail

# Validate the FlashInfer SM120-family GDN prefill backend on DGX Spark (SM121)
# in a disposable container. The production server on port 8000 is untouched.

PORT="${1:-8102}"
GDN_BACKEND="${2:-flashinfer}"
PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
CONTAINER="vlx-gdn-sm121-${GDN_BACKEND}-${PORT}"
RESULT_DIR="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/gdn-sm121-${PORT}}"
FLASHINFER_WHEEL="/workspace/to_nvidia/third_party/wheels/flashinfer_python-0.6.18-py3-none-any.whl"

mkdir -p "${RESULT_DIR}"
cleanup() { docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --network host \
    -v "${PROJECT_DIR}:/workspace/to_nvidia" \
    -v "${RESULT_DIR}:/workspace/results" \
    -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
    -e VLLM_HUMMING_LM_HEAD_W4A16=0 \
    -e VLLM_USE_FLASHRT_OPS=0 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e ATTENTION_BACKEND=FLASH_ATTN \
    -e "GDN_PREFILL_BACKEND=${GDN_BACKEND}" \
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
        python3 -m pip install --no-deps --force-reinstall \
            "'"${FLASHINFER_WHEEL}"'"
        cp -a /workspace/to_nvidia/vllm_/. \
            /usr/local/lib/python3.12/dist-packages/vllm/
        python3 - <<"PY" > /workspace/results/versions.txt
import importlib.metadata as md
import torch
import flashinfer
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name())
print("capability", torch.cuda.get_device_capability())
print("flashinfer", flashinfer.__version__)
for package in ("nvidia-cutlass-dsl", "cuda-tile", "cuda-python"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "not-installed")
PY
        cd /workspace/to_nvidia
        python3 benchmarks/bench_gdn_prefill.py \
            --warmup 20 --repetitions 100 \
            --output /workspace/results/gdn_microbench.json \
            > /workspace/results/gdn_microbench.log 2>&1
        exec bash raw_vllm_server/start_vllm_server.sh \
            --quantization modelopt_fp4 \
            --attention-backend FLASH_ATTN \
            --gdn-prefill-backend "${GDN_PREFILL_BACKEND}"
    ' >/dev/null

ready=0
for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
        break
    fi
    sleep 1
done

docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true
if (( ! ready )); then
    echo "FlashInfer SM121 GDN candidate failed to become healthy" >&2
    tail -160 "${RESULT_DIR}/gdn_microbench.log" >&2 || true
    tail -160 "${RESULT_DIR}/server.log" >&2 || true
    exit 1
fi

python3 "${PROJECT_DIR}/benchmarks/bench_qwen35.py" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --input-tokens 2200 --output-tokens 27 \
    --warmup 3 --latency-runs 10 \
    --concurrency 1,8 --requests-per-concurrency 10 \
    --out "${RESULT_DIR}/benchmark.json"

# Repeat the high-concurrency shape after the first pass has compiled and
# autotuned every kernel variant. This separates steady state from JIT cost.
python3 "${PROJECT_DIR}/benchmarks/bench_qwen35.py" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --input-tokens 2200 --output-tokens 27 \
    --warmup 0 --latency-runs 3 \
    --concurrency 8 --requests-per-concurrency 10 \
    --out "${RESULT_DIR}/benchmark-repeat.json"

if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    python3 "${PROJECT_DIR}/benchmarks/compare_qwen35_accuracy.py" \
        --baseline-url http://127.0.0.1:8000/v1/completions \
        --candidate-url "http://127.0.0.1:${PORT}/v1/completions" \
        --model "${MODEL_NAME}" --max-tokens 27 \
        --out "${RESULT_DIR}/accuracy.json"
else
    printf '%s\n' "Skipped: baseline server on port 8000 is unavailable." \
        > "${RESULT_DIR}/accuracy-skipped.txt"
fi

docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1 || true
grep -E "GDN prefill|FlashInfer|NvFp4LinearBackend|FlashAttention" \
    "${RESULT_DIR}/server.log" | tail -80 || true
