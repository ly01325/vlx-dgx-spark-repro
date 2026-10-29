#!/usr/bin/env bash
set -euo pipefail

# Full-stack deployment A/B for the exact Qwen3.5 2200-token prefill and
# 27-token decode workload. All concurrency shapes are warmed before timing.

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
RESULT_ROOT="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/qwen35-deployment-ab}"
BASE_PORT="${BASE_PORT:-8113}"
OPT_PORT="${OPT_PORT:-8114}"
REQUESTS_PER_CONCURRENCY="${REQUESTS_PER_CONCURRENCY:-16}"
LATENCY_RUNS="${LATENCY_RUNS:-12}"
OUTPUT_LENGTHS="${OUTPUT_LENGTHS:-27}"
HUMMING_ARCHIVE="${PROJECT_DIR}/third_party/sources/humming-a74973b.tar.gz"

mkdir -p "${RESULT_ROOT}/baseline" "${RESULT_ROOT}/optimized"

for required in \
    "${HUMMING_ARCHIVE}" \
    "${PROJECT_DIR}/third_party/wheels/flashinfer_python-0.6.18-py3-none-any.whl" \
    "${PROJECT_DIR}/third_party/wheels/apache_tvm_ffi-0.1.13.post3-"*.whl \
    "${PROJECT_DIR}/third_party/wheels/nvidia_cuda_nvdisasm-13.4.49-"*.whl \
    "${PROJECT_DIR}/third_party/wheels/nvidia_cutlass_dsl-4.7.1-"*.whl \
    "${PROJECT_DIR}/third_party/wheels/nvidia_cutlass_dsl_libs_core-4.7.1-"*.whl \
    "${PROJECT_DIR}/third_party/wheels/nvidia_cutlass_dsl_libs_base-4.7.1-"*.whl \
    "${PROJECT_DIR}/third_party/wheels/nvidia_cutlass_dsl_libs_cu13-4.7.1-"*.whl; do
    [[ -f "${required}" ]] || { echo "Missing dependency: ${required}" >&2; exit 2; }
done

active_container=""
cleanup() {
    if [[ -n "${active_container}" ]]; then
        docker rm -f "${active_container}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

run_variant() {
    local label="$1"
    local port="$2"
    local gdn_backend="$3"
    local lm_head_w4a16="$4"
    local result_dir="${RESULT_ROOT}/${label}"
    active_container="vlx-qwen35-ab-${label}-${port}"
    docker rm -f "${active_container}" >/dev/null 2>&1 || true

    printf 'Starting %s: GDN=%s LM_HEAD_W4A16=%s\n' \
        "${label}" "${gdn_backend}" "${lm_head_w4a16}"
    docker run -d \
        --name "${active_container}" \
        --gpus all \
        --network host \
        -v "${PROJECT_DIR}:/workspace/to_nvidia" \
        -v "${result_dir}:/workspace/results" \
        -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
        -e "VLLM_HUMMING_LM_HEAD_W4A16=${lm_head_w4a16}" \
        -e VLLM_USE_FLASHRT_OPS=0 \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e ATTENTION_BACKEND=FLASH_ATTN \
        -e "GDN_PREFILL_BACKEND=${gdn_backend}" \
        -e "MODEL_NAME=${MODEL_NAME}" \
        -e "MODEL_PATH=/workspace/to_nvidia/resources/${MODEL_NAME}" \
        -e "PORT=${port}" \
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
            cp -a /workspace/to_nvidia/vllm_/. \
                /usr/local/lib/python3.12/dist-packages/vllm/
            python3 - <<"PY" > /workspace/results/versions.txt
import importlib.metadata as md
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name())
print("capability", torch.cuda.get_device_capability())
for package in ("flashinfer-python", "nvidia-cutlass-dsl", "humming-kernels"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "not-installed")
PY
            cd /workspace/to_nvidia
            exec bash raw_vllm_server/start_vllm_server.sh \
                --quantization modelopt_fp4 \
                --attention-backend FLASH_ATTN \
                --gdn-prefill-backend "${GDN_PREFILL_BACKEND}"
        ' >/dev/null

    local ready=0
    for _ in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${active_container}" 2>/dev/null || true)" != true ]]; then
            break
        fi
        sleep 1
    done
    docker logs "${active_container}" > "${result_dir}/server.log" 2>&1 || true
    if (( ! ready )); then
        echo "${label} server failed to become healthy" >&2
        tail -160 "${result_dir}/server.log" >&2 || true
        return 1
    fi

    IFS=, read -ra output_lengths <<< "${OUTPUT_LENGTHS}"
    for output_tokens in "${output_lengths[@]}"; do
        # Compile/autotune all packed-batch shapes. Results are discarded.
        python3 "${PROJECT_DIR}/benchmarks/bench_qwen35.py" \
            --url "http://127.0.0.1:${port}/v1/completions" \
            --input-tokens 2200 --output-tokens "${output_tokens}" \
            --warmup 2 --latency-runs 2 \
            --concurrency 1,2,4,8 --requests-per-concurrency 8 \
            --out "${result_dir}/shape-warmup-${output_tokens}.json"

        python3 "${PROJECT_DIR}/benchmarks/bench_qwen35.py" \
            --url "http://127.0.0.1:${port}/v1/completions" \
            --input-tokens 2200 --output-tokens "${output_tokens}" \
            --warmup 2 --latency-runs "${LATENCY_RUNS}" \
            --concurrency 1,2,4,8 \
            --requests-per-concurrency "${REQUESTS_PER_CONCURRENCY}" \
            --out "${result_dir}/benchmark-${output_tokens}.json"
    done

    docker logs "${active_container}" > "${result_dir}/server.log" 2>&1 || true
    grep -E "Using (FlashInfer|Triton/FLA) GDN|Humming|NvFp4LinearBackend|FlashAttention" \
        "${result_dir}/server.log" | tail -100 || true
    docker rm -f "${active_container}" >/dev/null 2>&1 || true
    active_container=""
}

run_variant baseline "${BASE_PORT}" triton 0
run_variant optimized "${OPT_PORT}" flashinfer 1

IFS=, read -ra output_lengths <<< "${OUTPUT_LENGTHS}"
for output_tokens in "${output_lengths[@]}"; do
    python3 "${PROJECT_DIR}/benchmarks/compare_qwen35_benchmarks.py" \
        --baseline "${RESULT_ROOT}/baseline/benchmark-${output_tokens}.json" \
        --optimized "${RESULT_ROOT}/optimized/benchmark-${output_tokens}.json" \
        --out-json "${RESULT_ROOT}/comparison-${output_tokens}.json" \
        --out-md "${RESULT_ROOT}/comparison-${output_tokens}.md"
done

python3 - "${RESULT_ROOT}" "${OUTPUT_LENGTHS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lengths = [int(v) for v in sys.argv[2].split(",")]
reports = [json.loads((root / f"comparison-{length}.json").read_text())
           for length in lengths]
combined = {
    "input_tokens": 2200,
    "output_lengths": lengths,
    "comparisons": {str(length): report for length, report in zip(lengths, reports)},
}
(root / "comparison.json").write_text(json.dumps(combined, indent=2) + "\n")
print(json.dumps(combined, indent=2))
PY
