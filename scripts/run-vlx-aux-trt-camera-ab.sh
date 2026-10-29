#!/usr/bin/env bash
set -euo pipefail

# Full VLX A/B at the only three camera resolutions accepted by the TRT router.

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
RESULT_ROOT="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/vlx-aux-trt-camera-ab}"
ENGINE="${PROJECT_DIR}/.trt-aux-artifacts/camera_480p_720p_1080p.bf16.plan"
HUMMING_ARCHIVE="${PROJECT_DIR}/third_party/sources/humming-a74973b.tar.gz"
PORT=8000

mkdir -p "${RESULT_ROOT}/baseline" "${RESULT_ROOT}/tensorrt"
for required in "${HUMMING_ARCHIVE}" "${ENGINE}"; do
    [[ -f "${required}" ]] || { echo "Missing dependency: ${required}" >&2; exit 2; }
done

active_container=""
log_pid=""
cleanup() {
    [[ -z "${log_pid}" ]] || kill "${log_pid}" >/dev/null 2>&1 || true
    [[ -z "${active_container}" ]] || docker rm -f "${active_container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_variant() {
    local label="$1" enable_trt="$2"
    local result_dir="${RESULT_ROOT}/${label}"
    active_container="vlx-aux-trt-camera-${label}-$$"
    docker rm -f "${active_container}" >/dev/null 2>&1 || true

    local trt_env=()
    if [[ "${enable_trt}" == 1 ]]; then
        trt_env+=(
            -e OMCHAT_AUX_TRT_STRICT=1
            -e OMCHAT_AUX_TRT_CAMERA_ENGINE=/workspace/to_nvidia/.trt-aux-artifacts/camera_480p_720p_1080p.bf16.plan
            -e OMCHAT_AUX_TRT_CAMERA_LENGTHS=1600,3610,8170
        )
    fi

    docker run -d \
        --name "${active_container}" --gpus all --network host \
        -v "${PROJECT_DIR}:/workspace/to_nvidia" \
        -v "${result_dir}:/workspace/results" \
        -v /usr/lib/python3.12/dist-packages/tensorrt:/opt/trt-python/tensorrt:ro \
        -e PYTHONPATH=/opt/trt-python \
        -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
        -e VLLM_HUMMING_LM_HEAD_W4A16=1 \
        -e VLLM_USE_FLASHRT_OPS=0 \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e ATTENTION_BACKEND=FLASH_ATTN \
        -e GDN_PREFILL_BACKEND=flashinfer \
        -e OMCHAT_PROFILE_MODEL_PHASES=1 \
        -e OMCHAT_PROFILE_OBJECT_FEATURES=1 \
        -e "MODEL_NAME=${MODEL_NAME}" \
        -e "MODEL_PATH=/workspace/to_nvidia/resources/${MODEL_NAME}" \
        -e "PORT=${PORT}" -e HOST=0.0.0.0 \
        -e GPU_MEMORY_UTILIZATION=0.10 \
        -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=16 \
        -e MAX_NUM_BATCHED_TOKENS=4096 \
        -e NO_PREFIX_CACHE=1 -e NO_MM_CACHE=1 \
        "${trt_env[@]}" \
        --entrypoint bash "${IMAGE}" -lc '
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
    --quantization modelopt_fp4 --attention-backend FLASH_ATTN \
    --gdn-prefill-backend flashinfer
' >/dev/null

    docker logs -f "${active_container}" > "${result_dir}/server.log" 2>&1 &
    log_pid=$!
    local ready=0
    for _ in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${active_container}" 2>/dev/null || true)" != true ]]; then
            break
        fi
        sleep 1
    done
    if (( ! ready )); then
        tail -180 "${result_dir}/server.log" >&2 || true
        return 1
    fi

    docker exec "${active_container}" \
        python3 /workspace/to_nvidia/benchmarks/bench_camera_resolutions.py \
        --model "${MODEL_NAME}" \
        --image /workspace/to_nvidia/resources/demo_1920_1080.jpeg \
        --server-log /workspace/results/server.log \
        --out /workspace/results/camera_resolutions.json \
        --warmup 2 --runs 5

    kill "${log_pid}" >/dev/null 2>&1 || true
    wait "${log_pid}" 2>/dev/null || true
    log_pid=""
    docker logs "${active_container}" > "${result_dir}/server.log" 2>&1 || true
    grep -E "routed to TensorRT|routed to PyTorch" "${result_dir}/server.log" || true
    docker rm -f "${active_container}" >/dev/null 2>&1 || true
    active_container=""
}

run_variant baseline 0
run_variant tensorrt 1

python3 - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "baseline/camera_resolutions.json").read_text())
trt = json.loads((root / "tensorrt/camera_resolutions.json").read_text())

def reduction(a, b):
    return (a - b) / a * 100

rows = []
for b, t in zip(base["results"], trt["results"]):
    row = {
        "input_resolution": b["input_resolution"],
        "prepared_resolution": b["prepared_resolution"],
        "token_length": b["token_length"],
    }
    for key in ("aux_vision_tower_ms", "multimodal_embed_total_ms", "e2e_ms"):
        before = b[key]["mean"]
        after = t[key]["mean"]
        row[key] = {
            "baseline": before,
            "tensorrt": after,
            "reduction_pct": reduction(before, after),
        }
    rows.append(row)
report = {"runs_per_resolution": base["runs"], "results": rows}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
