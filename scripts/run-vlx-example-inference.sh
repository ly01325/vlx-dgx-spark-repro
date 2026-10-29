#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
RESULT_DIR="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/vlx-example-inference}"
ENGINE="${PROJECT_DIR}/.trt-aux-artifacts/camera_480p_720p_1080p.bf16.plan"
DEMO_IMAGE="${PROJECT_DIR}/resources/demo_1920_1080.jpeg"
CONTAINER="vlx-example-inference-$$"
PORT=8000
MAX_TOKENS="${INFERENCE_MAX_TOKENS:-512}"

mkdir -p "${RESULT_DIR}"
for required in "${ENGINE}" "${DEMO_IMAGE}" "${PROJECT_DIR}/third_party/sources/humming-a74973b.tar.gz"; do
    [[ -f "${required}" ]] || { echo "Missing dependency: ${required}" >&2; exit 2; }
done
cp "${DEMO_IMAGE}" "${RESULT_DIR}/demo_1920_1080.jpeg"

cleanup() { docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d \
    --name "${CONTAINER}" --gpus all --network host \
    -v "${PROJECT_DIR}:/workspace/to_nvidia" \
    -v "${RESULT_DIR}:/workspace/results" \
    -v /usr/lib/python3.12/dist-packages/tensorrt:/opt/trt-python/tensorrt:ro \
    -e PYTHONPATH=/opt/trt-python \
    -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
    -e VLLM_HUMMING_LM_HEAD_W4A16=1 \
    -e VLLM_USE_FLASHRT_OPS=0 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e ATTENTION_BACKEND=FLASH_ATTN \
    -e GDN_PREFILL_BACKEND=flashinfer \
    -e OMCHAT_AUX_TRT_STRICT=1 \
    -e OMCHAT_AUX_TRT_CAMERA_ENGINE=/workspace/to_nvidia/.trt-aux-artifacts/camera_480p_720p_1080p.bf16.plan \
    -e OMCHAT_AUX_TRT_CAMERA_LENGTHS=1600,8170 \
    -e "MODEL_NAME=${MODEL_NAME}" \
    -e "MODEL_PATH=/workspace/to_nvidia/resources/${MODEL_NAME}" \
    -e "PORT=${PORT}" -e HOST=0.0.0.0 \
    -e GPU_MEMORY_UTILIZATION=0.10 \
    -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=16 \
    -e MAX_NUM_BATCHED_TOKENS=4096 \
    -e NO_PREFIX_CACHE=1 -e NO_MM_CACHE=1 \
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

ready=0
for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
        break
    fi
    sleep 1
done
if (( ! ready )); then
    docker logs "${CONTAINER}" | tail -200 >&2
    exit 1
fi

docker exec "${CONTAINER}" python3 /workspace/to_nvidia/raw_vllm_server/vllm_request.py \
    --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --model "${MODEL_NAME}" \
    --image /workspace/to_nvidia/resources/demo_1920_1080.jpeg \
    --max-tokens "${MAX_TOKENS}" --temperature 0 --raw \
    > "${RESULT_DIR}/response.json"

docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1
grep -E "routed to TensorRT|Using FlashInfer GDN|Humming NVFP4 W4A16|FlashAttention" \
    "${RESULT_DIR}/server.log" | tail -40 || true
cat "${RESULT_DIR}/response.json"
