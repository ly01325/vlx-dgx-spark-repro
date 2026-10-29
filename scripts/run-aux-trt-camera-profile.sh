#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
MODEL_PATH="${PROJECT_DIR}/resources/${MODEL_NAME}"
ARTIFACT_ROOT="${AUX_TRT_ARTIFACT_ROOT:-${PROJECT_DIR}/.trt-aux-artifacts}"
RESULT_DIR="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/aux-trt-camera-profile}"
CONTAINER="vlx-aux-trt-camera-$$"

mkdir -p "${RESULT_DIR}" "${ARTIFACT_ROOT}"
cleanup() { docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run --name "${CONTAINER}" --gpus all --network host \
    -v "${PROJECT_DIR}:/workspace/to_nvidia" \
    -v "${RESULT_DIR}:/workspace/results" \
    -v /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm120.so.10.16.1:/usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm120.so.10.16.1:ro \
    -v /usr/lib/python3.12/dist-packages/tensorrt:/opt/trt-python/tensorrt:ro \
    -e PYTHONPATH=/opt/trt-python \
    --entrypoint bash "${IMAGE}" -lc '
set -euo pipefail
cp -a /workspace/to_nvidia/vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
artifact_root=/workspace/to_nvidia/.trt-aux-artifacts
onnx_path="${artifact_root}/c_radio_blocks_dynamic.onnx"
engine="${artifact_root}/camera_480p_720p_1080p.bf16.plan"
[[ -f "${onnx_path}" ]] || { echo "Missing ${onnx_path}" >&2; exit 2; }

/usr/bin/trtexec \
    --onnx="${onnx_path}" --saveEngine="${engine}" --bf16 \
    --minShapes=tokens:1x1600x1152 \
    --optShapes=tokens:1x3610x1152 \
    --maxShapes=tokens:1x8170x1152 \
    --inputIOFormats=bf16:chw \
    --outputIOFormats=bf16:chw,bf16:chw,bf16:chw,bf16:chw \
    --builderOptimizationLevel=4 --memPoolSize=workspace:24576 \
    --timingCacheFile="${artifact_root}/timing.cache" \
    --skipInference --verbose > /workspace/results/build.log 2>&1

for length in 1600 3610 8170; do
    /usr/bin/trtexec --loadEngine="${engine}" \
        --shapes="tokens:1x${length}x1152" \
        --warmUp=1000 --duration=5 --iterations=20 \
        --useCudaGraph --noDataTransfers \
        --exportTimes="/workspace/results/times-L${length}.json" \
        > "/workspace/results/bench-L${length}.log" 2>&1
done

python3 /workspace/to_nvidia/trt_aux/validate_aux_trt.py \
    --model-path "/workspace/to_nvidia/resources/'"${MODEL_NAME}"'" \
    --artifact-root "${artifact_root}" \
    --engine camera_480p_720p_1080p.bf16.plan \
    --lengths 1600,3610,8170 \
    --output /workspace/results/validation.json

stat -c "%n %s" "${engine}" > /workspace/results/artifact-size.txt
'
