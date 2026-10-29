#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
MODEL_PATH="${PROJECT_DIR}/resources/${MODEL_NAME}"
ARTIFACT_ROOT="${AUX_TRT_ARTIFACT_ROOT:-${PROJECT_DIR}/.trt-aux-artifacts}"
RESULT_DIR="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/aux-trt-poc}"
CONTAINER="vlx-aux-trt-poc-$$"

mkdir -p "${ARTIFACT_ROOT}" "${RESULT_DIR}"
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
mkdir -p /workspace/to_nvidia/.pip-cache
PIP_CONSTRAINT= python3 -m pip install \
    --cache-dir /workspace/to_nvidia/.pip-cache \
    "onnx==1.19.1" "ml_dtypes==0.5.3"

model_path="/workspace/to_nvidia/resources/'"${MODEL_NAME}"'"
artifact_root="/workspace/to_nvidia/.trt-aux-artifacts"
onnx_path="${artifact_root}/c_radio_blocks_dynamic.onnx"
mkdir -p "${artifact_root}"

if [[ ! -f "${onnx_path}" ]]; then
    python3 /workspace/to_nvidia/trt_aux/export_aux_blocks.py \
        --model-path "${model_path}" \
        --output "${onnx_path}" \
        --metadata /workspace/results/export-metadata.json
else
    cp "${artifact_root}/c_radio_blocks_dynamic.metadata.json" \
        /workspace/results/export-metadata.json 2>/dev/null || true
fi

build_engine() {
    local name="$1" min_shape="$2" opt_shape="$3" max_shape="$4"
    local engine="${artifact_root}/${name}.bf16.plan"
    /usr/bin/trtexec \
        --onnx="${onnx_path}" \
        --saveEngine="${engine}" \
        --bf16 \
        --minShapes="tokens:${min_shape}" \
        --optShapes="tokens:${opt_shape}" \
        --maxShapes="tokens:${max_shape}" \
        --builderOptimizationLevel=4 \
        --inputIOFormats=bf16:chw \
        --outputIOFormats=bf16:chw,bf16:chw,bf16:chw,bf16:chw \
        --memPoolSize=workspace:24576 \
        --timingCacheFile="${artifact_root}/timing.cache" \
        --skipInference \
        --verbose \
        > "/workspace/results/build-${name}.log" 2>&1
}

# Start with the useful dynamic range that covers common 256p/720p cameras.
# The dedicated 1080p profile is built separately so wide-range tactic choices
# cannot slow down the production shape.
build_engine small_dynamic 1x266x1152 1x3610x1152 1x8169x1152
build_engine main_1920x1088 1x8170x1152 1x8170x1152 1x8170x1152
build_engine universal_dynamic 1x266x1152 1x8170x1152 1x16394x1152

benchmark_engine() {
    local engine="$1" name="$2" length="$3"
    /usr/bin/trtexec \
        --loadEngine="${engine}" \
        --shapes="tokens:1x${length}x1152" \
        --warmUp=1000 --duration=5 --iterations=20 \
        --useCudaGraph --noDataTransfers \
        --exportTimes="/workspace/results/times-${name}-L${length}.json" \
        --exportLayerInfo="/workspace/results/layers-${name}.json" \
        > "/workspace/results/bench-${name}-L${length}.log" 2>&1
}

benchmark_engine "${artifact_root}/small_dynamic.bf16.plan" small_dynamic 266
benchmark_engine "${artifact_root}/small_dynamic.bf16.plan" small_dynamic 3610
benchmark_engine "${artifact_root}/small_dynamic.bf16.plan" small_dynamic 8169
benchmark_engine "${artifact_root}/main_1920x1088.bf16.plan" main_1920x1088 8170
benchmark_engine "${artifact_root}/universal_dynamic.bf16.plan" universal_dynamic 266
benchmark_engine "${artifact_root}/universal_dynamic.bf16.plan" universal_dynamic 3610
benchmark_engine "${artifact_root}/universal_dynamic.bf16.plan" universal_dynamic 8170
benchmark_engine "${artifact_root}/universal_dynamic.bf16.plan" universal_dynamic 12298
benchmark_engine "${artifact_root}/universal_dynamic.bf16.plan" universal_dynamic 16394

python3 /workspace/to_nvidia/trt_aux/validate_aux_trt.py \
    --model-path "${model_path}" \
    --artifact-root "${artifact_root}" \
    --engine universal_dynamic.bf16.plan \
    --lengths 266,3610,8170,12298,16394 \
    --output /workspace/results/validation.json

python3 - <<PY > /workspace/results/artifacts.json
import json
from pathlib import Path
root = Path("${artifact_root}")
print(json.dumps({p.name: p.stat().st_size for p in root.iterdir()}, indent=2))
PY
'
