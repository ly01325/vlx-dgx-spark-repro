#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SPARK_PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${SPARK_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
HUMMING_ARCHIVE="${HUMMING_ARCHIVE:-/workspace/to_nvidia/third_party/sources/humming-a74973b.tar.gz}"
OUTPUT="${1:-${SPARK_JOB_DIR:-$PROJECT_DIR}/humming_nvfp4.json}"
OUTPUT_CONTAINER="${OUTPUT/#$PROJECT_DIR/\/workspace\/to_nvidia}"

docker run --rm \
  --gpus all \
  --network host \
  -v "$PROJECT_DIR:/workspace/to_nvidia" \
  -w /workspace/to_nvidia \
  --entrypoint bash \
  "$IMAGE" \
  -lc "
    set -euo pipefail
    python3 -m pip install --disable-pip-version-check --no-deps --no-build-isolation \
      --target /tmp/humming-site \
      '$HUMMING_ARCHIVE'
    PYTHONPATH=/tmp/humming-site python3 benchmarks/bench_humming_nvfp4.py \
      --output '$OUTPUT_CONTAINER' --m 1 8 27 2200 \
      --shapes 6144x2048 12288x2048 2048x6144 2048x2048
  "
