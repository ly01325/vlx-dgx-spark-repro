#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SPARK_PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${SPARK_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
HUMMING_ARCHIVE="${HUMMING_ARCHIVE:-/workspace/to_nvidia/third_party/sources/humming-a74973b.tar.gz}"
OUTPUT="${1:-${SPARK_JOB_DIR:-$PROJECT_DIR}/lm_head_precision.json}"
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
    PYTHONPATH=/tmp/humming-site python3 benchmarks/bench_lm_head_precision.py \
      --output '$OUTPUT_CONTAINER' --m 1 8 27 --n 248182 --k 2048 \
      --warmup 20 --rep 100
  "
