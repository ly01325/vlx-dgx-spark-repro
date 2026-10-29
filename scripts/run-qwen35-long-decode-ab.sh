#!/usr/bin/env bash
set -euo pipefail
export OUTPUT_LENGTHS=256,512
export REQUESTS_PER_CONCURRENCY="${REQUESTS_PER_CONCURRENCY:-8}"
export LATENCY_RUNS="${LATENCY_RUNS:-8}"
exec "$(dirname "$0")/run-qwen35-deployment-ab.sh"
