#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

sync_requested=0
if [[ "${1:-}" == "--sync" ]]; then
    sync_requested=1
    shift
fi

benchmark_args=""
if (( $# )); then
    printf -v benchmark_args '%q ' "$@"
fi
benchmark_command="python3 benchmarks/bench_latency_throughput.py ${benchmark_args}--out \"\${SPARK_JOB_DIR}/benchmark.json\""

if (( sync_requested )); then
    exec "${SCRIPT_DIR}/spark-submit" --sync \
        --name benchmark --shell "${benchmark_command}"
fi

exec "${SCRIPT_DIR}/spark-submit" \
    --name benchmark --shell "${benchmark_command}"
