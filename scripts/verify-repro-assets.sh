#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
require_humming=0

if [[ "${1:-}" == "--require-humming" ]]; then
    require_humming=1
    shift
fi
(( $# == 0 )) || { echo "Usage: scripts/verify-repro-assets.sh [--require-humming]" >&2; exit 2; }

cd "${PROJECT_ROOT}"

required=(
    README.md
    benchmarks/bench_qwen35.py
    benchmarks/bench_latency_throughput.py
    benchmarks/bench_module_latency.py
    scripts/spark-submit
    scripts/run-gdn-sm121-validation.sh
    scripts/run-vlx-full-stack-ab.sh
    raw_vllm_server/start_vllm_server.sh
    third_party/wheels/SHA256SUMS
    docs/reproduction/DGX_SPARK_REPRODUCTION.md
)

failed=0
for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "MISSING  ${path}" >&2
        failed=1
    fi
done

if command -v sha256sum >/dev/null; then
    sha256sum -c third_party/wheels/SHA256SUMS || failed=1
elif command -v shasum >/dev/null; then
    while read -r expected file; do
        actual="$(shasum -a 256 "${file}" | awk '{print $1}')"
        if [[ "${actual}" == "${expected}" ]]; then
            echo "${file}: OK"
        else
            echo "${file}: FAILED" >&2
            failed=1
        fi
    done < third_party/wheels/SHA256SUMS
else
    echo "Neither sha256sum nor shasum is available" >&2
    failed=1
fi

if (( require_humming )); then
    archive="third_party/sources/humming-a74973b.tar.gz"
    if [[ ! -f "${archive}" ]]; then
        echo "MISSING  ${archive}" >&2
        failed=1
    elif command -v sha256sum >/dev/null; then
        sha256sum -c third_party/sources/SHA256SUMS || failed=1
    elif command -v shasum >/dev/null; then
        expected="$(awk '{print $1}' third_party/sources/SHA256SUMS)"
        actual="$(shasum -a 256 "${archive}" | awk '{print $1}')"
        [[ "${actual}" == "${expected}" ]] || {
            echo "${archive}: FAILED" >&2
            failed=1
        }
    fi
fi

if (( failed )); then
    echo "Reproduction asset verification failed." >&2
    exit 1
fi

echo "Reproduction assets verified."
