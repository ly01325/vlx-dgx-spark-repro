#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_NAME="vlx-dgx-spark-repro"
OUTPUT="${PROJECT_ROOT}/deliverables/${PACKAGE_NAME}-$(date +%Y%m%d).tar.gz"
include_wheels=0
include_raw_results=0

usage() {
    cat <<'EOF'
Usage: scripts/build-repro-package.sh [options]

Options:
  --output FILE          Output .tar.gz path.
  --include-wheel-cache  Include downloaded third_party wheel/source caches.
  --include-raw-results  Include .spark-results and artifacts (large).
  -h, --help             Show this help.

Model weight formats, TensorRT engines, ONNX exports and local secrets are
always excluded. By default the package downloads third-party dependencies.
EOF
}

while (( $# )); do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || { echo "--output needs a value" >&2; exit 2; }
            OUTPUT="$2"; shift
            ;;
        --include-wheel-cache) include_wheels=1 ;;
        --include-raw-results) include_raw_results=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

bash "${SCRIPT_DIR}/check-dgx-spark-env.sh" --source-only

mkdir -p "$(dirname -- "${OUTPUT}")"
stage_root="$(mktemp -d)"
cleanup() { rm -rf "${stage_root}"; }
trap cleanup EXIT
package_root="${stage_root}/${PACKAGE_NAME}"
mkdir -p "${package_root}"

items=(
    README.md
    .gitignore
    .sparkignore
    .spark.env.example
    requirements
    benchmarks
    scripts
    raw_vllm_server
    vllm_
    trt_aux
    quant
    docs
    results
    resources
    deliverables
    third_party/wheels/README.md
    third_party/wheels/SHA256SUMS
    third_party/sources/README.md
    third_party/sources/SHA256SUMS
)

if (( include_raw_results )); then
    items+=(.spark-results artifacts)
fi
if (( include_wheels )); then
    items+=(third_party/wheels third_party/sources)
fi

tar -C "${PROJECT_ROOT}" -cf - \
    --exclude='.DS_Store' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.spark.env' \
    --exclude='.spark-jobs' \
    --exclude='.codex-build' \
    --exclude='~$*' \
    --exclude='vlx-dgx-spark-repro-*.tar.gz' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.engine' \
    --exclude='*.plan' \
    --exclude='*.onnx' \
    --exclude='*.timing' \
    "${items[@]}" | tar -C "${package_root}" -xf -

if (( ! include_wheels )); then
    find "${package_root}/third_party" -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) -delete
fi

cat > "${package_root}/PACKAGE_INFO.txt" <<EOF
Package: ${PACKAGE_NAME}
Created UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Target: NVIDIA DGX Spark / Linux aarch64 / Python 3.12 / CUDA 13 / SM121
Weights included: no
Third-party cache included: $([[ ${include_wheels} -eq 1 ]] && echo yes || echo no)
Raw benchmark results included: $([[ ${include_raw_results} -eq 1 ]] && echo yes || echo no)

Start with docs/reproduction/PORTABLE_PACKAGE.md.
EOF

(
    cd "${package_root}"
    find . -type f ! -name PACKAGE_MANIFEST.sha256 -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 > PACKAGE_MANIFEST.sha256
)

COPYFILE_DISABLE=1 tar -C "${stage_root}" -czf "${OUTPUT}" "${PACKAGE_NAME}"
archive_size="$(du -h "${OUTPUT}" | awk '{print $1}')"
archive_hash="$(shasum -a 256 "${OUTPUT}" | awk '{print $1}')"
printf 'Created: %s\nSize: %s\nSHA256: %s\n' "${OUTPUT}" "${archive_size}" "${archive_hash}"
