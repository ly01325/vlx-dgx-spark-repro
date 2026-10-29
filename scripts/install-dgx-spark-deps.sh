#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels"
SOURCE_DIR="${PROJECT_ROOT}/third_party/sources"

install_flashinfer=0
install_humming=0

usage() {
    printf '%s\n' \
        "Usage: scripts/install-dgx-spark-deps.sh [--flashinfer] [--humming] [--all]" \
        "Run this inside the DGX Spark Python 3.12 container."
}

while (( $# )); do
    case "$1" in
        --flashinfer) install_flashinfer=1 ;;
        --humming) install_humming=1 ;;
        --all) install_flashinfer=1; install_humming=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

if (( ! install_flashinfer && ! install_humming )); then
    usage >&2
    exit 2
fi

if (( install_flashinfer )); then
    command -v sha256sum >/dev/null || {
        echo "sha256sum is required inside the container" >&2
        exit 2
    }
    (cd "${PROJECT_ROOT}" && sha256sum -c third_party/wheels/SHA256SUMS)

    PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
        "${WHEEL_DIR}"/apache_tvm_ffi-0.1.13.post3-*.whl
    PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
        "${WHEEL_DIR}"/nvidia_cuda_nvdisasm-13.4.49-*.whl \
        "${WHEEL_DIR}"/nvidia_cutlass_dsl_libs_core-4.7.1-*.whl \
        "${WHEEL_DIR}"/nvidia_cutlass_dsl_libs_base-4.7.1-*.whl \
        "${WHEEL_DIR}"/nvidia_cutlass_dsl_libs_cu13-4.7.1-*.whl \
        "${WHEEL_DIR}"/nvidia_cutlass_dsl-4.7.1-*.whl
    PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
        "${WHEEL_DIR}"/flashinfer_python-0.6.18-py3-none-any.whl
fi

if (( install_humming )); then
    archive="${HUMMING_ARCHIVE:-${SOURCE_DIR}/humming-a74973b.tar.gz}"
    [[ -f "${archive}" ]] || {
        echo "Missing Humming archive: ${archive}" >&2
        exit 2
    }
    PIP_CONSTRAINT= python3 -m pip install --no-deps --no-build-isolation \
        "${archive}"
fi

python3 - <<'PY'
import importlib.util
for name in ("flashinfer", "tvm_ffi", "cutlass"):
    print(f"{name}: {'available' if importlib.util.find_spec(name) else 'not installed'}")
PY
