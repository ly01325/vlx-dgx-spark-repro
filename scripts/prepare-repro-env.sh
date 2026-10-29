#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels"
SOURCE_DIR="${PROJECT_ROOT}/third_party/sources"
REQUIREMENTS="${PROJECT_ROOT}/requirements/dgx-spark-kernels.txt"
HUMMING_COMMIT="a74973b"
HUMMING_ARCHIVE="${SOURCE_DIR}/humming-${HUMMING_COMMIT}.tar.gz"
HUMMING_URL="https://github.com/inclusionAI/humming/archive/${HUMMING_COMMIT}.tar.gz"

download_only=0
skip_humming=0
force_download=0
offline=0

usage() {
    cat <<'EOF'
Usage: scripts/prepare-repro-env.sh [options]

Download the pinned DGX Spark dependencies, cache them under third_party/, and
install them into the active Python environment.

Options:
  --download-only  Populate the cache without installing packages.
  --offline        Install only from the included cache; never access network.
  --skip-humming   Do not download or install Humming.
  --force-download Refresh files that already exist in the cache.
  -h, --help       Show this help.

Run this inside the VLX DGX Spark container (Linux aarch64, Python 3.12).
EOF
}

while (( $# )); do
    case "$1" in
        --download-only) download_only=1 ;;
        --offline) offline=1 ;;
        --skip-humming) skip_humming=1 ;;
        --force-download) force_download=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if (( offline && download_only )); then
    echo "--offline and --download-only cannot be used together" >&2
    exit 2
fi

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }
[[ -f "${REQUIREMENTS}" ]] || { echo "Missing ${REQUIREMENTS}" >&2; exit 2; }

machine="$(uname -m)"
py_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$(uname -s)" != "Linux" || "${machine}" != "aarch64" || "${py_version}" != "3.12" ]]; then
    echo "WARNING: validated target is Linux/aarch64/Python 3.12; detected $(uname -s)/${machine}/Python ${py_version}." >&2
fi

mkdir -p "${WHEEL_DIR}" "${SOURCE_DIR}"

if (( force_download )); then
    find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -delete
    [[ ! -f "${HUMMING_ARCHIVE}" ]] || rm -f "${HUMMING_ARCHIVE}"
fi

if (( offline )); then
    echo "Using the included dependency cache."
    if command -v sha256sum >/dev/null; then
        (cd "${PROJECT_ROOT}" && sha256sum -c third_party/wheels/SHA256SUMS)
    else
        (cd "${PROJECT_ROOT}" && shasum -a 256 -c third_party/wheels/SHA256SUMS)
    fi
else
    echo "Downloading pinned wheels to ${WHEEL_DIR}"
    python3 -m pip download \
        --disable-pip-version-check \
        --no-deps \
        --only-binary=:all: \
        --dest "${WHEEL_DIR}" \
        -r "${REQUIREMENTS}"
fi

if (( offline && ! skip_humming )) && [[ ! -f "${HUMMING_ARCHIVE}" ]]; then
    echo "Offline cache is missing ${HUMMING_ARCHIVE}" >&2
    exit 2
elif (( ! offline && ! skip_humming )) && [[ ! -f "${HUMMING_ARCHIVE}" ]]; then
    echo "Downloading Humming ${HUMMING_COMMIT}"
    if command -v curl >/dev/null; then
        curl --fail --location --retry 3 --output "${HUMMING_ARCHIVE}" "${HUMMING_URL}"
    elif command -v wget >/dev/null; then
        wget --output-document="${HUMMING_ARCHIVE}" "${HUMMING_URL}"
    else
        echo "curl or wget is required to download Humming" >&2
        exit 2
    fi
fi

if (( ! skip_humming )); then
    expected_humming_sha="7b804904c7644d45cdeff77ea578eaddc245cf1802d2d1e217702f4feb11cfc8"
    if command -v sha256sum >/dev/null; then
        actual_humming_sha="$(sha256sum "${HUMMING_ARCHIVE}" | awk '{print $1}')"
    else
        actual_humming_sha="$(shasum -a 256 "${HUMMING_ARCHIVE}" | awk '{print $1}')"
    fi
    [[ "${actual_humming_sha}" == "${expected_humming_sha}" ]] || {
        echo "Humming archive checksum mismatch" >&2
        exit 1
    }
    echo "Humming archive checksum verified."
fi

if (( download_only )); then
    echo "Dependency cache is ready. No packages were installed."
    exit 0
fi

install_args=(--disable-pip-version-check --no-deps --force-reinstall)
python3 -m pip install "${install_args[@]}" \
    "${WHEEL_DIR}"/apache_tvm_ffi-0.1.13.post3-*.whl \
    "${WHEEL_DIR}"/nvidia_cuda_nvdisasm-13.4.49-*.whl \
    "${WHEEL_DIR}"/nvidia_cutlass_dsl_libs_core-4.7.1-*.whl \
    "${WHEEL_DIR}"/nvidia_cutlass_dsl_libs_base-4.7.1-*.whl \
    "${WHEEL_DIR}"/nvidia_cutlass_dsl_libs_cu13-4.7.1-*.whl \
    "${WHEEL_DIR}"/nvidia_cutlass_dsl-4.7.1-*.whl \
    "${WHEEL_DIR}"/flashinfer_python-0.6.18-*.whl

if (( ! skip_humming )); then
    python3 -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --no-build-isolation \
        --force-reinstall \
        "${HUMMING_ARCHIVE}"
fi

echo "Dependency installation finished. Running environment checks..."
bash "${SCRIPT_DIR}/check-dgx-spark-env.sh" --skip-model
