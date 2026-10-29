#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
skip_model=0
source_only=0

usage() {
    cat <<'EOF'
Usage: scripts/check-dgx-spark-env.sh [options]

Options:
  --model-path PATH  Validate an externally supplied VLX model directory.
  --skip-model       Do not require model weights.
  --source-only      Validate only files contained in the portable archive.
  -h, --help         Show this help.
EOF
}

while (( $# )); do
    case "$1" in
        --model-path)
            [[ $# -ge 2 ]] || { echo "--model-path needs a value" >&2; exit 2; }
            MODEL_PATH="$2"; shift
            ;;
        --skip-model) skip_model=1 ;;
        --source-only) source_only=1; skip_model=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*" >&2; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*" >&2; failures=$((failures + 1)); }

required_files=(
    README.md
    requirements/dgx-spark-kernels.txt
    benchmarks/bench_qwen35.py
    benchmarks/bench_module_latency.py
    scripts/prepare-repro-env.sh
    scripts/run-gdn-sm121-validation.sh
    scripts/run-vlx-full-stack-ab.sh
    raw_vllm_server/start_vllm_server.sh
    vllm_/__init__.py
    trt_aux/export_aux_blocks.py
    docs/reproduction/PORTABLE_PACKAGE.md
)

for path in "${required_files[@]}"; do
    [[ -e "${PROJECT_ROOT}/${path}" ]] && pass "source file ${path}" || fail "missing source file ${path}"
done

if (( source_only )); then
    if (( failures )); then
        echo "Source package check failed (${failures} failure(s), ${warnings} warning(s))." >&2
        exit 1
    fi
    echo "Source package check passed."
    exit 0
fi

[[ "$(uname -s)" == "Linux" ]] && pass "operating system Linux" || fail "DGX Spark target requires Linux; found $(uname -s)"
[[ "$(uname -m)" == "aarch64" ]] && pass "architecture aarch64" || fail "DGX Spark target requires aarch64; found $(uname -m)"

if command -v python3 >/dev/null; then
    py_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${py_version}" == "3.12" ]] && pass "Python ${py_version}" || fail "expected Python 3.12; found ${py_version}"
else
    fail "python3 is not installed"
fi

if command -v nvidia-smi >/dev/null; then
    pass "nvidia-smi is available"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || warn "nvidia-smi could not query the GPU"
else
    fail "nvidia-smi is unavailable"
fi

if command -v docker >/dev/null; then
    docker info >/dev/null 2>&1 && pass "Docker daemon is reachable" || warn "Docker exists but the daemon is not reachable"
else
    warn "Docker CLI is unavailable (acceptable when already inside the container)"
fi

if command -v python3 >/dev/null; then
    python3 - <<'PY'
import importlib.metadata as md
import importlib.util
import sys

expected = {
    "flashinfer-python": "0.6.18",
    "apache-tvm-ffi": "0.1.13.post3",
    "nvidia-cutlass-dsl": "4.7.1",
}
modules = ("torch", "vllm", "tensorrt", "flashinfer", "tvm_ffi", "cutlass", "humming")
bad = False
for name in modules:
    found = importlib.util.find_spec(name) is not None
    print(f"{'PASS' if found else 'FAIL'}  Python module {name}")
    bad |= not found
for package, wanted in expected.items():
    try:
        actual = md.version(package)
    except md.PackageNotFoundError:
        print(f"FAIL  package {package} is not installed")
        bad = True
    else:
        ok = actual == wanted
        print(f"{'PASS' if ok else 'FAIL'}  {package} {actual} (expected {wanted})")
        bad |= not ok
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"{'PASS' if cuda_ok else 'FAIL'}  PyTorch CUDA availability")
    bad |= not cuda_ok
    if cuda_ok:
        capability = torch.cuda.get_device_capability()
        print(f"INFO  GPU: {torch.cuda.get_device_name(0)}, capability {capability[0]}.{capability[1]}")
        if capability != (12, 1):
            print("FAIL  expected DGX Spark compute capability 12.1 (SM121)")
            bad = True
except Exception as exc:
    print(f"FAIL  PyTorch CUDA probe: {exc}")
    bad = True
sys.exit(1 if bad else 0)
PY
    (( $? == 0 )) || failures=$((failures + 1))
fi

if (( ! skip_model )); then
    if [[ -z "${MODEL_PATH}" ]]; then
        fail "set MODEL_PATH or pass --model-path; model weights are intentionally not in the archive"
    elif [[ ! -d "${MODEL_PATH}" ]]; then
        fail "model directory does not exist: ${MODEL_PATH}"
    else
        [[ -f "${MODEL_PATH}/config.json" ]] && pass "model config.json" || fail "model config.json is missing"
        if find "${MODEL_PATH}" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -print -quit | grep -q .; then
            pass "model weight files found"
        else
            fail "no model weight files found under ${MODEL_PATH}"
        fi
    fi
fi

if (( failures )); then
    echo "Environment check failed (${failures} failure group(s), ${warnings} warning(s))." >&2
    exit 1
fi
echo "Environment check passed (${warnings} warning(s))."
