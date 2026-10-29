#!/usr/bin/env bash
set -euo pipefail

# End-to-end VLX benchmark: original W4A4 deployment vs all validated
# C-RADIO and Qwen3.5 optimizations, for the 1080p / 27-token workload.

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
IMAGE="${KERNEL_TEST_IMAGE:-registry.linker.cc/linker/vlx_seek1_5_spark:1.0.0_encrypted}"
MODEL_NAME="${KERNEL_TEST_MODEL:-vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct}"
RESULT_ROOT="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/vlx-full-stack-ab}"
ENGINE="${PROJECT_DIR}/.trt-aux-artifacts/camera_480p_720p_1080p.bf16.plan"
HUMMING_ARCHIVE="${PROJECT_DIR}/third_party/sources/humming-a74973b.tar.gz"
PORT=8000
LONG_DECODE_BENCH="${LONG_DECODE_BENCH:-0}"

mkdir -p "${RESULT_ROOT}/baseline" "${RESULT_ROOT}/optimized"
for required in "${HUMMING_ARCHIVE}" "${ENGINE}"; do
    [[ -f "${required}" ]] || { echo "Missing dependency: ${required}" >&2; exit 2; }
done

active_container=""
log_pid=""
cleanup() {
    [[ -z "${log_pid}" ]] || kill "${log_pid}" >/dev/null 2>&1 || true
    [[ -z "${active_container}" ]] || docker rm -f "${active_container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_variant() {
    local label="$1" optimized="$2"
    local result_dir="${RESULT_ROOT}/${label}"
    local gdn_backend="triton"
    local lm_head_w4a16=0
    local trt_env=()
    if [[ "${optimized}" == 1 ]]; then
        gdn_backend="flashinfer"
        lm_head_w4a16=1
        trt_env+=(
            -e OMCHAT_AUX_TRT_STRICT=1
            -e OMCHAT_AUX_TRT_CAMERA_ENGINE=/workspace/to_nvidia/.trt-aux-artifacts/camera_480p_720p_1080p.bf16.plan
            -e OMCHAT_AUX_TRT_CAMERA_LENGTHS=1600,8170
        )
    fi

    active_container="vlx-full-stack-${label}-$$"
    docker rm -f "${active_container}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${active_container}" --gpus all --network host \
        -v "${PROJECT_DIR}:/workspace/to_nvidia" \
        -v "${result_dir}:/workspace/results" \
        -v /usr/lib/python3.12/dist-packages/tensorrt:/opt/trt-python/tensorrt:ro \
        -e PYTHONPATH=/opt/trt-python \
        -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
        -e "VLLM_HUMMING_LM_HEAD_W4A16=${lm_head_w4a16}" \
        -e VLLM_USE_FLASHRT_OPS=0 \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e ATTENTION_BACKEND=FLASH_ATTN \
        -e "GDN_PREFILL_BACKEND=${gdn_backend}" \
        -e OMCHAT_PROFILE_MODEL_PHASES=1 \
        -e OMCHAT_PROFILE_OBJECT_FEATURES=1 \
        -e "MODEL_NAME=${MODEL_NAME}" \
        -e "MODEL_PATH=/workspace/to_nvidia/resources/${MODEL_NAME}" \
        -e "PORT=${PORT}" -e HOST=0.0.0.0 \
        -e GPU_MEMORY_UTILIZATION=0.10 \
        -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=16 \
        -e MAX_NUM_BATCHED_TOKENS=4096 \
        -e NO_PREFIX_CACHE=1 -e NO_MM_CACHE=1 \
        "${trt_env[@]}" \
        --entrypoint bash "${IMAGE}" -lc '
set -euo pipefail
PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
    /workspace/to_nvidia/third_party/wheels/apache_tvm_ffi-0.1.13.post3-*.whl
PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
    /workspace/to_nvidia/third_party/wheels/nvidia_cuda_nvdisasm-13.4.49-*.whl \
    /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl_libs_core-4.7.1-*.whl \
    /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl_libs_base-4.7.1-*.whl \
    /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl_libs_cu13-4.7.1-*.whl \
    /workspace/to_nvidia/third_party/wheels/nvidia_cutlass_dsl-4.7.1-*.whl
PIP_CONSTRAINT= python3 -m pip install --no-deps --force-reinstall \
    /workspace/to_nvidia/third_party/wheels/flashinfer_python-0.6.18-py3-none-any.whl
PIP_CONSTRAINT= python3 -m pip install --no-deps --no-build-isolation \
    /workspace/to_nvidia/third_party/sources/humming-a74973b.tar.gz
cp -a /workspace/to_nvidia/vllm_/. /usr/local/lib/python3.12/dist-packages/vllm/
cd /workspace/to_nvidia
exec bash raw_vllm_server/start_vllm_server.sh \
    --quantization modelopt_fp4 --attention-backend FLASH_ATTN \
    --gdn-prefill-backend "${GDN_PREFILL_BACKEND}"
' >/dev/null

    docker logs -f "${active_container}" > "${result_dir}/server.log" 2>&1 &
    log_pid=$!
    local ready=0
    for _ in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${active_container}" 2>/dev/null || true)" != true ]]; then
            break
        fi
        sleep 1
    done
    if (( ! ready )); then
        tail -200 "${result_dir}/server.log" >&2 || true
        return 1
    fi

    if [[ "${LONG_DECODE_BENCH}" == 1 ]]; then
        docker exec "${active_container}" \
            python3 /workspace/to_nvidia/benchmarks/bench_vlx_long_decode.py \
            --model "${MODEL_NAME}" \
            --image /workspace/to_nvidia/resources/demo_1920_1080.jpeg \
            --output-lengths 256,512 --warmup 2 --runs 5 \
            --out /workspace/results/long_decode.json
        kill "${log_pid}" >/dev/null 2>&1 || true
        wait "${log_pid}" 2>/dev/null || true
        log_pid=""
        docker logs "${active_container}" > "${result_dir}/server.log" 2>&1 || true
        grep -E "routed to TensorRT|Using (FlashInfer|Triton/FLA) GDN|Humming NVFP4 W4A16|FlashAttention" \
            "${result_dir}/server.log" | tail -100 || true
        docker rm -f "${active_container}" >/dev/null 2>&1 || true
        active_container=""
        return
    fi

    # Warm every packed/concurrent shape once. This output is not used in A/B.
    docker exec -e "MODEL_NAME=${MODEL_NAME}" -e "BENCH_SERVER_LABEL=${label}-warmup" \
        "${active_container}" python3 /workspace/to_nvidia/benchmarks/bench_latency_throughput.py \
        --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --image /workspace/to_nvidia/resources/demo_1920_1080.jpeg \
        --model "${MODEL_NAME}" --warmup 2 --latency-runs 2 \
        --fixed-decode \
        --out /workspace/results/shape_warmup.json

    docker exec \
        -e VLLM_SERVER_LOG=/workspace/results/server.log \
        -e MODULE_BENCH_OUT=/workspace/results/module_latency.json \
        -e MODULE_BENCH_IMAGE=/workspace/to_nvidia/resources/demo_1920_1080.jpeg \
        -e MODULE_BENCH_FIXED_DECODE=1 \
        -e MODULE_BENCH_WARMUP=3 -e MODULE_BENCH_RUNS=10 \
        -e "MODEL_NAME=${MODEL_NAME}" \
        "${active_container}" python3 /workspace/to_nvidia/benchmarks/bench_module_latency.py

    docker exec -e "MODEL_NAME=${MODEL_NAME}" -e "BENCH_SERVER_LABEL=${label}" \
        "${active_container}" python3 /workspace/to_nvidia/benchmarks/bench_latency_throughput.py \
        --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --image /workspace/to_nvidia/resources/demo_1920_1080.jpeg \
        --model "${MODEL_NAME}" --warmup 3 --latency-runs 12 \
        --fixed-decode \
        --out /workspace/results/latency_throughput.json

    kill "${log_pid}" >/dev/null 2>&1 || true
    wait "${log_pid}" 2>/dev/null || true
    log_pid=""
    docker logs "${active_container}" > "${result_dir}/server.log" 2>&1 || true
    grep -E "routed to TensorRT|Using (FlashInfer|Triton/FLA) GDN|Humming|NvFp4LinearBackend|FlashAttention" \
        "${result_dir}/server.log" | tail -100 || true
    docker rm -f "${active_container}" >/dev/null 2>&1 || true
    active_container=""
}

run_variant baseline 0
run_variant optimized 1

if [[ "${LONG_DECODE_BENCH}" == 1 ]]; then
python3 - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline = json.loads((root / "baseline/long_decode.json").read_text())
optimized = json.loads((root / "optimized/long_decode.json").read_text())

def reduction(a, b): return (a - b) / a * 100
def gain(a, b): return (b / a - 1) * 100

rows = []
for b, o in zip(baseline["results"], optimized["results"]):
    assert b["output_tokens"] == o["output_tokens"]
    bs, os = b["stats"], o["stats"]
    row = {
        "output_tokens": b["output_tokens"],
        "prompt_tokens": bs["prompt_tokens_mean"],
        "baseline": {
            "e2e_ms": bs["e2e_ms"]["mean"],
            "ttft_ms": bs["ttft_ms"]["mean"],
            "tpot_ms": bs["tpot_ms"]["mean"],
            "rps": b["request_throughput_rps"],
            "output_tok_s": b["output_token_throughput_tok_s"],
            "completion_tokens": bs["completion_tokens_mean"],
        },
        "optimized": {
            "e2e_ms": os["e2e_ms"]["mean"],
            "ttft_ms": os["ttft_ms"]["mean"],
            "tpot_ms": os["tpot_ms"]["mean"],
            "rps": o["request_throughput_rps"],
            "output_tok_s": o["output_token_throughput_tok_s"],
            "completion_tokens": os["completion_tokens_mean"],
        },
    }
    row["improvement_pct"] = {
        "e2e": reduction(row["baseline"]["e2e_ms"], row["optimized"]["e2e_ms"]),
        "ttft": reduction(row["baseline"]["ttft_ms"], row["optimized"]["ttft_ms"]),
        "tpot": reduction(row["baseline"]["tpot_ms"], row["optimized"]["tpot_ms"]),
        "request_throughput": gain(row["baseline"]["rps"], row["optimized"]["rps"]),
        "output_throughput": gain(row["baseline"]["output_tok_s"], row["optimized"]["output_tok_s"]),
    }
    rows.append(row)
report = {"scenario": "full VLX, 1080p, exact fixed decode, concurrency=1", "rows": rows}
(root / "long_decode_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
exit 0
fi

python3 - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_m = json.loads((root / "baseline/module_latency.json").read_text())
opt_m = json.loads((root / "optimized/module_latency.json").read_text())
base_e = json.loads((root / "baseline/latency_throughput.json").read_text())
opt_e = json.loads((root / "optimized/latency_throughput.json").read_text())

def reduction(a, b): return (a - b) / a * 100
def gain(a, b): return (b / a - 1) * 100

module_names = (
    "aux_vision_tower", "multimodal_embed_total", "lm_prefill_forward",
    "serving_prefill", "lm_decode_forward_mean", "serving_decode", "client_e2e",
)
modules = {}
for name in module_names:
    before = base_m["module_stats_ms"][name]["mean"]
    after = opt_m["module_stats_ms"][name]["mean"]
    modules[name] = {"baseline_ms": before, "optimized_ms": after,
                     "reduction_pct": reduction(before, after)}

def compact(row):
    s = row["stats"]
    return {
        "rps": row["request_throughput_rps"],
        "output_tok_s": row["output_token_throughput_tok_s"],
        "e2e_ms": s["e2e_ms"]["mean"],
        "ttft_ms": s["ttft_ms"]["mean"],
        "tpot_ms": s["tpot_ms"]["mean"],
        "prompt_tokens": s["prompt_tokens_mean"],
        "completion_tokens": s["completion_tokens_mean"],
        "ok": s["ok"], "failed": s["failed"],
    }

concurrency = []
for b, o in zip(base_e["concurrency_sweep"], opt_e["concurrency_sweep"]):
    bc, oc = compact(b), compact(o)
    concurrency.append({
        "concurrency": b["concurrency"], "baseline": bc, "optimized": oc,
        "improvement_pct": {
            "throughput": gain(bc["rps"], oc["rps"]),
            "e2e": reduction(bc["e2e_ms"], oc["e2e_ms"]),
            "ttft": reduction(bc["ttft_ms"], oc["ttft_ms"]),
            "tpot": reduction(bc["tpot_ms"], oc["tpot_ms"]),
        },
    })

report = {
    "scenario": {
        "image": "1920x1080 (prepared C-RADIO shape 1920x1088, L=8170)",
        "bboxes": 100, "max_output_tokens": 27,
        "baseline": "C-RADIO PyTorch + W4A4 FlashInfer-CUTLASS + BF16 LM head + Triton GDN + FA2",
        "optimized": "C-RADIO TensorRT + W4A4 FlashInfer-CUTLASS + Humming W4A16 LM head + FlashInfer GDN + FA2",
    },
    "modules": modules,
    "sequential": {
        "baseline": compact(base_e["sequential_latency"]),
        "optimized": compact(opt_e["sequential_latency"]),
    },
    "concurrency": concurrency,
}
b, o = report["sequential"]["baseline"], report["sequential"]["optimized"]
report["sequential"]["improvement_pct"] = {
    "throughput": gain(b["rps"], o["rps"]),
    "e2e": reduction(b["e2e_ms"], o["e2e_ms"]),
    "ttft": reduction(b["ttft_ms"], o["ttft_ms"]),
    "tpot": reduction(b["tpot_ms"], o["tpot_ms"]),
}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
