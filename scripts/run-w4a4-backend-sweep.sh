#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
RESULT_ROOT="${SPARK_JOB_DIR:-${PROJECT_DIR}/artifacts/results/w4a4-backend-sweep}"
BACKENDS="${BACKENDS:-flashinfer-cutlass,cutlass,flashinfer-cudnn,flashinfer-trtllm,humming}"
OUTPUT_LENGTHS="${OUTPUT_LENGTHS:-27,256,512}"
mkdir -p "${RESULT_ROOT}"

IFS=, read -ra backends <<< "${BACKENDS}"
port=8120
for backend in "${backends[@]}"; do
    result_dir="${RESULT_ROOT}/${backend}"
    mkdir -p "${result_dir}"
    echo "=== W4A4 backend ${backend} ==="
    if SPARK_JOB_DIR="${result_dir}" \
       OUTPUT_LENGTHS="${OUTPUT_LENGTHS}" \
       CONCURRENCIES=1,8 REQUESTS_PER_CONCURRENCY=8 LATENCY_RUNS=6 \
       bash "${PROJECT_DIR}/scripts/run-qwen35-kernel-candidate.sh" \
           "${backend}" FLASH_ATTN "${port}"; then
        echo succeeded > "${result_dir}/candidate-state"
    else
        echo failed > "${result_dir}/candidate-state"
    fi
    port=$((port + 1))
done

python3 - "${RESULT_ROOT}" "${BACKENDS}" "${OUTPUT_LENGTHS}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
backends = sys.argv[2].split(",")
lengths = [int(v) for v in sys.argv[3].split(",")]
report = {"input_tokens": 2200, "output_lengths": lengths, "candidates": {}, "winners": {}}
for backend in backends:
    directory = root / backend
    state = (directory / "candidate-state").read_text().strip()
    item = {"state": state, "results": {}}
    if state == "succeeded":
        for length in lengths:
            data = json.loads((directory / f"benchmark-{length}.json").read_text())
            item["results"][str(length)] = {
                str(row["concurrency"]): {
                    "request_throughput_rps": row["request_throughput_rps"],
                    "output_token_throughput_tok_s": row["output_token_throughput_tok_s"],
                    "e2e_mean_ms": row["e2e_ms"]["mean"],
                    "ttft_mean_ms": row["ttft_ms"]["mean"],
                    "tpot_mean_ms": row["tpot_ms"]["mean"],
                }
                for row in data["throughput"]
            }
    report["candidates"][backend] = item

for length in lengths:
    report["winners"][str(length)] = {}
    for concurrency in (1, 8):
        viable = []
        for backend, item in report["candidates"].items():
            row = item["results"].get(str(length), {}).get(str(concurrency))
            if row:
                viable.append((row["output_token_throughput_tok_s"], backend, row))
        if viable:
            throughput, backend, row = max(viable)
            report["winners"][str(length)][str(concurrency)] = {
                "backend": backend, **row
            }
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
