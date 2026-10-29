#!/usr/bin/env python3
"""Sequential full-VLX benchmark with exact long decode lengths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench_latency_throughput import build_payload, run_batch, stats_from_rows
from raw_vllm_server.vllm_request import image_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-lengths", default="256,512")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    encoded = image_source(args.image)
    results = []
    request_id = 0
    for output_tokens in (int(v) for v in args.output_lengths.split(",")):
        payload = build_payload(
            encoded, output_tokens, args.model, fixed_decode=True
        )
        warm = run_batch(
            args.url, payload, args.warmup, 1, 900.0, request_id
        )
        request_id += args.warmup
        if any(not row["ok"] for row in warm):
            raise RuntimeError(f"warmup failed for output_tokens={output_tokens}")

        started = time.perf_counter()
        rows = run_batch(args.url, payload, args.runs, 1, 900.0, request_id)
        wall_s = time.perf_counter() - started
        request_id += args.runs
        stats = stats_from_rows(rows)
        if stats["failed"]:
            raise RuntimeError(f"measured request failed for {output_tokens=}")
        if stats["completion_tokens_mean"] != output_tokens:
            raise RuntimeError(
                f"expected {output_tokens} tokens, got "
                f"{stats['completion_tokens_mean']}"
            )
        result = {
            "output_tokens": output_tokens,
            "wall_s": wall_s,
            "request_throughput_rps": args.runs / wall_s,
            "output_token_throughput_tok_s": args.runs * output_tokens / wall_s,
            "stats": stats,
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    report = {
        "setup": {
            "image": args.image,
            "model": args.model,
            "warmup": args.warmup,
            "runs": args.runs,
            "concurrency": 1,
            "fixed_decode": True,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
