#!/usr/bin/env python3
"""Compare two bench_qwen35.py JSON reports and emit JSON plus Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def improvement(baseline: float, optimized: float, lower_is_better: bool) -> float:
    if baseline == 0:
        return float("nan")
    if lower_is_better:
        return (baseline - optimized) / baseline * 100
    return (optimized - baseline) / baseline * 100


def mean_metric(row: dict[str, Any], name: str) -> float:
    return float(row[name]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    optimized = json.loads(Path(args.optimized).read_text(encoding="utf-8"))
    if baseline["setup"]["input_tokens"] != optimized["setup"]["input_tokens"]:
        raise ValueError("input token counts differ")
    if baseline["setup"]["output_tokens"] != optimized["setup"]["output_tokens"]:
        raise ValueError("output token counts differ")

    base_rows = {int(row["concurrency"]): row for row in baseline["throughput"]}
    opt_rows = {int(row["concurrency"]): row for row in optimized["throughput"]}
    if base_rows.keys() != opt_rows.keys():
        raise ValueError("concurrency sets differ")

    rows = []
    for concurrency in sorted(base_rows):
        b = base_rows[concurrency]
        o = opt_rows[concurrency]
        if b["failed"] or o["failed"]:
            raise RuntimeError(f"failed requests at concurrency {concurrency}")
        item = {
            "concurrency": concurrency,
            "baseline": {
                "request_throughput_rps": b["request_throughput_rps"],
                "output_token_throughput_tok_s": b["output_token_throughput_tok_s"],
                "e2e_mean_ms": mean_metric(b, "e2e_ms"),
                "ttft_mean_ms": mean_metric(b, "ttft_ms"),
                "tpot_mean_ms": mean_metric(b, "tpot_ms"),
            },
            "optimized": {
                "request_throughput_rps": o["request_throughput_rps"],
                "output_token_throughput_tok_s": o["output_token_throughput_tok_s"],
                "e2e_mean_ms": mean_metric(o, "e2e_ms"),
                "ttft_mean_ms": mean_metric(o, "ttft_ms"),
                "tpot_mean_ms": mean_metric(o, "tpot_ms"),
            },
        }
        item["improvement_pct"] = {
            "request_throughput": improvement(
                item["baseline"]["request_throughput_rps"],
                item["optimized"]["request_throughput_rps"],
                False,
            ),
            "output_token_throughput": improvement(
                item["baseline"]["output_token_throughput_tok_s"],
                item["optimized"]["output_token_throughput_tok_s"],
                False,
            ),
            "e2e_latency": improvement(
                item["baseline"]["e2e_mean_ms"],
                item["optimized"]["e2e_mean_ms"],
                True,
            ),
            "ttft": improvement(
                item["baseline"]["ttft_mean_ms"],
                item["optimized"]["ttft_mean_ms"],
                True,
            ),
            "tpot": improvement(
                item["baseline"]["tpot_mean_ms"],
                item["optimized"]["tpot_mean_ms"],
                True,
            ),
        }
        rows.append(item)

    report = {
        "scenario": {
            "input_tokens": baseline["setup"]["input_tokens"],
            "output_tokens": baseline["setup"]["output_tokens"],
            "concurrencies": sorted(base_rows),
            "baseline": "W4A4 FlashInfer-CUTLASS + BF16 LM head + Triton GDN",
            "optimized": "W4A4 FlashInfer-CUTLASS + Humming W4A16 LM head + FlashInfer GDN",
        },
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen3.5 deployment A/B benchmark",
        "",
        f"Fixed workload: {report['scenario']['input_tokens']} input tokens, "
        f"{report['scenario']['output_tokens']} output tokens.",
        "",
        "| Concurrency | Baseline req/s | Optimized req/s | Throughput gain | "
        "Baseline E2E ms | Optimized E2E ms | E2E reduction | TTFT reduction | TPOT reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        b = row["baseline"]
        o = row["optimized"]
        p = row["improvement_pct"]
        lines.append(
            f"| {row['concurrency']} | {b['request_throughput_rps']:.3f} | "
            f"{o['request_throughput_rps']:.3f} | {p['request_throughput']:+.2f}% | "
            f"{b['e2e_mean_ms']:.2f} | {o['e2e_mean_ms']:.2f} | "
            f"{p['e2e_latency']:+.2f}% | {p['ttft']:+.2f}% | {p['tpot']:+.2f}% |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
