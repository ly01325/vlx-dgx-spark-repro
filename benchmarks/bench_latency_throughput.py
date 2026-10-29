#!/usr/bin/env python3
"""Latency and throughput bench for the live vlx-seek-1.5-3b-sanitized server.

Uses the same image / prompt / 100-bbox payload as raw_vllm_server/vllm_request.py.
Streaming is enabled so TTFT and TPOT can be measured client-side.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "raw_vllm_server"))
from vllm_request import (  # noqa: E402
    DEFAULT_BBOX_LIST,
    DEFAULT_IMAGE_URL,
    DEFAULT_MODEL,
    DEFAULT_QUESTION,
    DEFAULT_URL,
    image_source,
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def build_payload(
    image_data_url: str, max_tokens: int, model: str, fixed_decode: bool = False
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": DEFAULT_QUESTION},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"disable_thinking_tags": True},
        "skip_special_tokens": False,
        "stop": ["<|im_end|>"],
        "stop_token_ids": [248046],
        "include_stop_str_in_output": False,
        "mm_processor_kwargs": {"bbox_list": DEFAULT_BBOX_LIST},
    }
    if fixed_decode:
        payload["min_tokens"] = max_tokens
        payload["ignore_eos"] = True
        payload.pop("stop", None)
        payload.pop("stop_token_ids", None)
    return payload


def one_stream_request(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    request_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ttft_s: float | None = None
    token_times: list[float] = []
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    error = None
    status = None
    try:
        with requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=timeout,
        ) as response:
            status = response.status_code
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                body = raw_line[6:]
                if body == "[DONE]":
                    break
                data = json.loads(body)
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (
                    delta.get("content") or ""
                )
                if not piece:
                    continue
                now = time.perf_counter()
                if ttft_s is None:
                    ttft_s = now - started
                token_times.append(now)
                text_parts.append(piece)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    e2e_ms = (ended - started) * 1000.0
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    visible_tokens = len(token_times)
    if completion_tokens <= 0:
        completion_tokens = visible_tokens

    itls_ms: list[float] = []
    for prev, cur in zip(token_times, token_times[1:]):
        itls_ms.append((cur - prev) * 1000.0)

    tpot_ms = float("nan")
    if completion_tokens > 1 and ttft_s is not None:
        tpot_ms = (e2e_ms - ttft_s * 1000.0) / (completion_tokens - 1)
    elif itls_ms:
        tpot_ms = statistics.fmean(itls_ms)

    decode_tok_s = float("nan")
    if ttft_s is not None and completion_tokens > 0:
        decode_s = (ended - started) - ttft_s
        if decode_s > 0:
            decode_tok_s = completion_tokens / decode_s

    return {
        "id": request_id,
        "ok": error is None and status == 200 and ttft_s is not None,
        "status": status,
        "error": error,
        "e2e_ms": e2e_ms,
        "ttft_ms": None if ttft_s is None else ttft_s * 1000.0,
        "tpot_ms": tpot_ms,
        "mean_itl_ms": statistics.fmean(itls_ms) if itls_ms else float("nan"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "visible_chunks": visible_tokens,
        "decode_tok_s": decode_tok_s,
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
    }


def run_batch(
    url: str,
    payload: dict[str, Any],
    n: int,
    concurrency: int,
    timeout: float,
    start_id: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(one_stream_request, url, payload, timeout, start_id + i)
            for i in range(n)
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda row: row["id"])
    return results


def stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["ok"]]
    e2e = [row["e2e_ms"] for row in ok_rows]
    ttft = [row["ttft_ms"] for row in ok_rows if row["ttft_ms"] is not None]
    tpot = [row["tpot_ms"] for row in ok_rows if row["tpot_ms"] == row["tpot_ms"]]
    itl = [
        row["mean_itl_ms"]
        for row in ok_rows
        if row["mean_itl_ms"] == row["mean_itl_ms"]
    ]
    decode_tok = [
        row["decode_tok_s"]
        for row in ok_rows
        if row["decode_tok_s"] == row["decode_tok_s"]
    ]
    prompt = [row["prompt_tokens"] for row in ok_rows if row["prompt_tokens"]]
    completion = [row["completion_tokens"] for row in ok_rows]
    return {
        "requests": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "e2e_ms": summarize(e2e),
        "ttft_ms": summarize(ttft),
        "tpot_ms": summarize(tpot),
        "mean_itl_ms": summarize(itl),
        "decode_tok_s": summarize(decode_tok),
        "prompt_tokens_mean": statistics.fmean(prompt) if prompt else float("nan"),
        "completion_tokens_mean": (
            statistics.fmean(completion) if completion else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL)
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", DEFAULT_MODEL),
    )
    parser.add_argument("--max-tokens", type=int, default=27)
    parser.add_argument(
        "--fixed-decode",
        action="store_true",
        help="Force exactly max_tokens by setting min_tokens and ignore_eos.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--latency-runs", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "artifacts/results" / "latency_throughput.json"),
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Encoding image {args.image}", flush=True)
    image_data_url = image_source(args.image)
    payload = build_payload(
        image_data_url, args.max_tokens, args.model, fixed_decode=args.fixed_decode
    )

    report: dict[str, Any] = {
        "started_unix": time.time(),
        "setup": {
            "url": args.url,
            "model": args.model,
            "image": args.image,
            "prompt": DEFAULT_QUESTION,
            "bbox_count": len(DEFAULT_BBOX_LIST),
            "max_tokens": args.max_tokens,
            "fixed_decode": args.fixed_decode,
            "temperature": 0.0,
            "warmup": args.warmup,
            "latency_runs": args.latency_runs,
            "gpu": "NVIDIA GB10 x1",
            "tp": 1,
            "server": os.getenv("BENCH_SERVER_LABEL", "vlx-seek-15-3b / vLLM 0.19.0"),
            "notes": [
                "NO_PREFIX_CACHE=1 NO_MM_CACHE=1",
                "OMCHAT_PROFILE_MODEL_PHASES=1",
                "VLLM_USE_FLASHRT_OPS=0",
            ],
        },
    }

    if args.warmup > 0:
        print(f"Warmup x{args.warmup}", flush=True)
        warm_t0 = time.perf_counter()
        warm = run_batch(args.url, payload, args.warmup, 1, args.timeout, 0)
        report["warmup"] = {
            "wall_s": time.perf_counter() - warm_t0,
            "stats": stats_from_rows(warm),
        }
        print(json.dumps(report["warmup"]["stats"], indent=2), flush=True)

    print(f"Sequential latency x{args.latency_runs}", flush=True)
    lat_t0 = time.perf_counter()
    latency_rows = run_batch(
        args.url, payload, args.latency_runs, 1, args.timeout, 1000
    )
    lat_wall = time.perf_counter() - lat_t0
    latency_stats = stats_from_rows(latency_rows)
    ok_lat = [row for row in latency_rows if row["ok"]]
    total_out = sum(row["completion_tokens"] for row in ok_lat)
    total_in = sum(row["prompt_tokens"] for row in ok_lat)
    report["sequential_latency"] = {
        "concurrency": 1,
        "wall_s": lat_wall,
        "request_throughput_rps": len(ok_lat) / lat_wall if lat_wall else 0.0,
        "output_token_throughput_tok_s": total_out / lat_wall if lat_wall else 0.0,
        "total_token_throughput_tok_s": (total_in + total_out) / lat_wall
        if lat_wall
        else 0.0,
        "stats": latency_stats,
        "runs": [
            {
                "id": row["id"],
                "e2e_ms": row["e2e_ms"],
                "ttft_ms": row["ttft_ms"],
                "tpot_ms": row["tpot_ms"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "finish_reason": row["finish_reason"],
                "ok": row["ok"],
                "error": row["error"],
            }
            for row in latency_rows
        ],
    }
    print(json.dumps(report["sequential_latency"]["stats"], indent=2), flush=True)

    sweep = []
    concurrency_plan = [
        (1, 8),
        (2, 16),
        (4, 16),
        (8, 16),
    ]
    req_id = 2000
    for concurrency, n in concurrency_plan:
        print(f"Throughput concurrency={concurrency} n={n}", flush=True)
        t0 = time.perf_counter()
        rows = run_batch(args.url, payload, n, concurrency, args.timeout, req_id)
        wall = time.perf_counter() - t0
        req_id += n
        ok_rows = [row for row in rows if row["ok"]]
        out_tok = sum(row["completion_tokens"] for row in ok_rows)
        in_tok = sum(row["prompt_tokens"] for row in ok_rows)
        item = {
            "concurrency": concurrency,
            "num_prompts": n,
            "wall_s": wall,
            "request_throughput_rps": len(ok_rows) / wall if wall else 0.0,
            "output_token_throughput_tok_s": out_tok / wall if wall else 0.0,
            "total_token_throughput_tok_s": (in_tok + out_tok) / wall if wall else 0.0,
            "stats": stats_from_rows(rows),
        }
        sweep.append(item)
        print(
            json.dumps(
                {
                    "concurrency": concurrency,
                    "rps": item["request_throughput_rps"],
                    "out_tok_s": item["output_token_throughput_tok_s"],
                    "e2e_mean_ms": item["stats"]["e2e_ms"]["mean"],
                    "ttft_mean_ms": item["stats"]["ttft_ms"]["mean"],
                    "ok": item["stats"]["ok"],
                    "failed": item["stats"]["failed"],
                },
                indent=2,
            ),
            flush=True,
        )
    report["concurrency_sweep"] = sweep
    report["finished_unix"] = time.time()
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
