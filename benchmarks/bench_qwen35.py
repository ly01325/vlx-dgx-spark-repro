#!/usr/bin/env python3
"""Benchmark only the Qwen3.5 language path through vLLM completions.

The prompt is supplied as token IDs, so its length is exact and no image,
vision tower, ROI extractor, or multimodal projector is invoked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q / 100
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def discover_model(url: str, timeout: float) -> str:
    models_url = url.rsplit("/v1/completions", 1)[0] + "/v1/models"
    response = requests.get(models_url, timeout=timeout)
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        raise RuntimeError(f"No served models returned by {models_url}")
    return str(models[0]["id"])


def run_request(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    request_id: int,
    expected_input: int,
    expected_output: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ttft_s: float | None = None
    token_arrivals: list[float] = []
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
                chunk = json.loads(body)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if choice.get("text"):
                    now = time.perf_counter()
                    if ttft_s is None:
                        ttft_s = now - started
                    token_arrivals.append(now)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if error is None and prompt_tokens != expected_input:
        error = f"Expected {expected_input} prompt tokens, got {prompt_tokens}"
    if error is None and completion_tokens != expected_output:
        error = f"Expected {expected_output} completion tokens, got {completion_tokens}"

    e2e_ms = (ended - started) * 1000
    ttft_ms = None if ttft_s is None else ttft_s * 1000
    tpot_ms = None
    if ttft_ms is not None and completion_tokens > 1:
        tpot_ms = (e2e_ms - ttft_ms) / (completion_tokens - 1)

    return {
        "id": request_id,
        "ok": error is None and status == 200 and ttft_s is not None,
        "status": status,
        "error": error,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "stream_chunks": len(token_arrivals),
    }


def run_batch(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    count: int,
    concurrency: int,
    start_id: int,
    expected_input: int,
    expected_output: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_request,
                url=url,
                payload=payload,
                timeout=timeout,
                request_id=start_id + idx,
                expected_input=expected_input,
                expected_output=expected_output,
            )
            for idx in range(count)
        ]
        rows = [future.result() for future in as_completed(futures)]
    rows.sort(key=lambda row: row["id"])
    return rows, time.perf_counter() - started


def batch_report(
    rows: list[dict[str, Any]], wall_s: float, concurrency: int
) -> dict[str, Any]:
    ok = [row for row in rows if row["ok"]]
    e2e = [float(row["e2e_ms"]) for row in ok]
    ttft = [float(row["ttft_ms"]) for row in ok if row["ttft_ms"] is not None]
    tpot = [float(row["tpot_ms"]) for row in ok if row["tpot_ms"] is not None]
    output_tokens = sum(int(row["completion_tokens"]) for row in ok)
    input_tokens = sum(int(row["prompt_tokens"]) for row in ok)
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "wall_s": wall_s,
        "request_throughput_rps": len(ok) / wall_s if wall_s else 0.0,
        "input_token_throughput_tok_s": input_tokens / wall_s if wall_s else 0.0,
        "output_token_throughput_tok_s": output_tokens / wall_s if wall_s else 0.0,
        "e2e_ms": summary(e2e),
        "ttft_ms": summary(ttft),
        "tpot_ms": summary(tpot),
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--input-tokens", type=int, default=2200)
    parser.add_argument("--output-tokens", type=int, default=27)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--latency-runs", type=int, default=15)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests-per-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="artifacts/results/qwen35_2200_27.json")
    args = parser.parse_args()

    if args.input_tokens < 1 or args.output_tokens < 1:
        parser.error("input and output token counts must be positive")
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    if not concurrencies or any(value < 1 for value in concurrencies):
        parser.error("--concurrency must contain positive integers")

    model = (
        discover_model(args.url, args.timeout)
        if args.model == "auto"
        else args.model
    )
    payload = {
        "model": model,
        "prompt": [args.token_id] * args.input_tokens,
        "max_tokens": args.output_tokens,
        "min_tokens": args.output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    report: dict[str, Any] = {
        "started_unix": time.time(),
        "setup": {
            "url": args.url,
            "model": model,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "token_id": args.token_id,
            "warmup": args.warmup,
            "latency_runs": args.latency_runs,
            "concurrencies": concurrencies,
            "vision_path": "disabled by text-only completions request",
            "fixed_decode": "min_tokens=max_tokens and ignore_eos=true",
        },
    }
    print(json.dumps(report["setup"], indent=2), flush=True)

    if args.warmup:
        rows, wall = run_batch(
            url=args.url,
            payload=payload,
            timeout=args.timeout,
            count=args.warmup,
            concurrency=1,
            start_id=0,
            expected_input=args.input_tokens,
            expected_output=args.output_tokens,
        )
        report["warmup"] = batch_report(rows, wall, 1)
        if report["warmup"]["failed"]:
            raise RuntimeError(f"Warmup failed: {rows}")

    rows, wall = run_batch(
        url=args.url,
        payload=payload,
        timeout=args.timeout,
        count=args.latency_runs,
        concurrency=1,
        start_id=1000,
        expected_input=args.input_tokens,
        expected_output=args.output_tokens,
    )
    report["sequential"] = batch_report(rows, wall, 1)
    print(json.dumps({k: v for k, v in report["sequential"].items() if k != "runs"}, indent=2), flush=True)

    report["throughput"] = []
    next_id = 2000
    for concurrency in concurrencies:
        count = max(args.requests_per_concurrency, concurrency)
        rows, wall = run_batch(
            url=args.url,
            payload=payload,
            timeout=args.timeout,
            count=count,
            concurrency=concurrency,
            start_id=next_id,
            expected_input=args.input_tokens,
            expected_output=args.output_tokens,
        )
        next_id += count
        item = batch_report(rows, wall, concurrency)
        report["throughput"].append(item)
        print(json.dumps({k: v for k, v in item.items() if k != "runs"}, indent=2), flush=True)

    report["finished_unix"] = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return 0 if report["sequential"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

