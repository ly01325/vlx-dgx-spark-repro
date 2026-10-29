#!/usr/bin/env python3
"""Collect per-module OmChat/vLLM profiler timings from sequential live requests."""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
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

LOG_PATH = Path(
    os.getenv("VLLM_SERVER_LOG", "/workspace/to_nvidia/artifacts/logs/vllm_server.log")
)
OUT_PATH = Path(
    os.getenv("MODULE_BENCH_OUT", "/workspace/to_nvidia/artifacts/results/module_latency.json")
)
SERVED_MODEL = os.getenv("MODEL_NAME", DEFAULT_MODEL)

OBJECT_RE = re.compile(r"OmChat object-feature (\S+) took ([0-9.]+) ms")
BREAKDOWN_RE = re.compile(
    r"Model step breakdown step_index=(\d+) phase=(\S+) total=([0-9.]+) ms "
    r"multimodal_encoder_and_input_embed=([0-9.]+) ms "
    r"model_input_prepare=([0-9.]+) ms "
    r"language_model_forward=([0-9.]+) ms other=([0-9.]+) ms"
)
PHASES_RE = re.compile(
    r"OmChat model phases request_id=(\S+) prompt_tokens=(\d+) "
    r"completion_tokens=(\d+) prefill=([0-9.]+) ms decode=([0-9.]+) ms"
)
DECODE_SUM_RE = re.compile(
    r"OmChat decode intervals summary request_id=(\S+) decode_steps=(\d+) "
    r"total=([0-9.]+) ms model_step=([0-9.]+) ms other=([-+0-9.]+) ms"
)
LM_FORWARD_RE = re.compile(
    r"Model forward step_index=\d+ phase=(\S+) scope=language_model_forward "
    r"took ([0-9.]+) ms"
)

LEAF_STAGES = [
    "shared_image_pack",
    "main_mm_preprocess",
    "aux_image_preprocess",
    "main_vision_input_prepare",
    "main_vision_tower",
    "main_vision_postprocess",
    "aux_vision_tower",
    "roi_extract",
    "aux_projector",
    "append_object_features",
]


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
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def build_payload(image_data_url: str) -> dict[str, Any]:
    payload = {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": DEFAULT_QUESTION},
                ],
            }
        ],
        "max_tokens": 27,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"disable_thinking_tags": True},
        "skip_special_tokens": False,
        "stop": ["<|im_end|>"],
        "stop_token_ids": [248046],
        "mm_processor_kwargs": {"bbox_list": DEFAULT_BBOX_LIST},
    }
    if os.getenv("MODULE_BENCH_FIXED_DECODE", "0") == "1":
        payload["min_tokens"] = 27
        payload["ignore_eos"] = True
        payload.pop("stop", None)
        payload.pop("stop_token_ids", None)
    return payload


def send_one(url: str, payload: dict[str, Any], timeout: float, profile_id: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = requests.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "x-omchat-profile-id": profile_id,
        },
        timeout=timeout,
    )
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "e2e_ms": e2e_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
    }


def parse_block(text: str) -> dict[str, Any]:
    stages: dict[str, float] = {}
    for name, ms in OBJECT_RE.findall(text):
        stages[name] = float(ms)

    prefill_bd = None
    decode_lm: list[float] = []
    decode_step_total: list[float] = []
    for phase, total, mm, prep, lm, other in (
        (m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5)),
         float(m.group(6)), float(m.group(7)))
        for m in BREAKDOWN_RE.finditer(text)
    ):
        if phase == "prefill_only" and prefill_bd is None:
            prefill_bd = {
                "total_ms": total,
                "multimodal_encoder_and_input_embed_ms": mm,
                "model_input_prepare_ms": prep,
                "language_model_forward_ms": lm,
                "other_ms": other,
            }
        elif phase == "decode_only":
            decode_lm.append(lm)
            decode_step_total.append(total)

    phases = None
    m = PHASES_RE.search(text)
    if m:
        phases = {
            "request_id": m.group(1),
            "prompt_tokens": int(m.group(2)),
            "completion_tokens": int(m.group(3)),
            "prefill_ms": float(m.group(4)),
            "decode_ms": float(m.group(5)),
        }

    decode_sum = None
    m = DECODE_SUM_RE.search(text)
    if m:
        decode_sum = {
            "request_id": m.group(1),
            "decode_steps": int(m.group(2)),
            "total_ms": float(m.group(3)),
            "model_step_ms": float(m.group(4)),
            "other_ms": float(m.group(5)),
        }

    lm_prefill = None
    lm_decode: list[float] = []
    for phase, ms in LM_FORWARD_RE.findall(text):
        value = float(ms)
        if phase == "prefill_only" and lm_prefill is None:
            lm_prefill = value
        elif phase == "decode_only":
            lm_decode.append(value)

    return {
        "object_feature_stages_ms": stages,
        "prefill_step_breakdown": prefill_bd,
        "decode_language_model_forward_ms": decode_lm or lm_decode,
        "decode_step_total_ms": decode_step_total,
        "serving_phases": phases,
        "decode_summary": decode_sum,
        "lm_prefill_forward_ms": lm_prefill,
    }


def wait_for_phases(log_path: Path, start: int, timeout_s: float = 30.0) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = log_path.read_bytes()
        chunk = data[start:]
        text = chunk.decode("utf-8", errors="replace")
        if PHASES_RE.search(text):
            return text, len(data)
        time.sleep(0.05)
    data = log_path.read_bytes()
    return data[start:].decode("utf-8", errors="replace"), len(data)


def main() -> int:
    warmup = 3
    runs = 10
    image_path = os.getenv("MODULE_BENCH_IMAGE", DEFAULT_IMAGE_URL)
    image_data_url = image_source(image_path)
    payload = build_payload(image_data_url)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_pos = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    print(f"log start offset {log_pos}", flush=True)

    for i in range(warmup):
        print(f"warmup {i+1}/{warmup}", flush=True)
        send_one(DEFAULT_URL, payload, 600.0, f"warmup-{i}")
        text, log_pos = wait_for_phases(LOG_PATH, log_pos)
        print("  skipped", bool(PHASES_RE.search(text)), flush=True)

    measured: list[dict[str, Any]] = []
    for i in range(runs):
        print(f"measure {i+1}/{runs}", flush=True)
        client = send_one(DEFAULT_URL, payload, 600.0, f"module-{i}")
        text, log_pos = wait_for_phases(LOG_PATH, log_pos)
        parsed = parse_block(text)
        parsed["client_e2e_ms"] = client["e2e_ms"]
        parsed["client_ok"] = client["ok"]
        measured.append(parsed)
        stages = parsed["object_feature_stages_ms"]
        print(
            json.dumps(
                {
                    "e2e_ms": round(client["e2e_ms"], 1),
                    "main_vit": stages.get("main_vision_tower"),
                    "aux_vit": stages.get("aux_vision_tower"),
                    "roi": stages.get("roi_extract"),
                    "lm_prefill": (parsed.get("prefill_step_breakdown") or {}).get(
                        "language_model_forward_ms"
                    ),
                }
            ),
            flush=True,
        )

    def collect(path: list[str]) -> list[float]:
        out: list[float] = []
        for row in measured:
            cur: Any = row
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    cur = None
                    break
                cur = cur[key]
            if isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    module_stats: dict[str, Any] = {}
    for stage in LEAF_STAGES:
        module_stats[stage] = summarize(
            [row["object_feature_stages_ms"][stage]
             for row in measured
             if stage in row["object_feature_stages_ms"]]
        )
    module_stats["multimodal_embed_total"] = summarize(
        [row["object_feature_stages_ms"]["multimodal_embed_total"]
         for row in measured
         if "multimodal_embed_total" in row["object_feature_stages_ms"]]
    )
    module_stats["base_embed_multimodal"] = summarize(
        [row["object_feature_stages_ms"]["base_embed_multimodal"]
         for row in measured
         if "base_embed_multimodal" in row["object_feature_stages_ms"]]
    )
    module_stats["mm_encoder_step"] = summarize(
        collect(["prefill_step_breakdown", "multimodal_encoder_and_input_embed_ms"])
    )
    module_stats["lm_prefill_forward"] = summarize(
        collect(["prefill_step_breakdown", "language_model_forward_ms"])
    )
    module_stats["prefill_step_total"] = summarize(
        collect(["prefill_step_breakdown", "total_ms"])
    )
    module_stats["serving_prefill"] = summarize(
        collect(["serving_phases", "prefill_ms"])
    )
    module_stats["serving_decode"] = summarize(
        collect(["serving_phases", "decode_ms"])
    )
    decode_means = []
    decode_sums = []
    for row in measured:
        vals = row.get("decode_language_model_forward_ms") or []
        if vals:
            decode_means.append(statistics.fmean(vals))
            decode_sums.append(sum(vals))
    module_stats["lm_decode_forward_mean"] = summarize(decode_means)
    module_stats["lm_decode_forward_sum"] = summarize(decode_sums)
    module_stats["client_e2e"] = summarize([row["client_e2e_ms"] for row in measured])

    report = {
        "setup": {
            "model": SERVED_MODEL,
            "image": image_path,
            "bbox_count": len(DEFAULT_BBOX_LIST),
            "max_tokens": 27,
            "warmup": warmup,
            "runs": runs,
            "notes": [
                "Parsed from OMCHAT_PROFILE_OBJECT_FEATURES and OMCHAT_PROFILE_MODEL_PHASES logs",
                "roi_extract includes SimpleFPN + ROI align + object pooling/pos-embed",
                "Caches off: NO_PREFIX_CACHE=1 NO_MM_CACHE=1",
            ],
        },
        "module_stats_ms": module_stats,
        "runs": measured,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", OUT_PATH, flush=True)
    print(json.dumps({k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
                      for k, v in module_stats.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
