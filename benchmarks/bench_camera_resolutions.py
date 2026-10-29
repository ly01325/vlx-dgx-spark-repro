#!/usr/bin/env python3
"""Benchmark full VLX at the three camera resolutions used by the TRT router."""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench_module_latency import parse_block, wait_for_phases
from raw_vllm_server.vllm_request import DEFAULT_BBOX_LIST, DEFAULT_QUESTION


RESOLUTIONS = ((854, 480), (1280, 720), (1920, 1080))


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def data_url(image: Image.Image, size: tuple[int, int]) -> str:
    resized = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def scaled_boxes(width: int, height: int) -> list[list[float]]:
    sx, sy = width / 1920.0, height / 1080.0
    return [
        [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
        for box in DEFAULT_BBOX_LIST
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    source = Image.open(args.image).convert("RGB")
    log_pos = args.server_log.stat().st_size
    results = []
    for width, height in RESOLUTIONS:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(source, (width, height))}},
                {"type": "text", "text": DEFAULT_QUESTION},
            ]}],
            "max_tokens": 27,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"disable_thinking_tags": True},
            "mm_processor_kwargs": {"bbox_list": scaled_boxes(width, height)},
        }
        for _ in range(args.warmup):
            response = requests.post(args.url, json=payload, timeout=600)
            response.raise_for_status()
            _, log_pos = wait_for_phases(args.server_log, log_pos)
        e2e, aux, mm = [], [], []
        for _ in range(args.runs):
            start = time.perf_counter()
            response = requests.post(args.url, json=payload, timeout=600)
            elapsed = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            text, log_pos = wait_for_phases(args.server_log, log_pos)
            parsed = parse_block(text)
            stages = parsed["object_feature_stages_ms"]
            e2e.append(elapsed)
            aux.append(stages["aux_vision_tower"])
            mm.append(stages["multimodal_embed_total"])
        prepared_w = round(width / 16) * 16
        prepared_h = round(height / 16) * 16
        results.append({
            "input_resolution": [width, height],
            "prepared_resolution": [prepared_w, prepared_h],
            "token_length": 10 + (prepared_w // 16) * (prepared_h // 16),
            "e2e_ms": summary(e2e),
            "aux_vision_tower_ms": summary(aux),
            "multimodal_embed_total_ms": summary(mm),
        })
    report = {"warmup": args.warmup, "runs": args.runs, "results": results}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
