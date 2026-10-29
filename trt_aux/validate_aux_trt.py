#!/usr/bin/env python3
"""Validate TensorRT C-RADIO tap outputs against the PyTorch block stack."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import tensorrt as trt
import torch

from export_aux_blocks import AuxBlockStack, load_aux_state
from vllm.model_executor.models.omchat.object_modules import build_vision_tower_aux
from vllm.transformers_utils.configs.omchat_qwen3_5 import OmChatQwen3_5Config


OUTPUT_NAMES = ("feat_l06", "feat_l13", "feat_l19", "feat_l26")


def time_cuda(function, warmup: int = 3, repetitions: int = 10) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repetitions


def compare(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    got = actual.float().reshape(-1)
    delta = got - ref
    dot = torch.dot(ref, got).item()
    norm = math.sqrt(torch.dot(ref, ref).item() * torch.dot(got, got).item())
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": torch.sqrt(torch.mean(delta * delta)).item(),
        "cosine": dot / norm if norm else 1.0,
        "reference_abs_max": ref.abs().max().item(),
    }


def load_engine(path: Path) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {path}")
    return engine


def run_engine(
    engine: trt.ICudaEngine, tokens: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], callable]:
    context = engine.create_execution_context()
    if not context.set_input_shape("tokens", tuple(tokens.shape)):
        raise RuntimeError(f"Engine rejected input shape {tuple(tokens.shape)}")
    if engine.get_tensor_dtype("tokens") != trt.bfloat16:
        raise RuntimeError(f"Engine input is not BF16: {engine.get_tensor_dtype('tokens')}")
    outputs = tuple(torch.empty_like(tokens) for _ in OUTPUT_NAMES)
    context.set_tensor_address("tokens", tokens.data_ptr())
    for name, output in zip(OUTPUT_NAMES, outputs):
        if engine.get_tensor_dtype(name) != trt.bfloat16:
            raise RuntimeError(f"Engine output {name} is not BF16")
        context.set_tensor_address(name, output.data_ptr())

    def execute() -> None:
        ok = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

    execute()
    torch.cuda.synchronize()
    return outputs, execute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="266,3610,8170")
    parser.add_argument("--engine", default="universal_dynamic.bf16.plan")
    args = parser.parse_args()

    config = OmChatQwen3_5Config.from_pretrained(args.model_path)
    tower = build_vision_tower_aux(config, str(args.model_path))
    load_report = load_aux_state(args.model_path, tower)
    indices = tuple(int(value) for value in tower._out_indices_tuple)
    model = AuxBlockStack(
        tower.image_tower.radio_model.model.blocks, indices
    ).eval().to(device="cuda", dtype=torch.bfloat16)

    engine = load_engine(args.artifact_root / args.engine)
    rows = []
    torch.manual_seed(20260916)
    for length in (int(item) for item in args.lengths.split(",")):
        tokens = torch.randn(1, length, 1152, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            reference = model(tokens)
            actual, execute = run_engine(engine, tokens)
            metrics = [compare(ref, got) for ref, got in zip(reference, actual)]
            repetitions = 3 if length > 8170 else 10
            pytorch_ms = time_cuda(lambda: model(tokens), repetitions=repetitions)
            trt_ms = time_cuda(execute, repetitions=repetitions)
        rows.append(
            {
                "length": length,
                "engine": args.engine,
                "pytorch_timm_sdpa_bf16_ms": pytorch_ms,
                "tensorrt_bf16_ms": trt_ms,
                "speedup": pytorch_ms / trt_ms,
                "outputs": dict(zip(OUTPUT_NAMES, metrics)),
            }
        )

    report = {
        "load": load_report,
        "tap_indices_zero_based": list(indices),
        "rows": rows,
        "acceptance": {
            "minimum_per_tap_cosine": min(
                item["cosine"] for row in rows for item in row["outputs"].values()
            ),
            "all_cosine_at_least_0_999": all(
                item["cosine"] >= 0.999
                for row in rows
                for item in row["outputs"].values()
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
