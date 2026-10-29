#!/usr/bin/env python3
"""Smoke-test Humming NVFP4 GEMM on Qwen3.5-sized dense layers."""

from __future__ import annotations

import argparse
import json

import torch
import triton

from humming.forward import may_process_input
from humming.layer import HummingLayer
from vllm._custom_ops import scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


def benchmark(shape_m: int, shape_n: int, shape_k: int) -> dict[str, float | int]:
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 16,
            "scale_dtype": "float8e4m3",
            "weight_scale_2_type": "tensor",
        },
        input_config={"dtype": "float4e2m1", "group_size": 16},
        torch_dtype=torch.bfloat16,
    ).cuda()
    source_weight = torch.randn(
        (shape_n, shape_k), dtype=torch.bfloat16, device="cuda"
    ) / (shape_k**0.5)
    layer.load_from_unquantized(source_weight)
    layer.transform()

    global_scale = torch.ones((), dtype=torch.float32, device="cuda")
    fi_weight, fi_weight_scale = scaled_fp4_quant(
        source_weight,
        global_scale,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )

    inputs = torch.randn((shape_m, shape_k), dtype=torch.bfloat16, device="cuda")
    inputs_fp4, group_scales, token_scales = may_process_input(layer.humming_config, inputs)
    input_scale = group_scales if group_scales is not None else token_scales
    if token_scales is not None:
        input_scale = token_scales.unsqueeze(-1)

    def kernel_only():
        return layer(inputs_fp4, input_scale=input_scale)

    fi_inputs, fi_input_scale = scaled_fp4_quant(
        inputs,
        global_scale,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )

    def flashinfer_kernel_only():
        return flashinfer_scaled_fp4_mm(
            fi_inputs,
            fi_weight,
            fi_input_scale,
            fi_weight_scale,
            global_scale,
            torch.bfloat16,
            backend="cutlass",
        )

    output = kernel_only()
    torch.cuda.synchronize()
    if output.shape != (shape_m, shape_n) or not torch.isfinite(output).all():
        finite_ratio = torch.isfinite(output).float().mean().item()
        raise RuntimeError(
            f"invalid output for M={shape_m}, N={shape_n}, K={shape_k}: "
            f"shape={tuple(output.shape)}, finite_ratio={finite_ratio}"
        )

    fi_output = flashinfer_kernel_only()
    torch.cuda.synchronize()
    if fi_output.shape != (shape_m, shape_n) or not torch.isfinite(fi_output).all():
        raise RuntimeError(f"invalid FlashInfer output for M={shape_m}, N={shape_n}, K={shape_k}")

    humming_kernel_ms = float(triton.testing.do_bench(kernel_only, warmup=50, rep=300))
    flashinfer_kernel_ms = float(
        triton.testing.do_bench(flashinfer_kernel_only, warmup=50, rep=300)
    )

    def humming_end_to_end():
        quanted, scales, token = may_process_input(layer.humming_config, inputs)
        scale = scales if scales is not None else token
        if token is not None:
            scale = token.unsqueeze(-1)
        return layer(quanted, input_scale=scale)

    def flashinfer_end_to_end():
        quanted, scales = scaled_fp4_quant(
            inputs,
            global_scale,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
        )
        return flashinfer_scaled_fp4_mm(
            quanted,
            fi_weight,
            scales,
            fi_weight_scale,
            global_scale,
            torch.bfloat16,
            backend="cutlass",
        )

    humming_e2e_ms = float(triton.testing.do_bench(humming_end_to_end, warmup=50, rep=300))
    flashinfer_e2e_ms = float(
        triton.testing.do_bench(flashinfer_end_to_end, warmup=50, rep=300)
    )

    return {
        "m": shape_m,
        "n": shape_n,
        "k": shape_k,
        "humming_kernel_ms": humming_kernel_ms,
        "flashinfer_kernel_ms": flashinfer_kernel_ms,
        "humming_e2e_ms": humming_e2e_ms,
        "flashinfer_e2e_ms": flashinfer_e2e_ms,
        "kernel_speedup_vs_flashinfer": flashinfer_kernel_ms / humming_kernel_ms,
        "e2e_speedup_vs_flashinfer": flashinfer_e2e_ms / humming_e2e_ms,
        "humming_kernel_tops": 2 * shape_m * shape_n * shape_k / humming_kernel_ms / 1e9,
        "flashinfer_kernel_tops": 2 * shape_m * shape_n * shape_k / flashinfer_kernel_ms / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 27, 128, 2200])
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument(
        "--shapes",
        nargs="+",
        help="Optional N×K pairs such as 12288x2048 2048x6144",
    )
    args = parser.parse_args()

    shapes = [(args.n, args.k)]
    if args.shapes:
        shapes = [tuple(map(int, value.lower().split("x", 1))) for value in args.shapes]

    payload = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": "NVFP4 W4A4, group_size=16, BF16 output",
        "results": [benchmark(m, n, k) for n, k in shapes for m in args.m],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
