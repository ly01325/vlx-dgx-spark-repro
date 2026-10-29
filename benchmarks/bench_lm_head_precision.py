#!/usr/bin/env python3
"""Compare BF16, FP8 W8A8, and Humming W4A16 for the VLX LM head."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import torch.nn.functional as F

from humming.layer import HummingLayer
from vllm._custom_ops import cutlass_scaled_mm, scaled_fp8_quant


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def bench(fn, warmup: int, rep: int) -> float:
    output = fn()
    torch.cuda.synchronize()
    if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
        raise RuntimeError(
            f"invalid output: shape={tuple(output.shape)}, dtype={output.dtype}, "
            f"finite={torch.isfinite(output).float().mean().item()}"
        )
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 27])
    parser.add_argument("--n", type=int, default=248182)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(20260915)
    weight_bf16 = (
        torch.randn((args.n, args.k), dtype=torch.bfloat16, device="cuda")
        / (args.k**0.5)
    )

    weight_fp8, weight_fp8_scale = scaled_fp8_quant(weight_bf16, scale=None)
    weight_fp8_t = weight_fp8.t()

    humming_n = ((args.n + 63) // 64) * 64
    humming_weight = F.pad(weight_bf16, (0, 0, 0, humming_n - args.n))
    humming = HummingLayer(
        shape_n=humming_n,
        shape_k=args.k,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 16,
            "scale_dtype": "float8e4m3",
            "weight_scale_2_type": "tensor",
        },
        input_config={"dtype": "bfloat16"},
        torch_dtype=torch.bfloat16,
    ).cuda()
    humming.load_from_unquantized(humming_weight)
    humming.transform()

    humming_weight_bytes = sum(
        tensor_bytes(value)
        for name, value in humming.state_dict().items()
        if name != "locks"
    )
    sizes = {
        "bf16_weight_bytes": tensor_bytes(weight_bf16),
        "fp8_weight_and_scale_bytes": tensor_bytes(weight_fp8)
        + tensor_bytes(weight_fp8_scale),
        "humming_w4a16_weight_and_scale_bytes": humming_weight_bytes,
    }

    results = []
    for m in args.m:
        inputs = torch.randn((m, args.k), dtype=torch.bfloat16, device="cuda")

        def bf16():
            return torch.nn.functional.linear(inputs, weight_bf16)

        def fp8_kernel_only():
            return cutlass_scaled_mm(
                inputs_fp8,
                weight_fp8_t,
                inputs_fp8_scale,
                weight_fp8_scale,
                torch.bfloat16,
            )

        def fp8_end_to_end():
            quanted, scale = scaled_fp8_quant(
                inputs, scale=None, use_per_token_if_dynamic=True
            )
            return cutlass_scaled_mm(
                quanted,
                weight_fp8_t,
                scale,
                weight_fp8_scale,
                torch.bfloat16,
            )

        def w4a16():
            return humming(inputs)[..., : args.n]

        inputs_fp8, inputs_fp8_scale = scaled_fp8_quant(
            inputs, scale=None, use_per_token_if_dynamic=True
        )

        bf16_ms = bench(bf16, args.warmup, args.rep)
        fp8_kernel_ms = bench(fp8_kernel_only, args.warmup, args.rep)
        fp8_e2e_ms = bench(fp8_end_to_end, args.warmup, args.rep)
        w4a16_ms = bench(w4a16, args.warmup, args.rep)

        results.append(
            {
                "m": m,
                "n": args.n,
                "k": args.k,
                "bf16_ms": bf16_ms,
                "fp8_kernel_only_ms": fp8_kernel_ms,
                "fp8_w8a8_e2e_ms": fp8_e2e_ms,
                "humming_w4a16_ms": w4a16_ms,
                "fp8_speedup_vs_bf16": bf16_ms / fp8_e2e_ms,
                "w4a16_speedup_vs_bf16": bf16_ms / w4a16_ms,
                "w4a16_speedup_vs_fp8": fp8_e2e_ms / w4a16_ms,
            }
        )

    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "problem": {
            "operation": "LM head projection",
            "weight_shape_nk": [args.n, args.k],
            "output_dtype": "bfloat16",
            "fp8": "W8A8 E4M3, per-tensor weight scale, dynamic per-token activation scale",
            "humming": "W4A16 E2M1 weight, group_size=16, BF16 activation",
            "warmup_ms": args.warmup,
            "rep_ms": args.rep,
        },
        "sizes": sizes,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
