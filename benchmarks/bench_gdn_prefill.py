#!/usr/bin/env python3
"""Correctness and latency comparison for VLX Qwen3.5 GDN prefill kernels."""

from __future__ import annotations

import argparse
import json

import torch
import triton

from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.triton_utils.allocation import set_triton_allocator


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    a = actual.float().flatten()
    b = reference.float().flatten()
    diff = (a - b).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rmse": torch.mean((a - b) ** 2).sqrt().item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


def run_case(
    lengths: list[int], heads: int, dim: int, warmup: int, repetitions: int
) -> dict:
    torch.manual_seed(20260915 + sum(lengths))
    total = sum(lengths)
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, total, heads, dim, dtype=dtype, device=device) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1
    g = -(torch.rand(1, total, heads, dtype=torch.float32, device=device) * 0.08)
    beta = torch.sigmoid(
        torch.randn(1, total, heads, dtype=torch.float32, device=device)
    )
    initial_state = torch.zeros(
        len(lengths), heads, dim, dim, dtype=torch.float32, device=device
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )

    def flashinfer():
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    def triton_fla():
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    fi_output, fi_state = flashinfer()
    ref_output, ref_state = triton_fla()
    torch.cuda.synchronize()
    fi_ms = float(
        triton.testing.do_bench(flashinfer, warmup=warmup, rep=repetitions)
    )
    fla_ms = float(
        triton.testing.do_bench(triton_fla, warmup=warmup, rep=repetitions)
    )
    return {
        "lengths": lengths,
        "total_tokens": total,
        "output": metrics(fi_output, ref_output),
        "final_state": metrics(fi_state, ref_state),
        "flashinfer_ms": fi_ms,
        "triton_fla_ms": fla_ms,
        "speedup": fla_ms / fi_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    set_triton_allocator(torch.device("cuda"))
    cases = [[128], [512], [2200], [700, 1500]]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "heads": args.heads,
        "head_dim": args.dim,
        "cases": [
            run_case(
                lengths,
                args.heads,
                args.dim,
                args.warmup,
                args.repetitions,
            )
            for lengths in cases
        ],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
