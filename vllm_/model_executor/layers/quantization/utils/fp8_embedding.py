# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Row-major FP8 weight-only kernels for tied token embeddings."""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fp8_rowwise_linear_kernel(
    input_ptr,
    weight_bits_ptr,
    scale_ptr,
    fp8_lut_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = input_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    weight_ptrs = (
        weight_bits_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k_start * BLOCK_K + offs_k < K
        input_block = tl.load(
            input_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        weight_bits = tl.load(
            weight_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0,
        )
        weight_block = tl.load(fp8_lut_ptr + weight_bits).to(input_block.dtype)
        accumulator = tl.dot(input_block, weight_block, accumulator)
        input_ptrs += BLOCK_K * stride_ak
        weight_ptrs += BLOCK_K * stride_wk

    scale = tl.load(
        scale_ptr + offs_n * stride_sn,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    accumulator *= scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :].to(tl.float32)

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        output_ptrs,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _fp8_rowwise_linear(
    input_: torch.Tensor,
    weight_bits: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_lut: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_.device.type != "cuda":
        raise ValueError("FP8 row-wise linear requires CUDA input.")
    if weight_bits.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 FP8 storage, got {weight_bits.dtype}.")
    if fp8_lut.shape != (256,) or fp8_lut.dtype != input_.dtype:
        raise ValueError(
            f"Expected a 256-element {input_.dtype} FP8 lookup table, got "
            f"{tuple(fp8_lut.shape)} {fp8_lut.dtype}."
        )
    if weight_bits.ndim != 2 or weight_scale.shape != (weight_bits.shape[0], 1):
        raise ValueError(
            "Expected weight [out_features, in_features] and scale "
            f"[out_features, 1], got {tuple(weight_bits.shape)} and "
            f"{tuple(weight_scale.shape)}."
        )
    if input_.shape[-1] != weight_bits.shape[1]:
        raise ValueError(
            f"Input width {input_.shape[-1]} does not match weight width "
            f"{weight_bits.shape[1]}."
        )

    input_2d = input_.reshape(-1, input_.shape[-1]).contiguous()
    if input_2d.shape[0] == 0:
        return torch.empty(
            (*input_.shape[:-1], weight_bits.shape[0]),
            dtype=input_.dtype,
            device=input_.device,
        )

    m, k = input_2d.shape
    n = weight_bits.shape[0]
    output = torch.empty((m, n), dtype=input_.dtype, device=input_.device)

    if m <= 8:
        block_m, block_n, num_warps = 16, 64, 4
    elif m <= 32:
        block_m, block_n, num_warps = 32, 64, 4
    else:
        block_m, block_n, num_warps = 32, 128, 8
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    _fp8_rowwise_linear_kernel[grid](
        input_2d,
        weight_bits,
        weight_scale,
        fp8_lut,
        bias,
        output,
        m,
        n,
        k,
        input_2d.stride(0),
        input_2d.stride(1),
        weight_bits.stride(0),
        weight_bits.stride(1),
        weight_scale.stride(0),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return output.reshape(*input_.shape[:-1], n)


def _fp8_rowwise_linear_fake(
    input_: torch.Tensor,
    weight_bits: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_lut: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del weight_scale, fp8_lut, bias
    return torch.empty(
        (*input_.shape[:-1], weight_bits.shape[0]),
        dtype=input_.dtype,
        device=input_.device,
    )


direct_register_custom_op(
    op_name="fp8_rowwise_linear",
    op_func=_fp8_rowwise_linear,
    mutates_args=[],
    fake_impl=_fp8_rowwise_linear_fake,
    dispatch_key=current_platform.dispatch_key,
)


def fp8_rowwise_linear(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_lut: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Expected float8_e4m3fn weight, got {weight.dtype}.")
    return torch.ops.vllm.fp8_rowwise_linear(
        input_,
        weight.view(torch.uint8),
        weight_scale,
        fp8_lut,
        bias,
    )
