# SPDX-License-Identifier: Apache-2.0
"""Optional FlashRT-backed CUDA operators.

FlashRT exposes its kernels as pointer-oriented pybind functions rather than
``torch.ops``.  This module is the small adapter layer needed by vLLM:
register tensor-shaped custom ops, pass vLLM's current CUDA stream through to
FlashRT, and keep the normal vLLM implementation available as a fallback.

The adapter is intentionally opt-in.  Set ``VLLM_USE_FLASHRT_OPS=1`` and make
the FlashRT source tree importable (``VLLM_FLASHRT_PATH`` or the default
``/workspace/FlashRT``).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _flashrt_root() -> str:
    return os.getenv("VLLM_FLASHRT_PATH", "/workspace/FlashRT")


def _ensure_flashrt_path() -> None:
    root = _flashrt_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _load_kernels() -> Any:
    """Load the generic FlashRT kernel module once."""
    _ensure_flashrt_path()
    from flash_rt import flash_rt_kernels as fvk
    return fvk


@lru_cache(maxsize=1)
def _load_fp4_module() -> Any:
    """Load the separately built NVFP4 extension."""
    _ensure_flashrt_path()
    from flash_rt import flash_rt_fp4

    if not flash_rt_fp4.has_nvfp4():
        raise RuntimeError("flash_rt_fp4.so was built without NVFP4 support")
    return flash_rt_fp4


def _stream(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _nvfp4_linear_impl(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_sfb: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise RuntimeError("FlashRT NVFP4 Linear requires CUDA BF16 activations")
    if weight_packed.dtype != torch.uint8 or not weight_packed.is_cuda:
        raise RuntimeError("FlashRT NVFP4 packed weights must be CUDA uint8")
    if weight_sfb.dtype != torch.uint8 or not weight_sfb.is_cuda:
        raise RuntimeError("FlashRT NVFP4 SFB scales must be CUDA uint8")
    if x.ndim < 2 or weight_packed.ndim != 2:
        raise RuntimeError(
            "FlashRT NVFP4 Linear expects x[..., K] and packed weight[N, K/2]"
        )

    fp4 = _load_fp4_module()
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # ``contiguous()`` may return the original tensor when it is a contiguous
    # view with a non-zero storage offset. FlashRT's vectorized BF16 quantizer
    # requires the source address to be 16-byte aligned, so materialize such
    # views into a fresh CUDA allocation before launching it.
    if x_2d.data_ptr() % 16:
        x_2d = x_2d.clone(memory_format=torch.contiguous_format)
    weight_packed = weight_packed.contiguous()
    weight_sfb = weight_sfb.contiguous()
    m, k = x_2d.shape
    n = weight_packed.shape[0]
    if weight_packed.shape[1] * 2 != k:
        raise RuntimeError(
            "FlashRT NVFP4 Linear dimension mismatch: "
            f"activation K={k}, packed weight K/2={weight_packed.shape[1]}"
        )
    if k % 16:
        raise RuntimeError("FlashRT NVFP4 Linear requires K divisible by 16")
    if bias is not None:
        if (
            bias.dtype != torch.bfloat16
            or not bias.is_cuda
            or not bias.is_contiguous()
            or bias.numel() != n
        ):
            raise RuntimeError(
                "FlashRT NVFP4 bias must be contiguous CUDA BF16 with N elements"
            )

    # A is dynamically quantized once per fused vLLM projection. QKV and
    # gate/up therefore each share one quantization launch and one GEMM.
    activation_packed = torch.empty(
        (m, k // 2), dtype=torch.uint8, device=x.device
    )
    activation_sfa = torch.zeros(
        fp4.sfa_size_bytes(m, k, False), dtype=torch.uint8, device=x.device
    )
    out_2d = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    stream = _stream(x.device)

    rc = fp4.quantize_fp4_dynamic_sfa_bf16_vec(
        x_2d.data_ptr(),
        activation_packed.data_ptr(),
        activation_sfa.data_ptr(),
        m,
        k,
        False,
        stream,
    )
    if rc != 0:
        raise RuntimeError(
            "FlashRT activation NVFP4 quantization failed: "
            f"rc={rc}, M={m}, K={k}, "
            f"src_alignment={x_2d.data_ptr() % 16}, "
            f"dst_alignment={activation_packed.data_ptr() % 8}"
        )

    _run_nvfp4_gemm(
        activation_packed.data_ptr(),
        activation_sfa.data_ptr(),
        weight_packed.data_ptr(),
        weight_sfb.data_ptr(),
        out_2d.data_ptr(),
        m,
        n,
        k,
        bias,
        stream,
    )
    return out_2d.reshape(*x.shape[:-1], n)


def _nvfp4_linear_fake(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_sfb: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight_packed.shape[0]))


def _fp8_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlashRT per-tensor W8A8 FP8 GEMM with BF16 output."""
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"FlashRT FP8 expects BF16 activation, got {x.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"FlashRT FP8 expects E4M3 weight, got {weight.dtype}")
    if weight_scale.numel() != 1:
        raise RuntimeError(
            f"FlashRT per-tensor FP8 expects one weight scale, got {weight_scale.numel()}"
        )
    fvk = _load_kernels()
    required = ("quantize_fp8_device", "fp8_gemm_descale_bf16out")
    if not all(hasattr(fvk, name) for name in required):
        raise RuntimeError(
            "flash_rt_kernels lacks the per-tensor FP8 API: "
            + ", ".join(required)
        )

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # FlashRT's cuBLASLt B operand is laid out as [K, N]. The quantization
    # method transposes vLLM's conventional [N, K] parameter once after load.
    weight = weight.contiguous()
    m, k = x_2d.shape
    if weight.shape[0] != k:
        raise RuntimeError(
            f"FlashRT FP8 dimension mismatch: activation K={k}, weight K={weight.shape[0]}"
        )
    n = weight.shape[1]
    activation_fp8 = torch.empty(
        (m, k), dtype=torch.float8_e4m3fn, device=x.device
    )
    activation_scale = torch.empty(1, dtype=torch.float32, device=x.device)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    stream = _stream(x.device)
    fvk.quantize_fp8_device(
        x_2d.data_ptr(), activation_fp8.data_ptr(), activation_scale.data_ptr(),
        m * k, stream
    )
    fvk.fp8_gemm_descale_bf16out(
        activation_fp8.data_ptr(), weight.data_ptr(), output.data_ptr(),
        m, n, k, activation_scale.data_ptr(), weight_scale.data_ptr(), stream
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*x.shape[:-1], n)


def _fp8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del weight_scale, bias
    return x.new_empty((*x.shape[:-1], weight.shape[1]))


def _run_nvfp4_gemm(
    activation_packed_ptr: int,
    activation_sfa_ptr: int,
    weight_packed_ptr: int,
    weight_sfb_ptr: int,
    output_ptr: int,
    m: int,
    n: int,
    k: int,
    bias: torch.Tensor | None,
    stream: int,
) -> None:
    """Dispatch the fastest measured Thor W4A4 kernel for the projection."""
    fvk = _load_kernels()
    if bias is None:
        regular = fvk.fp4_w4a16_gemm_sm120_bf16out
        widen = getattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_widen", regular)
        if n >= 8192:
            gemm = widen if m <= 128 else regular
        elif n >= 4096 or (m >= 256 and k >= 4096):
            gemm = widen
        else:
            gemm = regular
        gemm(
            activation_packed_ptr,
            weight_packed_ptr,
            output_ptr,
            m,
            n,
            k,
            activation_sfa_ptr,
            weight_sfb_ptr,
            1.0,
            stream,
        )
        return

    fp4 = _load_fp4_module()
    rc = fp4.cutlass_fp4_gemm_bias_bf16(
        activation_packed_ptr,
        activation_sfa_ptr,
        weight_packed_ptr,
        weight_sfb_ptr,
        bias.data_ptr(),
        output_ptr,
        m,
        n,
        k,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"FlashRT NVFP4 GEMM failed, rc={rc}")


def prepare_nvfp4_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Convert linear UE4M3 scales from the checkpoint to CUTLASS SFB."""
    if (
        weight_scale.dtype != torch.float8_e4m3fn
        or not weight_scale.is_cuda
        or weight_scale.ndim != 2
        or not weight_scale.is_contiguous()
    ):
        raise RuntimeError(
            "FlashRT NVFP4 weight_scale must be contiguous CUDA float8_e4m3fn"
        )
    rows, blocks = weight_scale.shape
    k = blocks * 16
    fp4 = _load_fp4_module()
    weight_sfb = torch.zeros(
        fp4.sfa_size_bytes(rows, k, True),
        dtype=torch.uint8,
        device=weight_scale.device,
    )
    rc = fp4.reshape_linear_scales_to_sfa(
        weight_scale.data_ptr(),
        weight_sfb.data_ptr(),
        rows,
        k,
        True,
        _stream(weight_scale.device),
    )
    if rc != 0:
        raise RuntimeError(f"FlashRT weight scale SFB conversion failed, rc={rc}")
    return weight_sfb


def _register_ops() -> None:
    # FlashRT is a CUDA-only backend.  Using an explicit key also keeps local
    # source-tree launches working when vLLM package metadata is unavailable
    # and platform detection temporarily falls back to UnspecifiedPlatform.
    dispatch_key = "CUDA"
    direct_register_custom_op(
        op_name="flashrt_nvfp4_linear",
        op_func=_nvfp4_linear_impl,
        fake_impl=_nvfp4_linear_fake,
        dispatch_key=dispatch_key,
    )
    direct_register_custom_op(
        op_name="flashrt_fp8_linear",
        op_func=_fp8_linear_impl,
        fake_impl=_fp8_linear_fake,
        dispatch_key=dispatch_key,
    )


_register_ops()


def nvfp4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_sfb: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flashrt_nvfp4_linear(
        x, weight_packed, weight_sfb, bias
    )


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flashrt_fp8_linear(x, weight, weight_scale, bias)
