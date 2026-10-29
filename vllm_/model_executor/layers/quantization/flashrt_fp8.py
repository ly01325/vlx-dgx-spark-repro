# SPDX-License-Identifier: Apache-2.0
"""FlashRT per-tensor FP8 backend for dense Qwen3.5 projections."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm import _flashrt_ops, envs
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


class FlashRTFp8Config(QuantizationConfig):
    _MLP_TARGETS = (".mlp.gate_up_proj", ".mlp.down_proj")
    _ATTN_TARGETS = (".self_attn.qkv_proj", ".self_attn.o_proj")

    def __init__(self, scope: str = "mlp_full_attn"):
        super().__init__()
        if scope not in ("mlp_only", "mlp_full_attn"):
            raise ValueError(f"Unsupported FlashRT FP8 scope: {scope}")
        self.scope = scope

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "flashrt_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FlashRTFp8Config":
        return cls(scope=str(config.get("scope", "mlp_full_attn")))

    def get_compile_hash(self) -> str:
        """Identify the parameter layout used by compiled model graphs."""
        return f"{self.get_name()}:scope={self.scope}"

    def _is_quantized_linear(self, prefix: str) -> bool:
        if prefix.endswith(self._MLP_TARGETS):
            return True
        return self.scope == "mlp_full_attn" and prefix.endswith(self._ATTN_TARGETS)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, Attention):
            return BaseKVCacheMethod(self)
        if not isinstance(layer, LinearBase):
            return None
        if self._is_quantized_linear(prefix):
            return FlashRTFp8LinearMethod(self)
        return UnquantizedLinearMethod()


@register_weight_loader_v2_supported_method
class FlashRTFp8LinearMethod(LinearMethodBase):
    """Load FP8 E4M3 weights and run FlashRT FP8 GEMM with BF16 output."""

    def __init__(self, quant_config: FlashRTFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size
        if params_dtype != torch.bfloat16:
            raise ValueError(f"FlashRT FP8 requires BF16 activations, got {params_dtype}")
        weight_loader = extra_weight_attrs["weight_loader"]
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ), input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not bool(envs.VLLM_USE_FLASHRT_OPS) or not _env_flag("VLLM_FLASHRT_FP8"):
            raise RuntimeError(
                "flashrt_fp8 checkpoint requires VLLM_USE_FLASHRT_OPS=1 and "
                "VLLM_FLASHRT_FP8=1"
            )
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"FlashRT FP8 expects E4M3 weights, got {layer.weight.dtype}")

        # Fused QKV and gate/up layers can have one source scale per logical
        # shard. FlashRT's per-tensor GEMM accepts one weight scale, so merge
        # the shards once after loading and requantize to a common scale.
        scales = layer.weight_scale.float().reshape(-1)
        widths = list(getattr(layer, "logical_widths", [layer.weight.shape[0]]))
        if len(scales) != len(widths):
            raise ValueError(f"FP8 scale count {len(scales)} != shard count {len(widths)}")
        dequant = layer.weight.float()
        start = 0
        for width, scale in zip(widths, scales):
            dequant[start:start + width].mul_(scale)
            start += width
        common_scale = dequant.abs().amax().clamp_min(1e-12) / 448.0
        requant = (dequant / common_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        # FlashRT's cuBLASLt wrapper treats its second operand as a row-major
        # [K, N] matrix. vLLM stores Linear weights as [N, K], so transpose
        # once after loading instead of paying for a transpose on every call.
        flashrt_weight = requant.transpose(0, 1).contiguous()
        replace_parameter(layer, "weight", flashrt_weight)
        replace_parameter(layer, "weight_scale", common_scale.reshape(1).contiguous())
        layer.weight.requires_grad_(False)
        layer.weight_scale.requires_grad_(False)
        logger.info_once("FlashRT FP8 Linear enabled (no fused operators)")

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _flashrt_ops.fp8_linear(x, layer.weight, layer.weight_scale, bias)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


__all__ = ["FlashRTFp8Config", "FlashRTFp8LinearMethod"]
