# SPDX-License-Identifier: Apache-2.0
"""FlashRT NVFP4 quantization backend for Qwen3.5-style dense projections."""

from __future__ import annotations

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
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


class FlashRTNvFp4Config(QuantizationConfig):
    """Configuration for FlashRT's dynamic-activation NVFP4 Linear path."""

    _MLP_TARGETS = (".mlp.gate_up_proj", ".mlp.down_proj")
    _ATTN_TARGETS = (".self_attn.qkv_proj", ".self_attn.o_proj")

    def __init__(self, block_size: int = 16, scope: str = "mlp_full_attn"):
        super().__init__()
        if block_size != 16:
            raise ValueError(
                f"FlashRT NVFP4 requires block_size=16, got {block_size}"
            )
        if scope not in (
            "mlp_only",
            "mlp_full_attn",
            "language_all_linear",
            "all_linear",
        ):
            raise ValueError(f"Unsupported FlashRT NVFP4 scope: {scope}")
        self.block_size = block_size
        self.scope = scope

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "flashrt_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FlashRTNvFp4Config":
        return cls(
            block_size=int(config.get("block_size", config.get("group_size", 16))),
            scope=str(config.get("scope", "mlp_full_attn")),
        )

    def get_compile_hash(self) -> str:
        """Identify the parameter layout used by compiled model graphs."""
        return f"{self.get_name()}:block_size={self.block_size}:scope={self.scope}"

    def _is_quantized_linear(self, prefix: str) -> bool:
        if self.scope == "language_all_linear":
            # Multimodal Qwen3.5 builds the text stack under
            # ``model.language_model.model.layers``. Keep the shorter form for
            # frontends that omit the inner causal-LM ``model`` component.
            return (
                "language_model.model.layers." in prefix
                or "language_model.layers." in prefix
            )
        if self.scope == "all_linear":
            return not any(
                token in prefix for token in ("embed_tokens", "lm_head", "pos_embed")
            )
        if prefix.endswith(self._MLP_TARGETS):
            return True
        return self.scope == "mlp_full_attn" and prefix.endswith(
            self._ATTN_TARGETS
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, Attention):
            return BaseKVCacheMethod(self)
        if not isinstance(layer, LinearBase):
            return None
        if self._is_quantized_linear(prefix):
            return FlashRTNvFp4LinearMethod(self)
        return UnquantizedLinearMethod()


@register_weight_loader_v2_supported_method
class FlashRTNvFp4LinearMethod(LinearMethodBase):
    """Load packed FP4 weights and run BF16 -> FP4 -> BF16 Linear."""

    def __init__(self, quant_config: FlashRTNvFp4Config) -> None:
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
            raise ValueError(
                f"FlashRT NVFP4 requires BF16 model activations, got {params_dtype}"
            )
        if input_size_per_partition % self.quant_config.block_size:
            raise ValueError(
                "FlashRT NVFP4 input size must be divisible by "
                f"{self.quant_config.block_size}, got {input_size_per_partition}"
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs["weight_loader"]
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight_packed = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight_packed)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.block_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_global_scale = PerTensorScaleParameter(
            # FlashRT's NVFP4 path currently supports only unit global scale.
            # Initialize to the checkpoint invariant so a loader that omits
            # this scalar does not leave an uninitialized value behind.
            data=torch.ones(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not bool(envs.VLLM_USE_FLASHRT_OPS):
            raise RuntimeError(
                "flashrt_nvfp4 checkpoint requires VLLM_USE_FLASHRT_OPS=1"
            )
        if not torch.allclose(
            layer.weight_global_scale.float(),
            torch.ones_like(layer.weight_global_scale, dtype=torch.float32),
        ):
            raise ValueError(
                "FlashRT checkpoint uses unsupported non-unit weight_global_scale"
            )

        weight_sfb = _flashrt_ops.prepare_nvfp4_weight_scale(
            layer.weight_scale.contiguous()
        )
        layer.weight_sfb = Parameter(weight_sfb, requires_grad=False)
        # ``weight_packed`` is already a vLLM ModelWeightParameter registered
        # during create_weights(). Re-wrapping it with torch.nn.Parameter
        # breaks on newer Torch versions because ModelWeightParameter has a
        # custom __torch_dispatch__ detach path. Keep the parameter subclass
        # and normalize only its underlying storage.
        if not layer.weight_packed.is_contiguous():
            layer.weight_packed.data = layer.weight_packed.data.contiguous()
        layer.weight_packed.requires_grad_(False)
        del layer.weight_scale
        del layer.weight_global_scale
        logger.info_once("FlashRT NVFP4 Linear enabled")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _flashrt_ops.nvfp4_linear(
            x, layer.weight_packed, layer.weight_sfb, bias
        )

__all__ = ["FlashRTNvFp4Config", "FlashRTNvFp4LinearMethod"]
