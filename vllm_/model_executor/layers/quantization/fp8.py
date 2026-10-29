# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    maybe_post_process_fp8_weight_block,
    process_fp8_weight_channel_strategy,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    gptq_pack,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    quantize_weights,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        ignored_layer_substrs: list[str] | None = None,
        weight_only_layer_substrs: list[str] | None = None,
        int4_weight_only_layer_substrs: list[str] | None = None,
        int4_group_size: int = 128,
        quantize_embeddings: bool = False,
        weight_block_size: list[int] | None = None,
        weight_scale_strategy: str = "tensor",
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        weight_scale_strategy = os.getenv(
            "FP8_WEIGHT_SCALE_STRATEGY",
            weight_scale_strategy,
        ).strip()

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        if weight_scale_strategy not in {"tensor", "channel"}:
            raise ValueError(
                "Unsupported FP8 weight_scale_strategy "
                f"{weight_scale_strategy!r}; expected 'tensor' or 'channel'."
            )
        self.activation_scheme = activation_scheme
        self.weight_scale_strategy = weight_scale_strategy
        self.ignored_layers = ignored_layers or []
        self.ignored_layer_substrs = [
            *(ignored_layer_substrs or []),
            *[
                layer.strip()
                for layer in os.getenv("FP8_IGNORED_LAYER_SUBSTRS", "").split(",")
                if layer.strip()
            ],
        ]
        self.ignored_layer_substrs = [
            layer.strip()
            for layer in dict.fromkeys(self.ignored_layer_substrs)
            if layer.strip()
        ]
        self.weight_only_layer_substrs = [
            *(weight_only_layer_substrs or []),
            *[
                layer.strip()
                for layer in os.getenv("FP8_WEIGHT_ONLY_LAYER_SUBSTRS", "").split(",")
                if layer.strip()
            ],
        ]
        self.int4_weight_only_layer_substrs = [
            *(int4_weight_only_layer_substrs or []),
            *[
                layer.strip()
                for layer in os.getenv("FP8_INT4_WEIGHT_ONLY_LAYER_SUBSTRS", "").split(
                    ","
                )
                if layer.strip()
            ],
        ]
        self.int4_weight_only_layer_substrs = [
            layer.strip()
            for layer in dict.fromkeys(self.int4_weight_only_layer_substrs)
            if layer.strip()
        ]
        int4_group_size = int(os.getenv("FP8_INT4_GROUP_SIZE", str(int4_group_size)))
        if int4_group_size not in {32, 64, 128}:
            raise ValueError(
                "Unsupported int4_group_size "
                f"{int4_group_size}; expected one of 32, 64, 128."
            )
        self.int4_group_size = int4_group_size
        self.weight_only_layer_substrs = [
            layer.strip()
            for layer in dict.fromkeys(self.weight_only_layer_substrs)
            if layer.strip()
        ]
        quantize_embeddings_env = os.getenv("FP8_QUANTIZE_EMBEDDINGS")
        if quantize_embeddings_env is not None:
            quantize_embeddings = quantize_embeddings_env.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.quantize_embeddings = quantize_embeddings
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.use_deep_gemm: bool | None = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        ignored_layer_substrs = cls.get_from_keys_or(
            config, ["ignored_layer_substrs"], None
        )
        weight_only_layer_substrs = cls.get_from_keys_or(
            config, ["weight_only_layer_substrs"], None
        )
        int4_weight_only_layer_substrs = cls.get_from_keys_or(
            config, ["int4_weight_only_layer_substrs"], None
        )
        int4_group_size = cls.get_from_keys_or(config, ["int4_group_size"], 128)
        quantize_embeddings = cls.get_from_keys_or(
            config, ["quantize_embeddings"], False
        )
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        weight_scale_strategy = cls.get_from_keys_or(
            config, ["weight_scale_strategy"], "tensor"
        )
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            ignored_layer_substrs=ignored_layer_substrs,
            weight_only_layer_substrs=weight_only_layer_substrs,
            int4_weight_only_layer_substrs=int4_weight_only_layer_substrs,
            int4_group_size=int4_group_size,
            quantize_embeddings=quantize_embeddings,
            weight_block_size=weight_block_size,
            weight_scale_strategy=weight_scale_strategy,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if any(substr in prefix for substr in self.int4_weight_only_layer_substrs):
                if self.is_checkpoint_fp8_serialized:
                    return _create_gptq_marlin_linear_method(self.int4_group_size)
                return Fp8OnlineInt4LinearMethod(self.int4_group_size)
            if any(substr in prefix for substr in self.weight_only_layer_substrs):
                if self.is_checkpoint_fp8_serialized:
                    return Fp8WeightOnlyLinearMethod(self)
                return Fp8OnlineWeightOnlyLinearMethod(self)
            if any(substr in prefix for substr in self.ignored_layer_substrs):
                logger.info_once(
                    "Skipping FP8 quantization for layer %s because it matches "
                    "FP8_IGNORED_LAYER_SUBSTRS.",
                    prefix,
                )
                return UnquantizedLinearMethod()
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                online_method = Fp8OnlineLinearMethod(self)
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            else:
                offline_method = Fp8LinearMethod(self)
                offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return offline_method
        elif isinstance(layer, FusedMoE):
            if any(substr in prefix for substr in self.ignored_layer_substrs):
                logger.info_once(
                    "Skipping FP8 quantization for layer %s because it matches "
                    "FP8_IGNORED_LAYER_SUBSTRS.",
                    prefix,
                )
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                moe_quant_method = Fp8MoEMethod(self, layer)
            else:
                moe_quant_method = Fp8OnlineMoEMethod(self, layer)
            return moe_quant_method
        elif isinstance(layer, VocabParallelEmbedding):
            if not self.quantize_embeddings:
                return None
            if self.is_checkpoint_fp8_serialized:
                return Fp8EmbeddingMethod(self)
            return Fp8OnlineEmbeddingMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class CopyNumelCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        if func == torch.ops.aten.copy_.default:
            self.copied_numel += args[0].numel()
        return out


def _copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


class Fp8WeightOnlyLinearMethod(LinearMethodBase):
    """Per-output-channel FP8 weights with BF16/FP16 activations."""

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                output_size_per_partition,
                input_size_per_partition,
                weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            create_fp8_scale_parameter(
                ChannelQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "FP8 weight-only checkpoint must store float8_e4m3fn weights, "
                f"but got {layer.weight.dtype}."
            )
        if layer.weight_scale.shape != (layer.weight.shape[0], 1):
            raise ValueError(
                "FP8 weight-only scales must have shape [out_features, 1], "
                f"but got {tuple(layer.weight_scale.shape)} for weight "
                f"{tuple(layer.weight.shape)}."
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)


class Fp8OnlineWeightOnlyLinearMethod(Fp8WeightOnlyLinearMethod):
    """Load BF16/FP16 weights and export per-channel FP8 weight-only layers."""

    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    device="meta",
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_info = torch.finfo(torch.float8_e4m3fn)
        weight = layer.weight.detach().float()
        weight_scale = (
            weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_info.max
        )
        qweight = (
            (weight / weight_scale)
            .clamp(
                min=fp8_info.min,
                max=fp8_info.max,
            )
            .to(torch.float8_e4m3fn)
        )
        replace_parameter(layer, "weight", qweight.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)
        layer._already_called_process_weights_after_loading = True


class _Fp8EmbeddingMixin:
    def process_weights_after_loading(self, layer: Module) -> None:
        super().process_weights_after_loading(layer)
        fp8_lut = (
            torch.arange(256, dtype=torch.uint8, device=layer.weight.device)
            .view(torch.float8_e4m3fn)
            .to(layer.orig_dtype)
        )
        if hasattr(layer, "_fp8_lut"):
            layer._fp8_lut = fp8_lut
        else:
            layer.register_buffer("_fp8_lut", fp8_lut, persistent=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_triton = os.getenv("FP8_EMBEDDING_USE_TRITON", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_triton and x.device.type == "cuda":
            from vllm.model_executor.layers.quantization.utils.fp8_embedding import (
                fp8_rowwise_linear,
            )

            return fp8_rowwise_linear(
                x,
                layer.weight,
                layer.weight_scale,
                layer._fp8_lut,
                bias,
            )
        return super().apply(layer, x, bias)

    def embedding(
        self,
        layer: torch.nn.Module,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.nn.functional.embedding(input_, layer.weight).to(
            layer.orig_dtype
        )
        scale = torch.nn.functional.embedding(input_, layer.weight_scale).to(
            layer.orig_dtype
        )
        return weight * scale


class Fp8EmbeddingMethod(_Fp8EmbeddingMixin, Fp8WeightOnlyLinearMethod):
    """Tied row-wise FP8 token embedding and language-model head."""


class Fp8OnlineEmbeddingMethod(
    _Fp8EmbeddingMixin,
    Fp8OnlineWeightOnlyLinearMethod,
):
    """Online exporter for tied row-wise FP8 embeddings."""


def _create_gptq_marlin_linear_method(group_size: int) -> LinearMethodBase:
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinConfig,
        GPTQMarlinLinearMethod,
    )

    config = GPTQMarlinConfig(
        weight_bits=4,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={
            "quant_method": "gptq",
            "bits": 4,
            "group_size": group_size,
            "desc_act": False,
            "sym": True,
            "lm_head": False,
        },
        modules_in_block_to_quantize=[],
    )
    return GPTQMarlinLinearMethod(config)


class Fp8OnlineInt4LinearMethod(LinearMethodBase):
    """Online exporter for symmetric group-wise INT4 W4A16 layers."""

    uses_meta_device: bool = True

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.pack_factor = 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    device="meta",
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.input_size_per_partition % self.group_size != 0:
            raise ValueError(
                "INT4 group-wise quantization requires input size "
                f"{layer.input_size_per_partition} to be divisible by "
                f"group size {self.group_size}."
            )
        if layer.input_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                "INT4 GPTQ packing requires input size "
                f"{layer.input_size_per_partition} to be divisible by "
                f"pack factor {self.pack_factor}."
            )
        if layer.output_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                "INT4 GPTQ zero-point packing requires output size "
                f"{layer.output_size_per_partition} to be divisible by "
                f"pack factor {self.pack_factor}."
            )

        weight = layer.weight.detach().to(torch.float32).t().contiguous()
        size_k, size_n = weight.shape
        _, qweight_unpacked, scales, _ = quantize_weights(
            weight,
            scalar_types.uint4b8,
            self.group_size,
        )
        qweight = gptq_pack(
            qweight_unpacked,
            num_bits=4,
            size_k=size_k,
            size_n=size_n,
        ).contiguous()
        num_groups = size_k // self.group_size
        qzeros = torch.zeros(
            num_groups,
            size_n // self.pack_factor,
            dtype=torch.int32,
            device=qweight.device,
        )
        g_idx = torch.arange(
            size_k,
            dtype=torch.int32,
            device=qweight.device,
        ).div_(self.group_size, rounding_mode="floor")

        del layer._parameters["weight"]
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(qweight, requires_grad=False),
        )
        layer.register_parameter(
            "scales",
            torch.nn.Parameter(scales.to(layer.orig_dtype), requires_grad=False),
        )
        layer.register_parameter(
            "qzeros",
            torch.nn.Parameter(qzeros, requires_grad=False),
        )
        layer.register_parameter(
            "g_idx",
            torch.nn.Parameter(g_idx, requires_grad=False),
        )
        layer._already_called_process_weights_after_loading = True

        if os.getenv("FP8_EXPORT_CANONICAL", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return

        raise RuntimeError(
            "Online INT4 is intended for checkpoint export. Export with "
            "FP8_EXPORT_CANONICAL=1, then reload the offline checkpoint to use "
            "GPTQ-Marlin W4A16 kernels."
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "Online INT4 layers must be exported and reloaded before inference."
        )


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None

        self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enabled()
        if self.quant_config.use_deep_gemm is not None:
            self.use_deep_gemm = self.quant_config.use_deep_gemm
        else:
            self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        self.weight_scale_strategy = self.quant_config.weight_scale_strategy

        # Use per-token quantization for better perf if dynamic and cutlass
        if self.act_q_static:
            activation_quant_key = kFp8StaticTensorSym
        elif cutlass_fp8_supported():
            activation_quant_key = kFp8DynamicTokenSym
        else:
            activation_quant_key = kFp8DynamicTensorSym

        if self.block_quant:
            weight_quant_key = kFp8Static128BlockSym
        else:
            weight_quant_key = kFp8StaticTensorSym

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=activation_quant_key,
            weight_quant_key=weight_quant_key,
            out_dtype=torch.get_default_dtype(),
            module_name=self.__class__.__name__,
        )
        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

        if self.block_quant and not self.use_marlin:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(*self.weight_block_size),
                act_quant_group_shape=GroupShape(1, self.weight_block_size[0]),
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported,
                use_deep_gemm=self.use_deep_gemm,
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if not self.block_quant:
            parameter_type = (
                ChannelQuantScaleParameter
                if self.weight_scale_strategy == "channel"
                else PerTensorScaleParameter
            )
            scale = create_fp8_scale_parameter(
                parameter_type,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
            )
            # The weight_scale_inv name is intentional for deepseekv3
            layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.use_marlin:
            if not self.block_quant:
                if self.weight_scale_strategy == "channel":
                    weight, weight_scale, input_scale = (
                        process_fp8_weight_channel_strategy(
                            layer.weight,
                            layer.weight_scale,
                            getattr(layer, "input_scale", None),
                        )
                    )
                else:
                    weight, weight_scale, input_scale = (
                        process_fp8_weight_tensor_strategy(
                            layer.weight,
                            layer.weight_scale,
                            layer.logical_widths,
                            getattr(layer, "input_scale", None),
                        )
                    )
                if self.act_q_static:
                    assert input_scale is not None
                    input_scale = input_scale.max()
                    replace_parameter(layer, "input_scale", input_scale)
                replace_parameter(layer, "weight", weight.data)
                replace_parameter(layer, "weight_scale", weight_scale.data)

            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
            return

        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static

            weight, weight_scale_inv = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale_inv
            )

            # Update layer with new values
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)

        # If checkpoint not serialized fp8, quantize the weights.
        else:
            # If checkpoint is fp8 per-tensor, handle that there are N scales for N
            # shards in a fused module
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            if self.weight_scale_strategy == "channel":
                weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                    weight,
                    weight_scale,
                    getattr(layer, "input_scale", None),
                )
            else:
                weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                    weight,
                    weight_scale,
                    layer.logical_widths,
                    getattr(layer, "input_scale", None),
                )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

            # Update layer with new values.
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        if self.block_quant and self.use_deep_gemm:
            maybe_post_process_fp8_weight_block(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled, prefer DeepGEMM FP8 path
        # we will use BF16 dequant when DeepGEMM is not supported.
        if envs.VLLM_BATCH_INVARIANT:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.w8a8_block_fp8_linear.apply(
                    input=x,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=layer.input_scale,
                    bias=bias,
                )
            else:
                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        if self.use_marlin:
            return self.fp8_linear.apply_weights(layer, x, bias)

        if self.block_quant:
            assert self.weight_block_size is not None

            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
            )

        return self.fp8_linear.apply_weights(layer, x, bias)


class Fp8OnlineLinearMethod(Fp8LinearMethod):
    """Online version of Fp8LinearMethod which loads a full precision checkpoint
    and quantizes weights during loading."""

    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # TODO(future): support block_quant in online quant path
        assert not self.block_quant

        layer.input_scale = None
        if self.weight_scale_strategy == "channel":
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            weight = layer.weight.detach().float()
            weight_scale = (
                weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_info.max
            )
            qweight = (
                (weight / weight_scale)
                .clamp(
                    min=fp8_info.min,
                    max=fp8_info.max,
                )
                .to(torch.float8_e4m3fn)
            )
        else:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

        # Update layer with new values.
        replace_parameter(layer, "weight", qweight.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        if os.getenv("FP8_EXPORT_CANONICAL", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            layer._already_called_process_weights_after_loading = True
            return

        if self.use_marlin:
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
        else:
            weight = qweight.t()
            replace_parameter(layer, "weight", weight.data)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = (
                kFp8StaticTensorSym
                if self.quant_config.activation_scheme == "static"
                else kFp8DynamicTensorSym
            )

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        assert self.quant_config.is_checkpoint_fp8_serialized
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=layer.orig_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # For per-tensor quant, the scales are per expert and weight.
            w13_scale_data = torch.ones(num_experts, 2, dtype=torch.float32)
            w2_scale_data = torch.ones(num_experts, dtype=torch.float32)
        else:
            # For block quant, the scales are per block (typically 128x128).
            w13_scale_data = torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_scale_data = torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            )
        w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
        # Note: name is weight_scale for tensor, weight_scale_inv for block.
        layer.register_parameter(f"w13_{self.weight_scale_name}", w13_weight_scale)
        layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _setup_kernel(
        self,
        layer: FusedMoE,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13,
                w13_scale,
                w13_input_scale,
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2,
                w2_scale,
                w2_input_scale,
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per tensor kernels require single weight scale for w13 per expert, but
        # on disk there is a scale for w1 and w3. Use the max to requantize.
        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        quant_config = make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.weight_block_size,
        )

        # Inject biases into the quant config if the model has them
        # (e.g. GPT-OSS biased MoE)
        if quant_config is not None and self.moe.has_bias:
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if w13_bias is not None:
                quant_config._w1.bias = w13_bias
            if w2_bias is not None:
                quant_config._w2.bias = w2_bias

        return quant_config

    @property
    def supports_eplb(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )


class Fp8OnlineMoEMethod(Fp8MoEMethod):
    """MoE method for online FP8 quantization.
    Supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)
        assert not quant_config.is_checkpoint_fp8_serialized
        assert quant_config.activation_scheme == "dynamic"
        assert quant_config.weight_block_size is None

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO(@ksayers): inplace fp8 quant kernel, initialize scales with ones
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        w13_scale = torch.ones(
            layer.num_experts, device=w13.device, dtype=torch.float32
        )
        w2_scale = torch.ones(layer.num_experts, device=w2.device, dtype=torch.float32)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = ops.scaled_fp8_quant(
                layer.w13_weight[expert, :, :]
            )
            w2[expert, :, :], w2_scale[expert] = ops.scaled_fp8_quant(
                layer.w2_weight[expert, :, :]
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
        )

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
