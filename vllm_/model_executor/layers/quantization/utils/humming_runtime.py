# SPDX-License-Identifier: Apache-2.0
"""Small Humming adapter for the vLLM version bundled with VLX.

This backports the linear integration without pulling a newer vLLM runtime.
It supports ModelOpt NVFP4 W4A4 checkpoint tensors and online W4A16
quantization for the (padded) language-model head.
"""

from __future__ import annotations

import json

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)


def _install_humming_tensors(
    layer: torch.nn.Module,
    *,
    weight_schema,
    input_schema,
    tensors: dict[str, torch.Tensor],
    shape_n: int,
    shape_k: int,
    pad_n_to_multiple: int,
    pad_k_to_multiple: int,
) -> None:
    from humming.layer import HummingLayerMethod
    from humming.transform import transform_humming_tensors

    meta = HummingLayerMethod.prepare_layer_meta(
        layer,
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=pad_n_to_multiple,
        pad_k_to_multiple=pad_k_to_multiple,
        has_bias=False,
        torch_dtype=layer.params_dtype,
    )
    input_scales = {
        name: tensors.pop(name)
        for name in ("input_scale", "input_scale_2")
        if name in tensors
    }
    transformed = transform_humming_tensors(meta, tensors)

    for name in (
        "weight",
        "weight_scale",
        "weight_scale_2",
        "weight_global_scale",
        "input_scale",
        "input_global_scale",
        "input_global_scale_inv",
        "alpha",
    ):
        if hasattr(layer, name):
            delattr(layer, name)
    for name, tensor in transformed.items():
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))
    for name, tensor in input_scales.items():
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))
    if "locks" not in layer._buffers:
        layer.register_buffer(
            "locks",
            torch.zeros(1024, dtype=torch.int32, device=layer.weight.device),
            persistent=False,
        )
    layer.humming_logical_size_n = shape_n
    layer.humming_layer_config = meta.to_str()
    layer.humming_compute_config = json.dumps(
        {"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"}
    )
    layer.humming_should_quantize = meta.input_quant_mode.should_quantize
    layer.humming_has_group_scale = meta.input_quant_mode.has_group_scale
    layer.humming_has_secondary_scale = meta.input_quant_mode.has_secondary_scale
    layer.humming_quant_mode = meta.input_quant_mode.value
    layer.humming_quant_dtype = str(meta.a_dtype)
    layer.humming_quant_group_size = meta.input_scale_group_size or None
    layer.humming_group_scale_dtype = (
        str(meta.as_dtype) if meta.as_dtype is not None else None
    )


def prepare_modelopt_nvfp4_humming(layer: torch.nn.Module) -> None:
    """Convert loaded ModelOpt NVFP4 W4A4 tensors to Humming format."""
    from humming.schema import BaseInputSchema, BaseWeightSchema

    source = dict(layer.named_parameters(recurse=False))
    device = layer.weight.device
    shape_n = layer.output_size_per_partition
    shape_k = layer.input_size_per_partition
    shape_n_stacks = list(layer.logical_widths)
    shape_k_stacks = [shape_k]

    weight_schema = BaseWeightSchema.from_config(
        {"quant_method": "modelopt", "quant_algo": "nvfp4"}
    )
    input_schema = BaseInputSchema.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "nvfp4",
            "dynamic": False,
            "group_size": 16,
        }
    )
    weight_schema, weight_tensors = weight_schema.convert_humming(
        tensors=source,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
        device=device,
    )
    input_schema, input_tensors = input_schema.convert_humming(
        tensors=source,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
        device=device,
    )
    _install_humming_tensors(
        layer,
        weight_schema=weight_schema,
        input_schema=input_schema,
        tensors=weight_tensors | input_tensors,
        shape_n=shape_n,
        shape_k=shape_k,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
    )


def apply_humming_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from humming import ops

    flat = x.reshape(-1, x.shape[-1])
    input_scale = getattr(layer, "input_scale", None)
    input_scale_2 = getattr(layer, "input_scale_2", None)
    if layer.humming_should_quantize:
        group_scales = input_scale if layer.humming_has_group_scale else None
        token_scales = (
            input_scale_2 if layer.humming_has_secondary_scale else input_scale
        )
        flat, group_scales, token_scales = ops.process_input(
            inputs=flat,
            quant_mode=layer.humming_quant_mode,
            quant_dtype=layer.humming_quant_dtype,
            quant_group_size=layer.humming_quant_group_size,
            group_scales=group_scales,
            group_scale_dtype=layer.humming_group_scale_dtype,
            token_scales=token_scales,
            use_pdl=False,
        )
        input_scale = group_scales if layer.humming_has_group_scale else token_scales
        input_scale_2 = token_scales if layer.humming_has_secondary_scale else None

    output = ops.humming_gemm(
        layer_config=layer.humming_layer_config,
        inputs=flat,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_scale_2=getattr(layer, "weight_scale_2", None),
        input_scale=input_scale,
        input_scale_2=input_scale_2,
        locks=layer.locks,
        compute_config=layer.humming_compute_config,
    )
    output = output[..., : layer.humming_logical_size_n]
    if bias is not None:
        output = output + bias
    return output.view(*x.shape[:-1], layer.humming_logical_size_n)


class HummingW4A16LMHeadMethod(UnquantizedEmbeddingMethod):
    """Online-quantize a loaded BF16 LM head to Humming NVFP4 W4A16."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from humming import dtypes
        from humming.config.enum import WeightScale2Type, WeightScaleType
        from humming.schema import BaseInputSchema, HummingWeightSchema

        source_weight = layer.weight.detach()
        shape_n, shape_k = source_weight.shape
        # This bundled vLLM does not retain the constructor's params_dtype on
        # VocabParallelEmbedding/ParallelLMHead; the loaded tensor is canonical.
        layer.params_dtype = source_weight.dtype
        weight_schema = HummingWeightSchema(
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_type=WeightScaleType.GROUP,
            weight_scale_2_type=WeightScale2Type.TENSOR,
        )
        input_schema = BaseInputSchema.from_config(
            {"quant_method": "humming", "dtype": "bfloat16"}
        )
        tensors = HummingWeightSchema.quant_tensor(
            source_weight, weight_schema, layer.params_dtype
        )
        _install_humming_tensors(
            layer,
            weight_schema=weight_schema,
            input_schema=input_schema,
            tensors=tensors,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_n_to_multiple=64,
            pad_k_to_multiple=128,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_humming_linear(layer, x, bias)
