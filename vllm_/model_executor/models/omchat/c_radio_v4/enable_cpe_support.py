# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import MethodType
from typing import List, Tuple, Union

import torch
from torch import nn
from timm.models import VisionTransformer, checkpoint_seq

from .forward_intermediates import forward_intermediates
from .vit_patch_generator import ViTPatchGenerator


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_generator(x)
    if getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


def _forward_intermediates_cpe(
    self: VisionTransformer,
    x: torch.Tensor,
    norm: bool = False,
    **kwargs,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    return forward_intermediates(
        self,
        patch_extractor=self.patch_generator,
        num_summary_tokens=self.patch_generator.num_skip,
        num_cls_tokens=self.patch_generator.num_cls_tokens,
        norm=self.norm if norm else lambda y: y,
        x=x,
        **kwargs,
    )


def _enable_cpe_for_timm_vit(
    model: VisionTransformer,
    max_img_size: Union[int, Tuple[int, int]] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: int | None = None,
    num_registers: int | None = None,
):
    if not isinstance(model, VisionTransformer):
        raise ValueError("CPE only support for VisionTransformer models!")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
    )

    model.patch_generator = patch_generator
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.patch_size = patch_size
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.forward_features = MethodType(_forward_cpe, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe, model)


def enable_cpe(model: nn.Module, *args, **kwargs):
    if isinstance(model, VisionTransformer):
        _enable_cpe_for_timm_vit(model, *args, **kwargs)
        return
    raise ValueError(f"CPE not supported for this model type: {type(model)}")
