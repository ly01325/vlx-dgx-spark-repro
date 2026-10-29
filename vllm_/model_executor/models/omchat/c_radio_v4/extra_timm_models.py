# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal timm registrations needed for C-RADIOv4-SO400M inference."""

from __future__ import annotations

import torch
from torch import nn
from timm.models import PretrainedCfg, register_model
from timm.models.vision_transformer import (
    Block,
    LayerScale as TIMMLayerScale,
    VisionTransformer,
    _create_vision_transformer as _timm_create_vision_transformer,
)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.grandma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.grandma) if self.inplace else x * self.grandma


def _create_vision_transformer(*args, **kwargs):
    if kwargs.get("pretrained_cfg", None) is None:
        kwargs["pretrained_cfg"] = PretrainedCfg()

    model = _timm_create_vision_transformer(*args, **kwargs)
    for mod in model.modules():
        if isinstance(mod, Block):
            if isinstance(mod.ls1, TIMMLayerScale):
                new_ls = LayerScale(mod.ls1.gamma.shape[0], inplace=mod.ls1.inplace)
                new_ls.load_state_dict(mod.ls1.state_dict())
                mod.ls1 = new_ls
            if isinstance(mod.ls2, TIMMLayerScale):
                new_ls = LayerScale(mod.ls2.gamma.shape[0], inplace=mod.ls2.inplace)
                new_ls.load_state_dict(mod.ls2.state_dict())
                mod.ls2 = new_ls
    return model


@register_model
def vit_so400m_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    if pretrained:
        raise ValueError("There is no pretrained weights for vit_so400m_patch16_224")
    mlp_ratio = 4304 / 1152
    model_args = dict(
        patch_size=16, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=mlp_ratio
    )
    return _create_vision_transformer(
        "vit_so400m_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
