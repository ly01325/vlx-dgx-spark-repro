# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OmChat Qwen3.5-VL model configuration."""

from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config


class OmChatQwen3_5Config(Qwen3_5Config):
    """Config for OmChat models built on the Qwen3.5-VL backbone."""

    model_type = "omchat_qwen3_5"

    def __init__(
        self,
        mm_vision_tower: str | None = None,
        mm_vision_tower_aux: str | None = None,
        mm_projector_type: str = "identity",
        mm_projector_aux_type: str = "linear",
        mm_hidden_size: int | None = None,
        mm_object_hidden_size: int | None = None,
        mm_use_vision_tower_object_feature: bool = False,
        **kwargs,
    ):
        self.mm_vision_tower = mm_vision_tower
        self.mm_vision_tower_aux = mm_vision_tower_aux
        self.mm_projector_type = mm_projector_type
        self.mm_projector_aux_type = mm_projector_aux_type
        self.mm_hidden_size = mm_hidden_size
        self.mm_object_hidden_size = mm_object_hidden_size
        self.mm_use_vision_tower_object_feature = mm_use_vision_tower_object_feature
        super().__init__(**kwargs)
