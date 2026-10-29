# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build OmChat object-feature modules from vendored vLLM code."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

import vllm.envs as envs
from vllm import _flashrt_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper
from vllm.model_executor.models.omchat.c_radio_v4.hf_model import (
    RADIOConfig,
    RADIOModel,
)
from vllm.model_executor.models.omchat.c_radio_v4.feature_normalizer import (
    NullIntermediateFeatureNormalizer,
)
from vllm.model_executor.models.omchat.visual_prompt.multilevel_roi_visual_prompt import (
    MultiLevelROIVisualPrompt,
)

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.omchat_qwen3_5 import OmChatQwen3_5Config

logger = init_logger(__name__)

_DEFAULT_RADIO_CONFIG = os.path.join(
    os.path.dirname(__file__), "c_radio_v4", "radio_so400m_config.json"
)

_AUX_FP8_MLP_SUFFIXES = (".mlp.fc1", ".mlp.fc2")
_AUX_FP8_ATTN_SUFFIXES = (".attn.qkv", ".attn.proj")
_AUX_FP8_MARLIN_FALLBACK_SHAPES: set[tuple[int, int]] = set()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class AuxFlashAttention(nn.Module):
    """Run an existing timm attention module with vLLM FlashAttention."""

    def __init__(
        self,
        source: nn.Module,
        cu_seqlens_cache: dict[tuple[object, ...], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.attn_dim = source.attn_dim
        self.scale = source.scale
        self.qkv = source.qkv
        self.q_norm = source.q_norm
        self.k_norm = source.k_norm
        self.attn_drop = source.attn_drop
        self.norm = source.norm
        self.proj = source.proj
        self.proj_drop = source.proj_drop
        self._fa_version = get_flash_attn_version(head_size=self.head_dim)
        self._cu_seqlens_cache = (
            cu_seqlens_cache if cu_seqlens_cache is not None else {}
        )

    def _get_cached_cu_seqlens(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (device.type, device.index, batch_size, seq_len)
        cu_seqlens = self._cu_seqlens_cache.get(key)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=device,
            )
            self._cu_seqlens_cache[key] = cu_seqlens
        return cu_seqlens

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = self.q_norm(query)
        key = self.k_norm(key)

        dropout_p = self.attn_drop.p if self.training else 0.0
        use_flash_attn = (
            self._fa_version is not None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and attn_mask is None
            and not is_causal
            and dropout_p == 0.0
        )
        if use_flash_attn:
            if cu_seqlens is None:
                cu_seqlens = self._get_cached_cu_seqlens(
                    batch_size,
                    seq_len,
                    x.device,
                )
            hidden_states = vit_flash_attn_wrapper(
                query,
                key,
                value,
                batch_size=batch_size,
                is_rocm_aiter=False,
                fa_version=self._fa_version,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif cu_seqlens is not None:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            query_chunks = torch.split(query, lengths, dim=1)
            key_chunks = torch.split(key, lengths, dim=1)
            value_chunks = torch.split(value, lengths, dim=1)
            hidden_states = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        query_chunk.permute(0, 2, 1, 3),
                        key_chunk.permute(0, 2, 1, 3),
                        value_chunk.permute(0, 2, 1, 3),
                        dropout_p=dropout_p,
                        scale=self.scale,
                    ).transpose(1, 2)
                    for query_chunk, key_chunk, value_chunk in zip(
                        query_chunks,
                        key_chunks,
                        value_chunks,
                    )
                ],
                dim=1,
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.scale,
            ).transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, seq_len, self.attn_dim)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj(hidden_states)
        return self.proj_drop(hidden_states)


def _replace_aux_attention_with_flash(module: nn.Module) -> int:
    replaced = 0
    cu_seqlens_cache: dict[tuple[object, ...], torch.Tensor] = {}
    for block in module.blocks:
        attention = getattr(block, "attn", None)
        if isinstance(attention, AuxFlashAttention):
            attention._cu_seqlens_cache = cu_seqlens_cache
            replaced += 1
            continue
        required = (
            "num_heads",
            "head_dim",
            "attn_dim",
            "scale",
            "qkv",
            "q_norm",
            "k_norm",
            "attn_drop",
            "norm",
            "proj",
            "proj_drop",
        )
        if attention is None or any(
            not hasattr(attention, attr) for attr in required
        ):
            continue
        block.attn = AuxFlashAttention(attention, cu_seqlens_cache)
        replaced += 1
    return replaced


class AuxFp8Linear(nn.Module):
    """Weight-only FP8 Linear for OmChat aux vision tower modules."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        orig_dtype: torch.dtype = torch.bfloat16,
        marlin_eligible: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_partition = in_features
        self.output_size_per_partition = out_features
        self.logical_widths = [out_features]
        self.orig_dtype = orig_dtype
        self.marlin_eligible = marlin_eligible
        self.weight_block_size = None
        self._marlin_processed = False
        self.register_buffer("_dequant_weight", None, persistent=False)

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=orig_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def process_weights_after_loading(self) -> None:
        if self._marlin_processed:
            return
        use_marlin = _env_flag("OMCHAT_AUX_FP8_USE_MARLIN")
        if use_marlin and self.marlin_eligible and (
            self.weight.dtype == torch.float8_e4m3fn
            and is_fp8_marlin_supported()
            and not envs.VLLM_BATCH_INVARIANT
        ):
            try:
                self.input_scale = None
                prepare_fp8_layer_for_marlin(
                    self,
                    size_k_first=False,
                    input_dtype=self.orig_dtype,
                )
                self._marlin_processed = True
            except RuntimeError as exc:
                shape = (self.out_features, self.in_features)
                if shape not in _AUX_FP8_MARLIN_FALLBACK_SHAPES:
                    _AUX_FP8_MARLIN_FALLBACK_SHAPES.add(shape)
                    logger.warning(
                        "Falling back to BF16 dequant for aux FP8 Linear "
                        "with shape=(%d, %d): %s",
                        self.out_features,
                        self.in_features,
                        exc,
                    )
                self._marlin_processed = False
            finally:
                if hasattr(self, "input_scale"):
                    del self.input_scale

        if not self._marlin_processed:
            self._prepare_dequant_weight()

    def _prepare_dequant_weight(self) -> None:
        if not _env_flag("OMCHAT_AUX_FP8_CACHE_DEQUANT"):
            self._dequant_weight = None
            return
        if self.weight.dtype != torch.float8_e4m3fn:
            self._dequant_weight = None
            return
        self._dequant_weight = (
            self.weight.to(self.orig_dtype) * self.weight_scale.to(self.orig_dtype)
        ).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._marlin_processed:
            return apply_fp8_marlin_linear(
                input=x,
                weight=self.weight,
                weight_scale=self.weight_scale,
                workspace=self.workspace,
                size_n=self.out_features,
                size_k=self.in_features,
                input_dtype=self.orig_dtype,
                bias=self.bias,
            )

        weight = self._dequant_weight
        if weight is None or weight.dtype != x.dtype or weight.device != x.device:
            weight = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        return torch.nn.functional.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class FlashRTNvFp4TorchLinear(nn.Module):
    """FlashRT NVFP4 replacement for object modules using nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if in_features % 16:
            raise ValueError("FlashRT NVFP4 Linear requires K divisible by 16")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_packed = nn.Parameter(
            torch.empty(
                out_features,
                in_features // 2,
                device=device,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                out_features,
                in_features // 16,
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_global_scale = nn.Parameter(
            torch.empty(1, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer("weight_sfb", None, persistent=False)

    def process_weights_after_loading(self) -> None:
        if not torch.allclose(
            self.weight_global_scale.float(),
            torch.ones_like(self.weight_global_scale, dtype=torch.float32),
        ):
            raise ValueError("FlashRT NVFP4 object Linear requires unit global scale")
        self.weight_sfb = _flashrt_ops.prepare_nvfp4_weight_scale(
            self.weight_scale.contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_sfb is None:
            self.process_weights_after_loading()
        return _flashrt_ops.nvfp4_linear(
            x,
            self.weight_packed,
            self.weight_sfb,
            self.bias,
        )


def replace_torch_linear_with_flashrt_nvfp4(module: nn.Module) -> int:
    """Replace every compatible ordinary Linear below module."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0:
            quant_linear = FlashRTNvFp4TorchLinear(
                child.in_features,
                child.out_features,
                child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            setattr(module, child_name, quant_linear)
            replaced += 1
            continue
        replaced += replace_torch_linear_with_flashrt_nvfp4(child)
    return replaced


def process_flashrt_nvfp4_torch_linears(module: nn.Module | None) -> None:
    if module is None:
        return
    for submodule in module.modules():
        if isinstance(submodule, FlashRTNvFp4TorchLinear):
            submodule.process_weights_after_loading()


def _normalise_aux_fp8_targets(raw_targets: object) -> set[str]:
    if isinstance(raw_targets, str):
        targets = [target.strip() for target in raw_targets.split(",")]
    elif isinstance(raw_targets, (list, tuple, set)):
        targets = [str(target).strip() for target in raw_targets]
    else:
        targets = []
    targets = {target for target in targets if target}
    if "all" in targets:
        targets.update({"mlp", "attn"})
    return targets


def get_aux_vision_fp8_targets(config: OmChatQwen3_5Config) -> set[str]:
    quant_config = getattr(config, "aux_vision_quantization_config", None)
    if not isinstance(quant_config, dict):
        return set()
    if quant_config.get("quant_method") != "fp8":
        return set()
    return _normalise_aux_fp8_targets(quant_config.get("targets", ()))


def _is_aux_fp8_linear_target(name: str, targets: set[str]) -> bool:
    if "mlp" in targets and name.endswith(_AUX_FP8_MLP_SUFFIXES):
        return True
    if "attn" in targets and name.endswith(_AUX_FP8_ATTN_SUFFIXES):
        return True
    return False


def replace_aux_vision_linear_with_fp8(
    module: nn.Module,
    targets: set[str],
    prefix: str = "",
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _is_aux_fp8_linear_target(
            child_prefix, targets
        ):
            quant_linear = AuxFp8Linear(
                child.in_features,
                child.out_features,
                child.bias is not None,
                device=child.weight.device,
                orig_dtype=child.weight.dtype,
                # Marlin has no valid A100 runtime configuration for the
                # high-token C-RADIO MLP fc2 shape. Keep it on dequant GEMM.
                marlin_eligible=not child_prefix.endswith(
                    _AUX_FP8_MLP_SUFFIXES
                ),
            )
            setattr(module, child_name, quant_linear)
            replaced += 1
            continue
        replaced += replace_aux_vision_linear_with_fp8(
            child,
            targets,
            child_prefix,
        )
    return replaced


def process_aux_vision_fp8_modules(module: nn.Module | None) -> None:
    if module is None:
        return
    for submodule in module.modules():
        if isinstance(submodule, AuxFp8Linear):
            submodule.process_weights_after_loading()


def _infer_aux_spatial_scale(aux_tower, default: float = 0.25) -> float:
    if aux_tower is None:
        return default
    spatial_scale = getattr(aux_tower, "spatial_scale", None)
    if isinstance(spatial_scale, (int, float)):
        return float(spatial_scale)
    cfg = getattr(aux_tower, "config", None)
    if cfg is not None:
        patch_size = getattr(cfg, "patch_size", None)
        if isinstance(patch_size, int) and patch_size > 0:
            return 1.0 / patch_size
    return default


def build_mm_projector_aux(config: OmChatQwen3_5Config) -> nn.Module:
    projector_type = getattr(config, "mm_projector_aux_type", "linear")
    hidden_size = config.text_config.hidden_size
    in_size = config.mm_object_hidden_size

    if projector_type == "linear":
        return nn.Linear(in_size, hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        depth = int(mlp_gelu_match.group(1))
        layers: list[nn.Module] = [nn.Linear(in_size, hidden_size)]
        for _ in range(1, depth):
            layers.extend([nn.GELU(), nn.Linear(hidden_size, hidden_size)])
        return nn.Sequential(*layers)

    if projector_type == "identity":

        class _Identity(nn.Module):
            def forward(self, x, *args, **kwargs):
                return x

        return _Identity()

    raise ValueError(f"Unsupported mm_projector_aux_type: {projector_type}")


def _get_vt_attr(vision_tower, name: str, default=None):
    if hasattr(vision_tower, "config"):
        return getattr(vision_tower.config, name, default)
    return getattr(vision_tower, name, default)


def build_object_vp_extractor(
    config: OmChatQwen3_5Config,
    vision_tower,
    vision_tower_aux,
) -> nn.Module:
    vt_hidden = _get_vt_attr(vision_tower, "hidden_size")
    vt_out_hidden = _get_vt_attr(vision_tower, "out_hidden_size", vt_hidden)
    use_simple_fpn = getattr(config, "mm_use_simpleFPN_for_vt", False)
    use_simple_fpn_multi = getattr(config, "mm_use_simpleFPN_multiLayer_for_vt", False)
    vt_object_dim = (
        vt_out_hidden if use_simple_fpn and not use_simple_fpn_multi else vt_hidden * 4
    )

    use_vt_object_feature_only = getattr(
        config, "mm_use_vt_object_feature_only", False
    )
    aux_hidden = (
        getattr(vision_tower_aux.config, "hidden_size", None)
        if vision_tower_aux is not None
        else None
    )
    return MultiLevelROIVisualPrompt(
        output_size=getattr(config, "mm_roi_output_size", 7),
        # The aux scale is unused by the VT-only forward branch. Keep a
        # harmless default so the extractor can be built without C-RADIO.
        spatial_scale=(
            _infer_aux_spatial_scale(vision_tower_aux)
            if vision_tower_aux is not None
            else 1.0
        ),
        add_pos_embedding=getattr(config, "mm_add_pos_embed", True),
        pos_embedding_dim=config.mm_object_hidden_size,
        use_vision_tower_object_feature=getattr(
            config, "mm_use_vision_tower_object_feature", False
        ),
        vision_tower_object_feature_dim=vt_object_dim,
        vision_tower_spatial_scale=1.0
        / (
            _get_vt_attr(vision_tower, "patch_size")
            * _get_vt_attr(vision_tower, "spatial_merge_size")
        ),
        object_feature_combination=getattr(
            config, "mm_object_feature_combination", "mean"
        ),
        use_vt_object_feature_only=use_vt_object_feature_only,
        use_panet=getattr(config, "mm_use_panet", False),
        use_hybrid_encoder=getattr(config, "mm_use_hybrid_encoder", False),
        use_simpleFPN_for_vt=use_simple_fpn,
        use_simpleFPN_multiLayer_for_vt=use_simple_fpn_multi,
        use_separate_mlp_for_object=getattr(
            config, "mm_use_separate_mlp_for_object", False
        ),
        obj_pooling_type=getattr(config, "mm_obj_pooling_type", "mean"),
        pos_embedding_strategy=getattr(
            config, "mm_pos_embedding_strategy", "bbox_based"
        ),
        use_multi_scale_roi_align=getattr(
            config, "mm_use_multi_scale_roi_align", False
        ),
        apply_object_layer_norm=getattr(config, "mm_apply_object_layer_norm", False),
        roi_algined=getattr(config, "mm_roi_algined", False),
        use_simpleFPN_for_vt_aux=getattr(config, "mm_use_simpleFPN_for_vt_aux", False),
        use_simpleFPN_multiLayer_for_vt_aux=getattr(
            config, "mm_use_simpleFPN_multiLayer_for_vt_aux", False
        ),
        aux_feature_dims=[] if use_vt_object_feature_only else [aux_hidden],
        use_pos_embed_v2=getattr(config, "mm_use_pos_embed_v2", False),
        use_fourier_pos_embed=getattr(config, "mm_use_fourier_pos_embed", False),
        simpleFPN_out_channels_for_vt=getattr(
            config, "mm_simpleFPN_out_channels_for_vt", 512
        ),
    )


def _load_radio_config(config_path: str) -> RADIOConfig:
    with open(config_path, encoding="utf-8") as f:
        return RADIOConfig(**json.load(f))


def resolve_tower_path(model_dir: str, tower_path: str | None) -> str | None:
    if not tower_path:
        return None

    candidates = []
    if os.path.isabs(tower_path):
        candidates.append(tower_path)
    elif model_dir:
        candidates.append(os.path.join(model_dir, tower_path))
        candidates.append(os.path.join(model_dir, os.path.basename(tower_path)))
        candidates.append(os.path.join(model_dir, "C-RADIOv4-SO400M"))

    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def _resolve_radio_config_path(model_dir: str, tower_path: str | None) -> str | None:
    resolved_tower = resolve_tower_path(model_dir, tower_path)
    candidates = [
        os.environ.get("OMCHAT_RADIO_CONFIG"),
        _DEFAULT_RADIO_CONFIG,
    ]
    if resolved_tower:
        candidates.insert(0, os.path.join(resolved_tower, "config.json"))
    if model_dir:
        candidates.extend(
            [
                os.path.join(model_dir, "C-RADIOv4-SO400M", "config.json"),
                os.path.join(model_dir, "resources", "C-RADIOv4-SO400M", "config.json"),
            ]
        )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


class _AuxVisionCudaGraphRunner:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_input: torch.Tensor,
        static_outputs: list[torch.Tensor],
    ) -> None:
        self.graph = graph
        self.static_input = static_input
        self.static_outputs = static_outputs
        self._replay_lock = threading.Lock()
        self._replay_done = torch.cuda.Event()
        self._has_replayed = False

    def replay(self, image: torch.Tensor) -> list[torch.Tensor]:
        stream = torch.cuda.current_stream(image.device)
        with self._replay_lock:
            if self._has_replayed:
                stream.wait_event(self._replay_done)
            self.static_input.copy_(image)
            self.graph.replay()
            if _env_flag("OMCHAT_AUX_VIT_CUDAGRAPH_CLONE_OUTPUT", True):
                outputs = [output.clone() for output in self.static_outputs]
            else:
                outputs = self.static_outputs
            self._replay_done.record(stream)
            self._has_replayed = True
        return outputs


class OmChatVisionTowerAux(nn.Module):
    """Lightweight aux tower wrapper when weights come from the main checkpoint."""

    def __init__(self, radio_cfg: RADIOConfig) -> None:
        super().__init__()
        self.image_tower = RADIOModel(radio_cfg)
        backbone = self.image_tower.radio_model.model
        num_layers = len(backbone.blocks)
        requested_varlen_batch = _env_flag("OMCHAT_AUX_VARLEN_BATCH", True)
        use_flash_attention = _env_flag("OMCHAT_AUX_FLASH_ATTN", True)
        flash_attention_blocks = 0
        if use_flash_attention or requested_varlen_batch:
            flash_attention_blocks = _replace_aux_attention_with_flash(backbone)
            logger.info(
                "Enabled vLLM FlashAttention for %d/%d OmChat aux vision "
                "blocks (varlen_batch=%s).",
                flash_attention_blocks,
                num_layers,
                requested_varlen_batch,
            )
        supports_varlen_batch = radio_cfg.vitdet_window_size is None
        self._varlen_batch_enabled = (
            requested_varlen_batch
            and supports_varlen_batch
            and flash_attention_blocks == num_layers
        )
        if requested_varlen_batch and not self._varlen_batch_enabled:
            logger.warning(
                "OmChat aux varlen batching is unavailable "
                "(flash_attention_blocks=%d/%d, vitdet_window_size=%s); "
                "falling back to exact-shape groups.",
                flash_attention_blocks,
                num_layers,
                radio_cfg.vitdet_window_size,
            )
        self.out_indices = [round((i + 1) * num_layers / 4) - 1 for i in range(4)]
        self.out_indices[-1] = num_layers - 1
        self._out_indices_tuple = tuple(self.out_indices)
        self._backbone_blocks = tuple(backbone.blocks)
        self._single_block_parts = tuple(
            (
                block.norm1,
                block.attn,
                block.ls1,
                block.drop_path1,
                block.norm2,
                block.mlp,
                block.ls2,
                block.drop_path2,
            )
            for block in self._backbone_blocks
        )
        self._prefix_tokens = backbone.patch_generator.num_skip
        self._config = radio_cfg
        self._config.hidden_size = self.image_tower.radio_model.model.embed_dim
        self._compiled_forward_intermediates = None
        self._compile_failed = False
        self._compile_lock = threading.Lock()
        self._pure_tensor_validation_done = False
        self._pure_tensor_validation_lock = threading.Lock()
        self._stream_roi_validation_done = False
        self._stream_roi_validation_lock = threading.Lock()
        self._stream_roi_enabled_logged = False
        self._cudagraph_runners: dict[tuple[object, ...], _AuxVisionCudaGraphRunner] = {}
        self._cudagraph_failed_keys: set[tuple[object, ...]] = set()
        self._cudagraph_capture_lock = threading.Lock()
        self._cudagraph_capacity_warned = False
        self._max_input_pixels = max(0, _env_int("OMCHAT_AUX_MAX_PIXELS", 0))
        self._trt_block_router = None
        try:
            from vllm.model_executor.models.omchat.trt_aux_runtime import (
                create_aux_tensorrt_router,
            )

            self._trt_block_router = create_aux_tensorrt_router()
        except Exception:
            if _env_flag("OMCHAT_AUX_TRT_STRICT"):
                raise
            logger.exception(
                "TensorRT C-RADIO initialization failed; using PyTorch fallback."
            )
        if self._trt_block_router is not None:
            # TensorRT profiles are B=1. Heterogeneous image lists are handled
            # one image at a time and retain per-image dynamic token lengths.
            self._varlen_batch_enabled = False
            logger.info("Enabled dynamic-shape TensorRT routing for C-RADIO blocks.")
        if self._max_input_pixels:
            logger.info(
                "Capping OmChat aux vision inputs at %d pixels.",
                self._max_input_pixels,
            )
        if _env_flag("OMCHAT_PURE_TENSOR_AUX_VIT") or _env_flag(
            "OMCHAT_COMPILE_AUX_VIT"
        ):
            logger.info(
                "OmChat pure-tensor aux path configured "
                "(compile=%s, validate=%s, strict=%s).",
                _env_flag("OMCHAT_COMPILE_AUX_VIT"),
                _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT")
                or _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT_STRICT"),
                _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT_STRICT"),
            )
        self._configure_input_conditioner()

    def _configure_input_conditioner(self) -> None:
        conditioner = self.image_tower.radio_model.input_conditioner
        if conditioner is None or isinstance(conditioner, nn.Identity):
            return
        backbone_dtype = next(self.image_tower.radio_model.model.parameters()).dtype
        conditioner.to(dtype=torch.float32)
        conditioner.dtype = backbone_dtype

    @property
    def config(self):
        return self._config

    @property
    def varlen_batch_enabled(self) -> bool:
        return self._varlen_batch_enabled

    def _extract_feature_map(
        self,
        intermediates,
        input_size: tuple[int, int],
    ):
        if not isinstance(intermediates, (list, tuple)):
            intermediates = [intermediates]
        return {
            "image_features": list(intermediates),
            "input_size": input_size,
        }

    def _forward_intermediates_legacy(self, image: torch.Tensor):
        return self.image_tower.radio_model.forward_intermediates(
            image,
            indices=self.out_indices,
            return_prefix_tokens=False,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )

    def _prepare_token_inputs(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        radio = self.image_tower.radio_model
        backbone = radio.model
        conditioned = radio.input_conditioner(image)
        height, width = conditioned.shape[-2:]
        tokens = backbone.patch_generator(conditioned)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(
                "OmChat aux patch generator must return a tensor, got "
                f"{type(tokens)!r}"
            )
        return tokens, (height // backbone.patch_size, width // backbone.patch_size)

    def _forward_token_intermediates_eager(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._trt_block_router is not None:
            outputs = self._trt_block_router.run(tokens)
            if outputs is not None:
                prefix_tokens = self._prefix_tokens
                return tuple(output[:, prefix_tokens:] for output in outputs)  # type: ignore[return-value]

        backbone = self.image_tower.radio_model.model
        index0, index1, index2, index3 = self._out_indices_tuple
        output0 = output1 = output2 = output3 = None
        hidden_states = tokens
        for layer_index, block in enumerate(backbone.blocks):
            hidden_states = block(hidden_states)
            if layer_index == index0:
                output0 = hidden_states
            elif layer_index == index1:
                output1 = hidden_states
            elif layer_index == index2:
                output2 = hidden_states
            elif layer_index == index3:
                output3 = hidden_states

        if output0 is None or output1 is None or output2 is None or output3 is None:
            raise RuntimeError(
                "OmChat aux intermediate indices do not match the vision blocks"
            )
        prefix_tokens = self._prefix_tokens
        return (
            output0[:, prefix_tokens:],
            output1[:, prefix_tokens:],
            output2[:, prefix_tokens:],
            output3[:, prefix_tokens:],
        )

    @staticmethod
    def _scale_roi_boxes(
        boxes: list[torch.Tensor],
        *,
        original_size: tuple[int, int],
        prepared_size: tuple[int, int],
        device: torch.device,
    ) -> list[torch.Tensor]:
        original_h, original_w = original_size
        prepared_h, prepared_w = prepared_size
        scaled = []
        for item in boxes:
            item = item.to(device=device, dtype=torch.float32).reshape(-1, 4).clone()
            item[:, 0::2].mul_(prepared_w / original_w)
            item[:, 1::2].mul_(prepared_h / original_h)
            scaled.append(item)
        return scaled

    @staticmethod
    def _roi_align_tokens(
        hidden_states: torch.Tensor,
        *,
        grid_size: tuple[int, int],
        prefix_tokens: int,
        boxes: list[torch.Tensor],
        output_size: int,
        spatial_scale: float,
        aligned: bool,
    ) -> torch.Tensor:
        height, width = grid_size
        spatial_tokens = hidden_states[:, prefix_tokens:]
        feature = (
            spatial_tokens.reshape(hidden_states.shape[0], height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return roi_align(
            feature.float(),
            boxes,
            output_size=output_size,
            spatial_scale=spatial_scale,
            aligned=aligned,
        )

    def _forward_token_rois_eager(
        self,
        tokens: torch.Tensor,
        *,
        grid_size: tuple[int, int],
        boxes: list[torch.Tensor],
        output_size: int,
        spatial_scale: float,
        aligned: bool,
    ) -> torch.Tensor:
        prefix_tokens = self._prefix_tokens
        if self._trt_block_router is not None:
            outputs = self._trt_block_router.run(tokens)
            if outputs is not None:
                roi_features = [
                    self._roi_align_tokens(
                        output,
                        grid_size=grid_size,
                        prefix_tokens=prefix_tokens,
                        boxes=boxes,
                        output_size=output_size,
                        spatial_scale=spatial_scale,
                        aligned=aligned,
                    )
                    for output in outputs
                ]
                concatenated = torch.cat(roi_features, dim=1)
                return concatenated.mean(dim=(2, 3)).reshape(
                    1, concatenated.shape[0], concatenated.shape[1]
                )

        roi_features = []
        hidden_states = tokens
        next_output = 0
        for layer_index, (
            norm1,
            attention,
            layer_scale1,
            drop_path1,
            norm2,
            mlp,
            layer_scale2,
            drop_path2,
        ) in enumerate(self._single_block_parts):
            attention_output = attention(norm1(hidden_states))
            hidden_states = hidden_states + drop_path1(
                layer_scale1(attention_output)
            )
            hidden_states = hidden_states + drop_path2(
                layer_scale2(mlp(norm2(hidden_states)))
            )
            if (
                next_output < len(self._out_indices_tuple)
                and layer_index == self._out_indices_tuple[next_output]
            ):
                roi_features.append(
                    self._roi_align_tokens(
                        hidden_states,
                        grid_size=grid_size,
                        prefix_tokens=prefix_tokens,
                        boxes=boxes,
                        output_size=output_size,
                        spatial_scale=spatial_scale,
                        aligned=aligned,
                    )
                )
                next_output += 1
        if next_output != len(self._out_indices_tuple):
            raise RuntimeError("OmChat aux ROI checkpoints do not match vision blocks")
        concatenated = torch.cat(roi_features, dim=1)
        return concatenated.mean(dim=(2, 3)).reshape(
            1,
            concatenated.shape[0],
            concatenated.shape[1],
        )

    def _forward_rois_tensor(
        self,
        image: torch.Tensor,
        *,
        boxes: list[torch.Tensor],
        output_size: int,
        spatial_scale: float,
        aligned: bool,
    ) -> dict[str, object]:
        if not self._stream_roi_enabled_logged:
            self._stream_roi_enabled_logged = True
            logger.info(
                "Enabled streamed OmChat aux ROI extraction for tensor batches."
            )
        if image.dim() == 3:
            image = image.unsqueeze(0)
        original_size = (int(image.shape[-2]), int(image.shape[-1]))
        image = self._prepare_image(image)
        prepared_size = (int(image.shape[-2]), int(image.shape[-1]))
        if len(boxes) != image.shape[0]:
            raise ValueError(
                "Aux ROI box batches must match image batch: "
                f"got boxes={len(boxes)}, images={image.shape[0]}"
            )
        scaled_boxes = self._scale_roi_boxes(
            boxes,
            original_size=original_size,
            prepared_size=prepared_size,
            device=image.device,
        )
        tokens, grid_size = self._prepare_token_inputs(image)
        roi_features = self._forward_token_rois_eager(
            tokens,
            grid_size=grid_size,
            boxes=scaled_boxes,
            output_size=output_size,
            spatial_scale=spatial_scale,
            aligned=aligned,
        )
        self._validate_streamed_roi_features(
            image,
            scaled_boxes,
            roi_features,
            output_size=output_size,
            spatial_scale=spatial_scale,
            aligned=aligned,
        )
        return {
            "roi_features": roi_features,
            "roi_feature_counts": [int(item.shape[0]) for item in scaled_boxes],
            "input_size": prepared_size,
            "feature_map_size": grid_size,
        }

    def _validate_streamed_roi_features(
        self,
        image: torch.Tensor,
        boxes: list[torch.Tensor],
        actual: torch.Tensor,
        *,
        output_size: int,
        spatial_scale: float,
        aligned: bool,
    ) -> None:
        if (
            not _env_flag("OMCHAT_VALIDATE_STREAM_AUX_ROI_STRICT")
            or self._stream_roi_validation_done
        ):
            return
        with self._stream_roi_validation_lock:
            if self._stream_roi_validation_done:
                return
            reference_maps = self._forward_intermediates_legacy(image)
            reference_rois = torch.cat(
                [
                    roi_align(
                        feature.float(),
                        boxes,
                        output_size=output_size,
                        spatial_scale=spatial_scale,
                        aligned=aligned,
                    )
                    for feature in reference_maps
                ],
                dim=1,
            )
            reference = reference_rois.mean(dim=(2, 3)).reshape(
                1,
                reference_rois.shape[0],
                reference_rois.shape[1],
            )
            exact = torch.equal(reference, actual)
            max_abs = float((reference - actual).abs().max().item())
            logger.warning(
                "OmChat streamed aux ROI validation: shape=%s exact=%s "
                "max_abs=%.8f",
                tuple(image.shape),
                exact,
                max_abs,
            )
            self._stream_roi_validation_done = True
            if not exact:
                raise RuntimeError(
                    "OmChat streamed aux ROI output differs from dense path: "
                    f"max_abs={max_abs}"
                )

    def _forward_token_intermediates_varlen(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone = self.image_tower.radio_model.model
        index0, index1, index2, index3 = self._out_indices_tuple
        output0 = output1 = output2 = output3 = None
        hidden_states = tokens
        for layer_index, block in enumerate(backbone.blocks):
            if not isinstance(block.attn, AuxFlashAttention):
                raise TypeError(
                    "Packed aux execution requires AuxFlashAttention for all blocks"
                )
            attention_output = block.attn(
                block.norm1(hidden_states),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hidden_states = hidden_states + block.drop_path1(
                block.ls1(attention_output)
            )
            hidden_states = hidden_states + block.drop_path2(
                block.ls2(block.mlp(block.norm2(hidden_states)))
            )
            if layer_index == index0:
                output0 = hidden_states
            elif layer_index == index1:
                output1 = hidden_states
            elif layer_index == index2:
                output2 = hidden_states
            elif layer_index == index3:
                output3 = hidden_states

        if output0 is None or output1 is None or output2 is None or output3 is None:
            raise RuntimeError(
                "OmChat aux intermediate indices do not match the vision blocks"
            )
        return output0, output1, output2, output3

    @staticmethod
    def _reshape_token_intermediates(
        intermediates: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = grid_size

        def reshape(feature: torch.Tensor) -> torch.Tensor:
            return (
                feature.reshape(feature.shape[0], height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        return tuple(reshape(feature) for feature in intermediates)  # type: ignore[return-value]

    def _forward_intermediates_pure_tensor(
        self,
        image: torch.Tensor,
        *,
        use_compiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, grid_size = self._prepare_token_inputs(image)
        compiled = self._get_compiled_forward_intermediates() if use_compiled else None
        if compiled is None:
            intermediates = self._forward_token_intermediates_eager(tokens)
        else:
            intermediates = compiled(tokens)
        return self._reshape_token_intermediates(intermediates, grid_size)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        param = next(self.image_tower.parameters())
        is_uint8 = image.dtype == torch.uint8
        image = image.to(device=param.device, dtype=torch.float32)
        if is_uint8:
            image.mul_(1.0 / 255.0)
        return self._resize_to_supported(image)

    def _forward_varlen(self, images: list[torch.Tensor]):
        prepared_images = [self._prepare_image(image) for image in images]
        token_inputs = [
            self._prepare_token_inputs(image) for image in prepared_images
        ]
        tokens = [item[0] for item in token_inputs]
        grid_sizes = [item[1] for item in token_inputs]
        lengths = [int(item.shape[1]) for item in tokens]
        if any(item.shape[0] != 1 for item in tokens):
            raise ValueError("Packed aux execution expects one image per tensor")

        packed_tokens = torch.cat(tokens, dim=1)
        cumulative_lengths = [0]
        for length in lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + length)
        cu_seqlens = torch.tensor(
            cumulative_lengths,
            device=packed_tokens.device,
            dtype=torch.int32,
        )
        max_seqlen = torch.tensor(
            max(lengths),
            device=packed_tokens.device,
            dtype=torch.int32,
        )
        packed_intermediates = self._forward_token_intermediates_varlen(
            packed_tokens,
            cu_seqlens,
            max_seqlen,
        )
        split_intermediates = [
            torch.split(intermediate, lengths, dim=1)
            for intermediate in packed_intermediates
        ]

        prefix_tokens = self.image_tower.radio_model.model.patch_generator.num_skip
        outputs = []
        for image_index, (image, grid_size) in enumerate(
            zip(prepared_images, grid_sizes)
        ):
            image_intermediates = tuple(
                level[image_index][:, prefix_tokens:]
                for level in split_intermediates
            )
            reshaped = self._reshape_token_intermediates(
                image_intermediates,  # type: ignore[arg-type]
                grid_size,
            )
            outputs.append(
                self._extract_feature_map(
                    reshaped,
                    (image.shape[-2], image.shape[-1]),
                )
            )
        return outputs

    def _forward_intermediates_eager(self, image: torch.Tensor):
        if _env_flag("OMCHAT_PURE_TENSOR_AUX_VIT"):
            return self._forward_intermediates_pure_tensor(
                image,
                use_compiled=False,
            )
        return self._forward_intermediates_legacy(image)

    def _prepare_static_intermediate_normalizer(self) -> None:
        radio_model = self.image_tower.radio_model
        if radio_model.inter_feature_normalizer is not None:
            return
        backbone_param = next(radio_model.model.parameters())
        radio_model.inter_feature_normalizer = NullIntermediateFeatureNormalizer(
            backbone_param.dtype,
            backbone_param.device,
        )

    @staticmethod
    def _cudagraph_key(image: torch.Tensor) -> tuple[object, ...]:
        return (
            image.device.type,
            image.device.index,
            image.dtype,
            tuple(image.shape),
            tuple(image.stride()),
        )

    def _capture_aux_cudagraph(
        self,
        image: torch.Tensor,
    ) -> _AuxVisionCudaGraphRunner:
        self._prepare_static_intermediate_normalizer()
        static_input = torch.empty_like(image)
        capture_stream = torch.cuda.Stream(device=image.device)
        capture_stream.wait_stream(torch.cuda.current_stream(image.device))
        warmup_steps = max(1, _env_int("OMCHAT_AUX_VIT_CUDAGRAPH_WARMUPS", 2))

        with torch.inference_mode(), torch.cuda.stream(capture_stream):
            static_input.copy_(image)
            for _ in range(warmup_steps):
                self._forward_intermediates_eager(static_input)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, stream=capture_stream):
            static_outputs = self._forward_intermediates_eager(static_input)
        capture_stream.synchronize()
        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = [static_outputs]
        return _AuxVisionCudaGraphRunner(
            graph,
            static_input,
            list(static_outputs),
        )

    def _get_aux_cudagraph_runner(
        self,
        image: torch.Tensor,
    ) -> _AuxVisionCudaGraphRunner | None:
        key = self._cudagraph_key(image)
        runner = self._cudagraph_runners.get(key)
        if runner is not None:
            return runner
        if key in self._cudagraph_failed_keys:
            return None

        with self._cudagraph_capture_lock:
            runner = self._cudagraph_runners.get(key)
            if runner is not None:
                return runner
            max_graphs = max(0, _env_int("OMCHAT_AUX_VIT_CUDAGRAPH_MAX_GRAPHS", 2))
            if len(self._cudagraph_runners) >= max_graphs:
                if not self._cudagraph_capacity_warned:
                    self._cudagraph_capacity_warned = True
                    logger.warning(
                        "Aux vision CUDA graph cache is full (%d graphs); "
                        "new input shapes will use eager execution.",
                        max_graphs,
                    )
                return None
            try:
                runner = self._capture_aux_cudagraph(image)
            except Exception:
                self._cudagraph_failed_keys.add(key)
                logger.exception(
                    "Failed to capture aux vision CUDA graph for shape=%s; "
                    "falling back to eager execution.",
                    tuple(image.shape),
                )
                return None
            self._cudagraph_runners[key] = runner
            logger.info(
                "Captured aux vision CUDA graph for shape=%s dtype=%s device=%s.",
                tuple(image.shape),
                image.dtype,
                image.device,
            )
            return runner

    def _forward_intermediates_cudagraph(
        self,
        image: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        if not image.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        runner = self._get_aux_cudagraph_runner(image)
        if runner is None:
            return None
        return runner.replay(image)

    def _get_compiled_forward_intermediates(self):
        if not _env_flag("OMCHAT_COMPILE_AUX_VIT") or self._compile_failed:
            return None
        if self._compiled_forward_intermediates is not None:
            return self._compiled_forward_intermediates

        with self._compile_lock:
            if self._compiled_forward_intermediates is not None:
                return self._compiled_forward_intermediates
            mode = os.getenv("OMCHAT_COMPILE_AUX_VIT_MODE", "default")
            dynamic = _env_flag("OMCHAT_COMPILE_AUX_VIT_DYNAMIC")
            fullgraph = _env_flag("OMCHAT_COMPILE_AUX_VIT_FULLGRAPH", True)
            try:
                self._compiled_forward_intermediates = torch.compile(
                    self._forward_token_intermediates_eager,
                    mode=mode,
                    dynamic=dynamic,
                    fullgraph=fullgraph,
                )
                logger.info(
                    "Enabled pure-tensor torch.compile for OmChat aux vision "
                    "blocks (mode=%s, dynamic=%s, fullgraph=%s).",
                    mode,
                    dynamic,
                    fullgraph,
                )
            except Exception:
                self._compile_failed = True
                logger.exception("Failed to initialize aux vision torch.compile.")
                return None
        return self._compiled_forward_intermediates

    def _validate_pure_tensor_outputs(
        self,
        image: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        if (
            not (
                _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT")
                or _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT_STRICT")
            )
            or self._pure_tensor_validation_done
        ):
            return
        with self._pure_tensor_validation_lock:
            if self._pure_tensor_validation_done:
                return
            reference = self._forward_intermediates_legacy(image)
            if len(reference) != len(outputs):
                raise RuntimeError(
                    "OmChat pure-tensor aux output count differs from the legacy "
                    f"path: expected={len(reference)} actual={len(outputs)}"
                )
            exact = True
            max_abs = 0.0
            layer_metrics: list[dict[str, float | int | bool]] = []
            for layer_index, (expected, actual) in enumerate(
                zip(reference, outputs)
            ):
                exact &= torch.equal(expected, actual)
                expected_float = expected.float()
                actual_float = actual.float()
                abs_error = (expected_float - actual_float).abs()
                layer_max_abs = float(abs_error.max().item())
                max_abs = max(max_abs, layer_max_abs)
                cosine = torch.nn.functional.cosine_similarity(
                    expected_float.flatten(),
                    actual_float.flatten(),
                    dim=0,
                )
                layer_metrics.append(
                    {
                        "layer": layer_index,
                        "exact": torch.equal(expected, actual),
                        "max_abs": layer_max_abs,
                        "mean_abs": float(abs_error.mean().item()),
                        "reference_abs_max": float(expected_float.abs().max().item()),
                        "cosine": float(cosine.item()),
                    }
                )
            logger.warning(
                "OmChat pure-tensor aux validation: shape=%s exact=%s "
                "max_abs=%.8f layers=%s",
                tuple(image.shape),
                exact,
                max_abs,
                layer_metrics,
            )
            self._pure_tensor_validation_done = True
            if not exact and _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT_STRICT"):
                raise RuntimeError(
                    "OmChat pure-tensor aux output differs from the legacy path: "
                    f"max_abs={max_abs}"
                )

    def _forward_intermediates(self, image: torch.Tensor):
        if _env_flag("OMCHAT_AUX_VIT_CUDAGRAPH"):
            cudagraph_outputs = self._forward_intermediates_cudagraph(image)
            if cudagraph_outputs is not None:
                return cudagraph_outputs
        use_pure_tensor = _env_flag("OMCHAT_PURE_TENSOR_AUX_VIT") or _env_flag(
            "OMCHAT_COMPILE_AUX_VIT"
        )
        if not use_pure_tensor:
            return self._forward_intermediates_legacy(image)
        try:
            outputs = self._forward_intermediates_pure_tensor(
                image,
                use_compiled=_env_flag("OMCHAT_COMPILE_AUX_VIT"),
            )
            self._validate_pure_tensor_outputs(image, outputs)
            return outputs
        except Exception:
            self._compile_failed = True
            logger.exception(
                "Aux vision pure-tensor execution failed; falling back to legacy."
            )
            if _env_flag("OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT_STRICT"):
                raise
            return self._forward_intermediates_legacy(image)

    def _resize_to_supported(self, image: torch.Tensor) -> torch.Tensor:
        radio = self.image_tower.radio_model
        h, w = image.shape[-2], image.shape[-1]
        if self._max_input_pixels > 0 and h * w > self._max_input_pixels:
            scale = (self._max_input_pixels / (h * w)) ** 0.5
            step = radio.min_resolution_step
            target_h = max(step, int(h * scale) // step * step)
            target_w = max(step, int(w * scale) // step * step)
            target = radio.get_nearest_supported_resolution(target_h, target_w)
        else:
            target = radio.get_nearest_supported_resolution(h, w)
        if target.height == h and target.width == w:
            return image
        return torch.nn.functional.interpolate(
            image,
            size=(target.height, target.width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _forward_single(self, image: torch.Tensor):
        image = self._prepare_image(image)
        intermediates = self._forward_intermediates(image)
        return self._extract_feature_map(
            intermediates,
            (image.shape[-2], image.shape[-1]),
        )

    def forward(
        self,
        images,
        *,
        roi_boxes: list[torch.Tensor] | None = None,
        roi_output_size: int = 14,
        roi_spatial_scale: float = 1 / 16,
        roi_aligned: bool = True,
    ):
        if roi_boxes is not None and not isinstance(images, list):
            return self._forward_rois_tensor(
                images,
                boxes=roi_boxes,
                output_size=roi_output_size,
                spatial_scale=roi_spatial_scale,
                aligned=roi_aligned,
            )
        if isinstance(images, list):
            if self._varlen_batch_enabled and len(images) > 1:
                return self._forward_varlen(images)
            return [self._forward_single(img) for img in images]
        return self._forward_single(images)


def build_vision_tower_aux(
    config: OmChatQwen3_5Config,
    model_dir: str,
) -> nn.Module:
    raw_tower_path = getattr(config, "mm_vision_tower_aux", None)
    radio_config_path = _resolve_radio_config_path(model_dir, raw_tower_path)
    if radio_config_path is None:
        tower_path = resolve_tower_path(model_dir, raw_tower_path)
        raise FileNotFoundError(
            "C-RADIO aux tower config not found. Place config under "
            f"{tower_path or raw_tower_path}, or set OMCHAT_RADIO_CONFIG."
        )

    radio_cfg = _load_radio_config(radio_config_path)
    radio_cfg.vitdet_window_size = getattr(config, "mm_vitdet_window_size", None)
    return OmChatVisionTowerAux(radio_cfg)


def resolve_model_dir(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return model_name_or_path
