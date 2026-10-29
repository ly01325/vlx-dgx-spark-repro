# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only OmChat Qwen3.5-VL model for vLLM."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.configs.omchat_qwen3_5 import OmChatQwen3_5Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.utils.omchat_timing import (
    current_profile_id,
    record_shared_image_pack_start,
)
from vllm.utils.mistral import is_mistral_tokenizer

from .interfaces import MultiModalEmbeddings
from .omchat.object_modules import (
    build_mm_projector_aux,
    build_object_vp_extractor,
    build_vision_tower_aux,
    get_aux_vision_fp8_targets,
    process_flashrt_nvfp4_torch_linears,
    process_aux_vision_fp8_modules,
    replace_aux_vision_linear_with_fp8,
    replace_torch_linear_with_flashrt_nvfp4,
    resolve_model_dir,
    resolve_tower_path,
)
from .qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from .utils import AutoWeightsLoader, WeightsMapper
from .vision import run_dp_sharded_mrope_vision_model

logger = init_logger(__name__)

# OmChat Qwen3.5 object placeholder token id.
QWEN3_5_OBJECT_FEATURE_TOKEN_INDEX = 248181

_AUX_IMAGE_PROCESSOR = None
_SINGLE_IMAGE_QWEN_PROCESSOR_CLASS = None
_OBJECT_SUFFIX_CACHE_MAX_SIZE = 512
_OBJECT_SUFFIX_CACHE: dict[
    tuple[int, str, int],
    tuple[str, list[int], torch.Tensor],
] = {}


def _build_single_image_qwen_processor():
    """Create a Qwen image processor that avoids single-image regrouping."""
    global _SINGLE_IMAGE_QWEN_PROCESSOR_CLASS
    if _SINGLE_IMAGE_QWEN_PROCESSOR_CLASS is not None:
        return _SINGLE_IMAGE_QWEN_PROCESSOR_CLASS

    from transformers import BatchFeature
    from transformers.image_utils import SizeDict
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
        smart_resize,
    )

    class _SingleImageQwenProcessor(Qwen2VLImageProcessor):
        def _preprocess(
            self,
            images,
            do_resize,
            size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            patch_size,
            temporal_patch_size,
            merge_size,
            disable_grouping,
            return_tensors,
            **kwargs,
        ):
            if len(images) != 1 or not isinstance(images[0], torch.Tensor):
                return super()._preprocess(
                    images,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    disable_grouping=disable_grouping,
                    return_tensors=return_tensors,
                    **kwargs,
                )

            image = images[0].unsqueeze(0)
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image=image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )

            patches = self.rescale_and_normalize(
                image,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            _, channel = patches.shape[:2]
            resized_height, resized_width = patches.shape[-2:]
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            patches = patches.reshape(
                1,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
            pixel_values = (
                patches.unsqueeze(6)
                .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
                .reshape(
                    1,
                    grid_h * grid_w,
                    channel * temporal_patch_size * patch_size * patch_size,
                )
                .reshape(-1, channel * temporal_patch_size * patch_size * patch_size)
            )
            return BatchFeature(
                data={
                    "pixel_values": pixel_values,
                    "image_grid_thw": torch.tensor(
                        [[1, grid_h, grid_w]], dtype=torch.long
                    ),
                },
                tensor_type=return_tensors,
            )

    _SINGLE_IMAGE_QWEN_PROCESSOR_CLASS = _SingleImageQwenProcessor
    return _SINGLE_IMAGE_QWEN_PROCESSOR_CLASS


def _enable_single_image_qwen_processor(processor: object) -> None:
    if not _env_flag("OMCHAT_FAST_QWEN_PROCESSOR", True):
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or getattr(
        image_processor, "_omchat_single_image_fast_path", False
    ):
        return
    try:
        processor_cls = _build_single_image_qwen_processor()
        if not isinstance(image_processor, processor_cls):
            fast_processor = processor_cls(**image_processor.to_dict())
            fast_processor._omchat_single_image_fast_path = True
            processor.image_processor = fast_processor
    except Exception as exc:
        logger.warning("Unable to enable OmChat single-image processor fast path: %s", exc)


class _AsyncAuxVisionTowerResult(NamedTuple):
    outputs: object
    active_pixel_values_aux: torch.Tensor
    active_count: int
    stream: torch.cuda.Stream
    start_time: float
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None


class _AsyncVtFpnResult(NamedTuple):
    outputs: list[torch.Tensor]
    stream: torch.cuda.Stream
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None


def _get_aux_image_processor():
    global _AUX_IMAGE_PROCESSOR
    if _AUX_IMAGE_PROCESSOR is None:
        from transformers import CLIPImageProcessor

        _AUX_IMAGE_PROCESSOR = CLIPImageProcessor(
            do_resize=False,
            do_center_crop=False,
            do_rescale=True,
            do_normalize=False,
            do_convert_rgb=True,
            resample=3,
        )
    return _AUX_IMAGE_PROCESSOR


def _pack_pil_images(images: object) -> list[torch.Tensor] | None:
    if not isinstance(images, (list, tuple)):
        images = [images]
    image_list = list(images)
    if not image_list:
        return None

    packed_images: list[torch.Tensor] = []
    for image in image_list:
        if isinstance(image, Image.Image):
            packed_images.append(
                pil_to_tensor(image if image.mode == "RGB" else image.convert("RGB"))
            )
        elif (
            isinstance(image, torch.Tensor)
            and image.dtype == torch.uint8
            and image.ndim == 3
            and image.shape[0] == 3
        ):
            packed_images.append(image)
        else:
            return None
    return packed_images


def _preprocess_aux_images(
    images: object,
    *,
    packed_images: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if not isinstance(images, (list, tuple)):
        images = [images]
    start_time = (
        time.perf_counter() if _env_flag("OMCHAT_PROFILE_OBJECT_FEATURES") else 0.0
    )
    image_list = list(images)
    if packed_images is None and _env_flag("OMCHAT_PACK_MM_INPUTS", True):
        packed_images = _pack_pil_images(image_list)
    if packed_images is not None:
        pixel_values = (
            packed_images[0].unsqueeze(0)
            if len(packed_images) == 1
            else torch.stack(packed_images)
        )
    else:
        processor = _get_aux_image_processor()
        pixel_values = processor.preprocess(image_list, return_tensors="pt")[
            "pixel_values"
        ]
    if start_time:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "OmChat object-feature aux_image_preprocess took %.3f ms",
            elapsed_ms,
        )
    return pixel_values


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _is_bbox(values: object) -> bool:
    return (
        isinstance(values, (list, tuple))
        and len(values) == 4
        and all(isinstance(value, (int, float)) for value in values)
    )


def _get_object_suffix(
    tokenizer: object,
    vision_end_text: str,
    object_count: int,
) -> tuple[str, list[int], torch.Tensor]:
    cache_key = (id(tokenizer), vision_end_text, object_count)
    cached = _OBJECT_SUFFIX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    object_text = "".join(f"<obj{obj_idx}><objfeat>" for obj_idx in range(object_count))
    suffix_text = vision_end_text + object_text
    suffix_token_ids = tokenizer.encode(  # type: ignore[attr-defined]
        suffix_text,
        add_special_tokens=False,
    )
    suffix_mask = torch.tensor(suffix_token_ids, dtype=torch.long).eq(
        QWEN3_5_OBJECT_FEATURE_TOKEN_INDEX
    )
    cached = (suffix_text, suffix_token_ids, suffix_mask)
    if len(_OBJECT_SUFFIX_CACHE) >= _OBJECT_SUFFIX_CACHE_MAX_SIZE:
        _OBJECT_SUFFIX_CACHE.clear()
    _OBJECT_SUFFIX_CACHE[cache_key] = cached
    return cached


@lru_cache(maxsize=None)
def _cached_get_omchat_qwen3_vl_processor(
    processor_path: str,
    revision: str | None,
    trust_remote_code: bool,
    use_fast: bool,
    tokenizer: object,
):
    from transformers.models.qwen3_vl import (
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    image_processor = cached_get_image_processor(
        processor_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )
    video_processor = Qwen3VLVideoProcessor.from_pretrained(
        processor_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
    )


class OmChatQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OmChatQwen3_5Config)

    def _resolve_processor_path(self) -> str:
        config = self.get_hf_config()
        model_dir = self.ctx.model_config.model
        processor_path = resolve_tower_path(
            model_dir, getattr(config, "mm_vision_tower", None)
        )
        return processor_path or model_dir

    def get_hf_processor(self, **kwargs: object):
        model_config = self.ctx.model_config

        tokenizer = self.ctx.tokenizer
        if is_mistral_tokenizer(tokenizer):
            tokenizer = tokenizer.transformers_tokenizer  # type: ignore[union-attr]

        processor = _cached_get_omchat_qwen3_vl_processor(
            self._resolve_processor_path(),
            model_config.revision,
            model_config.trust_remote_code,
            bool(kwargs.pop("use_fast", True)),
            tokenizer,
        )
        _enable_single_image_qwen_processor(processor)
        return processor


class OmChatQwen3_5VLMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Extends Qwen3-VL processor with bbox / aux-image fields for object features."""

    @staticmethod
    def _num_images_from_processor_output(
        hf_inputs: Mapping[str, object],
    ) -> int:
        image_grid_thw = hf_inputs.get("image_grid_thw")
        if isinstance(image_grid_thw, torch.Tensor):
            if image_grid_thw.ndim == 1:
                return 1
            return int(image_grid_thw.shape[0])
        return 0

    @staticmethod
    def _empty_bbox_tensor() -> torch.Tensor:
        return torch.empty((0, 4), dtype=torch.float32)

    @classmethod
    def _boxes_to_tensor(cls, boxes: object) -> torch.Tensor:
        if isinstance(boxes, torch.Tensor):
            tensor = boxes.to(dtype=torch.float32)
        elif _is_bbox(boxes):
            tensor = torch.tensor([boxes], dtype=torch.float32)
        elif isinstance(boxes, (list, tuple)) and len(boxes) == 0:
            tensor = cls._empty_bbox_tensor()
        else:
            tensor = torch.tensor(boxes, dtype=torch.float32)

        if tensor.numel() == 0:
            return cls._empty_bbox_tensor()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != 4:
            raise ValueError(
                "bbox_list entries must use [xmin, ymin, xmax, ymax] format, "
                f"got shape={tuple(tensor.shape)}"
            )
        return tensor.reshape(-1, 4)

    @classmethod
    def _normalize_bboxes_for_processor(
        cls,
        bboxes: object,
        *,
        num_images: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_images <= 0:
            raise ValueError("bbox_list was provided but no image input was found")
        if num_images == 1:
            boxes = cls._boxes_to_tensor(bboxes)
            counts = torch.tensor([int(boxes.shape[0])], dtype=torch.long)
            return boxes, counts

        per_image_boxes: list[torch.Tensor]
        if isinstance(bboxes, torch.Tensor):
            boxes = bboxes.to(dtype=torch.float32)
            if boxes.numel() == 0:
                per_image_boxes = [cls._empty_bbox_tensor() for _ in range(num_images)]
            elif boxes.ndim == 1:
                if num_images != 1:
                    raise ValueError(
                        "A single bbox cannot be mapped to multiple images; "
                        f"num_images={num_images}"
                    )
                per_image_boxes = [cls._boxes_to_tensor(boxes)]
            elif boxes.ndim == 2:
                if num_images == 1:
                    per_image_boxes = [cls._boxes_to_tensor(boxes)]
                elif boxes.shape[0] == num_images and boxes.shape[-1] == 4:
                    per_image_boxes = [
                        cls._boxes_to_tensor(boxes[i]) for i in range(num_images)
                    ]
                else:
                    raise ValueError(
                        "2-D bbox_list for multi-image input must contain one "
                        "bbox per image. Use a per-image nested list for "
                        "variable object counts."
                    )
            elif boxes.ndim == 3:
                if boxes.shape[0] != num_images:
                    raise ValueError(
                        "Batched bbox_list does not match image count: "
                        f"got {boxes.shape[0]}, expected {num_images}"
                    )
                per_image_boxes = [
                    cls._boxes_to_tensor(boxes[i]) for i in range(num_images)
                ]
            else:
                raise ValueError(f"Unsupported bbox_list shape: {tuple(boxes.shape)}")
        elif _is_bbox(bboxes):
            if num_images != 1:
                raise ValueError(
                    "A single bbox cannot be mapped to multiple images; "
                    f"num_images={num_images}"
                )
            per_image_boxes = [cls._boxes_to_tensor(bboxes)]
        elif isinstance(bboxes, (list, tuple)):
            if len(bboxes) == 0:
                per_image_boxes = [cls._empty_bbox_tensor() for _ in range(num_images)]
            elif num_images == 1:
                per_image_boxes = [cls._boxes_to_tensor(bboxes)]
            elif len(bboxes) == num_images:
                per_image_boxes = [cls._boxes_to_tensor(boxes) for boxes in bboxes]
            else:
                raise ValueError(
                    "bbox_list length does not match image count: "
                    f"got {len(bboxes)}, expected {num_images}"
                )
        else:
            raise TypeError(f"Unsupported bbox_list type: {type(bboxes)!r}")

        counts = torch.tensor(
            [int(boxes.shape[0]) for boxes in per_image_boxes],
            dtype=torch.long,
        )
        if int(counts.sum().item()) == 0:
            flat_boxes = cls._empty_bbox_tensor()
        else:
            flat_boxes = torch.cat(per_image_boxes, dim=0)
        return flat_boxes, counts

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        profile_object_features = _env_flag("OMCHAT_PROFILE_OBJECT_FEATURES")
        mm_data = dict(mm_data)
        mm_kwargs = dict(mm_kwargs)
        bboxes = mm_data.pop("bboxes", None)
        if bboxes is None:
            bboxes = mm_data.pop("bbox_list", None)
        if bboxes is None:
            bboxes = mm_kwargs.pop("bboxes", None)
        else:
            mm_kwargs.pop("bboxes", None)
        if bboxes is None:
            bboxes = mm_kwargs.pop("bbox_list", None)
        else:
            mm_kwargs.pop("bbox_list", None)
        mm_kwargs.pop("bbox_counts", None)

        raw_images = mm_data.get("images")
        packed_images = None
        if (
            raw_images is not None
            and _env_flag("OMCHAT_PACK_MM_INPUTS", True)
            and _env_flag("OMCHAT_SHARE_IMAGE_TENSOR", True)
        ):
            record_shared_image_pack_start(
                current_profile_id(), time.monotonic_ns()
            )
            pack_start = time.perf_counter() if profile_object_features else 0.0
            packed_images = _pack_pil_images(raw_images)
            if packed_images is not None:
                # Qwen's torchvision processor accepts uint8 CHW tensors and
                # produces the same patches as PIL input. Reuse this conversion
                # for the full-resolution aux tower input below.
                mm_data["images"] = packed_images
            if pack_start:
                logger.info(
                    "OmChat object-feature shared_image_pack took %.3f ms",
                    (time.perf_counter() - pack_start) * 1000,
                )

        main_preprocess_start = (
            time.perf_counter() if profile_object_features else 0.0
        )
        out = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if _env_flag("OMCHAT_PACK_MM_INPUTS", True):
            pixel_values = out.get("pixel_values")
            vision_config = getattr(self.info.get_hf_config(), "vision_config", None)
            vision_dtype = getattr(vision_config, "dtype", None)
            if isinstance(pixel_values, torch.Tensor) and vision_dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                out["pixel_values"] = pixel_values.to(dtype=vision_dtype)

        if main_preprocess_start:
            logger.info(
                "OmChat object-feature main_mm_preprocess took %.3f ms",
                (time.perf_counter() - main_preprocess_start) * 1000,
            )

        if bboxes is not None and raw_images is not None:
            num_images = self._num_images_from_processor_output(out)
            flat_bboxes, bbox_counts = self._normalize_bboxes_for_processor(
                bboxes,
                num_images=num_images,
            )
            if int(bbox_counts.sum().item()) > 0:
                out["bbox_list"] = flat_bboxes
                out["bbox_counts"] = bbox_counts
                out["pixel_values_aux"] = _preprocess_aux_images(
                    raw_images,
                    packed_images=packed_images,
                )
        return out

    @staticmethod
    def _count_objects(bboxes: object) -> int:
        if bboxes is None:
            return 0
        if isinstance(bboxes, torch.Tensor):
            if bboxes.ndim == 0:
                return 0
            if bboxes.ndim == 1:
                return 1 if bboxes.numel() else 0
            return int(bboxes.shape[-2] if bboxes.ndim > 2 else bboxes.shape[0])
        if isinstance(bboxes, (list, tuple)):
            if len(bboxes) == 0:
                return 0
            if len(bboxes) == 4 and all(isinstance(v, (int, float)) for v in bboxes):
                return 1
            return len(bboxes)
        return 0

    def _get_object_feature_counts(
        self,
        out_mm_kwargs,
    ) -> list[int]:
        image_items = out_mm_kwargs.get("image", [])
        counts: list[int] = []
        for item in image_items:
            count_elem = item.get("bbox_counts")
            if count_elem is not None:
                count_data = count_elem.data
                if isinstance(count_data, torch.Tensor):
                    counts.append(int(count_data.item()))
                else:
                    counts.append(int(count_data))
                continue
            bbox_elem = item.get("bbox_list")
            bboxes = None if bbox_elem is None else bbox_elem.data
            counts.append(self._count_objects(bboxes))
        return counts

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        if "bbox_list" in hf_inputs:
            bbox_counts = hf_inputs.get("bbox_counts")
            if not isinstance(bbox_counts, torch.Tensor):
                bbox_counts = torch.tensor(
                    [self._count_objects(hf_inputs["bbox_list"])],
                    dtype=torch.long,
                )
            fields["bbox_list"] = MultiModalFieldConfig.flat_from_sizes(
                "image",
                bbox_counts,
            )
        if "bbox_counts" in hf_inputs:
            fields["bbox_counts"] = MultiModalFieldConfig.batched(
                "image",
                keep_on_cpu=True,
            )
        if "pixel_values_aux" in hf_inputs:
            fields["pixel_values_aux"] = MultiModalFieldConfig.batched("image")
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        updates = list(
            super()._get_prompt_updates(
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                out_mm_kwargs=out_mm_kwargs,
            )
        )
        if _env_flag("OMCHAT_SKIP_AUX_VIT"):
            return updates
        object_counts = self._get_object_feature_counts(out_mm_kwargs)
        if not any(object_counts):
            return updates

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        tokenizer = hf_processor.tokenizer
        vision_end_token_id = hf_config.vision_end_token_id
        vision_end_text = tokenizer.decode(
            [vision_end_token_id], skip_special_tokens=False
        )
        image_update_idx = next(
            (idx for idx, update in enumerate(updates) if update.modality == "image"),
            None,
        )
        if image_update_idx is None:
            return updates

        base_image_update = updates[image_update_idx]
        _, vision_end_token_ids, _ = _get_object_suffix(
            tokenizer,
            vision_end_text,
            0,
        )
        object_suffixes = [
            _get_object_suffix(tokenizer, vision_end_text, object_count)
            for object_count in object_counts
        ]

        def append_suffix(target: object, text: str, token_ids: list[int]):
            if not text:
                return target
            if isinstance(target, str):
                return target + text
            return list(target) + token_ids

        def get_image_target_with_vision_end(item_idx: int):
            base_target = base_image_update._resolve_target(item_idx)
            return append_suffix(
                base_target,
                vision_end_text,
                vision_end_token_ids,
            )

        def get_image_replacement_with_objects(item_idx: int):
            content = base_image_update._resolve_content(item_idx)
            suffix_text, suffix_token_ids, suffix_mask = object_suffixes[item_idx]
            full = append_suffix(content.full, suffix_text, suffix_token_ids)

            def is_embed(tokenizer_arg, full_seq):
                content_is_embed = content.is_embed
                if content_is_embed is None:
                    content_tokens = (
                        tokenizer.encode(
                            content.full,
                            add_special_tokens=False,
                        )
                        if isinstance(content.full, str)
                        else list(content.full)
                    )
                    base_mask = torch.ones(len(content_tokens), dtype=torch.bool)
                else:
                    base_mask = content_is_embed(tokenizer_arg, content.full)

                return torch.cat([base_mask, suffix_mask])

            return PromptUpdateDetails(full=full, is_embed=is_embed)

        object_update = PromptReplacement(
            modality="image",
            target=get_image_target_with_vision_end,
            replacement=get_image_replacement_with_objects,
        )
        updates[image_update_idx : image_update_idx + 1] = [object_update]
        return updates


@MULTIMODAL_REGISTRY.register_processor(
    OmChatQwen3_5VLMultiModalProcessor,
    info=OmChatQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class OmChatQwen3_5VLForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """OmChat Qwen3.5-VL with optional object-feature (bbox) support."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_tower.visual.": "visual.",
            "model.vision_tower_aux.": "vision_tower_aux.",
            "model.object_vp_extractor.": "object_vp_extractor.",
            "model.mm_projector_aux.": "mm_projector_aux.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "model.visual.": "visual.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.quant_config = vllm_config.quant_config
        self.vision_tower_aux: nn.Module | None = None
        self.object_vp_extractor: nn.Module | None = None
        self.mm_projector_aux: nn.Module | None = None
        self._mm_projector_aux_dtype: torch.dtype | None = None
        self._object_modules_initialized = False
        self._object_modules_init_error: Exception | None = None
        self._model_dir = resolve_model_dir(vllm_config.model_config.model)
        self._vt_fpn_streams: dict[tuple[str, int | None], torch.cuda.Stream] = {}

    def _uses_vt_only_object_features(self) -> bool:
        return bool(getattr(self.config, "mm_use_vt_object_feature_only", False))

    def _ensure_object_modules(self) -> bool:
        if self._object_modules_initialized:
            return self._supports_object_features()

        self._object_modules_initialized = True
        config: OmChatQwen3_5Config = self.config
        use_vt_only = self._uses_vt_only_object_features()
        if not use_vt_only and not getattr(config, "mm_vision_tower_aux", None):
            return False

        try:
            model_dir = self._model_dir
            if not use_vt_only:
                self.vision_tower_aux = build_vision_tower_aux(config, model_dir)
            self.object_vp_extractor = build_object_vp_extractor(
                config,
                vision_tower=self.visual,
                vision_tower_aux=self.vision_tower_aux,
            )
            self.mm_projector_aux = build_mm_projector_aux(config)
            param = next(
                (
                    candidate
                    for candidate in self.visual.parameters()
                    if candidate.is_floating_point()
                ),
                None,
            )
            if param is None:
                param = next(
                    (
                        candidate
                        for candidate in self.parameters()
                        if candidate.is_floating_point()
                    ),
                    None,
                )
            if param is not None:
                device = param.device
                dtype = param.dtype
                if self.vision_tower_aux is not None:
                    self.vision_tower_aux = self.vision_tower_aux.to(
                        device=device, dtype=dtype
                    )
                self.object_vp_extractor = self.object_vp_extractor.to(
                    device=device, dtype=dtype
                )
                self.mm_projector_aux = self.mm_projector_aux.to(
                    device=device, dtype=dtype
                )
                use_flashrt_all_linear = (
                    self.quant_config is not None
                    and self.quant_config.get_name() == "flashrt_nvfp4"
                    and getattr(self.quant_config, "scope", None) == "all_linear"
                )
                if use_flashrt_all_linear:
                    replaced = replace_torch_linear_with_flashrt_nvfp4(
                        self.mm_projector_aux
                    )
                    if self.vision_tower_aux is not None:
                        replaced += replace_torch_linear_with_flashrt_nvfp4(
                            self.vision_tower_aux
                        )
                    logger.info(
                        "Replaced %d object-module Linear layers with "
                        "FlashRT NVFP4 W4A4 Linear.",
                        replaced,
                    )
                self._mm_projector_aux_dtype = (
                    dtype
                    if use_flashrt_all_linear
                    else self._get_module_dtype(self.mm_projector_aux)
                )
                if self.vision_tower_aux is not None:
                    self.vision_tower_aux._configure_input_conditioner()
                    aux_fp8_targets = get_aux_vision_fp8_targets(config)
                    if aux_fp8_targets:
                        replaced = replace_aux_vision_linear_with_fp8(
                            self.vision_tower_aux,
                            aux_fp8_targets,
                        )
                        logger.info(
                            "Replaced %d aux vision tower Linear layers with "
                            "FP8 weight-only layers for targets=%s.",
                            replaced,
                            sorted(aux_fp8_targets),
                        )
            logger.info(
                "OmChat object-feature modules initialized (%s).",
                "VT ROI + projector; C-RADIO skipped"
                if use_vt_only
                else "aux tower + ROI + projector",
            )
            self._object_modules_init_error = None
            return True
        except Exception as exc:
            self._object_modules_init_error = exc
            logger.exception(
                "Failed to initialize OmChat object-feature modules; "
                "image-only inference will still work."
            )
            self.vision_tower_aux = None
            self.object_vp_extractor = None
            self.mm_projector_aux = None
            self._mm_projector_aux_dtype = None
            return False

    def _supports_object_features(self) -> bool:
        if self._uses_vt_only_object_features():
            return (
                self.object_vp_extractor is not None
                and self.mm_projector_aux is not None
            )
        return (
            self.vision_tower_aux is not None
            and self.object_vp_extractor is not None
            and self.mm_projector_aux is not None
        )

    @staticmethod
    def _get_module_dtype(module: nn.Module | None) -> torch.dtype | None:
        if module is None:
            return None
        param = next(module.parameters(), None)
        if param is not None:
            return param.dtype
        buffer = next(module.buffers(), None)
        if buffer is not None:
            return buffer.dtype
        return None

    def _get_mm_projector_aux_dtype(
        self,
        default: torch.dtype,
    ) -> torch.dtype:
        if self._mm_projector_aux_dtype is not None:
            return self._mm_projector_aux_dtype
        self._mm_projector_aux_dtype = self._get_module_dtype(self.mm_projector_aux)
        return self._mm_projector_aux_dtype or default

    @staticmethod
    def _extract_vt_multi_level_features(
        image_embeds: torch.Tensor,
        grid_thw: torch.Tensor | tuple[int, int, int],
        spatial_merge_size: int,
    ) -> list[torch.Tensor]:
        if isinstance(grid_thw, torch.Tensor):
            t, h, w = grid_thw.tolist()
        else:
            t, h, w = grid_thw
        h = h // spatial_merge_size
        w = w // spatial_merge_size
        last_feature = image_embeds.view(t, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return [last_feature]

    @staticmethod
    def _grid_thw_to_tuples(grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if grid_thw.device.type != "cpu":
            grid_thw = grid_thw.detach().cpu()
        return [tuple(int(value) for value in grid.tolist()) for grid in grid_thw]

    @classmethod
    def _split_image_embeds_by_grid(
        cls,
        image_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
    ) -> list[torch.Tensor]:
        lengths = []
        grid_thw_shapes = cls._grid_thw_to_tuples(grid_thw)
        for t, h, w in grid_thw_shapes:
            lengths.append(
                int(t * (h // spatial_merge_size) * (w // spatial_merge_size))
            )

        total = sum(lengths)
        if total != image_embeds.shape[0]:
            raise ValueError(
                "Visual embedding length does not match image_grid_thw: "
                f"got {image_embeds.shape[0]} embeddings, expected {total} "
                f"from grid_thw={grid_thw_shapes}"
            )

        return list(torch.split(image_embeds, lengths, dim=0))

    @staticmethod
    def _split_aux_outputs_by_batch(
        aux_outputs: object,
        num_images: int,
    ) -> list[dict[str, object]]:
        if isinstance(aux_outputs, dict):
            roi_features = aux_outputs.get("roi_features")
            roi_feature_counts = aux_outputs.get("roi_feature_counts")
            if isinstance(roi_features, torch.Tensor) and isinstance(
                roi_feature_counts, (list, tuple)
            ):
                counts = [int(count) for count in roi_feature_counts]
                if len(counts) != num_images or sum(counts) != roi_features.shape[1]:
                    raise ValueError(
                        "Aux ROI feature counts do not match image batch: "
                        f"counts={counts}, num_images={num_images}, "
                        f"features={roi_features.shape[1]}"
                    )
                return [
                    {
                        **aux_outputs,
                        "roi_features": feature,
                        "roi_feature_counts": [count],
                    }
                    for feature, count in zip(
                        torch.split(roi_features, counts, dim=1),
                        counts,
                    )
                ]
            image_features = aux_outputs.get("image_features")
            if not isinstance(image_features, list) or not image_features:
                return [aux_outputs]

            batch_size = image_features[0].shape[0]
            if batch_size == 1 and num_images == 1:
                return [aux_outputs]
            if batch_size != num_images:
                raise ValueError(
                    "Aux vision output batch does not match image_grid_thw: "
                    f"got batch_size={batch_size}, num_images={num_images}"
                )

            return [
                {
                    **aux_outputs,
                    "image_features": [
                        feature[batch_idx : batch_idx + 1] for feature in image_features
                    ],
                }
                for batch_idx in range(num_images)
            ]

        if isinstance(aux_outputs, list):
            return aux_outputs

        raise TypeError(f"Unsupported aux output type: {type(aux_outputs)!r}")

    @staticmethod
    def _normalize_bbox_list(
        bbox_list: torch.Tensor | list[torch.Tensor],
        *,
        num_images: int,
        device: torch.device,
        bbox_counts: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(bbox_list, torch.Tensor):
            boxes = bbox_list.to(device=device, dtype=torch.float32)
            if num_images == 1:
                if boxes.numel() == 0:
                    return [boxes.reshape(0, 4)]
                if boxes.ndim == 1:
                    return [boxes.unsqueeze(0)]
                if boxes.shape[-1] != 4:
                    raise ValueError(
                        "bbox_list entries must use [xmin, ymin, xmax, ymax] "
                        f"format, got shape={tuple(boxes.shape)}"
                    )
                return [boxes.reshape(-1, 4)]
            if bbox_counts is not None:
                counts = bbox_counts.detach().cpu().to(dtype=torch.long).flatten()
                if counts.numel() != num_images:
                    raise ValueError(
                        "bbox_counts does not match image_grid_thw: "
                        f"got {counts.numel()} counts, num_images={num_images}"
                    )
                if boxes.numel() == 0:
                    boxes = boxes.reshape(0, 4)
                else:
                    boxes = boxes.reshape(-1, 4)
                total = int(counts.sum().item())
                if total != boxes.shape[0]:
                    raise ValueError(
                        "bbox_counts sum does not match bbox_list rows: "
                        f"got sum={total}, bbox_rows={boxes.shape[0]}"
                    )
                lengths = [int(count.item()) for count in counts]
                return list(torch.split(boxes, lengths, dim=0))

            if boxes.ndim == 1:
                return [boxes.unsqueeze(0)]
            if boxes.ndim == 2:
                if (
                    num_images > 1
                    and boxes.shape[0] == num_images
                    and boxes.shape[-1] == 4
                ):
                    return [boxes[i : i + 1] for i in range(num_images)]
                return [boxes]
            if boxes.ndim == 3:
                if boxes.shape[0] != num_images:
                    raise ValueError(
                        "Batched bbox_list does not match image_grid_thw: "
                        f"got {boxes.shape[0]} bbox batches, "
                        f"num_images={num_images}"
                    )
                return [boxes[i] for i in range(num_images)]
            raise ValueError(f"Unsupported bbox_list shape: {tuple(boxes.shape)}")

        if isinstance(bbox_list, (list, tuple)):
            if num_images == 1:
                boxes = torch.as_tensor(bbox_list, device=device, dtype=torch.float32)
                if boxes.ndim == 1:
                    boxes = boxes.unsqueeze(0)
                return [boxes]

            if len(bbox_list) != num_images:
                raise ValueError(
                    "bbox_list length does not match image_grid_thw: "
                    f"got len={len(bbox_list)}, num_images={num_images}"
                )

            per_image_boxes = []
            for item in bbox_list:
                boxes = torch.as_tensor(item, device=device, dtype=torch.float32)
                if boxes.ndim == 1:
                    boxes = boxes.unsqueeze(0)
                per_image_boxes.append(boxes)
            return per_image_boxes

        raise TypeError(f"Unsupported bbox_list type: {type(bbox_list)!r}")

    @staticmethod
    def _normalize_image_grid_thw(image_grid_thw: object) -> torch.Tensor:
        if isinstance(image_grid_thw, torch.Tensor):
            if image_grid_thw.ndim == 1:
                return image_grid_thw.unsqueeze(0)
            return image_grid_thw

        if isinstance(image_grid_thw, (list, tuple)):
            grids: list[torch.Tensor] = []
            target_device: torch.device | None = None
            for item in image_grid_thw:
                grid = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                if grid.ndim == 1:
                    grid = grid.unsqueeze(0)
                if target_device is None:
                    target_device = grid.device
                grids.append(grid.to(device=target_device, dtype=torch.long))
            if grids:
                return torch.cat(grids, dim=0)
            return torch.empty((0, 3), dtype=torch.long)

        raise TypeError(f"Unsupported image_grid_thw type: {type(image_grid_thw)!r}")

    @staticmethod
    def _normalize_bbox_counts(bbox_counts: object) -> torch.Tensor | None:
        if bbox_counts is None:
            return None
        if isinstance(bbox_counts, torch.Tensor):
            return bbox_counts
        if isinstance(bbox_counts, (list, tuple)):
            counts = []
            for item in bbox_counts:
                count = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                counts.append(count.detach().cpu().to(dtype=torch.long).reshape(-1))
            if counts:
                return torch.cat(counts, dim=0)
            return torch.empty((0,), dtype=torch.long)
        return torch.as_tensor(bbox_counts, dtype=torch.long).reshape(-1)

    def _expected_object_feature_counts(
        self,
        bbox_list: object,
        bbox_counts: object,
        *,
        num_images: int,
    ) -> list[int]:
        counts = self._normalize_bbox_counts(bbox_counts)
        if counts is not None:
            values = counts.detach().cpu().to(dtype=torch.long).reshape(-1).tolist()
            if len(values) != num_images:
                raise ValueError(
                    "bbox_counts does not match image_grid_thw: "
                    f"got {len(values)} counts for {num_images} images"
                )
            return [int(value) for value in values]

        if isinstance(bbox_list, torch.Tensor):
            if num_images == 1:
                if bbox_list.ndim == 1:
                    return [1 if bbox_list.numel() else 0]
                return [int(bbox_list.shape[0])]
            if bbox_list.ndim == 3 and bbox_list.shape[0] == num_images:
                return [int(bbox_list.shape[1])] * num_images
        if isinstance(bbox_list, (list, tuple)):
            if num_images == 1:
                return [self._count_objects(bbox_list)]
            if len(bbox_list) == num_images:
                return [self._count_objects(item) for item in bbox_list]

        raise ValueError(
            "Unable to map bbox_list to image_grid_thw while validating "
            "object-feature embeddings"
        )

    @staticmethod
    def _split_pixel_values_aux(
        pixel_values_aux: object,
        *,
        num_images: int,
    ) -> list[torch.Tensor]:
        if isinstance(pixel_values_aux, torch.Tensor):
            if num_images == 1:
                return [pixel_values_aux]
            if pixel_values_aux.ndim < 4:
                raise ValueError(
                    "pixel_values_aux tensor for multiple images must include "
                    f"a batch dimension, got shape={tuple(pixel_values_aux.shape)}"
                )
            if pixel_values_aux.shape[0] != num_images:
                raise ValueError(
                    "pixel_values_aux batch does not match image_grid_thw: "
                    f"got batch={pixel_values_aux.shape[0]}, "
                    f"num_images={num_images}"
                )
            return [pixel_values_aux[idx : idx + 1] for idx in range(num_images)]

        if isinstance(pixel_values_aux, (list, tuple)):
            pixels: list[torch.Tensor] = []
            for item in pixel_values_aux:
                if not isinstance(item, torch.Tensor):
                    item = torch.as_tensor(item)
                if item.ndim == 3:
                    pixels.append(item.unsqueeze(0))
                elif item.ndim >= 4:
                    pixels.extend(item[idx : idx + 1] for idx in range(item.shape[0]))
                else:
                    raise ValueError(
                        "Unsupported pixel_values_aux item shape: "
                        f"{tuple(item.shape)}"
                    )
            if len(pixels) != num_images:
                raise ValueError(
                    "pixel_values_aux list does not match image_grid_thw: "
                    f"got {len(pixels)} images, num_images={num_images}"
                )
            return pixels

        raise TypeError(f"Unsupported pixel_values_aux type: {type(pixel_values_aux)!r}")

    @staticmethod
    def _pixel_values_aux_device(pixel_values_aux: object) -> torch.device:
        if isinstance(pixel_values_aux, torch.Tensor):
            return pixel_values_aux.device
        if isinstance(pixel_values_aux, (list, tuple)):
            for item in pixel_values_aux:
                if isinstance(item, torch.Tensor):
                    return item.device
        return torch.device("cpu")

    @staticmethod
    def _log_object_feature_timing(
        stage: str,
        start_time: float,
        start_event: torch.cuda.Event | None,
        end_event: torch.cuda.Event | None,
    ) -> None:
        if start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("OmChat object-feature %s took %.3f ms", stage, elapsed_ms)

    def _start_object_feature_timing(
        self,
    ) -> tuple[float, torch.cuda.Event | None, torch.cuda.Event | None]:
        start_time = time.perf_counter()
        if not torch.cuda.is_available():
            return start_time, None, None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_time, start_event, end_event

    @staticmethod
    def _log_recorded_object_feature_timing(
        stage: str,
        start_time: float,
        start_event: torch.cuda.Event | None,
        end_event: torch.cuda.Event | None,
    ) -> None:
        if start_event is not None and end_event is not None:
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("OmChat object-feature %s took %.3f ms", stage, elapsed_ms)

    def _maybe_start_async_aux_vision_tower(
        self,
        *,
        pixel_values_aux: object,
        bbox_list: object,
        bbox_counts: object,
        image_grid_thw: object,
        profile: bool,
    ) -> _AsyncAuxVisionTowerResult | None:
        if self._uses_vt_only_object_features():
            return None
        if not _env_flag("OMCHAT_PARALLEL_AUX_VIT"):
            return None
        if not torch.cuda.is_available():
            return None
        if not isinstance(pixel_values_aux, torch.Tensor):
            return None
        if not pixel_values_aux.is_cuda:
            return None
        if not isinstance(image_grid_thw, torch.Tensor):
            return None
        if bbox_list is None or not self._ensure_object_modules():
            return None

        grid_thw = image_grid_thw
        if grid_thw.ndim == 1:
            num_images = 1
        else:
            num_images = int(grid_thw.shape[0])

        normalized_bbox_list = self._normalize_bbox_list(
            bbox_list,  # type: ignore[arg-type]
            num_images=num_images,
            device=pixel_values_aux.device,
            bbox_counts=(
                bbox_counts if isinstance(bbox_counts, torch.Tensor) else None
            ),
        )
        active_indices = [
            idx for idx, boxes in enumerate(normalized_bbox_list) if boxes.numel() > 0
        ]
        if not active_indices:
            return None

        if num_images == 1:
            active_pixel_values_aux = pixel_values_aux
        else:
            active_pixel_values_aux = self._select_active_aux_pixels(
                pixel_values_aux,
                active_indices,
            )

        aux_stream = torch.cuda.Stream(device=pixel_values_aux.device)
        aux_stream.wait_stream(torch.cuda.current_stream(pixel_values_aux.device))
        start_time = 0.0
        start_event = None
        end_event = None
        with torch.cuda.stream(aux_stream):
            if profile:
                start_time = time.perf_counter()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            aux_outputs = self._run_aux_vision_tower(
                active_pixel_values_aux,
                [normalized_bbox_list[idx] for idx in active_indices],
            )
            if profile and end_event is not None:
                end_event.record()

        active_pixel_values_aux.record_stream(aux_stream)
        return _AsyncAuxVisionTowerResult(
            outputs=aux_outputs,
            active_pixel_values_aux=active_pixel_values_aux,
            active_count=len(active_indices),
            stream=aux_stream,
            start_time=start_time,
            start_event=start_event,
            end_event=end_event,
        )

    def _can_stream_aux_rois(self) -> bool:
        if self._uses_vt_only_object_features():
            return False
        if not _env_flag("OMCHAT_STREAM_AUX_ROI", True):
            return False
        extractor = self.object_vp_extractor
        if extractor is None:
            return False
        if (
            getattr(extractor, "use_simpleFPN_for_vt_aux", False)
            or getattr(extractor, "use_simpleFPN_multiLayer_for_vt_aux", False)
            or getattr(extractor, "pos_embedding_strategy", None)
            in {"feature_map_based", "hybrid"}
            or _env_flag("OMCHAT_COMPILE_AUX_VIT")
            or _env_flag("OMCHAT_AUX_VIT_CUDAGRAPH")
        ):
            return False
        return True

    def _run_aux_vision_tower(
        self,
        images: object,
        boxes: list[torch.Tensor],
    ) -> object:
        assert self.vision_tower_aux is not None
        extractor = self.object_vp_extractor
        if not self._can_stream_aux_rois() or extractor is None:
            return self.vision_tower_aux(images)
        return self.vision_tower_aux(
            images,
            roi_boxes=boxes,
            roi_output_size=extractor.output_size,
            roi_spatial_scale=extractor.spatial_scale,
            roi_aligned=extractor.roi_algined,
        )

    def _maybe_start_async_vt_fpn(
        self,
        vt_feature: torch.Tensor,
        *,
        profile: bool,
    ) -> _AsyncVtFpnResult | None:
        if not _env_flag("OMCHAT_PARALLEL_VT_FPN"):
            return None
        if not vt_feature.is_cuda or self.object_vp_extractor is None:
            return None
        extractor = self.object_vp_extractor
        if not getattr(extractor, "use_simpleFPN_for_vt", False):
            return None
        if (
            getattr(extractor, "_compile_simple_fpn", False)
            or getattr(extractor, "_fused_simple_fpn", False)
            or getattr(extractor.simple_fpn, "_parallel_stages", False)
        ):
            return None

        device = vt_feature.device
        key = (device.type, device.index)
        stream = self._vt_fpn_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._vt_fpn_streams[key] = stream
        caller_stream = torch.cuda.current_stream(device)
        stream.wait_stream(caller_stream)
        start_event = torch.cuda.Event(enable_timing=True) if profile else None
        end_event = torch.cuda.Event(enable_timing=True) if profile else None
        with torch.cuda.stream(stream):
            if start_event is not None:
                start_event.record()
            outputs = extractor.prepare_vt_fpn_features(vt_feature)
            if end_event is not None:
                end_event.record()
        vt_feature.record_stream(stream)
        return _AsyncVtFpnResult(
            outputs=outputs,
            stream=stream,
            start_event=start_event,
            end_event=end_event,
        )

    def _finish_async_vt_fpn(
        self,
        result: _AsyncVtFpnResult | None,
        *,
        profile: bool,
    ) -> list[torch.Tensor] | None:
        if result is None:
            return None
        caller_stream = torch.cuda.current_stream(result.outputs[0].device)
        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        caller_stream.wait_stream(result.stream)
        for output in result.outputs:
            output.record_stream(caller_stream)
        if profile:
            self._log_object_feature_timing(
                "wait_vt_fpn_async",
                start_time,
                start_event,
                end_event,
            )
            self._log_recorded_object_feature_timing(
                "vt_fpn_async",
                0.0,
                result.start_event,
                result.end_event,
            )
        return result.outputs

    def _get_reused_image_embeds(
        self,
        embeddings: MultiModalEmbeddings,
        *,
        image_grid_thw: torch.Tensor,
        num_images: int,
    ) -> list[torch.Tensor]:
        if isinstance(embeddings, torch.Tensor):
            grid_thw = image_grid_thw
            if grid_thw.ndim == 1:
                grid_thw = grid_thw.unsqueeze(0)
            return self._split_image_embeds_by_grid(
                embeddings,
                grid_thw,
                self.visual.spatial_merge_size,
            )

        if len(embeddings) < num_images:
            raise ValueError(
                "Not enough multimodal embeddings to reuse for object features: "
                f"got {len(embeddings)}, expected at least {num_images}"
            )
        return list(embeddings[:num_images])

    @staticmethod
    def _select_active_aux_pixels(
        pixel_values_aux: torch.Tensor,
        active_indices: Sequence[int],
    ) -> torch.Tensor:
        if len(active_indices) == 1 and pixel_values_aux.ndim == 3:
            return pixel_values_aux
        if pixel_values_aux.ndim >= 4:
            if len(active_indices) == pixel_values_aux.shape[0] and all(
                idx == pos for pos, idx in enumerate(active_indices)
            ):
                return pixel_values_aux
            if len(active_indices) == 1:
                idx = active_indices[0]
                return pixel_values_aux[idx : idx + 1]
        indices = torch.tensor(
            active_indices,
            device=pixel_values_aux.device,
            dtype=torch.long,
        )
        return pixel_values_aux.index_select(0, indices)

    def _extract_object_features_for_image(
        self,
        *,
        aux_out: dict[str, object],
        boxes: torch.Tensor,
        vt_feats: list[torch.Tensor] | torch.Tensor,
        image_size: tuple[int, int],
        pixel_values_aux: torch.Tensor,
        profile: bool,
        vt_fpn_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        aux_feats = aux_out.get("image_features", [])
        aux_roi_features = aux_out.get("roi_features")
        if aux_roi_features is not None and not isinstance(
            aux_roi_features, torch.Tensor
        ):
            raise TypeError("Aux ROI features must be a tensor")
        if getattr(self.config, "mm_use_simpleFPN_for_vt", False) and not getattr(
            self.config, "mm_use_simpleFPN_multiLayer_for_vt", False
        ):
            vt_feats = vt_feats[-1]

        current_h = pixel_values_aux.shape[-2]
        current_w = pixel_values_aux.shape[-1]
        aux_h, aux_w = aux_out.get("input_size", (current_h, current_w))
        orig_h, orig_w = image_size
        aux_boxes = boxes.clone()
        aux_boxes[:, 0::2].mul_(aux_w / current_w)
        aux_boxes[:, 1::2].mul_(aux_h / current_h)
        vt_boxes = boxes.clone()
        vt_boxes[:, 0::2].mul_(orig_w / current_w)
        vt_boxes[:, 1::2].mul_(orig_h / current_h)

        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        extracted = self.object_vp_extractor(
            multi_level_features=aux_feats,
            vt_multi_level_features=vt_feats,
            boxes=[aux_boxes],
            vt_boxes=[vt_boxes],
            boxes_for_pos_emb=[aux_boxes],
            vt_boxes_for_pos_emb=[vt_boxes],
            add_pos_embed=True,
            vt_fpn_features=vt_fpn_features,
            aux_roi_features=aux_roi_features,
            aux_feature_map_size=aux_out.get("feature_map_size"),
        ).squeeze(0)
        if profile:
            self._log_object_feature_timing(
                "roi_extract",
                start_time,
                start_event,
                end_event,
            )
        return extracted

    def _extract_object_features_for_batch(
        self,
        *,
        aux_outputs: object,
        active_indices: Sequence[int],
        bbox_list: Sequence[torch.Tensor],
        image_embeds: Sequence[torch.Tensor],
        grid_thw_shapes: Sequence[tuple[int, int, int]],
        merge_size: int,
        image_sizes: Sequence[tuple[int, int]],
        active_pixel_values_aux: torch.Tensor,
        profile: bool,
        vt_fpn_features: list[torch.Tensor] | None = None,
    ) -> list[tuple[int, torch.Tensor]] | None:
        if not _env_flag("OMCHAT_BATCH_ROI_EXTRACT", True):
            return None
        if len(active_indices) <= 1:
            return None
        if not isinstance(aux_outputs, dict):
            return None
        aux_feats = aux_outputs.get("image_features")
        aux_roi_features = aux_outputs.get("roi_features")
        if aux_roi_features is None:
            if not isinstance(aux_feats, list) or not aux_feats:
                return None
        elif not isinstance(aux_roi_features, torch.Tensor):
            return None
        if active_pixel_values_aux.ndim < 4:
            return None
        active_count = len(active_indices)
        if active_pixel_values_aux.shape[0] != active_count:
            return None
        if aux_roi_features is None and any(
            not isinstance(feature, torch.Tensor) or feature.shape[0] != active_count
            for feature in aux_feats  # type: ignore[union-attr]
        ):
            return None

        active_image_sizes = [image_sizes[idx] for idx in active_indices]
        if len(set(active_image_sizes)) != 1:
            return None

        vt_levels_by_image = [
            self._extract_vt_multi_level_features(
                image_embeds[idx],
                grid_thw_shapes[idx],
                merge_size,
            )
            for idx in active_indices
        ]
        num_vt_levels = len(vt_levels_by_image[0])
        if any(len(levels) != num_vt_levels for levels in vt_levels_by_image):
            return None

        batched_vt_levels: list[torch.Tensor] = []
        for level_idx in range(num_vt_levels):
            level_features = [levels[level_idx] for levels in vt_levels_by_image]
            first_shape = level_features[0].shape
            if first_shape[0] != 1:
                return None
            if any(feature.shape != first_shape for feature in level_features):
                return None
            batched_vt_levels.append(torch.cat(level_features, dim=0))

        vt_feats: list[torch.Tensor] | torch.Tensor = batched_vt_levels
        if getattr(self.config, "mm_use_simpleFPN_for_vt", False) and not getattr(
            self.config, "mm_use_simpleFPN_multiLayer_for_vt", False
        ):
            vt_feats = batched_vt_levels[-1]

        prepared_vt_fpn_features = vt_fpn_features
        if prepared_vt_fpn_features is not None:
            prepared_batch = prepared_vt_fpn_features[0].shape[0]
            all_images_are_active = (
                len(active_indices) == len(image_embeds)
                and all(idx == pos for pos, idx in enumerate(active_indices))
            )
            if prepared_batch == len(image_embeds) and not all_images_are_active:
                active_index_tensor = torch.tensor(
                    active_indices,
                    device=prepared_vt_fpn_features[0].device,
                    dtype=torch.long,
                )
                prepared_vt_fpn_features = [
                    feature.index_select(0, active_index_tensor)
                    for feature in prepared_vt_fpn_features
                ]
            elif prepared_batch != active_count:
                prepared_vt_fpn_features = None

        current_h = active_pixel_values_aux.shape[-2]
        current_w = active_pixel_values_aux.shape[-1]
        aux_h, aux_w = aux_outputs.get("input_size", (current_h, current_w))
        boxes_for_roi: list[torch.Tensor] = []
        vt_boxes_for_roi: list[torch.Tensor] = []
        counts: list[int] = []
        for batch_idx, image_size in zip(active_indices, active_image_sizes):
            boxes = bbox_list[batch_idx].to(
                device=active_pixel_values_aux.device,
                dtype=torch.float32,
            )
            if boxes.numel() == 0:
                return None
            boxes = boxes.reshape(-1, 4)
            orig_h, orig_w = image_size
            aux_boxes = boxes.clone()
            aux_boxes[:, 0::2].mul_(aux_w / current_w)
            aux_boxes[:, 1::2].mul_(aux_h / current_h)
            vt_boxes = boxes.clone()
            vt_boxes[:, 0::2].mul_(orig_w / current_w)
            vt_boxes[:, 1::2].mul_(orig_h / current_h)
            boxes_for_roi.append(aux_boxes)
            vt_boxes_for_roi.append(vt_boxes)
            counts.append(int(boxes.shape[0]))

        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        extracted = self.object_vp_extractor(
            multi_level_features=aux_feats,
            vt_multi_level_features=vt_feats,
            boxes=boxes_for_roi,
            vt_boxes=vt_boxes_for_roi,
            boxes_for_pos_emb=boxes_for_roi,
            vt_boxes_for_pos_emb=vt_boxes_for_roi,
            add_pos_embed=True,
            vt_fpn_features=prepared_vt_fpn_features,
            aux_roi_features=aux_roi_features,
            aux_feature_map_size=aux_outputs.get("feature_map_size"),
        ).squeeze(0)
        if profile:
            self._log_object_feature_timing(
                "roi_extract_batch",
                start_time,
                start_event,
                end_event,
            )

        if extracted.shape[0] != sum(counts):
            return None
        return list(zip(active_indices, torch.split(extracted, counts, dim=0)))

    def _encode_vt_only_object_features(
        self,
        *,
        image_embeds: Sequence[torch.Tensor],
        image_grid_thw: object,
        pixel_values_aux: object,
        bbox_list: torch.Tensor | list[torch.Tensor],
        bbox_counts: object | None = None,
        image_sizes: list[tuple[int, int]] | None = None,
        async_vt_fpn: _AsyncVtFpnResult | None = None,
        profile: bool = False,
    ) -> list[torch.Tensor]:
        """Encode bbox features from the main ViT without constructing C-RADIO."""
        assert self._supports_object_features()
        assert self.object_vp_extractor is not None
        assert self.mm_projector_aux is not None

        grid_thw = self._normalize_image_grid_thw(image_grid_thw)
        grid_thw_shapes = self._grid_thw_to_tuples(grid_thw)
        num_images = len(grid_thw_shapes)
        bbox_counts = self._normalize_bbox_counts(bbox_counts)
        bbox_device = image_embeds[0].device
        bbox_list = self._normalize_bbox_list(
            bbox_list,
            num_images=num_images,
            device=bbox_device,
            bbox_counts=bbox_counts,
        )
        active_indices = [
            idx for idx, boxes in enumerate(bbox_list) if boxes.numel() > 0
        ]
        if not active_indices:
            return []

        pixel_values_aux_items = self._split_pixel_values_aux(
            pixel_values_aux,
            num_images=num_images,
        )
        patch_size = self.visual.patch_size
        if image_sizes is None:
            image_sizes = [
                (
                    grid_thw_shapes[i][1] * patch_size,
                    grid_thw_shapes[i][2] * patch_size,
                )
                for i in range(num_images)
            ]

        merge_size = self.visual.spatial_merge_size
        # A batched prepared FPN tensor has a different batch layout from the
        # per-image ROI call. Reuse it only for the single-image fast path.
        if num_images == 1:
            vt_fpn_features = self._finish_async_vt_fpn(
                async_vt_fpn,
                profile=profile,
            )
        else:
            vt_fpn_features = None
            if async_vt_fpn is not None:
                self._finish_async_vt_fpn(async_vt_fpn, profile=profile)

        projector_inputs: list[tuple[int, torch.Tensor]] = []
        for batch_idx in active_indices:
            vt_feats = self._extract_vt_multi_level_features(
                image_embeds[batch_idx],
                grid_thw_shapes[batch_idx],
                merge_size,
            )
            extracted = self._extract_object_features_for_image(
                aux_out={},
                boxes=bbox_list[batch_idx],
                vt_feats=vt_feats,
                image_size=image_sizes[batch_idx],
                pixel_values_aux=pixel_values_aux_items[batch_idx],
                profile=profile,
                vt_fpn_features=vt_fpn_features if num_images == 1 else None,
            )
            projector_inputs.append((batch_idx, extracted))

        if len(projector_inputs) == 1:
            projector_input = projector_inputs[0][1]
        else:
            projector_input = torch.cat(
                [item[1] for item in projector_inputs],
                dim=0,
            )
        proj_dtype = self._get_mm_projector_aux_dtype(projector_input.dtype)
        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        projected = self.mm_projector_aux(projector_input.to(proj_dtype))
        if profile:
            self._log_object_feature_timing(
                "aux_projector", start_time, start_event, end_event
            )

        object_features: list[torch.Tensor | None] = [None] * num_images
        offset = 0
        for batch_idx, extracted in projector_inputs:
            next_offset = offset + extracted.shape[0]
            object_features[batch_idx] = projected[offset:next_offset]
            offset = next_offset
        empty_feature = projected.new_empty((0, projected.shape[-1]))
        return [
            feature if feature is not None else empty_feature
            for feature in object_features
        ]

    def _encode_object_features(
        self,
        *,
        image_embeds: Sequence[torch.Tensor],
        image_grid_thw: object,
        pixel_values_aux: object,
        bbox_list: torch.Tensor | list[torch.Tensor],
        bbox_counts: object | None = None,
        image_sizes: list[tuple[int, int]] | None = None,
        aux_outputs: object | None = None,
        active_pixel_values_aux: torch.Tensor | None = None,
        active_aux_count: int | None = None,
        async_vt_fpn: _AsyncVtFpnResult | None = None,
        profile: bool = False,
    ) -> list[torch.Tensor]:
        assert self._supports_object_features()

        if self._uses_vt_only_object_features():
            return self._encode_vt_only_object_features(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                pixel_values_aux=pixel_values_aux,
                bbox_list=bbox_list,
                bbox_counts=bbox_counts,
                image_sizes=image_sizes,
                async_vt_fpn=async_vt_fpn,
                profile=profile,
            )

        grid_thw = self._normalize_image_grid_thw(image_grid_thw)
        grid_thw_shapes = self._grid_thw_to_tuples(grid_thw)
        num_images = len(grid_thw_shapes)
        bbox_counts = self._normalize_bbox_counts(bbox_counts)
        pixel_values_aux_items = self._split_pixel_values_aux(
            pixel_values_aux,
            num_images=num_images,
        )

        bbox_list = self._normalize_bbox_list(
            bbox_list,
            num_images=num_images,
            device=self._pixel_values_aux_device(pixel_values_aux),
            bbox_counts=bbox_counts,
        )
        active_indices = [
            idx for idx, boxes in enumerate(bbox_list) if boxes.numel() > 0
        ]
        if not active_indices:
            return []
        if aux_outputs is not None and active_aux_count != len(active_indices):
            raise ValueError(
                "Async aux output active image count does not match bbox_list: "
                f"got active_aux_count={active_aux_count}, "
                f"expected {len(active_indices)}"
            )

        patch_size = self.visual.patch_size
        if image_sizes is None:
            image_sizes = [
                (
                    grid_thw_shapes[i][1] * patch_size,
                    grid_thw_shapes[i][2] * patch_size,
                )
                for i in range(num_images)
            ]

        merge_size = self.visual.spatial_merge_size
        if num_images == 1:
            single_pixel_values_aux = pixel_values_aux_items[0]
            boxes = bbox_list[0]
            vt_feats = self._extract_vt_multi_level_features(
                image_embeds[0],
                grid_thw_shapes[0],
                merge_size,
            )
            if aux_outputs is None:
                if profile:
                    start_time, start_event, end_event = (
                        self._start_object_feature_timing()
                    )
                aux_outputs = self._run_aux_vision_tower(
                    single_pixel_values_aux,
                    [boxes],
                )
                if profile:
                    self._log_object_feature_timing(
                        "aux_vision_tower",
                        start_time,
                        start_event,
                        end_event,
                    )
            # The single-image tensor path already returns a batch of one.
            # Re-splitting its ROI tensor only rebuilds a dict and creates a
            # short-lived view before the extractor consumes it.
            aux_out = (
                aux_outputs
                if isinstance(aux_outputs, dict)
                else self._split_aux_outputs_by_batch(aux_outputs, num_images=1)[0]
            )
            vt_fpn_features = self._finish_async_vt_fpn(
                async_vt_fpn,
                profile=profile,
            )
            projector_input = self._extract_object_features_for_image(
                aux_out=aux_out,
                boxes=boxes,
                vt_feats=vt_feats,
                image_size=image_sizes[0],
                pixel_values_aux=single_pixel_values_aux,
                profile=profile,
                vt_fpn_features=vt_fpn_features,
            )
            proj_dtype = self._get_mm_projector_aux_dtype(projector_input.dtype)
            if profile:
                start_time, start_event, end_event = self._start_object_feature_timing()
            projected = self.mm_projector_aux(projector_input.to(proj_dtype))
            if profile:
                self._log_object_feature_timing(
                    "aux_projector",
                    start_time,
                    start_event,
                    end_event,
            )
            return [projected]

        if not isinstance(pixel_values_aux, torch.Tensor):
            object_features: list[torch.Tensor | None] = [None] * num_images
            projector_inputs: list[tuple[int, torch.Tensor]] = []
            grouped_active_indices: dict[
                tuple[str, int | None, torch.dtype, tuple[int, ...]],
                list[int],
            ] = {}
            for batch_idx in active_indices:
                pixel_values_for_image = pixel_values_aux_items[batch_idx]
                key = (
                    pixel_values_for_image.device.type,
                    pixel_values_for_image.device.index,
                    pixel_values_for_image.dtype,
                    tuple(pixel_values_for_image.shape[1:]),
                )
                grouped_active_indices.setdefault(key, []).append(batch_idx)

            use_varlen_batch = (
                getattr(self.vision_tower_aux, "varlen_batch_enabled", False)
                and len(active_indices) > 1
                and len(grouped_active_indices) > 1
            )
            if use_varlen_batch:
                if profile:
                    start_time, start_event, end_event = (
                        self._start_object_feature_timing()
                    )
                packed_aux_outputs = self._run_aux_vision_tower(
                    [pixel_values_aux_items[idx] for idx in active_indices],
                    [bbox_list[idx] for idx in active_indices],
                )
                if profile:
                    self._log_object_feature_timing(
                        "aux_vision_tower_varlen",
                        start_time,
                        start_event,
                        end_event,
                    )
                if not isinstance(packed_aux_outputs, list) or len(
                    packed_aux_outputs
                ) != len(active_indices):
                    raise ValueError(
                        "Packed aux output count does not match active images"
                    )
                for batch_idx, aux_out in zip(
                    active_indices,
                    packed_aux_outputs,
                ):
                    pixel_values_for_image = pixel_values_aux_items[batch_idx]
                    vt_feats = self._extract_vt_multi_level_features(
                        image_embeds[batch_idx],
                        grid_thw_shapes[batch_idx],
                        merge_size,
                    )
                    extracted = self._extract_object_features_for_image(
                        aux_out=aux_out,
                        boxes=bbox_list[batch_idx],
                        vt_feats=vt_feats,
                        image_size=image_sizes[batch_idx],
                        pixel_values_aux=pixel_values_for_image,
                        profile=profile,
                    )
                    projector_inputs.append((batch_idx, extracted))
            else:
                for group_indices in grouped_active_indices.values():
                    if len(group_indices) == 1:
                        batched_pixel_values_aux = pixel_values_aux_items[
                            group_indices[0]
                        ]
                    else:
                        batched_pixel_values_aux = torch.cat(
                            [pixel_values_aux_items[idx] for idx in group_indices],
                            dim=0,
                        )
                    if profile:
                        start_time, start_event, end_event = (
                            self._start_object_feature_timing()
                        )
                    aux_outputs_for_group = self._run_aux_vision_tower(
                        batched_pixel_values_aux,
                        [bbox_list[idx] for idx in group_indices],
                    )
                    if profile:
                        self._log_object_feature_timing(
                            "aux_vision_tower",
                            start_time,
                            start_event,
                            end_event,
                        )
                    aux_feature_list = self._split_aux_outputs_by_batch(
                        aux_outputs_for_group,
                        num_images=len(group_indices),
                    )

                    batched_projector_inputs = self._extract_object_features_for_batch(
                        aux_outputs=aux_outputs_for_group,
                        active_indices=group_indices,
                        bbox_list=bbox_list,
                        image_embeds=image_embeds,
                        grid_thw_shapes=grid_thw_shapes,
                        merge_size=merge_size,
                        image_sizes=image_sizes,
                        active_pixel_values_aux=batched_pixel_values_aux,
                        profile=profile,
                    )
                    if batched_projector_inputs is not None:
                        projector_inputs.extend(batched_projector_inputs)
                    else:
                        for active_pos, batch_idx in enumerate(group_indices):
                            pixel_values_for_image = pixel_values_aux_items[batch_idx]
                            aux_out = aux_feature_list[active_pos]
                            vt_feats = self._extract_vt_multi_level_features(
                                image_embeds[batch_idx],
                                grid_thw_shapes[batch_idx],
                                merge_size,
                            )
                            extracted = self._extract_object_features_for_image(
                                aux_out=aux_out,
                                boxes=bbox_list[batch_idx],
                                vt_feats=vt_feats,
                                image_size=image_sizes[batch_idx],
                                pixel_values_aux=pixel_values_for_image,
                                profile=profile,
                            )
                            projector_inputs.append((batch_idx, extracted))

        if not isinstance(pixel_values_aux, torch.Tensor):
            if not projector_inputs:
                return []

            if len(projector_inputs) == 1:
                projector_input = projector_inputs[0][1]
            else:
                projector_input = torch.cat(
                    [item[1] for item in projector_inputs],
                    dim=0,
                )
            proj_dtype = self._get_mm_projector_aux_dtype(projector_input.dtype)
            if profile:
                start_time, start_event, end_event = (
                    self._start_object_feature_timing()
                )
            projected = self.mm_projector_aux(projector_input.to(proj_dtype))
            if profile:
                self._log_object_feature_timing(
                    "aux_projector", start_time, start_event, end_event
                )

            offset = 0
            for batch_idx, extracted in projector_inputs:
                next_offset = offset + extracted.shape[0]
                object_features[batch_idx] = projected[offset:next_offset]
                offset = next_offset

            empty_feature = projected.new_empty((0, projected.shape[-1]))
            return [
                feature if feature is not None else empty_feature
                for feature in object_features
            ]

        if active_pixel_values_aux is None:
            active_pixel_values_aux = self._select_active_aux_pixels(
                pixel_values_aux,
                active_indices,
            )
        if aux_outputs is None:
            if profile:
                start_time, start_event, end_event = self._start_object_feature_timing()
            aux_outputs = self._run_aux_vision_tower(
                active_pixel_values_aux,
                [bbox_list[idx] for idx in active_indices],
            )
            if profile:
                self._log_object_feature_timing(
                    "aux_vision_tower", start_time, start_event, end_event
                )
        aux_feature_list = self._split_aux_outputs_by_batch(
            aux_outputs,
            num_images=len(active_indices),
        )
        vt_fpn_features = self._finish_async_vt_fpn(
            async_vt_fpn,
            profile=profile,
        )

        object_features: list[torch.Tensor | None] = [None] * num_images
        projector_inputs = self._extract_object_features_for_batch(
            aux_outputs=aux_outputs,
            active_indices=active_indices,
            bbox_list=bbox_list,
            image_embeds=image_embeds,
            grid_thw_shapes=grid_thw_shapes,
            merge_size=merge_size,
            image_sizes=image_sizes,
            active_pixel_values_aux=active_pixel_values_aux,
            profile=profile,
            vt_fpn_features=vt_fpn_features,
        )
        if projector_inputs is None:
            vt_multi_level = {
                idx: self._extract_vt_multi_level_features(
                    image_embeds[idx], grid_thw_shapes[idx], merge_size
                )
                for idx in active_indices
            }
            projector_inputs = []
            for active_pos, batch_idx in enumerate(active_indices):
                aux_out = aux_feature_list[active_pos]
                boxes = bbox_list[batch_idx]
                vt_feats = vt_multi_level[batch_idx]
                extracted = self._extract_object_features_for_image(
                    aux_out=aux_out,
                    boxes=boxes,
                    vt_feats=vt_feats,
                    image_size=image_sizes[batch_idx],
                    pixel_values_aux=active_pixel_values_aux,
                    profile=profile,
                )
                projector_inputs.append((batch_idx, extracted))

        if not projector_inputs:
            return []

        if len(projector_inputs) == 1:
            projector_input = projector_inputs[0][1]
        else:
            projector_input = torch.cat(
                [item[1] for item in projector_inputs],
                dim=0,
            )
        proj_dtype = self._get_mm_projector_aux_dtype(projector_input.dtype)
        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        projected = self.mm_projector_aux(projector_input.to(proj_dtype))
        if profile:
            self._log_object_feature_timing(
                "aux_projector", start_time, start_event, end_event
            )

        offset = 0
        for batch_idx, extracted in projector_inputs:
            next_offset = offset + extracted.shape[0]
            object_features[batch_idx] = projected[offset:next_offset]
            offset = next_offset

        empty_feature = projected.new_empty((0, projected.shape[-1]))
        return [
            feature if feature is not None else empty_feature
            for feature in object_features
        ]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        bbox_list = kwargs.get("bbox_list")
        bbox_counts = kwargs.get("bbox_counts")
        pixel_values_aux = kwargs.get("pixel_values_aux")
        image_grid_thw_arg = kwargs.get("image_grid_thw")

        profile_object_features = _env_flag("OMCHAT_PROFILE_OBJECT_FEATURES")
        if profile_object_features:
            total_start_time, total_start_event, total_end_event = (
                self._start_object_feature_timing()
            )
        else:
            total_start_time = 0.0
            total_start_event = None
            total_end_event = None
        async_aux = self._maybe_start_async_aux_vision_tower(
            pixel_values_aux=pixel_values_aux,
            bbox_list=bbox_list,
            bbox_counts=bbox_counts,
            image_grid_thw=image_grid_thw_arg,
            profile=profile_object_features,
        )
        if profile_object_features:
            start_time, start_event, end_event = self._start_object_feature_timing()
        embeddings = super().embed_multimodal(**kwargs)
        if profile_object_features:
            self._log_object_feature_timing(
                "base_embed_multimodal", start_time, start_event, end_event
            )

        # Keep processor inputs and the main ViT workload intact for ablation
        # measurements while bypassing the optional object-feature encoder.
        if _env_flag("OMCHAT_SKIP_AUX_VIT"):
            if profile_object_features:
                self._log_object_feature_timing(
                    "multimodal_embed_total",
                    total_start_time,
                    total_start_event,
                    total_end_event,
                )
            return embeddings

        image_grid_thw = None
        image_embeds = None
        async_vt_fpn = None
        expected_object_counts: list[int] | None = None
        if bbox_list is not None:
            if image_grid_thw_arg is None:
                raise ValueError(
                    "bbox_list requires image_grid_thw to encode object features"
                )
            image_grid_thw = self._normalize_image_grid_thw(image_grid_thw_arg)
            expected_object_counts = self._expected_object_feature_counts(
                bbox_list,
                bbox_counts,
                num_images=int(image_grid_thw.shape[0]),
            )

        if (
            _env_flag("OMCHAT_PARALLEL_VT_FPN")
            and bbox_list is not None
            and (
                isinstance(pixel_values_aux, torch.Tensor)
                or self._uses_vt_only_object_features()
            )
            and image_grid_thw_arg is not None
            and embeddings is not None
            and self._ensure_object_modules()
        ):
            assert image_grid_thw is not None
            num_images = int(image_grid_thw.shape[0])
            image_embeds = self._get_reused_image_embeds(
                embeddings,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
            )
            grid_shapes = self._grid_thw_to_tuples(image_grid_thw)
            vt_levels = [
                self._extract_vt_multi_level_features(
                    image_embeds[idx],
                    grid_shapes[idx],
                    self.visual.spatial_merge_size,
                )[-1]
                for idx in range(num_images)
            ]
            if vt_levels and all(
                feature.shape == vt_levels[0].shape for feature in vt_levels[1:]
            ):
                vt_feature = (
                    vt_levels[0]
                    if len(vt_levels) == 1
                    else torch.cat(vt_levels, dim=0)
                )
                async_vt_fpn = self._maybe_start_async_vt_fpn(
                    vt_feature,
                    profile=profile_object_features,
                )

        if async_aux is not None:
            if profile_object_features:
                start_time, start_event, end_event = (
                    self._start_object_feature_timing()
                )
            torch.cuda.current_stream(
                async_aux.active_pixel_values_aux.device
            ).wait_stream(async_aux.stream)
            if profile_object_features:
                self._log_object_feature_timing(
                    "wait_aux_vision_tower_async",
                    start_time,
                    start_event,
                    end_event,
                )
                self._log_recorded_object_feature_timing(
                    "aux_vision_tower_async",
                    async_aux.start_time,
                    async_aux.start_event,
                    async_aux.end_event,
                )

        if expected_object_counts is not None and any(expected_object_counts):
            if pixel_values_aux is None:
                raise ValueError(
                    "bbox_list produced object placeholders but pixel_values_aux "
                    "is missing"
                )
            if embeddings is None:
                raise ValueError(
                    "bbox_list produced object placeholders but the main vision "
                    "tower returned no embeddings"
                )
            if async_aux is None and not self._ensure_object_modules():
                error = RuntimeError(
                    "bbox_list requires OmChat object-feature modules, but their "
                    "initialization failed"
                )
                if self._object_modules_init_error is not None:
                    raise error from self._object_modules_init_error
                raise error

            assert image_grid_thw is not None
            expected_image_counts = (
                image_grid_thw.prod(-1)
                // self.visual.spatial_merge_size
                // self.visual.spatial_merge_size
            ).detach().cpu().to(dtype=torch.long).tolist()
            actual_image_counts = [int(embedding.shape[0]) for embedding in embeddings]
            if actual_image_counts != expected_image_counts:
                raise ValueError(
                    "OmChat main-vision embedding count mismatch before merge: "
                    f"expected {expected_image_counts}, got {actual_image_counts}"
                )
            if image_embeds is None:
                num_images = int(image_grid_thw.shape[0])
                image_embeds = self._get_reused_image_embeds(
                    embeddings,
                    image_grid_thw=image_grid_thw,
                    num_images=num_images,
                )
            object_feats = self._encode_object_features(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                pixel_values_aux=pixel_values_aux,
                bbox_list=bbox_list,
                bbox_counts=bbox_counts,
                aux_outputs=None if async_aux is None else async_aux.outputs,
                active_pixel_values_aux=(
                    None if async_aux is None else async_aux.active_pixel_values_aux
                ),
                active_aux_count=None if async_aux is None else async_aux.active_count,
                async_vt_fpn=async_vt_fpn,
                profile=profile_object_features,
            )
            actual_object_counts = [int(feature.shape[0]) for feature in object_feats]
            if actual_object_counts != expected_object_counts:
                raise ValueError(
                    "OmChat object-feature embedding count mismatch before merge: "
                    f"expected {expected_object_counts}, got {actual_object_counts}"
                )
            if len(embeddings) != len(object_feats):
                raise ValueError(
                    "OmChat image/object embedding item count mismatch before merge: "
                    f"images={len(embeddings)}, objects={len(object_feats)}"
                )

            if profile_object_features:
                start_time, start_event, end_event = (
                    self._start_object_feature_timing()
                )
            embeddings = tuple(
                torch.cat((base_embedding, object_feature.to(base_embedding)), dim=0)
                for base_embedding, object_feature in zip(embeddings, object_feats)
            )
            if profile_object_features:
                self._log_object_feature_timing(
                    "append_object_features", start_time, start_event, end_event
                )

        if profile_object_features:
            self._log_object_feature_timing(
                "multimodal_embed_total",
                total_start_time,
                total_start_event,
                total_end_event,
            )
        return embeddings

    def _process_image_input(self, image_input: object) -> tuple[torch.Tensor, ...]:
        profile = _env_flag("OMCHAT_PROFILE_OBJECT_FEATURES")
        grid_thw = image_input["image_grid_thw"]  # type: ignore[index]
        assert grid_thw.ndim == 2

        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        if image_input["type"] == "image_embeds":  # type: ignore[index]
            image_embeds = image_input["image_embeds"].type(  # type: ignore[index]
                self.visual.dtype
            )
        else:
            pixel_values = image_input["pixel_values"].type(  # type: ignore[index]
                self.visual.dtype
            )
        if profile:
            self._log_object_feature_timing(
                "main_vision_input_prepare",
                start_time,
                start_event,
                end_event,
            )

        if image_input["type"] != "image_embeds":  # type: ignore[index]
            if profile:
                start_time, start_event, end_event = (
                    self._start_object_feature_timing()
                )
            if self.use_data_parallel:
                image_embeds_split = run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values,
                    grid_thw.tolist(),
                    rope_type="rope_3d",
                )
                if profile:
                    self._log_object_feature_timing(
                        "main_vision_tower",
                        start_time,
                        start_event,
                        end_event,
                    )
                return image_embeds_split
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
            if profile:
                self._log_object_feature_timing(
                    "main_vision_tower",
                    start_time,
                    start_event,
                    end_event,
                )

        if profile:
            start_time, start_event, end_event = self._start_object_feature_timing()
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        image_embeds_split = image_embeds.split(sizes)
        if profile:
            self._log_object_feature_timing(
                "main_vision_postprocess",
                start_time,
                start_event,
                end_event,
            )
        return image_embeds_split

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weight_list = list(weights)
        object_prefixes = (
            "model.vision_tower_aux.",
            "model.object_vp_extractor.",
            "model.mm_projector_aux.",
            "vision_tower_aux.",
            "object_vp_extractor.",
            "mm_projector_aux.",
        )
        if any(name.startswith(object_prefixes) for name, _ in weight_list):
            if self._uses_vt_only_object_features():
                # The checkpoint can still contain the training-time C-RADIO
                # weights. There is no matching module in VT-only inference.
                weight_list = [
                    (name, weight)
                    for name, weight in weight_list
                    if not name.startswith(
                        (
                            "model.vision_tower_aux.",
                            "vision_tower_aux.",
                        )
                    )
                ]
            if not self._ensure_object_modules():
                error = RuntimeError(
                    "Checkpoint contains OmChat object-feature weights, but the "
                    "required modules could not be initialized"
                )
                if self._object_modules_init_error is not None:
                    raise error from self._object_modules_init_error
                raise error

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp.", "model.mm_projector."],
        )
        loaded = loader.load_weights(iter(weight_list), mapper=self.hf_to_vllm_mapper)
        process_aux_vision_fp8_modules(self.vision_tower_aux)
        process_flashrt_nvfp4_torch_linears(self.vision_tower_aux)
        process_flashrt_nvfp4_torch_linears(self.mm_projector_aux)
        for module in self.modules():
            refresh = getattr(module, "_refresh_flashrt_effective_weight", None)
            if refresh is not None:
                refresh()
        return loaded
