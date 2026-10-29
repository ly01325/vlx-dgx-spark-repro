#!/usr/bin/env python3
"""Export a vLLM-native ModelOpt NVFP4 W4A4 VLX-Seek checkpoint.

The quantized scope is copied from the FlashRT ``language_all_linear`` export:
all language-model Linear weights are converted to ModelOpt/vLLM NVFP4 tensor
names, while the main vision tower, C-RADIO, object modules, embeddings, norms,
and lm_head stay in the source dtype.

This direct safetensors exporter intentionally avoids HuggingFace AutoModel
loading because the local VLX-Seek config uses the custom ``omchat_qwen3_5``
model type.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SRC = Path("/workspace/omchat/resources/vlx-seek-1.5-3b")
DEFAULT_FLASHRT = Path(
    "/workspace/omchat/resources/vlx-seek-1.5-3b-flashrt-nvfp4-language-max"
)
DEFAULT_DST = Path("/workspace/omchat/resources/vlx-seek-1.5-3b-native-nvfp4-w4a4")
DEFAULT_EXPECTED_COUNT = 186
DEFAULT_EXPECTED_FUSED_COUNT = 114
NVFP4_MAX = 6.0
FP8_MAX = 448.0

REQUIRED_VISION_SIDECARS = (
    Path("C-RADIOv4-SO400M/config.json"),
    Path("C-RADIOv4-SO400M/preprocessor_config.json"),
    Path("Qwen3.5-2B-Vision_Tower/config.json"),
    Path("Qwen3.5-2B-Vision_Tower/preprocessor_config.json"),
)

FORBIDDEN_TARGET_SUBSTRS = (
    "vision_tower",
    "vision_tower_aux",
    "radio_model",
    "object_vp_extractor",
    "mm_projector",
    ".visual.",
)

IGNORE_MODULES = [
    "lm_head",
    "lm_head*",
    "model.lm_head",
    "model.lm_head*",
    "language_model.lm_head",
    "language_model.lm_head*",
    "model.language_model.lm_head",
    "model.language_model.lm_head*",
    "embed_tokens",
    "embed_tokens*",
    "language_model.model.embed_tokens",
    "language_model.model.embed_tokens*",
    "model.language_model.model.embed_tokens",
    "model.language_model.model.embed_tokens*",
    "model.language_model.embed_tokens",
    "model.language_model.embed_tokens*",
    "visual",
    "visual*",
    "model.visual",
    "model.visual*",
    "model.vision_tower",
    "model.vision_tower*",
    "vision_tower",
    "vision_tower*",
    "vision_tower_aux",
    "vision_tower_aux*",
    "model.vision_tower_aux",
    "model.vision_tower_aux*",
    "object_vp_extractor",
    "object_vp_extractor*",
    "model.object_vp_extractor",
    "model.object_vp_extractor*",
    "mm_projector",
    "mm_projector*",
    "model.mm_projector",
    "model.mm_projector*",
    "mm_projector_aux",
    "mm_projector_aux*",
    "model.mm_projector_aux",
    "model.mm_projector_aux*",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _weight_files(src: Path) -> tuple[dict[str, str], list[str]]:
    index = src / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index)["weight_map"]
    else:
        filename = "model.safetensors"
        path = src / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            weight_map = {key: filename for key in handle.keys()}
    return weight_map, sorted(set(weight_map.values()))


def _flashrt_weight_files(reference: Path) -> list[Path]:
    index = reference / "model.safetensors.index.json"
    if index.is_file():
        filenames = sorted(set(_read_json(index)["weight_map"].values()))
        return [reference / name for name in filenames]
    path = reference / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


def _load_flashrt_targets(reference: Path) -> list[str]:
    manifest = reference / "flashrt_nvfp4_manifest.json"
    if manifest.is_file():
        data = _read_json(manifest)
        keys = data.get("quantized_keys")
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError(f"invalid quantized_keys in {manifest}")
        return list(dict.fromkeys(keys))

    target_keys: list[str] = []
    for path in _flashrt_weight_files(reference):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            target_keys.extend(
                key[: -len("_packed")]
                for key in handle.keys()
                if key.endswith(".weight_packed")
            )
    return list(dict.fromkeys(target_keys))


def _validate_targets(
    *,
    target_keys: list[str],
    weight_map: dict[str, str],
    handles: dict[str, Any],
    group_size: int,
    expected_count: int,
) -> None:
    if expected_count and len(target_keys) != expected_count:
        raise ValueError(
            f"expected {expected_count} FlashRT target weights, got {len(target_keys)}"
        )

    missing = [key for key in target_keys if key not in weight_map]
    if missing:
        raise ValueError(f"{len(missing)} FlashRT targets are missing from source")

    non_language = [
        key
        for key in target_keys
        if not key.startswith("model.language_model.layers.")
        or any(token in key for token in FORBIDDEN_TARGET_SUBSTRS)
    ]
    if non_language:
        raise ValueError(
            "FlashRT target list is not language-only; first offenders: "
            + ", ".join(non_language[:5])
        )

    incompatible: list[str] = []
    for key in target_keys:
        shape = handles[weight_map[key]].get_slice(key).get_shape()
        if len(shape) != 2 or shape[-1] % group_size != 0:
            incompatible.append(f"{key} shape={shape}")
    if incompatible:
        raise ValueError(
            "NVFP4 W4A4 requires 2D weights with K divisible by "
            f"{group_size}; first offenders: {', '.join(incompatible[:5])}"
        )


def _vllm_fused_group(base: str) -> str:
    """Return the vLLM runtime Linear name for an HF checkpoint Linear."""
    fused_suffixes = {
        ".mlp.gate_proj": ".mlp.gate_up_proj",
        ".mlp.up_proj": ".mlp.gate_up_proj",
        ".self_attn.q_proj": ".self_attn.qkv_proj",
        ".self_attn.k_proj": ".self_attn.qkv_proj",
        ".self_attn.v_proj": ".self_attn.qkv_proj",
        ".linear_attn.in_proj_qkv": ".linear_attn.in_proj_qkvz",
        ".linear_attn.in_proj_z": ".linear_attn.in_proj_qkvz",
        ".linear_attn.in_proj_b": ".linear_attn.in_proj_ba",
        ".linear_attn.in_proj_a": ".linear_attn.in_proj_ba",
    }
    for suffix, fused_suffix in fused_suffixes.items():
        if base.endswith(suffix):
            return base[: -len(suffix)] + fused_suffix
    return base


def _quantize(
    weight: torch.Tensor,
    group_size: int,
    weight_scale_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    if weight.ndim != 2 or weight.shape[-1] % group_size != 0:
        raise ValueError(
            "NVFP4 W4A4 requires 2D weights with K divisible by "
            f"{group_size}, got {tuple(weight.shape)}"
        )
    scale, scale2 = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        group_size,
        weight_scale_2,
    )
    packed = NVFP4QTensor.quantize(weight, group_size, scale, scale2)[0]._quantized_data
    return (
        packed.contiguous().to(torch.uint8),
        scale.contiguous().to(torch.float8_e4m3fn),
        scale2.reshape(1).contiguous().to(torch.float32),
    )


def _load_input_scale_map(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"input scale JSON must be an object: {path}")
    return data


def _resolve_input_scale(
    *,
    base: str,
    explicit_scale: float | None,
    input_amax: float | None,
    scale_map: dict[str, Any],
) -> tuple[torch.Tensor, str]:
    fused_group = _vllm_fused_group(base)
    raw = None
    matched_name = None
    for candidate in (
        base,
        base + ".weight",
        fused_group,
        fused_group + ".weight",
    ):
        if candidate in scale_map:
            raw = scale_map[candidate]
            matched_name = candidate
            break
    matched_fused_group = matched_name in (fused_group, fused_group + ".weight")
    source = "default_input_amax"
    if raw is None:
        if scale_map:
            raise KeyError(
                f"{base}: input scale JSON is missing source and fused names "
                f"(expected {base!r} or {fused_group!r})"
            )
        if explicit_scale is not None:
            scale = explicit_scale
            source = "explicit_input_scale"
        elif input_amax is not None:
            scale = input_amax / (NVFP4_MAX * FP8_MAX)
        else:
            # FlashRT writes direct per-16 UE4M3 block scales and uses unit
            # activation global scale. vLLM's native kernel has the same
            # per-block dynamic quantization when its global scale is 1.0.
            scale = 1.0
            source = "flashrt_direct_input_scale"
    elif isinstance(raw, dict):
        if "input_scale" in raw:
            scale = float(raw["input_scale"])
            source = (
                "fused_input_scale_json"
                if matched_fused_group
                else "input_scale_json"
            )
        elif "input_amax" in raw:
            scale = float(raw["input_amax"]) / (NVFP4_MAX * FP8_MAX)
            source = (
                "fused_input_amax_json"
                if matched_fused_group
                else "input_amax_json"
            )
        else:
            raise ValueError(f"{base}: expected input_scale or input_amax in JSON")
    else:
        scale = float(raw)
        source = (
            "fused_input_scale_json"
            if matched_fused_group
            else "input_scale_json"
        )

    if scale <= 0:
        raise ValueError(f"{base}: input scale must be positive, got {scale}")
    return torch.tensor([scale], dtype=torch.float32), source


def _copy_sidecar_files(src: Path, dst: Path) -> None:
    skipped_root_files = {
        "config.json",
        "hf_quant_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    renamed_source_audits = {
        "SANITIZATION.md": "SOURCE_SANITIZATION.md",
        "sanitization_report.json": "source_sanitization_report.json",
        "tensor_provenance.json": "source_tensor_provenance.json",
    }
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out = dst / rel
        out.mkdir(parents=True, exist_ok=True)
        for filename in files:
            is_weight_shard = filename.startswith("model-") and filename.endswith(
                ".safetensors"
            )
            is_root = rel == Path(".")
            if is_root and (
                filename in skipped_root_files
                or is_weight_shard
                or filename.endswith("_manifest.json")
            ):
                continue
            destination_name = (
                renamed_source_audits.get(filename, filename)
                if is_root
                else filename
            )
            destination = out / destination_name
            shutil.copy2(Path(root) / filename, destination)
            if destination_name == "SOURCE_SANITIZATION.md":
                text = destination.read_text(encoding="utf-8")
                text = text.replace(
                    "`sanitization_report.json` and `tensor_provenance.json`",
                    "`source_sanitization_report.json` and "
                    "`source_tensor_provenance.json`",
                )
                destination.write_text(text, encoding="utf-8")


def _build_quant_config(group_size: int) -> dict[str, Any]:
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": group_size,
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": group_size,
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": group_size,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": IGNORE_MODULES,
        "producer": {"name": "omchat-native-nvfp4-w4a4", "version": "1"},
    }


def _write_configs(
    *,
    src: Path,
    dst: Path,
    group_size: int,
    target_keys: list[str],
    reference: Path,
    input_scale_sources: dict[str, int],
    explicit_scale: float | None,
    input_amax: float | None,
    scale_map_path: Path | None,
) -> None:
    config = _read_json(src / "config.json")
    config["quantization_config"] = _build_quant_config(group_size)
    config["native_nvfp4_w4a4"] = {
        "format": "modelopt_fp4",
        "scope": "language_all_linear",
        "group_size": group_size,
        "reference_flashrt": str(reference),
        "flashrt_target_count": len(target_keys),
        "quantized_tensor_count": len(target_keys),
        "vllm_fused_linear_count": len(
            {_vllm_fused_group(key[: -len(".weight")]) for key in target_keys}
        ),
        "vision_quantized_count": 0,
        "input_scale": {
            "calibrated": scale_map_path is not None,
            "source_counts": input_scale_sources,
            "default_input_amax": input_amax,
            "explicit_input_scale": explicit_scale,
            "input_scale_json": str(scale_map_path) if scale_map_path else None,
            "note": (
                "Per-layer input scales were loaded from activation calibration."
                if scale_map_path
                else "Unit activation global scale matches FlashRT direct "
                "per-16 dynamic block scaling."
                if input_amax is None and explicit_scale is None
                else "A user-specified static activation scale was used."
            ),
        },
    }
    _write_json(dst / "config.json", config)
    _write_json(
        dst / "hf_quant_config.json",
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": group_size,
                "kv_cache_quant_algo": None,
                "exclude_modules": IGNORE_MODULES,
            }
        },
    )
    _write_json(
        dst / "native_nvfp4_w4a4_manifest.json",
        {
            "source": str(src),
            "reference_flashrt": str(reference),
            "format": "modelopt_fp4",
            "scope": "language_all_linear",
            "group_size": group_size,
            "quantized_keys": target_keys,
            "flashrt_target_count": len(target_keys),
            "quantized_tensor_count": len(target_keys),
            "vllm_fused_linear_count": len(
                {_vllm_fused_group(key[: -len(".weight")]) for key in target_keys}
            ),
            "vision_quantized_count": 0,
            "input_scale_calibrated": scale_map_path is not None,
            "input_scale_json": str(scale_map_path) if scale_map_path else None,
            "input_scale_sources": input_scale_sources,
        },
    )
    (dst / "QUANTIZATION.md").write_text(
        "# ModelOpt NVFP4 W4A4-direct checkpoint\n\n"
        "This checkpoint was exported from the sanitized BF16 source. It "
        "quantizes 186 language Linear weights into ModelOpt NVFP4 with group "
        "size 16 and direct input scale 1.0. Vision towers, object modules, "
        "embeddings, norms, and lm_head remain BF16.\n\n"
        "The inherited sanitization evidence is named "
        "`source_sanitization_report.json` and "
        "`source_tensor_provenance.json` because their checkpoint hash and "
        "tensor manifest refer to the BF16 source model.\n",
        encoding="utf-8",
    )


def _validate_export(dst: Path, target_keys: list[str]) -> dict[str, int]:
    missing_sidecars = [
        str(path) for path in REQUIRED_VISION_SIDECARS if not (dst / path).is_file()
    ]
    if missing_sidecars:
        raise ValueError(
            "export is missing required vision sidecars: "
            + ", ".join(missing_sidecars)
        )

    expected_bases = {key[: -len(".weight")] for key in target_keys}
    counts = {
        "uint8_weights": 0,
        "input_scale": 0,
        "weight_scale": 0,
        "weight_scale_2": 0,
        "bf16_vision_object_weights": 0,
    }
    with safe_open(
        str(dst / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
        actual_uint8_weights = {
            key
            for key in keys
            if key.endswith(".weight")
            and handle.get_slice(key).get_dtype() == "U8"
        }
        expected_uint8_weights = {base + ".weight" for base in expected_bases}
        if actual_uint8_weights != expected_uint8_weights:
            missing = sorted(expected_uint8_weights - actual_uint8_weights)
            unexpected = sorted(actual_uint8_weights - expected_uint8_weights)
            raise ValueError(
                "exported uint8 weight set differs from FlashRT scope: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        fused_scales: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for base in expected_bases:
            for suffix in (
                ".weight",
                ".input_scale",
                ".weight_scale",
                ".weight_scale_2",
            ):
                key = base + suffix
                if key not in keys:
                    raise ValueError(f"missing exported tensor: {key}")

            weight = handle.get_tensor(base + ".weight")
            input_scale = handle.get_tensor(base + ".input_scale")
            weight_scale = handle.get_tensor(base + ".weight_scale")
            weight_scale_2 = handle.get_tensor(base + ".weight_scale_2")
            if weight.ndim != 2 or weight.shape[1] % 8 != 0:
                raise ValueError(f"invalid packed NVFP4 weight shape: {base}.weight")
            expected_scale_shape = (weight.shape[0], weight.shape[1] // 8)
            if input_scale.dtype != torch.float32 or input_scale.numel() != 1:
                raise ValueError(f"invalid input scale: {base}.input_scale")
            if (
                weight_scale.dtype != torch.float8_e4m3fn
                or tuple(weight_scale.shape) != expected_scale_shape
            ):
                raise ValueError(f"invalid block scale: {base}.weight_scale")
            if (
                weight_scale_2.dtype != torch.float32
                or weight_scale_2.numel() != 1
            ):
                raise ValueError(f"invalid global scale: {base}.weight_scale_2")

            fused_group = _vllm_fused_group(base)
            scales = (input_scale, weight_scale_2)
            if fused_group in fused_scales:
                expected_input_scale, expected_weight_scale_2 = fused_scales[
                    fused_group
                ]
                if not torch.equal(input_scale, expected_input_scale):
                    raise ValueError(
                        f"fused input scales differ in group: {fused_group}"
                    )
                if not torch.equal(weight_scale_2, expected_weight_scale_2):
                    raise ValueError(
                        f"fused global weight scales differ in group: {fused_group}"
                    )
            else:
                fused_scales[fused_group] = scales
            counts["uint8_weights"] += 1
            counts["input_scale"] += 1
            counts["weight_scale"] += 1
            counts["weight_scale_2"] += 1

        for key in keys:
            if not key.endswith(".weight"):
                continue
            if any(token in key for token in FORBIDDEN_TARGET_SUBSTRS):
                dtype = handle.get_tensor(key).dtype
                if dtype != torch.bfloat16:
                    raise ValueError(
                        "vision/C-RADIO/object weight must remain BF16: "
                        f"{key} has dtype {dtype}"
                    )
                counts["bf16_vision_object_weights"] += 1
        if len(fused_scales) != DEFAULT_EXPECTED_FUSED_COUNT:
            raise ValueError(
                f"expected {DEFAULT_EXPECTED_FUSED_COUNT} vLLM fused Linear groups, "
                f"got {len(fused_scales)}"
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--reference-flashrt", type=Path, default=DEFAULT_FLASHRT)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    input_scale_group = parser.add_mutually_exclusive_group()
    input_scale_group.add_argument("--input-amax", type=float, default=None)
    input_scale_group.add_argument("--input-scale", type=float, default=None)
    input_scale_group.add_argument("--input-scale-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    reference = args.reference_flashrt.resolve()
    if src == dst:
        raise ValueError("--dst must differ from --src")
    if args.group_size != 16:
        raise ValueError("vLLM ModelOpt NVFP4 currently expects group size 16")
    if args.input_amax is not None and args.input_amax <= 0:
        raise ValueError("--input-amax must be positive")
    if args.input_scale is not None and args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")
    if not args.dry_run and dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"destination is not empty: {dst}")

    weight_map, filenames = _weight_files(src)
    target_keys = _load_flashrt_targets(reference)
    scale_map = _load_input_scale_map(args.input_scale_json)

    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(
                safe_open(str(src / name), framework="pt", device="cpu")
            )
            for name in filenames
        }
        _validate_targets(
            target_keys=target_keys,
            weight_map=weight_map,
            handles=handles,
            group_size=args.group_size,
            expected_count=args.expected_count,
        )

        print("format      : modelopt_fp4 (vLLM native)")
        print("mode        : NVFP4 W4A4, group size 16")
        print(f"source      : {src}")
        print(f"reference   : {reference}")
        print(f"destination : {dst}")
        print(
            f"quantized   : {len(target_keys)} "
            "FlashRT language_all_linear weights"
        )
        fused_groups = {
            _vllm_fused_group(key[: -len(".weight")]) for key in target_keys
        }
        if len(fused_groups) != DEFAULT_EXPECTED_FUSED_COUNT:
            raise ValueError(
                f"expected {DEFAULT_EXPECTED_FUSED_COUNT} vLLM fused Linear groups, "
                f"got {len(fused_groups)}"
            )
        print(f"vLLM fused : {len(fused_groups)} runtime Linear weights")
        print(
            "kept BF16   : main vision tower, C-RADIO/object modules, "
            "embeddings, lm_head"
        )
        if args.dry_run:
            return

        target_set = set(target_keys)
        tensors: dict[str, torch.Tensor] = {}
        input_scale_sources: dict[str, int] = {}
        fused_weight_amax: dict[str, torch.Tensor] = {}
        fused_input_scale: dict[str, torch.Tensor] = {}
        for key in target_keys:
            base = key[: -len(".weight")]
            fused_group = _vllm_fused_group(base)
            value = handles[weight_map[key]].get_tensor(key)
            weight_amax = value.float().abs().max()
            previous_amax = fused_weight_amax.get(fused_group)
            fused_weight_amax[fused_group] = (
                weight_amax
                if previous_amax is None
                else torch.maximum(previous_amax, weight_amax)
            )
            input_scale, scale_source = _resolve_input_scale(
                base=base,
                explicit_scale=args.input_scale,
                input_amax=args.input_amax,
                scale_map=scale_map,
            )
            input_scale_sources[scale_source] = (
                input_scale_sources.get(scale_source, 0) + 1
            )
            previous_input_scale = fused_input_scale.get(fused_group)
            fused_input_scale[fused_group] = (
                input_scale
                if previous_input_scale is None
                else torch.maximum(previous_input_scale, input_scale)
            )

        fused_weight_scale_2 = {
            group: (amax / (NVFP4_MAX * FP8_MAX)).reshape(1).to(torch.float32)
            for group, amax in fused_weight_amax.items()
        }
        dst.mkdir(parents=True, exist_ok=True)
        quantized_count = 0
        for key, filename in weight_map.items():
            value = handles[filename].get_tensor(key)
            if key not in target_set:
                tensors[key] = value.contiguous()
                continue

            base = key[: -len(".weight")]
            fused_group = _vllm_fused_group(base)
            packed, weight_scale, weight_scale_2 = _quantize(
                value.float(),
                args.group_size,
                fused_weight_scale_2[fused_group],
            )
            input_scale = fused_input_scale[fused_group]
            tensors[base + ".weight"] = packed
            tensors[base + ".input_scale"] = input_scale.clone()
            tensors[base + ".weight_scale"] = weight_scale
            tensors[base + ".weight_scale_2"] = weight_scale_2.clone()
            quantized_count += 1
            if quantized_count % 25 == 0:
                print(
                    f"quantized   : {quantized_count}/{len(target_keys)}",
                    flush=True,
                )

    save_file(
        tensors,
        str(dst / "model.safetensors"),
        metadata={"format": "pt", "quantization": "modelopt_fp4"},
    )
    _copy_sidecar_files(src, dst)
    _write_configs(
        src=src,
        dst=dst,
        group_size=args.group_size,
        target_keys=target_keys,
        reference=reference,
        input_scale_sources=input_scale_sources,
        explicit_scale=args.input_scale,
        input_amax=args.input_amax,
        scale_map_path=args.input_scale_json,
    )
    counts = _validate_export(dst, target_keys)
    print(f"wrote       : {dst}")
    print(
        "validated   : "
        f"{counts['uint8_weights']} uint8 weights, "
        f"{counts['input_scale']} input scales, "
        f"{counts['weight_scale']} block scales, "
        f"{counts['weight_scale_2']} global scales, "
        f"{counts['bf16_vision_object_weights']} BF16 vision/C-RADIO/object weights"
    )


if __name__ == "__main__":
    main()
