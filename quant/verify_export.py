#!/usr/bin/env python3
"""Audit the configured NVFP4 export against its sanitized BF16 source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
from safetensors import safe_open

from quantize_vlx_seek_native_nvfp4_w4a4 import _validate_export


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("quant_config.json"),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_root = _resolve(config_dir, str(config.get("model_root", ".")))
    source_dir = _resolve(model_root, str(config["source_model"]))
    output_dir = _resolve(model_root, str(config["output_model"]))
    scope_dir = _resolve(config_dir, str(config["scope_reference"]))
    target_keys = json.loads(
        (scope_dir / "flashrt_nvfp4_manifest.json").read_text(encoding="utf-8")
    )["quantized_keys"]
    target_set = set(target_keys)

    counts = _validate_export(output_dir, target_keys)
    source_checkpoint = source_dir / "model.safetensors"
    output_checkpoint = output_dir / "model.safetensors"
    with (
        safe_open(source_checkpoint, framework="pt", device="cpu") as source,
        safe_open(output_checkpoint, framework="pt", device="cpu") as output,
    ):
        source_keys = set(source.keys())
        output_keys = set(output.keys())
        expected_output_keys = source_keys | {
            key[: -len(".weight")] + suffix
            for key in target_set
            for suffix in (".input_scale", ".weight_scale", ".weight_scale_2")
        }
        if output_keys != expected_output_keys:
            raise AssertionError(
                "Output key mismatch: "
                f"missing={len(expected_output_keys - output_keys)}, "
                f"unexpected={len(output_keys - expected_output_keys)}"
            )

        equal_count = 0
        for key in sorted(source_keys - target_set):
            if not torch.equal(source.get_tensor(key), output.get_tensor(key)):
                raise AssertionError(f"Unquantized tensor changed: {key}")
            equal_count += 1

    report = {
        "status": "passed",
        "source_model": config["source_model"],
        "output_model": config["output_model"],
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "output_checkpoint_sha256": _sha256(output_checkpoint),
        "source_tensor_count": len(source_keys),
        "output_tensor_count": len(output_keys),
        "quantized_weight_count": len(target_set),
        "bitwise_equal_unquantized_tensor_count": equal_count,
        "added_scale_tensor_count": len(output_keys - source_keys),
        "export_validation_counts": counts,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "nvidia_modelopt": importlib.metadata.version("nvidia-modelopt"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "reproduction_file_sha256": {
            "quant_config.json": _sha256(config_path),
            "quantize_vlx_seek_native_nvfp4_w4a4.py": _sha256(
                config_dir / "quantize_vlx_seek_native_nvfp4_w4a4.py"
            ),
            "run_export.py": _sha256(config_dir / "run_export.py"),
            "scope_manifest": _sha256(
                scope_dir / "flashrt_nvfp4_manifest.json"
            ),
        },
    }
    report_path = config_dir / "export_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
