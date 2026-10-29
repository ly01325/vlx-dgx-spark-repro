#!/usr/bin/env python3
"""Run the bundled VLX ModelOpt NVFP4 exporter from a JSON config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("quant_config.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = _read_config(config_path)
    model_root = _resolve(config_dir, str(config.get("model_root", ".")))
    source = _resolve(model_root, str(config["source_model"]))
    output = _resolve(model_root, str(config["output_model"]))
    exporter = _resolve(config_dir, str(config["exporter"]))
    scope_reference = _resolve(config_dir, str(config["scope_reference"]))

    command = [
        sys.executable,
        str(exporter),
        "--src",
        str(source),
        "--reference-flashrt",
        str(scope_reference),
        "--dst",
        str(output),
        "--group-size",
        str(int(config.get("group_size", 16))),
        "--expected-count",
        str(int(config.get("expected_count", 186))),
    ]

    activation = config.get("activation_scale", {"mode": "direct", "value": None})
    if not isinstance(activation, dict):
        raise ValueError("activation_scale must be a JSON object")
    mode = activation.get("mode", "direct")
    value = activation.get("value")
    if mode == "direct":
        if value is not None:
            raise ValueError("direct activation mode requires a null value")
    elif mode == "input_scale":
        command.extend(["--input-scale", str(float(value))])
    elif mode == "input_amax":
        command.extend(["--input-amax", str(float(value))])
    elif mode == "input_scale_json":
        command.extend(["--input-scale-json", str(_resolve(config_dir, str(value)))])
    else:
        raise ValueError(f"Unsupported activation_scale mode: {mode}")

    if args.dry_run:
        command.append("--dry-run")

    print(f"Config : {config_path}", flush=True)
    print(f"Command: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=config_dir, check=True)


if __name__ == "__main__":
    main()
