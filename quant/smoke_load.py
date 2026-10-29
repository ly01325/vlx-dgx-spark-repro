#!/usr/bin/env python3
"""Load the configured NVFP4 export with the patched OmChat vLLM runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vllm import LLM


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("quant_config.json"),
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_root = _resolve(config_path.parent, str(config.get("model_root", ".")))
    model_path = _resolve(model_root, str(config["output_model"]))

    os.environ.setdefault("VLLM_USE_NATIVE_NVFP4", "1")
    LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=False,
        quantization="modelopt_fp4",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    print("NATIVE_NVFP4_FULL_MODEL_LOAD_OK", flush=True)


if __name__ == "__main__":
    main()
