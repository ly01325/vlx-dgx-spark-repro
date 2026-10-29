#!/usr/bin/env python3
"""Export the C-RADIO block stack with a dynamic token dimension.

The image conditioner, resize, patch embedding, CPE position embedding, and ROI
Align intentionally stay outside TensorRT. The engine contract is therefore
tokens [B, L, 1152] -> the complete outputs of blocks 6, 13, 19, and 26.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# These must be set before object_modules constructs the aux tower. The vLLM
# FlashAttention wrapper is not ONNX-exportable; timm SDPA is exportable.
os.environ["OMCHAT_AUX_FLASH_ATTN"] = "0"
os.environ["OMCHAT_AUX_VARLEN_BATCH"] = "0"

import torch
from safetensors import safe_open

from vllm.model_executor.models.omchat.object_modules import build_vision_tower_aux
from vllm.transformers_utils.configs.omchat_qwen3_5 import OmChatQwen3_5Config


OUTPUT_NAMES = ("feat_l06", "feat_l13", "feat_l19", "feat_l26")
CHECKPOINT_PREFIXES = ("model.vision_tower_aux.", "vision_tower_aux.")


class AuxBlockStack(torch.nn.Module):
    def __init__(self, blocks: torch.nn.ModuleList, indices: tuple[int, ...]) -> None:
        super().__init__()
        self.blocks = blocks
        self.indices = indices

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        hidden_states = tokens
        next_output = 0
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if layer_index == self.indices[next_output]:
                outputs.append(hidden_states)
                next_output += 1
                if next_output == len(self.indices):
                    break
        return tuple(outputs)


def load_aux_state(model_path: Path, tower: torch.nn.Module) -> dict[str, object]:
    checkpoint = model_path / "model.safetensors"
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            for prefix in CHECKPOINT_PREFIXES:
                if name.startswith(prefix):
                    state[name.removeprefix(prefix)] = handle.get_tensor(name)
                    break
    if not state:
        raise RuntimeError(f"No C-RADIO tensors found in {checkpoint}")
    result = tower.load_state_dict(state, strict=False)
    # Non-persistent runtime buffers are allowed, but learned tensors are not.
    missing = [name for name in result.missing_keys if not name.endswith("locks")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"C-RADIO checkpoint mismatch: missing={missing}, "
            f"unexpected={result.unexpected_keys}"
        )
    return {"loaded_tensors": len(state), "missing_runtime": result.missing_keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-length", type=int, default=266)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    config = OmChatQwen3_5Config.from_pretrained(args.model_path)
    tower = build_vision_tower_aux(config, str(args.model_path))
    load_report = load_aux_state(args.model_path, tower)
    indices = tuple(int(value) for value in tower._out_indices_tuple)
    if indices != (6, 13, 19, 26):
        raise RuntimeError(f"Unexpected C-RADIO tap indices: {indices}")

    backbone = tower.image_tower.radio_model.model
    block0 = backbone.blocks[0]
    model = AuxBlockStack(backbone.blocks, indices).eval().float().cuda()
    tokens = torch.randn(
        1, args.sample_length, backbone.embed_dim, device="cuda", dtype=torch.float32
    )
    with torch.inference_mode():
        outputs = model(tokens)
    if len(outputs) != 4:
        raise RuntimeError(f"Expected four outputs, got {len(outputs)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {"tokens": {1: "token_length"}}
    dynamic_axes.update({name: {1: "token_length"} for name in OUTPUT_NAMES})
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (tokens,),
            str(args.output),
            input_names=["tokens"],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
            external_data=True,
            dynamo=False,
        )

    metadata = {
        "contract": {
            "input": {"name": "tokens", "shape": [1, "L", 1152]},
            "outputs": [
                {"name": name, "shape": [1, "L", 1152]}
                for name in OUTPUT_NAMES
            ],
            "dtype": "builder-selectable; BF16 deployment",
            "token_length": "L = 10 + (prepared_H / 16) * (prepared_W / 16)",
        },
        "tap_indices_zero_based": list(indices),
        "export_sample_length": args.sample_length,
        "onnx_opset": args.opset,
        "load": load_report,
        "structure": {
            "blocks": len(backbone.blocks),
            "embed_dim": backbone.embed_dim,
            "num_heads": block0.attn.num_heads,
            "head_dim": block0.attn.head_dim,
            "layer_norm_eps": block0.norm1.eps,
            "activation": repr(block0.mlp.act),
        },
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
