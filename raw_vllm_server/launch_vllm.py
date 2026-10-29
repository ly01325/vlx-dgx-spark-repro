#!/usr/bin/env python3
"""Apply the model's CUDA compatibility setting, then run vLLM's CLI."""

import runpy

import torch


# Required by the OmChat SimpleFPN transpose convolutions on the target GPU.
torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
