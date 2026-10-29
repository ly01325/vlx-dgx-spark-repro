# SPDX-License-Identifier: Apache-2.0
"""Optional TensorRT runtime for C-RADIO token blocks.

Engines are lazy-loaded and selected by token length. Unsupported or
unprofitable shapes return None so the caller can preserve the PyTorch path.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_OUTPUT_NAMES = ("feat_l06", "feat_l13", "feat_l19", "feat_l26")


class _EngineState:
    def __init__(self, path: str) -> None:
        import tensorrt as trt

        self.path = path
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.lock = threading.Lock()
        if self.engine.get_tensor_dtype("tokens") != trt.bfloat16:
            raise TypeError(f"TensorRT C-RADIO input is not BF16: {path}")
        for name in _OUTPUT_NAMES:
            if self.engine.get_tensor_dtype(name) != trt.bfloat16:
                raise TypeError(f"TensorRT C-RADIO output {name} is not BF16: {path}")

    def execute(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = tuple(torch.empty_like(tokens) for _ in _OUTPUT_NAMES)
        with self.lock:
            if not self.context.set_input_shape("tokens", tuple(tokens.shape)):
                raise ValueError(
                    f"TensorRT engine {self.path} rejected shape {tuple(tokens.shape)}"
                )
            self.context.set_tensor_address("tokens", tokens.data_ptr())
            for name, output in zip(_OUTPUT_NAMES, outputs):
                self.context.set_tensor_address(name, output.data_ptr())
            current = torch.cuda.current_stream(tokens.device)
            self.stream.wait_stream(current)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed: {self.path}")
            # ROI Align consumes these tensors immediately. Synchronizing here
            # also makes reuse of the single execution context race-free.
            self.stream.synchronize()
        return outputs


class AuxTensorRTRouter:
    def __init__(self) -> None:
        self.paths = {
            "camera": os.getenv("OMCHAT_AUX_TRT_CAMERA_ENGINE", ""),
            "small": os.getenv("OMCHAT_AUX_TRT_SMALL_ENGINE", ""),
            "main": os.getenv("OMCHAT_AUX_TRT_MAIN_ENGINE", ""),
            "universal": os.getenv("OMCHAT_AUX_TRT_UNIVERSAL_ENGINE", ""),
        }
        self.min_tokens = int(os.getenv("OMCHAT_AUX_TRT_MIN_TOKENS", "4096"))
        self.max_tokens = int(os.getenv("OMCHAT_AUX_TRT_MAX_TOKENS", "15000"))
        self.camera_lengths = {
            int(value)
            for value in os.getenv(
                "OMCHAT_AUX_TRT_CAMERA_LENGTHS", "1600,3610,8170"
            ).split(",")
            if value.strip()
        }
        self._states: dict[str, _EngineState] = {}
        self._lock = threading.Lock()
        self._logged_routes: set[tuple[int, str]] = set()
        missing = [path for path in self.paths.values() if path and not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"TensorRT C-RADIO engine(s) missing: {missing}")
        if not any(self.paths.values()):
            raise ValueError("No TensorRT C-RADIO engine path is configured")

    def _select(self, length: int) -> str | None:
        if self.paths["camera"]:
            return "camera" if length in self.camera_lengths else None
        if length < self.min_tokens or length > self.max_tokens:
            return None
        if length == 8170 and self.paths["main"]:
            return "main"
        if length <= 8169 and self.paths["small"]:
            return "small"
        if 266 <= length <= 16394 and self.paths["universal"]:
            return "universal"
        return None

    def _state(self, name: str) -> _EngineState:
        state = self._states.get(name)
        if state is None:
            with self._lock:
                state = self._states.get(name)
                if state is None:
                    state = _EngineState(self.paths[name])
                    self._states[name] = state
        return state

    def run(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        if (
            tokens.ndim != 3
            or tokens.shape[0] != 1
            or tokens.shape[2] != 1152
            or tokens.dtype != torch.bfloat16
            or tokens.device.type != "cuda"
            or not tokens.is_contiguous()
        ):
            return None
        length = int(tokens.shape[1])
        selected = self._select(length)
        route = (length, selected or "pytorch")
        if route not in self._logged_routes:
            self._logged_routes.add(route)
            logger.info(
                "C-RADIO token length %d routed to %s.",
                length,
                f"TensorRT {selected}" if selected else "PyTorch fallback",
            )
        if selected is None:
            return None
        return self._state(selected).execute(tokens)


def create_aux_tensorrt_router() -> AuxTensorRTRouter | None:
    if not any(
        os.getenv(name)
        for name in (
            "OMCHAT_AUX_TRT_SMALL_ENGINE",
            "OMCHAT_AUX_TRT_CAMERA_ENGINE",
            "OMCHAT_AUX_TRT_MAIN_ENGINE",
            "OMCHAT_AUX_TRT_UNIVERSAL_ENGINE",
        )
    ):
        return None
    return AuxTensorRTRouter()
