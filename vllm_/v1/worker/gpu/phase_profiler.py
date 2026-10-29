# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelForwardProfileContext:
    step_index: int
    scope: str
    phase: str
    prefill_reqs: int
    decode_reqs: int
    prefill_tokens: int
    decode_tokens: int
    scheduled_tokens: int
    padded_tokens: int
    cudagraph_mode: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


class ModelForwardPhaseProfiler:
    """Optional CUDA timing for phase-aware model-forward diagnostics."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._step_index = 0
        self.enabled = os.getenv("OMCHAT_PROFILE_MODEL_PHASES", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def start(
        self,
        input_batch: Any,
        req_states: Any,
        cudagraph_mode: Any,
        *,
        scope: str = "model_forward",
        step_index: int | None = None,
    ) -> ModelForwardProfileContext | None:
        if not self.enabled or self.device.type != "cuda":
            return None

        req_indices = input_batch.idx_mapping_np
        return self.start_from_arrays(
            num_computed_tokens=req_states.num_computed_prefill_tokens[req_indices],
            prefill_lens=req_states.prefill_len.np[req_indices],
            scheduled_tokens=input_batch.num_scheduled_tokens,
            scheduled_token_count=int(input_batch.num_tokens),
            padded_token_count=int(input_batch.num_tokens_after_padding),
            cudagraph_mode=cudagraph_mode,
            scope=scope,
            step_index=step_index,
        )

    def start_from_arrays(
        self,
        *,
        num_computed_tokens: np.ndarray,
        prefill_lens: np.ndarray,
        scheduled_tokens: np.ndarray,
        scheduled_token_count: int,
        padded_token_count: int,
        cudagraph_mode: Any,
        scope: str = "model_forward",
        step_index: int | None = None,
    ) -> ModelForwardProfileContext | None:
        if not self.enabled or self.device.type != "cuda":
            return None

        if step_index is None:
            if scope == "model_step":
                self._step_index += 1
            step_index = self._step_index

        prefill_remaining = np.maximum(prefill_lens - num_computed_tokens, 0)
        prefill_parts = np.minimum(scheduled_tokens, prefill_remaining)
        decode_parts = scheduled_tokens - prefill_parts
        prefill_reqs = int(np.count_nonzero(prefill_parts))
        decode_reqs = int(np.count_nonzero(decode_parts))
        prefill_tokens = int(prefill_parts.sum())
        decode_tokens = int(decode_parts.sum())
        if prefill_reqs and decode_reqs:
            phase = "mixed"
        elif prefill_reqs:
            phase = "prefill_only"
        else:
            phase = "decode_only"

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return ModelForwardProfileContext(
            step_index=step_index,
            scope=scope,
            phase=phase,
            prefill_reqs=prefill_reqs,
            decode_reqs=decode_reqs,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            scheduled_tokens=scheduled_token_count,
            padded_tokens=padded_token_count,
            cudagraph_mode=str(cudagraph_mode),
            start_event=start_event,
            end_event=end_event,
        )

    @staticmethod
    def record_boundary(
        context: ModelForwardProfileContext | None,
    ) -> torch.cuda.Event | None:
        if context is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def finish(context: ModelForwardProfileContext | None) -> float | None:
        if context is None:
            return None
        context.end_event.record()
        context.end_event.synchronize()
        elapsed_ms = context.start_event.elapsed_time(context.end_event)
        logger.info(
            "Model forward step_index=%d phase=%s scope=%s took %.3f ms "
            "prefill_reqs=%d decode_reqs=%d prefill_tokens=%d "
            "decode_tokens=%d scheduled_tokens=%d padded_tokens=%d cudagraph=%s",
            context.step_index,
            context.phase,
            context.scope,
            elapsed_ms,
            context.prefill_reqs,
            context.decode_reqs,
            context.prefill_tokens,
            context.decode_tokens,
            context.scheduled_tokens,
            context.padded_tokens,
            context.cudagraph_mode,
        )
        return elapsed_ms

    @staticmethod
    def log_model_step_breakdown(
        context: ModelForwardProfileContext | None,
        *,
        mm_encoder_end_event: torch.cuda.Event | None,
        model_forward_start_event: torch.cuda.Event | None,
        model_forward_elapsed_ms: float | None,
        model_step_elapsed_ms: float | None,
    ) -> None:
        if (
            context is None
            or mm_encoder_end_event is None
            or model_forward_start_event is None
            or model_forward_elapsed_ms is None
            or model_step_elapsed_ms is None
        ):
            return

        mm_encoder_end_event.synchronize()
        model_forward_start_event.synchronize()
        mm_encoder_elapsed_ms = context.start_event.elapsed_time(mm_encoder_end_event)
        input_prepare_elapsed_ms = mm_encoder_end_event.elapsed_time(
            model_forward_start_event
        )
        other_elapsed_ms = (
            model_step_elapsed_ms
            - mm_encoder_elapsed_ms
            - input_prepare_elapsed_ms
            - model_forward_elapsed_ms
        )
        if abs(other_elapsed_ms) < 0.005:
            other_elapsed_ms = 0.0
        parts_sum_ms = (
            mm_encoder_elapsed_ms
            + input_prepare_elapsed_ms
            + model_forward_elapsed_ms
            + other_elapsed_ms
        )

        logger.info(
            "Model step breakdown step_index=%d phase=%s total=%.3f ms "
            "multimodal_encoder_and_input_embed=%.3f ms "
            "model_input_prepare=%.3f ms "
            "language_model_forward=%.3f ms other=%.3f ms "
            "parts_sum=%.3f ms prefill_reqs=%d decode_reqs=%d "
            "prefill_tokens=%d decode_tokens=%d scheduled_tokens=%d "
            "padded_tokens=%d cudagraph=%s",
            context.step_index,
            context.phase,
            model_step_elapsed_ms,
            mm_encoder_elapsed_ms,
            input_prepare_elapsed_ms,
            model_forward_elapsed_ms,
            other_elapsed_ms,
            parts_sum_ms,
            context.prefill_reqs,
            context.decode_reqs,
            context.prefill_tokens,
            context.decode_tokens,
            context.scheduled_tokens,
            context.padded_tokens,
            context.cudagraph_mode,
        )
