# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
import time

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.utils.omchat_timing import pop_shared_image_pack_start

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"
OMCHAT_PROFILE_ID_HEADER = "x-omchat-profile-id"


def _omchat_profile_response_headers(raw_request: Request) -> dict[str, str]:
    profile_id = raw_request.headers.get(OMCHAT_PROFILE_ID_HEADER, "")
    if not profile_id:
        return {}

    headers = {
        OMCHAT_PROFILE_ID_HEADER: profile_id,
        "x-omchat-vllm-response-ready-ns": str(time.monotonic_ns()),
    }
    shared_pack_start_ns = pop_shared_image_pack_start(profile_id)
    if shared_pack_start_ns is not None:
        headers["x-omchat-shared-pack-start-ns"] = str(shared_pack_start_ns)

    metadata = getattr(raw_request.state, "request_metadata", None)
    for name, header in (
        ("omchat_scheduled_ns", "x-omchat-scheduled-ns"),
        ("omchat_first_token_ns", "x-omchat-first-token-ns"),
        ("omchat_last_token_ns", "x-omchat-last-token-ns"),
    ):
        value = getattr(metadata, name, None)
        if value is not None:
            headers[header] = str(value)
    return headers


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    profile_id = raw_request.headers.get(OMCHAT_PROFILE_ID_HEADER, "")
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except BaseException:
        pop_shared_image_pack_start(profile_id)
        raise

    if isinstance(generator, ErrorResponse):
        pop_shared_image_pack_start(profile_id)
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        response_headers = dict(metrics_header(metrics_header_format) or {})
        response_headers.update(_omchat_profile_response_headers(raw_request))
        return JSONResponse(
            content=generator.model_dump(),
            headers=response_headers,
        )

    pop_shared_image_pack_start(profile_id)
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
