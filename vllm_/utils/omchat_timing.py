# SPDX-License-Identifier: Apache-2.0

"""Request timing handoff between OmChat's API and multimodal processor."""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token

_LOCK = threading.Lock()
_SHARED_IMAGE_PACK_START_NS: dict[str, int] = {}
_PROFILE_ID: ContextVar[str] = ContextVar("omchat_profile_id", default="")


def set_profile_id(profile_id: str) -> Token[str]:
    return _PROFILE_ID.set(profile_id)


def reset_profile_id(token: Token[str]) -> None:
    _PROFILE_ID.reset(token)


def current_profile_id() -> str:
    return _PROFILE_ID.get()


def record_shared_image_pack_start(profile_id: str, timestamp_ns: int) -> None:
    """Record the first pack start for one profiled request."""
    if not profile_id:
        return
    with _LOCK:
        _SHARED_IMAGE_PACK_START_NS.setdefault(profile_id, timestamp_ns)


def pop_shared_image_pack_start(profile_id: str) -> int | None:
    if not profile_id:
        return None
    with _LOCK:
        return _SHARED_IMAGE_PACK_START_NS.pop(profile_id, None)
