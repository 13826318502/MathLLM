"""Conversion from vLLM chunks to the public SSE schema."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


def extract_stream_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def extract_completion_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


async def public_sse_events(
    raw_events: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    async for payload in raw_events:
        content = extract_stream_content(payload)
        if content:
            yield {"content": content, "finished": False}
    yield {"content": "", "finished": True}


def encode_sse(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
