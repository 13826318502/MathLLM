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


def extract_usage(payload: dict[str, Any]) -> tuple[int, int, int] | None:
    """Return (prompt, completion, total) tokens when the server reports them."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    total = usage.get("total_tokens")
    if not isinstance(total, int):
        total = prompt + completion
    return prompt, completion, total


def extract_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first choice's message object, or an empty dict."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def extract_completion_content(payload: dict[str, Any]) -> str:
    content = extract_message(payload).get("content")
    return content if isinstance(content, str) else ""


def extract_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tool calls the model asked for (empty when it answered itself)."""
    calls = extract_message(payload).get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def parse_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Parse ``tool_call.function.arguments``.

    The engine sends arguments as a JSON string, so it still has to be parsed
    and later validated against the tool's Pydantic model. A malformed payload
    becomes an empty dict instead of an exception.
    """
    function = call.get("function")
    if not isinstance(function, dict):
        return {}
    raw = function.get("arguments")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
