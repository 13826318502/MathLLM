"""HTTP and SSE client for the local FastAPI service."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests

from .config import API_BASE_URL, REQUEST_TIMEOUT


def check_health() -> dict[str, Any]:
    """Return the backend health payload or raise a request error."""
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=(5, 10),
    )
    if not response.ok:
        raise RuntimeError(error_message(response))
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def iter_sse(response: requests.Response) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from a Server-Sent Events response."""
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _stream_chat_request(
    messages: list[dict[str, str]],
    summary: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Call FastAPI /api/chat and yield its SSE events."""
    payload: dict[str, Any] = {"messages": messages, "stream": True}
    if summary:
        payload["summary"] = summary
    with requests.post(
        f"{API_BASE_URL}/chat",
        json=payload,
        stream=True,
        timeout=(10, REQUEST_TIMEOUT),
    ) as response:
        if not response.ok:
            raise RuntimeError(error_message(response))
        yield from iter_sse(response)


def stream_chat(messages: list[dict[str, str]]) -> Iterator[dict[str, Any]]:
    """Call chat without a summary for backwards compatibility."""
    yield from _stream_chat_request(messages)


def stream_chat_with_summary(
    messages: list[dict[str, str]],
    summary: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Call chat with a compact summary plus recent messages."""
    yield from _stream_chat_request(messages, summary)


def stream_solve(question: str) -> Iterator[dict[str, Any]]:
    """Call FastAPI /api/solve without conversation history."""
    with requests.post(
        f"{API_BASE_URL}/solve",
        json={"question": question, "stream": True},
        stream=True,
        timeout=(10, REQUEST_TIMEOUT),
    ) as response:
        if not response.ok:
            raise RuntimeError(error_message(response))
        yield from iter_sse(response)


def summarize_chat(
    messages: list[dict[str, str]],
    existing_summary: str | None = None,
) -> str:
    """Ask the backend to compress older conversation messages."""
    response = requests.post(
        f"{API_BASE_URL}/memory/summarize",
        json={
            "messages": messages,
            "existing_summary": existing_summary or None,
        },
        timeout=(10, REQUEST_TIMEOUT),
    )
    if not response.ok:
        raise RuntimeError(error_message(response))
    payload = response.json()
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("摘要接口没有返回有效内容")
    return summary.strip()


def error_message(response: requests.Response) -> str:
    """Extract a safe, user-facing error from an HTTP response."""
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)
    except ValueError:
        pass
    return f"后端请求失败（HTTP {response.status_code}）"
