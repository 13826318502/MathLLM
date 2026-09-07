"""Frontend event handlers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from .api_client import (
    check_health,
    stream_chat_with_summary,
    stream_solve,
    summarize_chat,
)
from .conversation import history_to_messages
from .config import (
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MEMORY_TRIGGER_TOKENS,
    RECENT_HISTORY_MESSAGES,
)
from .memory import should_summarize, split_history


def solve_math(
    question: str,
    history: list[Any] | None,
    mode: str,
    memory_summary: str | None,
    memory_cursor: int | None,
) -> Iterator[tuple[list[dict[str, str]], str, str, int]]:
    """Stream one answer into the Gradio chatbot."""
    question = (question or "").strip()
    updated_history: list[dict[str, str]] = list(history or [])
    current_summary = (memory_summary or "").strip()
    current_cursor = max(0, int(memory_cursor or 0))
    if not question:
        yield updated_history, "", current_summary, current_cursor
        return

    updated_history.append({"role": "user", "content": question})
    updated_history.append({"role": "assistant", "content": ""})
    if mode == "solve":
        events = stream_solve(question)
        current_summary = ""
        current_cursor = 0
    else:
        all_messages = history_to_messages(updated_history[:-1])
        recent_budget = max(
            600,
            MAX_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS - 450,
        )
        old_messages, recent_messages = split_history(
            all_messages,
            RECENT_HISTORY_MESSAGES,
            recent_budget,
        )
        current_cursor = min(current_cursor, len(old_messages))
        new_old_messages = old_messages[current_cursor:]
        if should_summarize(
            current_summary,
            all_messages,
            MEMORY_TRIGGER_TOKENS,
            RECENT_HISTORY_MESSAGES,
        ) and new_old_messages:
            try:
                current_summary = summarize_chat(new_old_messages, current_summary)
                current_cursor = len(old_messages)
            except (requests.RequestException, RuntimeError, ValueError):
                # A summary failure should not prevent the user from asking a
                # question; recent context is still useful by itself.
                pass
        events = stream_chat_with_summary(recent_messages, current_summary)

    try:
        for payload in events:
            content = payload.get("content", "")
            if content:
                updated_history[-1]["content"] += str(content)
                yield updated_history, "", current_summary, current_cursor
            if payload.get("finished") is True:
                break
    except (requests.RequestException, RuntimeError) as exc:
        updated_history[-1]["content"] = f"无法完成请求：{exc}"
        yield updated_history, "", current_summary, current_cursor


def clear_history() -> tuple[list[Any], str, str, int]:
    """Clear the chatbot and input box."""
    return [], "", "", 0


def refresh_status() -> str:
    """Render a friendly service-status message for the header."""
    try:
        payload = check_health()
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        return f"🔴 **服务未连接**\n\n`{exc}`"

    model = payload.get("model", "当前模型")
    return f"🟢 **服务正常**\n\n模型：`{model}`"
