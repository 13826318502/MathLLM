"""Local conversation-memory helpers for the Gradio frontend.

The frontend keeps the complete conversation for display, but only sends a
bounded context to the model. Token counts are conservative approximations so
the frontend does not need to load the full model tokenizer.
"""

from __future__ import annotations

import math
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token usage for Chinese and math-heavy text."""
    value = str(text or "")
    cjk_count = sum(
        1
        for char in value
        if "\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff"
    )
    non_cjk_count = len(value) - cjk_count
    return max(1, math.ceil(cjk_count + non_cjk_count / 2.5))


def estimate_messages(messages: list[dict[str, str]]) -> int:
    """Estimate tokens for messages, including a small role overhead."""
    return sum(estimate_tokens(message.get("content", "")) + 4 for message in messages)


def split_history(
    messages: list[dict[str, str]],
    max_recent_messages: int,
    max_recent_tokens: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split messages into old messages for summarization and recent messages."""
    if not messages:
        return [], []

    recent: list[dict[str, str]] = []
    for message in reversed(messages):
        if len(recent) >= max_recent_messages:
            break
        candidate = [message, *recent]
        if recent and estimate_messages(candidate) > max_recent_tokens:
            break
        recent.insert(0, message)

    if not recent:
        recent = [messages[-1]]
    if recent and recent[0].get("role") == "assistant":
        recent = recent[1:] or [messages[-1]]

    old_count = max(0, len(messages) - len(recent))
    return messages[:old_count], recent


def should_summarize(
    summary: str,
    messages: list[dict[str, str]],
    trigger_tokens: int,
    max_recent_messages: int,
) -> bool:
    """Return whether old history should be compressed before the next call."""
    if len(messages) <= max_recent_messages:
        return False
    summary_tokens = estimate_tokens(summary) if summary else 0
    return summary_tokens + estimate_messages(messages) > trigger_tokens


def normalize_history(history: list[Any] | None) -> list[dict[str, str]]:
    """Convert supported Gradio history values to clean message dictionaries."""
    messages: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages
