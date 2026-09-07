"""Conversation-history conversion utilities for the Gradio frontend."""

from __future__ import annotations

from typing import Any


def history_to_messages(history: list[Any] | None) -> list[dict[str, str]]:
    """Convert Gradio tuple/message history to FastAPI chat messages."""
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
            continue

        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        question, answer = (
            str(value).strip() if value is not None else "" for value in item
        )
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages
