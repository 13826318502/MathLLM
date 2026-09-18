"""Generic structured completion with validation and error-feedback retries.

The model only proposes JSON. Parsing, validation and retrying live here, so a
malformed reply is corrected instead of crashing the caller.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.services.stream_service import extract_completion_content
from app.services.vllm_client import VLLMClient

ModelT = TypeVar("ModelT", bound=BaseModel)

# Reasoning models spend output tokens on thinking before writing the JSON, so a
# 256-token budget can be exhausted with nothing emitted. 1024 leaves room for
# the reasoning plus the small structured reply.
DEFAULT_STRUCTURED_MAX_TOKENS = 1024


def extract_json_object(text: str) -> str:
    """Return the JSON object substring, stripping code fences and prose."""
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else ""
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
        value = value.strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    start, end = value.find("{"), value.rfind("}")
    if start != -1 and end > start:
        return value[start : end + 1]
    return value


async def complete_structured(
    client: VLLMClient,
    messages: list[dict[str, str]],
    model_cls: type[ModelT],
    *,
    max_tokens: int = DEFAULT_STRUCTURED_MAX_TOKENS,
    max_retries: int = 2,
) -> ModelT | None:
    """Return a validated model instance, or None when every attempt fails.

    Each retry feeds the validation error back to the model so it can correct
    itself. The caller decides what to do when None is returned.
    """
    conversation = list(messages)
    for _ in range(max_retries + 1):
        payload = await client.complete_json(conversation, max_tokens=max_tokens)
        raw = extract_completion_content(payload)
        try:
            return model_cls.model_validate(json.loads(extract_json_object(raw)))
        except (json.JSONDecodeError, ValidationError) as exc:
            conversation.append({"role": "assistant", "content": raw})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"上面的输出不合法：{exc}。"
                        "请只输出符合字段要求的 JSON 对象，不要任何其他文字。"
                    ),
                }
            )
    return None
