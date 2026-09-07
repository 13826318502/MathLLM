"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str
    model_name: str
    api_key: str | None
    api_host: str
    api_port: int
    max_question_chars: int
    max_history_messages: int
    max_history_chars: int
    memory_trigger_tokens: int
    memory_summary_max_tokens: int
    recent_history_messages: int
    max_context_tokens: int
    max_output_tokens: int
    connect_timeout: float
    read_timeout: float
    allow_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv("MATHLLM_ALLOW_ORIGINS", "*").split(",")
            if origin.strip()
        )
        return cls(
            vllm_base_url=os.getenv(
                "MATHLLM_API_BASE_URL",
                os.getenv("MATHLLM_VLLM_BASE_URL", "http://localhost:8000/v1"),
            ).rstrip("/"),
            model_name=os.getenv("MATHLLM_MODEL_NAME", "mathllm-round5"),
            api_key=os.getenv("MATHLLM_API_KEY") or None,
            api_host=os.getenv("MATHLLM_API_HOST", "0.0.0.0"),
            api_port=_as_int("MATHLLM_API_PORT", 8080),
            max_question_chars=_as_int("MATHLLM_MAX_QUESTION_CHARS", 8000),
            max_history_messages=_as_int("MATHLLM_MAX_HISTORY_MESSAGES", 20),
            max_history_chars=_as_int("MATHLLM_MAX_HISTORY_CHARS", 24000),
            memory_trigger_tokens=_as_int("MATHLLM_MEMORY_TRIGGER_TOKENS", 1200),
            memory_summary_max_tokens=_as_int(
                "MATHLLM_MEMORY_SUMMARY_MAX_TOKENS", 256
            ),
            recent_history_messages=_as_int(
                "MATHLLM_RECENT_HISTORY_MESSAGES", 6
            ),
            max_context_tokens=_as_int("MATHLLM_MAX_CONTEXT_TOKENS", 2048),
            max_output_tokens=_as_int("MATHLLM_MAX_OUTPUT_TOKENS", 512),
            connect_timeout=_as_float("MATHLLM_CONNECT_TIMEOUT", 10.0),
            read_timeout=_as_float("MATHLLM_READ_TIMEOUT", 180.0),
            allow_origins=origins or ("*",),
        )


settings = Settings.from_env()
