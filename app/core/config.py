"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SOLVER_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_SOLVER_MODEL = "mathllm-round7"


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
class ModelConfig:
    """One OpenAI-compatible model endpoint."""

    role: str
    base_url: str
    model: str
    api_key: str | None
    connect_timeout: float
    read_timeout: float
    max_output_tokens: int


@dataclass(frozen=True)
class Settings:
    # Solver (local) endpoint. The field names are kept for backwards
    # compatibility with existing configs, launchers and tests.
    vllm_base_url: str
    model_name: str
    api_key: str | None
    # Orchestrator (cloud) endpoint. When unset it falls back to the solver, so
    # the system behaves exactly as before until it is configured.
    orchestrator_base_url: str
    orchestrator_model: str
    orchestrator_api_key: str | None
    orchestrator_fallback: str
    # Reasoning models spend output budget on thinking before writing anything,
    # so the orchestrator gets more headroom than the local solver.
    orchestrator_max_output_tokens: int
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
    knowledge_dir: str
    rag_persist_dir: str
    rag_collection: str
    rag_embedding_model: str
    rag_top_k: int
    trace_dir: str
    trace_enabled: bool
    # Knowledge-base answers are grounded on retrieved chunks. Running another
    # round of model calls to verify them is optional (and slow when the
    # orchestrator is a reasoning model), so it is off by default.
    verify_rag: bool

    @property
    def has_orchestrator(self) -> bool:
        """True when a separate cloud endpoint is configured."""
        return bool(self.orchestrator_base_url and self.orchestrator_model)

    @property
    def solver(self) -> ModelConfig:
        return ModelConfig(
            role="solver",
            base_url=self.vllm_base_url,
            model=self.model_name,
            api_key=self.api_key,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            max_output_tokens=self.max_output_tokens,
        )

    @property
    def orchestrator(self) -> ModelConfig:
        if self.has_orchestrator:
            base_url = self.orchestrator_base_url
            model = self.orchestrator_model
            api_key = self.orchestrator_api_key
        else:
            base_url = self.vllm_base_url
            model = self.model_name
            api_key = self.api_key
        return ModelConfig(
            role="orchestrator",
            base_url=base_url,
            model=model,
            api_key=api_key,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            max_output_tokens=self.orchestrator_max_output_tokens,
        )

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv("MATHLLM_ALLOW_ORIGINS", "*").split(",")
            if origin.strip()
        )
        solver_base_url = (
            os.getenv("MATHLLM_SOLVER_BASE_URL")
            or os.getenv("MATHLLM_API_BASE_URL")
            or os.getenv("MATHLLM_VLLM_BASE_URL")
            or DEFAULT_SOLVER_BASE_URL
        ).rstrip("/")
        solver_model = (
            os.getenv("MATHLLM_SOLVER_MODEL")
            or os.getenv("MATHLLM_MODEL_NAME")
            or DEFAULT_SOLVER_MODEL
        )
        solver_api_key = os.getenv("MATHLLM_SOLVER_API_KEY") or os.getenv(
            "MATHLLM_API_KEY"
        )
        fallback = os.getenv("MATHLLM_ORCHESTRATOR_FALLBACK", "local").strip().lower()
        if fallback not in {"local", "fail"}:
            fallback = "local"
        return cls(
            vllm_base_url=solver_base_url,
            model_name=solver_model,
            api_key=solver_api_key or None,
            orchestrator_base_url=os.getenv("MATHLLM_ORCHESTRATOR_BASE_URL", "").rstrip(
                "/"
            ),
            orchestrator_model=os.getenv("MATHLLM_ORCHESTRATOR_MODEL", "").strip(),
            orchestrator_api_key=os.getenv("MATHLLM_ORCHESTRATOR_API_KEY") or None,
            orchestrator_fallback=fallback,
            orchestrator_max_output_tokens=_as_int(
                "MATHLLM_ORCHESTRATOR_MAX_OUTPUT_TOKENS", 2048
            ),
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
            max_output_tokens=_as_int("MATHLLM_MAX_OUTPUT_TOKENS", 1024),
            connect_timeout=_as_float("MATHLLM_CONNECT_TIMEOUT", 10.0),
            read_timeout=_as_float("MATHLLM_READ_TIMEOUT", 180.0),
            allow_origins=origins or ("*",),
            knowledge_dir=os.getenv("MATHLLM_KNOWLEDGE_DIR", "knowledge"),
            rag_persist_dir=os.getenv("MATHLLM_RAG_PERSIST_DIR", "data/chroma"),
            rag_collection=os.getenv("MATHLLM_RAG_COLLECTION", "math_knowledge"),
            rag_embedding_model=os.getenv(
                "MATHLLM_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
            ),
            rag_top_k=_as_int("MATHLLM_RAG_TOP_K", 3),
            trace_dir=os.getenv("MATHLLM_TRACE_DIR", "data/traces"),
            trace_enabled=os.getenv("MATHLLM_TRACE_ENABLED", "1").strip().lower()
            not in {"0", "false", "no"},
            verify_rag=os.getenv("MATHLLM_VERIFY_RAG", "0").strip().lower()
            not in {"0", "false", "no"},
        )


settings = Settings.from_env()
