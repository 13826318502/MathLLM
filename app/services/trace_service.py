"""Append-only run traces, so a problem can be replayed after the fact.

One JSON line per agent run under ``data/traces/runs.jsonl`` (gitignored). The
record is deliberately structured rather than free text: it keeps the routing
decision, every tool call, the verification verdict, timings and token usage,
which is everything the metrics layer needs and everything a human needs to
locate where a run went wrong.

Anything written passes through :func:`redact`, so a credential that leaks into
a model error message cannot reach disk.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.agent.schema import AgentRun, RunTrace, TokenUsage

logger = logging.getLogger(__name__)

TRACE_FILENAME = "runs.jsonl"
MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 4000

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_\-]{4,}"), "sk-***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer ***"),
    (re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[^\"\s,}]+"), r"\1***"),
    (re.compile(r"(?i)(authorization\"?\s*[:=]\s*\"?)[^\"\s,}]+"), r"\1***"),
)


def redact(text: str) -> str:
    """Replace anything credential-shaped with a placeholder."""
    result = str(text or "")
    for pattern, replacement in _REDACTIONS:
        result = pattern.sub(replacement, result)
    return result


def trace_path(directory: Path | str) -> Path:
    return Path(directory) / TRACE_FILENAME


def build_trace(
    run: AgentRun,
    *,
    started_at: datetime,
    duration_ms: int,
    usage: TokenUsage | None = None,
    error: str | None = None,
    run_id: str | None = None,
    fallbacks: int = 0,
) -> RunTrace:
    return RunTrace(
        run_id=run_id or uuid.uuid4().hex[:12],
        question=redact(run.question)[:MAX_QUESTION_CHARS],
        started_at=started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        duration_ms=duration_ms,
        decision=run.decision,
        observations=run.observations,
        verification=run.verification,
        rag=run.rag,
        answer=redact(run.answer)[:MAX_ANSWER_CHARS],
        answer_model=run.answer_model,
        stopped_reason=run.stopped_reason,
        error=redact(error) if error else None,
        usage=usage or TokenUsage(),
        fallbacks=fallbacks,
    )


def build_error_trace(
    question: str,
    *,
    started_at: datetime,
    duration_ms: int,
    error: str,
    usage: TokenUsage | None = None,
    run_id: str | None = None,
) -> RunTrace:
    """A run that failed before it could produce an AgentRun."""
    return RunTrace(
        run_id=run_id or uuid.uuid4().hex[:12],
        question=redact(question)[:MAX_QUESTION_CHARS],
        started_at=started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        duration_ms=duration_ms,
        stopped_reason="error",
        error=redact(error),
        usage=usage or TokenUsage(),
    )


def record_trace(trace: RunTrace, path: Path | str) -> None:
    """Append one trace. Never raises: tracing must not break a request."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json() + "\n")
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to record run trace")


def clear_traces(path: Path | str) -> int:
    """Delete the trace file and return how many runs were removed.

    Backs the observability page's "clear history" action. Never raises:
    clearing history must not break a request.
    """
    target = Path(path)
    if not target.exists():
        return 0
    try:
        count = sum(
            1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        target.unlink()
        return count
    except OSError:  # pragma: no cover - defensive
        logger.exception("failed to clear run traces")
        return 0


def load_traces(path: Path | str, limit: int = 50) -> list[RunTrace]:
    """Return the most recent traces, oldest first within the requested window."""
    target = Path(path)
    if not target.exists():
        return []
    lines = target.read_text(encoding="utf-8").splitlines()
    traces: list[RunTrace] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            traces.append(RunTrace.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            continue
    return traces
