"""Aggregate run traces into operational metrics.

These are *observational* numbers: they say how the system behaved, not whether
it was right. Accuracy needs a labelled set and lives in the evaluation tooling.

Every rate is ``None`` when the denominator is zero, so an empty window is not
silently reported as "0% failure".
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from app.agent.schema import RunTrace

GUARD_STOPS = {"max_steps", "duplicate", "unknown_tool"}


class MetricsSummary(BaseModel):
    window_days: int
    generated_at: str
    runs: int = 0
    # Outcomes
    failure_rate: float | None = None
    completion_rate: float | None = None
    guard_stop_rate: float | None = None
    stopped_reasons: dict[str, int] = Field(default_factory=dict)
    # Routing and tools
    intents: dict[str, int] = Field(default_factory=dict)
    routes: dict[str, int] = Field(default_factory=dict)
    tool_calls: dict[str, int] = Field(default_factory=dict)
    tool_success_rate: float | None = None
    avg_tool_calls: float = 0.0
    # Verification
    verification: dict[str, int] = Field(default_factory=dict)
    refuted: int = 0
    ungrounded: int = 0
    hallucination_rate: float | None = None
    # Cost and latency
    latency_ms: dict[str, int] = Field(default_factory=dict)
    tokens: dict[str, int] = Field(default_factory=dict)
    models: dict[str, int] = Field(default_factory=dict)
    # Cloud health
    runs_with_fallback: int = 0
    fallback_rate: float | None = None


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile; empty input yields 0."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def filter_since(
    traces: list[RunTrace],
    days: int,
    *,
    now: datetime | None = None,
) -> list[RunTrace]:
    """Keep traces started within the last ``days`` days (0 = everything)."""
    if days <= 0:
        return list(traces)
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    kept: list[RunTrace] = []
    for trace in traces:
        try:
            started = datetime.fromisoformat(trace.started_at)
        except ValueError:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if started >= cutoff:
            kept.append(trace)
    return kept


def summarize(traces: list[RunTrace], *, window_days: int = 7) -> MetricsSummary:
    """Compute the operational summary for a set of traces."""
    total = len(traces)
    if total == 0:
        return MetricsSummary(
            window_days=window_days,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    failures = sum(1 for t in traces if t.error)
    completed = sum(1 for t in traces if t.answer.strip())
    guard_stops = sum(1 for t in traces if t.stopped_reason in GUARD_STOPS)

    observations = [obs for t in traces for obs in t.observations]
    succeeded = sum(1 for obs in observations if obs.success)

    statuses = [t.verification.status for t in traces if t.verification]
    refuted = sum(1 for status in statuses if status == "refuted")
    ungrounded = sum(
        1
        for t in traces
        if t.verification
        and t.verification.method == "grounding"
        and t.verification.status == "unknown"
    )

    durations = [t.duration_ms for t in traces]
    prompt_tokens = sum(t.usage.prompt_tokens for t in traces)
    completion_tokens = sum(t.usage.completion_tokens for t in traces)
    total_tokens = sum(t.usage.total_tokens for t in traces)

    with_fallback = sum(1 for t in traces if t.fallbacks > 0)

    return MetricsSummary(
        window_days=window_days,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        runs=total,
        failure_rate=_rate(failures, total),
        completion_rate=_rate(completed, total),
        guard_stop_rate=_rate(guard_stops, total),
        stopped_reasons=_count([t.stopped_reason or "unknown" for t in traces]),
        intents=_count([t.decision.intent for t in traces if t.decision]),
        routes=_count([t.decision.tool for t in traces if t.decision]),
        tool_calls=_count([obs.tool for obs in observations]),
        tool_success_rate=_rate(succeeded, len(observations)),
        avg_tool_calls=round(len(observations) / total, 2),
        verification=_count(statuses),
        refuted=refuted,
        ungrounded=ungrounded,
        hallucination_rate=_rate(refuted + ungrounded, len(statuses)),
        latency_ms={
            "p50": _percentile(durations, 0.5),
            "p95": _percentile(durations, 0.95),
            "max": max(durations),
            "avg": round(sum(durations) / total),
        },
        tokens={
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
            "avg_per_run": round(total_tokens / total),
        },
        models=_count([t.answer_model for t in traces if t.answer_model]),
        runs_with_fallback=with_fallback,
        fallback_rate=_rate(with_fallback, total),
    )
