"""Per-answer knowledge-base attribution.

The agent already records everything a run did. This module turns that record
into an answer to three questions a reviewer actually asks: did the answer
consult the knowledge base, what chunks were retrieved, and is the answer
grounded in them?

Attribution is *derived* from a :class:`RunTrace`, so it works for traces that
predate the ``rag`` field as well as for new ones. A trace written after the
attribution layer exists already carries the computed value.

Re-check results (a grounding judge run on demand) are kept in a small sidecar
JSONL next to the traces, keyed by ``run_id``, and override the stored verdict
when present.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agent.schema import (
    Observation,
    RagAttribution,
    RagGrounding,
    RetrievedSource,
    RouteDecision,
    RunTrace,
)

CHECKS_FILENAME = "rag_checks.jsonl"
SNIPPET_CHARS = 300

GROUNDING_VERIFIED = "verified"
GROUNDING_UNKNOWN = "unknown"
GROUNDING_NONE = "none"


class RagAttributionItem(BaseModel):
    """One run, flattened for the attribution list."""

    run_id: str
    question: str
    started_at: str
    duration_ms: int
    intent: str | None = None
    tool: str | None = None
    used_knowledge: bool = False
    query: str | None = None
    retrieved_count: int = 0
    sources: list[str] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    grounding: str = GROUNDING_NONE
    unsupported: list[str] = Field(default_factory=list)
    answer_model: str = ""
    error: str | None = None


class RagAttributionSummary(BaseModel):
    """Aggregate view over the requested window."""

    window_days: int
    generated_at: str
    runs: int = 0
    knowledge_runs: int = 0
    knowledge_rate: float | None = None
    avg_retrieved: float = 0.0
    grounding: dict[str, int] = Field(default_factory=dict)
    grounded_rate: float | None = None
    ungrounded_runs: int = 0


class RagAttributionListResponse(BaseModel):
    summary: RagAttributionSummary
    runs: list[RagAttributionItem] = Field(default_factory=list)


def checks_path(trace_dir: Path | str) -> Path:
    return Path(trace_dir) / CHECKS_FILENAME


def _parse_documents(summary: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(summary)
    except (json.JSONDecodeError, TypeError):
        return []
    documents = data.get("documents") if isinstance(data, dict) else None
    return [doc for doc in documents or [] if isinstance(doc, dict)]


def _to_source(document: dict[str, Any], fallback_rank: int) -> RetrievedSource:
    distance = document.get("distance")
    try:
        distance_value = float(distance) if distance is not None else None
    except (TypeError, ValueError):
        distance_value = None
    rank = document.get("rank")
    try:
        rank_value = int(rank)
    except (TypeError, ValueError):
        rank_value = fallback_rank
    return RetrievedSource(
        source=str(document.get("source", "unknown")),
        rank=rank_value,
        distance=distance_value,
        snippet=str(document.get("content", ""))[:SNIPPET_CHARS],
    )


def build_attribution(
    decision: RouteDecision | None,
    observations: list[Observation],
    answer: str,
) -> RagAttribution:
    """Derive the knowledge-base attribution for one answer."""
    retrieved: list[RetrievedSource] = []
    query: str | None = None
    for observation in observations:
        if observation.tool != "search_knowledge" or not observation.success:
            continue
        if query is None:
            raw_query = observation.arguments.get("query")
            query = str(raw_query) if raw_query else None
        for index, document in enumerate(_parse_documents(observation.summary), start=1):
            retrieved.append(_to_source(document, index))

    used = bool(retrieved) or (
        decision is not None
        and (decision.intent == "knowledge" or decision.tool == "search_knowledge")
    )
    if query is None and used and decision is not None:
        query = decision.query

    cited = [source.source for source in retrieved if _is_cited(source.source, answer)]
    # Preserve retrieval order while dropping duplicate sources.
    unique_cited = list(dict.fromkeys(cited))
    return RagAttribution(
        used_knowledge=used,
        query=query,
        retrieved=retrieved,
        cited_sources=unique_cited,
    )


def _is_cited(source: str, answer: str) -> bool:
    if not source or not answer:
        return False
    if source in answer:
        return True
    stem = Path(source).stem
    return bool(stem) and stem in answer


def attribution_from_trace(trace: RunTrace) -> RagAttribution:
    """Return the stored attribution, deriving it for older traces."""
    if trace.rag is not None:
        return trace.rag
    return build_attribution(trace.decision, trace.observations, trace.answer)


def _effective_grounding(
    attribution: RagAttribution,
    checks: dict[str, RagGrounding],
    run_id: str,
) -> RagGrounding | None:
    return checks.get(run_id) or attribution.grounding


def _status(grounding: RagGrounding | None) -> str:
    if grounding is None:
        return GROUNDING_NONE
    return GROUNDING_VERIFIED if grounding.grounded else GROUNDING_UNKNOWN


def to_item(
    trace: RunTrace,
    checks: dict[str, RagGrounding] | None = None,
) -> RagAttributionItem:
    checks = checks or {}
    attribution = attribution_from_trace(trace)
    grounding = _effective_grounding(attribution, checks, trace.run_id)
    return RagAttributionItem(
        run_id=trace.run_id,
        question=trace.question,
        started_at=trace.started_at,
        duration_ms=trace.duration_ms,
        intent=trace.decision.intent if trace.decision else None,
        tool=trace.decision.tool if trace.decision else None,
        used_knowledge=attribution.used_knowledge,
        query=attribution.query,
        retrieved_count=len(attribution.retrieved),
        sources=list(dict.fromkeys(item.source for item in attribution.retrieved)),
        cited_sources=attribution.cited_sources,
        grounding=_status(grounding),
        unsupported=list(grounding.unsupported) if grounding else [],
        answer_model=trace.answer_model,
        error=trace.error,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def summarize(
    traces: list[RunTrace],
    checks: dict[str, RagGrounding] | None = None,
    *,
    window_days: int = 7,
) -> RagAttributionSummary:
    checks = checks or {}
    total = len(traces)
    knowledge_runs = 0
    retrieved_total = 0
    verified = 0
    unknown = 0
    ungrounded = 0
    for trace in traces:
        attribution = attribution_from_trace(trace)
        if not attribution.used_knowledge:
            continue
        knowledge_runs += 1
        retrieved_total += len(attribution.retrieved)
        grounding = _effective_grounding(attribution, checks, trace.run_id)
        if grounding is None:
            continue
        if grounding.grounded:
            verified += 1
        else:
            unknown += 1
            ungrounded += 1
    return RagAttributionSummary(
        window_days=window_days,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        runs=total,
        knowledge_runs=knowledge_runs,
        knowledge_rate=_rate(knowledge_runs, total),
        avg_retrieved=round(retrieved_total / knowledge_runs, 2) if knowledge_runs else 0.0,
        grounding={
            GROUNDING_VERIFIED: verified,
            GROUNDING_UNKNOWN: unknown,
            GROUNDING_NONE: max(knowledge_runs - verified - unknown, 0),
        },
        grounded_rate=_rate(verified, verified + unknown),
        ungrounded_runs=ungrounded,
    )


def list_items(
    traces: list[RunTrace],
    checks: dict[str, RagGrounding] | None = None,
) -> list[RagAttributionItem]:
    return [to_item(trace, checks) for trace in traces]


def find_trace(traces: list[RunTrace], run_id: str) -> RunTrace | None:
    for trace in traces:
        if trace.run_id == run_id:
            return trace
    return None


def load_checks(path: Path | str) -> dict[str, RagGrounding]:
    """Load the latest grounding re-check per run id."""
    target = Path(path)
    if not target.exists():
        return {}
    checks: dict[str, RagGrounding] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            run_id = str(record["run_id"])
            checks[run_id] = RagGrounding.model_validate(record["grounding"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return checks


def save_check(
    path: Path | str,
    run_id: str,
    grounding: RagGrounding,
) -> None:
    """Append one re-check result. Never raises: a check must not break a request."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "grounding": grounding.model_dump(),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
