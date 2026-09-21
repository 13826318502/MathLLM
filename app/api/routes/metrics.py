"""Observability endpoints: aggregated metrics and recent run traces."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.agent.schema import RunTrace
from app.api.dependencies import get_settings
from app.core.config import Settings
from app.services import metrics_service, rag_attribution_service, trace_service
from app.services.metrics_service import MetricsSummary

router = APIRouter()

MAX_TRACES_SCANNED = 5000


class ClearHistoryResponse(BaseModel):
    """How much history a clear request removed."""

    cleared_runs: int = 0
    cleared_checks: int = 0


@router.get("/metrics", response_model=MetricsSummary)
async def metrics(
    days: int = Query(default=7, ge=0, le=365, description="0 表示全部"),
    settings: Settings = Depends(get_settings),
) -> MetricsSummary:
    """Aggregate the stored run traces into operational metrics."""
    path = trace_service.trace_path(settings.trace_dir)
    traces = trace_service.load_traces(path, limit=MAX_TRACES_SCANNED)
    return metrics_service.summarize(
        metrics_service.filter_since(traces, days), window_days=days
    )


@router.get("/traces", response_model=list[RunTrace])
async def traces(
    limit: int = Query(default=20, ge=1, le=200),
    settings: Settings = Depends(get_settings),
) -> list[RunTrace]:
    """Return the most recent runs, oldest first, for debugging."""
    path = trace_service.trace_path(settings.trace_dir)
    return trace_service.load_traces(path, limit=limit)


@router.delete("/traces", response_model=ClearHistoryResponse)
async def clear_traces(
    settings: Settings = Depends(get_settings),
) -> ClearHistoryResponse:
    """Delete all stored run traces and their grounding re-checks.

    Shared by the observability and knowledge-base attribution pages, since both
    are derived from the same trace store.
    """
    cleared_runs = trace_service.clear_traces(
        trace_service.trace_path(settings.trace_dir)
    )
    cleared_checks = rag_attribution_service.clear_checks(
        rag_attribution_service.checks_path(settings.trace_dir)
    )
    return ClearHistoryResponse(
        cleared_runs=cleared_runs, cleared_checks=cleared_checks
    )
