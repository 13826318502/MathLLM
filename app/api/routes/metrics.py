"""Observability endpoints: aggregated metrics and recent run traces."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.agent.schema import RunTrace
from app.api.dependencies import get_settings
from app.core.config import Settings
from app.services import metrics_service, trace_service
from app.services.metrics_service import MetricsSummary

router = APIRouter()

MAX_TRACES_SCANNED = 5000


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
