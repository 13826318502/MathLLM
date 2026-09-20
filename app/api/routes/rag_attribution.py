"""Knowledge-base attribution endpoints.

These answer, for every recorded run: did it consult the knowledge base, what
was retrieved, and is the answer grounded in it. Everything is derived from the
stored run traces, so the page works on runs that predate this feature.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.agent.schema import RagGrounding, RunTrace
from app.api.dependencies import get_orchestrator, get_settings
from app.core.config import Settings
from app.services import (
    metrics_service,
    rag_attribution_service,
    trace_service,
    verify_service,
)
from app.services.rag_attribution_service import RagAttributionListResponse
from app.services.vllm_client import VLLMServiceError

router = APIRouter()

MAX_TRACES_SCANNED = 5000


def _load(trace_dir: str) -> list[RunTrace]:
    return trace_service.load_traces(
        trace_service.trace_path(trace_dir), limit=MAX_TRACES_SCANNED
    )


@router.get("/rag/attribution", response_model=RagAttributionListResponse)
async def rag_attribution_list(
    limit: int = Query(default=50, ge=1, le=500),
    days: int = Query(default=7, ge=0, le=365, description="0 表示全部"),
    settings: Settings = Depends(get_settings),
) -> RagAttributionListResponse:
    """List recent runs with their knowledge-base attribution."""
    traces = metrics_service.filter_since(_load(settings.trace_dir), days)
    checks = rag_attribution_service.load_checks(
        rag_attribution_service.checks_path(settings.trace_dir)
    )
    summary = rag_attribution_service.summarize(
        traces, checks, window_days=days
    )
    recent = list(reversed(traces))[:limit]
    return RagAttributionListResponse(
        summary=summary,
        runs=rag_attribution_service.list_items(recent, checks),
    )


@router.get("/rag/attribution/{run_id}", response_model=RunTrace)
async def rag_attribution_detail(
    run_id: str,
    settings: Settings = Depends(get_settings),
) -> RunTrace:
    """Return one run with its knowledge-base attribution attached."""
    trace = rag_attribution_service.find_trace(_load(settings.trace_dir), run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="未找到该运行记录")
    checks = rag_attribution_service.load_checks(
        rag_attribution_service.checks_path(settings.trace_dir)
    )
    attribution = rag_attribution_service.attribution_from_trace(trace)
    grounding = checks.get(run_id)
    if grounding is not None:
        attribution = attribution.model_copy(update={"grounding": grounding})
    return trace.model_copy(update={"rag": attribution})


@router.post("/rag/attribution/{run_id}/check", response_model=RagGrounding)
async def rag_attribution_check(
    run_id: str,
    settings: Settings = Depends(get_settings),
    orchestrator=Depends(get_orchestrator),
) -> RagGrounding:
    """Re-run the grounding judge for one run and store the verdict."""
    trace = rag_attribution_service.find_trace(_load(settings.trace_dir), run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="未找到该运行记录")
    sources = verify_service.sources_from_observations(trace.observations)
    if not sources:
        raise HTTPException(status_code=400, detail="该运行没有检索到知识库片段")
    try:
        verdict = await verify_service.judge_grounding(
            orchestrator, trace.answer, sources
        )
    except VLLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if verdict is None:
        raise HTTPException(status_code=502, detail="无法得到来源核对结论")
    grounding = RagGrounding(
        grounded=verdict.grounded,
        unsupported=list(verdict.unsupported),
        reason=verdict.reason,
        citations=verify_service.citations_from_verdict(verdict, sources),
    )
    rag_attribution_service.save_check(
        rag_attribution_service.checks_path(settings.trace_dir), run_id, grounding
    )
    return grounding
