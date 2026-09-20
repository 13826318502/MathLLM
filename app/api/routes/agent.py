"""Agent endpoints: a plain run and a streaming run."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agent.loop import iter_agent_events, run_agent
from app.agent.schema import AgentRun
from app.agent.tools import ToolContext
from app.api.dependencies import get_orchestrator, get_settings, get_vllm_client
from app.api.models import AgentRunRequest
from app.core.config import Settings
from app.services.chat_service import build_history
from app.services.vllm_client import VLLMServiceError

router = APIRouter()


@router.post("/agent/run", response_model=AgentRun)
async def agent_run(
    request: AgentRunRequest,
    settings: Settings = Depends(get_settings),
    orchestrator=Depends(get_orchestrator),
    solver=Depends(get_vllm_client),
) -> AgentRun:
    ctx = ToolContext(client=orchestrator, solver=solver, settings=settings)
    try:
        history = build_history(
            request.messages, settings.max_history_messages, settings.max_history_chars
        )
        return await run_agent(
            orchestrator,
            request.question,
            ctx,
            history=history or None,
            summary=request.summary,
            max_steps=request.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VLLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/agent/stream")
async def agent_stream(
    request: AgentRunRequest,
    settings: Settings = Depends(get_settings),
    orchestrator=Depends(get_orchestrator),
    solver=Depends(get_vllm_client),
):
    ctx = ToolContext(client=orchestrator, solver=solver, settings=settings)
    try:
        history = build_history(
            request.messages, settings.max_history_messages, settings.max_history_chars
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = request.summary

    async def events():
        try:
            async for event in iter_agent_events(
                orchestrator,
                request.question,
                ctx,
                history=history or None,
                summary=summary,
                max_steps=request.max_steps,
            ):
                yield {"data": json.dumps(event, ensure_ascii=False)}
        except ValueError as exc:
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}
        except VLLMServiceError:
            yield {
                "data": json.dumps(
                    {"type": "error", "message": "模型服务暂时不可用，请稍后重试"},
                    ensure_ascii=False,
                )
            }

    return EventSourceResponse(events())
