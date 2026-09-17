"""Agent run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agent.loop import run_agent
from app.agent.schema import AgentRun
from app.agent.tools import ToolContext
from app.api.dependencies import get_settings, get_vllm_client
from app.api.models import AgentRunRequest
from app.core.config import Settings
from app.services.vllm_client import VLLMClient, VLLMServiceError

router = APIRouter()


@router.post("/agent/run", response_model=AgentRun)
async def agent_run(
    request: AgentRunRequest,
    settings: Settings = Depends(get_settings),
    client: VLLMClient = Depends(get_vllm_client),
) -> AgentRun:
    ctx = ToolContext(client=client, settings=settings)
    try:
        return await run_agent(
            client,
            request.question,
            ctx,
            max_steps=request.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VLLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
