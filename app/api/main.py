"""FastAPI application assembly.

Route and service logic lives in separate modules. This file only creates the
application, configures middleware, and registers routes.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    agent,
    chat,
    health,
    memory,
    metrics,
    rag_attribution,
    solve,
)
from app.core.config import settings
from app.services.model_gateway import ModelGateway
from app.services.vllm_client import VLLMClient


def create_app() -> FastAPI:
    application = FastAPI(
        title="MathLLM API",
        version="1.0.0",
        description="数学题目问答助手的 FastAPI 服务",
    )
    application.state.settings = settings
    # The local model serves the legacy single-shot endpoints and does the
    # actual math solving. Orchestration goes through the gateway, which prefers
    # the cloud endpoint and can fall back to the local one.
    solver_client = VLLMClient(settings.solver)
    application.state.vllm_client = solver_client
    application.state.orchestrator = ModelGateway(
        VLLMClient(settings.orchestrator),
        solver_client,
        settings.orchestrator_fallback,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allow_origins),
        allow_credentials=settings.allow_origins != ("*",),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(health.router, prefix="/api", tags=["health"])
    application.include_router(solve.router, prefix="/api", tags=["solve"])
    application.include_router(chat.router, prefix="/api", tags=["chat"])
    application.include_router(memory.router, prefix="/api", tags=["memory"])
    application.include_router(agent.router, prefix="/api", tags=["agent"])
    application.include_router(metrics.router, prefix="/api", tags=["observability"])
    application.include_router(
        rag_attribution.router, prefix="/api", tags=["observability"]
    )
    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
