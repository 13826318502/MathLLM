"""FastAPI application assembly.

Route and service logic lives in separate modules. This file only creates the
application, configures middleware, and registers routes.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, solve
from app.core.config import settings
from app.services.vllm_client import VLLMClient


def create_app() -> FastAPI:
    application = FastAPI(
        title="MathLLM API",
        version="1.0.0",
        description="数学题目问答助手的 FastAPI 服务",
    )
    application.state.settings = settings
    application.state.vllm_client = VLLMClient(settings)
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
    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
