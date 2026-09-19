"""Health-check endpoint."""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_settings, get_vllm_client
from app.api.models import HealthResponse
from app.core.config import Settings
from app.services import rag_service
from app.services.vllm_client import VLLMClient, VLLMServiceError

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    client: VLLMClient = Depends(get_vllm_client),
):
    rag_chunks = await asyncio.to_thread(rag_service.index_size, settings)
    rag = "ready" if rag_chunks else "missing"
    try:
        available_models = await client.list_models()
    except VLLMServiceError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "vllm": "unavailable",
                "model": settings.model_name,
                "available_models": [],
                "detail": str(exc),
                "rag": rag,
                "rag_chunks": rag_chunks,
            },
        )
    return HealthResponse(
        status="ok",
        vllm="connected",
        model=settings.model_name,
        available_models=available_models,
        rag=rag,
        rag_chunks=rag_chunks,
    )
