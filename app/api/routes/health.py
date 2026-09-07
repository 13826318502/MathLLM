"""Health-check endpoint."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_settings, get_vllm_client
from app.api.models import HealthResponse
from app.core.config import Settings
from app.services.vllm_client import VLLMClient, VLLMServiceError

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    client: VLLMClient = Depends(get_vllm_client),
):
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
            },
        )
    return HealthResponse(
        status="ok",
        vllm="connected",
        model=settings.model_name,
        available_models=available_models,
    )
