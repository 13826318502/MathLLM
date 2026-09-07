"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.services.vllm_client import VLLMClient


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vllm_client(request: Request) -> VLLMClient:
    return request.app.state.vllm_client
