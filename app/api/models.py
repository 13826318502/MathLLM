"""Pydantic request and response models for the application API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    stream: bool = True


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=20)
    stream: bool = True


class AnswerResponse(BaseModel):
    content: str
    finished: bool = True
    model: str


class HealthResponse(BaseModel):
    status: str
    vllm: str
    model: str
    available_models: list[str] = Field(default_factory=list)
    detail: str | None = None
