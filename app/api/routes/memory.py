"""Conversation-memory endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_settings, get_vllm_client
from app.api.models import MemorySummaryRequest, MemorySummaryResponse
from app.core.config import Settings
from app.services.memory_service import build_summary_messages
from app.services.stream_service import extract_completion_content
from app.services.vllm_client import VLLMClient, VLLMServiceError

router = APIRouter()


@router.post("/memory/summarize", response_model=MemorySummaryResponse)
async def summarize_memory(
    request: MemorySummaryRequest,
    settings: Settings = Depends(get_settings),
    client: VLLMClient = Depends(get_vllm_client),
) -> MemorySummaryResponse:
    messages = [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
    prompt_messages = build_summary_messages(messages, request.existing_summary)
    try:
        payload = await client.complete(
            prompt_messages,
            max_tokens=settings.memory_summary_max_tokens,
        )
    except VLLMServiceError as exc:
        raise HTTPException(status_code=502, detail="无法生成对话摘要") from exc

    summary = extract_completion_content(payload).strip()
    if not summary:
        raise HTTPException(status_code=502, detail="模型没有返回有效摘要")
    return MemorySummaryResponse(summary=summary)
