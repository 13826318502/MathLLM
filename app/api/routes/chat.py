"""Multi-turn chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import get_settings, get_vllm_client
from app.api.models import AnswerResponse, ChatRequest
from app.core.config import Settings
from app.services.chat_service import build_chat_messages
from app.services.stream_service import (
    encode_sse,
    extract_completion_content,
    public_sse_events,
)
from app.services.vllm_client import VLLMClient, VLLMServiceError

router = APIRouter()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    client: VLLMClient = Depends(get_vllm_client),
):
    try:
        messages = build_chat_messages(
            request.messages,
            settings.max_history_messages,
            settings.max_history_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.stream:
        async def events():
            try:
                async for event in public_sse_events(client.stream_raw(messages)):
                    yield {"data": encode_sse(event)}
            except VLLMServiceError:
                yield {
                    "data": encode_sse(
                        {
                            "content": "",
                            "finished": True,
                            "error": "模型服务暂时不可用，请稍后重试",
                        }
                    )
                }

        return EventSourceResponse(events())

    try:
        payload = await client.complete(messages)
    except VLLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    content = extract_completion_content(payload)
    if not content:
        raise HTTPException(status_code=502, detail="模型没有返回有效答案")
    return AnswerResponse(content=content, finished=True, model=settings.model_name)
