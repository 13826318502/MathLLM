"""Shared tool contract: validation, timeout, and structured failure."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from app.agent.schema import ToolResult
from app.core.config import Settings
from app.services.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Dependencies a tool may need, injected by the caller."""

    client: VLLMClient
    settings: Settings


@dataclass(frozen=True)
class ToolSpec:
    """Everything the runtime needs to validate and run one tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], Awaitable[ToolResult]]
    timeout: float = 30.0


async def run_tool(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: ToolContext,
) -> ToolResult:
    """Validate, execute with a timeout, and never raise to the agent."""
    try:
        args = spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        return ToolResult(
            success=False,
            error=f"参数不合法：{exc.error_count()} 处错误",
        )
    try:
        return await asyncio.wait_for(spec.handler(args, ctx), timeout=spec.timeout)
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            error=f"工具 {spec.name} 超时（{spec.timeout}s）",
        )
    except Exception as exc:  # 工具失败必须变成结构化结果，不能中断 Agent
        logger.exception("tool %s failed", spec.name)
        return ToolResult(success=False, error=f"工具 {spec.name} 执行失败：{exc}")
