"""Shared tool contract: validation, timeout, and structured failure."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from app.agent.schema import ToolResult
from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Dependencies a tool may need, injected by the caller.

    ``client`` is the orchestration endpoint (cloud when configured). ``solver``
    is the local model used for actual math solving; it defaults to ``client``
    so callers that only have one endpoint keep working unchanged.
    """

    client: Any
    settings: Settings
    solver: Any = None

    def __post_init__(self) -> None:
        if self.solver is None:
            self.solver = self.client


ToolHandler = Callable[[Any, ToolContext], Awaitable[ToolResult]]
ToolStreamer = Callable[[Any, ToolContext], AsyncIterator[str]]
ToolResultBuilder = Callable[[str, ToolContext], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """Everything the runtime needs to validate and run one tool.

    ``stream`` is optional: when set, the tool can push its output to the user
    as it is produced (used by the local solver so long answers are visible
    while they are written). Validation, timeout and error handling stay
    identical to the blocking path.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    timeout: float = 30.0
    stream: ToolStreamer | None = None
    stream_result: ToolResultBuilder | None = None


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


async def run_tool_streaming(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: ToolContext,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ``("delta", text)`` while a streaming tool writes, then ``("result", ToolResult)``.

    Tools without a ``stream`` implementation simply yield one result, so the
    caller has a single code path.
    """
    if spec.stream is None:
        yield ("result", await run_tool(spec, arguments, ctx))
        return
    try:
        args = spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        yield (
            "result",
            ToolResult(success=False, error=f"参数不合法：{exc.error_count()} 处错误"),
        )
        return

    parts: list[str] = []
    try:
        async with asyncio.timeout(spec.timeout):
            async for chunk in spec.stream(args, ctx):
                parts.append(chunk)
                yield ("delta", chunk)
    except TimeoutError:
        yield (
            "result",
            ToolResult(success=False, error=f"工具 {spec.name} 超时（{spec.timeout}s）"),
        )
        return
    except Exception as exc:  # 与阻塞路径一致：失败只返回结构化结果
        logger.exception("streaming tool %s failed", spec.name)
        yield (
            "result",
            ToolResult(success=False, error=f"工具 {spec.name} 执行失败：{exc}"),
        )
        return

    text = "".join(parts)
    if not text.strip():
        yield ("result", ToolResult(success=False, error="模型没有返回有效答案"))
        return
    builder = spec.stream_result
    if builder is None:
        yield ("result", ToolResult(success=True, data={"answer": text}))
        return
    yield ("result", builder(text, ctx))
