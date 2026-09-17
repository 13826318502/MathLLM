"""solve_math_problem: delegate a concrete problem to the fine-tuned model.

The tool offers both a blocking handler and a streaming one. The agent uses the
streaming path so a long local generation is visible while it happens, instead
of appearing all at once after a minute of silence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from app.agent.schema import ToolResult
from app.agent.tools.base import ToolContext, ToolSpec
from app.services.chat_service import build_solve_messages
from app.services.stream_service import (
    extract_completion_content,
    extract_stream_content,
)


class SolveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=8000)


def _solver_model(ctx: ToolContext) -> str:
    config = getattr(ctx.solver, "config", None)
    return getattr(config, "model", None) or ctx.settings.model_name


def _solve_messages(args: SolveInput, ctx: ToolContext) -> list[dict[str, str]]:
    return build_solve_messages(args.question, ctx.settings.max_question_chars)


async def _solve(args: SolveInput, ctx: ToolContext) -> ToolResult:
    try:
        messages = _solve_messages(args, ctx)
    except ValueError as exc:
        return ToolResult(success=False, error=str(exc))
    payload = await ctx.solver.complete(messages)
    content = extract_completion_content(payload)
    if not content:
        return ToolResult(success=False, error="模型没有返回有效答案")
    return ToolResult(
        success=True,
        data={"answer": content, "model": _solver_model(ctx)},
    )


async def _solve_stream(args: SolveInput, ctx: ToolContext) -> AsyncIterator[str]:
    messages = _solve_messages(args, ctx)
    async for payload in ctx.solver.stream_raw(messages):
        chunk = extract_stream_content(payload)
        if chunk:
            yield chunk


def _solve_result(text: str, ctx: ToolContext) -> ToolResult:
    return ToolResult(
        success=True,
        data={"answer": text, "model": _solver_model(ctx)},
    )


SPEC = ToolSpec(
    name="solve_math_problem",
    description="求解具体的数学题目，返回分步解答和最终答案。",
    input_model=SolveInput,
    handler=_solve,
    timeout=180.0,
    stream=_solve_stream,
    stream_result=_solve_result,
)
