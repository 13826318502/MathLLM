"""solve_math_problem: delegate a concrete problem to the fine-tuned model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agent.schema import ToolResult
from app.agent.tools.base import ToolContext, ToolSpec
from app.services.chat_service import build_solve_messages
from app.services.stream_service import extract_completion_content


class SolveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=8000)


async def _solve(args: SolveInput, ctx: ToolContext) -> ToolResult:
    try:
        messages = build_solve_messages(args.question, ctx.settings.max_question_chars)
    except ValueError as exc:
        return ToolResult(success=False, error=str(exc))
    payload = await ctx.client.complete(messages)
    content = extract_completion_content(payload)
    if not content:
        return ToolResult(success=False, error="模型没有返回有效答案")
    return ToolResult(
        success=True,
        data={"answer": content, "model": ctx.settings.model_name},
    )


SPEC = ToolSpec(
    name="solve_math_problem",
    description="求解具体的数学题目，返回分步解答和最终答案。",
    input_model=SolveInput,
    handler=_solve,
    timeout=180.0,
)
