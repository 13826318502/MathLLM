"""check_math_answer: rule-based verification first, LLM judge as fallback."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agent.schema import ToolResult, Verdict
from app.agent.structured import complete_structured
from app.agent.tools.base import ToolContext, ToolSpec
from app.services import verify_service

CHECKER_SYSTEM_PROMPT = """你是数学答案校验器。
判断给定的解答是否正确，只输出一个 JSON 对象，不要输出其他内容。

JSON 字段：
{
  "verdict": "correct" | "incorrect" | "uncertain",
  "reason": string
}

判定规则：
- 关键步骤和最终答案都正确 -> correct
- 存在明确数学错误 -> incorrect
- 信息不足或无法确认 -> uncertain
不要因为格式或写法不同就判错。"""


class CheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=8000)
    answer: str = Field(..., min_length=1, max_length=8000)


async def _judge(args: CheckInput, ctx: ToolContext) -> ToolResult:
    messages = [
        {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"题目：\n{args.question}\n\n待检查的解答：\n{args.answer}",
        },
    ]
    verdict = await complete_structured(ctx.client, messages, Verdict, max_tokens=256)
    if verdict is None:
        return ToolResult(success=False, error="无法得到有效的校验结论")
    data = verdict.model_dump()
    data["method"] = "llm_judge"
    return ToolResult(success=True, data=data)


async def _check(args: CheckInput, ctx: ToolContext) -> ToolResult:
    verification = await verify_service.verify_answer(
        ctx.client, args.question, args.answer, []
    )
    if verification.status == "verified":
        return ToolResult(
            success=True,
            data={
                "verdict": "correct",
                "reason": verification.detail,
                "method": verification.method,
            },
        )
    if verification.status == "refuted":
        return ToolResult(
            success=True,
            data={
                "verdict": "incorrect",
                "reason": verification.detail,
                "method": verification.method,
                "counterexample": verification.counterexample,
            },
        )
    return await _judge(args, ctx)


SPEC = ToolSpec(
    name="check_math_answer",
    description="检查一个数学解答是否正确，返回 correct/incorrect/uncertain 及理由。",
    input_model=CheckInput,
    handler=_check,
    timeout=120.0,
)
