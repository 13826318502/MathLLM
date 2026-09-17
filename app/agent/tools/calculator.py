"""calculate_expression: evaluate arithmetic safely with an AST whitelist."""

from __future__ import annotations

import ast
import math
import operator

from pydantic import BaseModel, ConfigDict, Field

from app.agent.schema import ToolResult
from app.agent.tools.base import ToolContext, ToolSpec

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000


class CalculateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(..., min_length=1, max_length=200)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("指数过大")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval(arg) for arg in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ValueError("表达式包含不支持的语法")


async def _calculate(args: CalculateInput, ctx: ToolContext) -> ToolResult:
    try:
        value = _eval(ast.parse(args.expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return ToolResult(success=False, error=f"无法计算：{exc}")
    return ToolResult(
        success=True,
        data={"expression": args.expression, "value": value},
    )


SPEC = ToolSpec(
    name="calculate_expression",
    description="计算纯算术表达式，支持 + - * / ** % sqrt log sin cos pi e 等。",
    input_model=CalculateInput,
    handler=_calculate,
    timeout=5.0,
)
