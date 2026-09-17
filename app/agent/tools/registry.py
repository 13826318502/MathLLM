"""Tool registry: name lookup, execution, and catalog rendering."""

from __future__ import annotations

from typing import Any

from app.agent.schema import ToolResult
from app.agent.tools import calculator, checker, knowledge, math_solver
from app.agent.tools.base import ToolContext, ToolSpec, run_tool

_REGISTRY: dict[str, ToolSpec] = {}

for spec in (math_solver.SPEC, calculator.SPEC, knowledge.SPEC, checker.SPEC):
    _REGISTRY[spec.name] = spec


def get_tool(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


async def call_tool(
    name: str,
    arguments: dict[str, Any],
    ctx: ToolContext,
) -> ToolResult:
    """Run a registered tool by name; unknown names fail like any other tool."""
    spec = _REGISTRY.get(name)
    if spec is None:
        return ToolResult(success=False, error=f"未知工具：{name}")
    return await run_tool(spec, arguments, ctx)


def tool_catalog() -> list[dict[str, Any]]:
    """Render the registry for planner prompts."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_model.model_json_schema(),
        }
        for spec in _REGISTRY.values()
    ]
