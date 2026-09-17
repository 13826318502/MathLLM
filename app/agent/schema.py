"""Structured data contracts shared by the agent layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_CHARS = 2000


class RouteDecision(BaseModel):
    """The routing decision the model must return as a single JSON object."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["math", "knowledge", "general"]
    tool: Literal["solve_math_problem", "search_knowledge", "none"] = "none"
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    answer_style: Literal["direct", "step_by_step", "hint"] = "step_by_step"


class ToolResult(BaseModel):
    """Uniform envelope every agent tool returns."""

    success: bool
    data: dict[str, Any] | list[Any] | str | None = None
    error: str | None = None


class Verdict(BaseModel):
    """Structured result of checking a math answer."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "incorrect", "uncertain"]
    reason: str = Field(..., max_length=500)


class AgentAction(BaseModel):
    """The next action the model chooses inside the agent loop."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool", "final"]
    tool: str | None = Field(default=None, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=300)


class Observation(BaseModel):
    """A record of one tool execution inside the loop."""

    step: int
    tool: str
    arguments: dict[str, Any]
    success: bool
    summary: str
    error: str | None = None
    duration_ms: int


class AgentRun(BaseModel):
    """The full trace and result of one agent run."""

    question: str
    decision: RouteDecision
    observations: list[Observation] = Field(default_factory=list)
    steps: int = 0
    stopped_reason: Literal["final", "max_steps", "duplicate", "unknown_tool"]
    answer: str
