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


class SympyForm(BaseModel):
    """A question rewritten as something SymPy can check by substitution.

    Produced from the question alone, never from the model's own solution, so
    the check cannot become circular.
    """

    model_config = ConfigDict(extra="forbid")

    applicable: bool = False
    lhs: str = Field(default="", max_length=500)
    rhs: str = Field(default="0", max_length=500)
    variables: list[str] = Field(default_factory=list, max_length=10)


class ExtractedAnswer(BaseModel):
    """The final answer pulled out of a free-form solution."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["numeric", "expression", "solution_set", "text", "none"]
    values: list[str] = Field(default_factory=list, max_length=20)


class GroundingVerdict(BaseModel):
    """Whether an answer's claims are supported by retrieved documents."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    unsupported: list[str] = Field(default_factory=list, max_length=10)
    reason: str = Field(default="", max_length=300)


class VerificationResult(BaseModel):
    """Outcome of an independent verification attempt."""

    status: Literal["verified", "refuted", "unknown"]
    method: Literal["substitution", "expression", "grounding", "llm_judge", "none"]
    detail: str = Field(..., max_length=500)
    counterexample: str | None = None


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


class TokenUsage(BaseModel):
    """Token counters, accumulated per client and per run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.calls += 1

    def merge(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls


class RunTrace(BaseModel):
    """One agent run, persisted so a problem can be replayed later."""

    run_id: str
    question: str
    started_at: str
    duration_ms: int
    decision: RouteDecision | None = None
    observations: list[Observation] = Field(default_factory=list)
    verification: VerificationResult | None = None
    answer: str = ""
    answer_model: str = ""
    stopped_reason: str = ""
    error: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    fallbacks: int = 0


class AgentRun(BaseModel):
    """The full trace and result of one agent run."""

    question: str
    decision: RouteDecision
    observations: list[Observation] = Field(default_factory=list)
    steps: int = 0
    stopped_reason: Literal["final", "max_steps", "duplicate", "unknown_tool"]
    answer: str
    verification: VerificationResult | None = None
    verify_attempts: int = 0
    answer_model: str = ""
