"""Routes orchestration calls to the cloud model, falling back to the local one.

The gateway exposes the same interface as ``VLLMClient`` so every orchestration
call site (routing, planning, verification) can take it unchanged. The solver
client stays separate and is used directly for actual math solving.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.agent.schema import TokenUsage
from app.services.stream_service import (
    extract_completion_content,
    extract_stream_content,
    extract_tool_calls,
)
from app.services.vllm_client import VLLMClient, VLLMServiceError


# A reasoning model can spend the whole output budget thinking and emit no
# content. Retrying the orchestrator once is far cheaper than falling back to a
# local CPU model, so every orchestration call gets two attempts.
ORCHESTRATOR_ATTEMPTS = 2


class ModelGateway:
    """Try the orchestrator; on failure optionally retry on the solver.

    ``last_role`` / ``last_model`` / ``fell_back`` describe the most recent call
    so the execution trace can show which model actually answered. They are
    per-instance state and therefore assume one in-flight request at a time,
    which holds for this local single-user service.
    """

    def __init__(
        self,
        orchestrator: VLLMClient,
        solver: VLLMClient,
        fallback: str = "local",
    ) -> None:
        self.orchestrator = orchestrator
        self.solver = solver
        self.fallback = fallback
        self.last_role = orchestrator.config.role
        self.last_model = orchestrator.config.model
        self.fell_back = False
        self.fallbacks = 0
        self.switches: list[dict[str, str]] = []

    @property
    def usage(self) -> TokenUsage:
        """Token totals across both endpoints, without double counting."""
        total = TokenUsage()
        total.merge(self.orchestrator.usage)
        if self.solver is not self.orchestrator:
            total.merge(self.solver.usage)
        return total

    def reset_counters(self) -> None:
        """Clear per-run counters so a run reports its own usage and fallbacks.

        ``fallbacks`` and the clients' token usage live on the shared gateway,
        so without this reset every later run inherits the previous runs' counts
        and the metrics over-report.
        """
        self.fallbacks = 0
        self.switches = []
        clients = [self.orchestrator]
        if self.solver is not self.orchestrator:
            clients.append(self.solver)
        for client in clients:
            reset = getattr(client, "reset_usage", None)
            if callable(reset):
                reset()

    def _record(self, client: VLLMClient, fell_back: bool) -> None:
        self.last_role = client.config.role
        self.last_model = client.config.model
        self.fell_back = fell_back
        if fell_back:
            self.fallbacks += 1

    def _record_switch(self, reason: str) -> None:
        """Remember that a call moved from the orchestrator to the solver."""
        self.switches.append(
            {
                "from_model": self.orchestrator.config.model,
                "from_role": self.orchestrator.config.role,
                "to_model": self.solver.config.model,
                "to_role": self.solver.config.role,
                "reason": reason,
            }
        )

    def take_switches(self) -> list[dict[str, str]]:
        """Return switches recorded since the last call, then clear them."""
        switches = self.switches
        self.switches = []
        return switches

    def _can_fall_back(self) -> bool:
        return self.fallback == "local" and self.solver is not self.orchestrator

    @staticmethod
    def _empty(result: Any) -> bool:
        """Reasoning models can spend the whole output budget thinking and
        return no content at all; that is a failure, not an answer.

        A tool-call reply has no text content by design, so it counts as a
        usable result rather than an empty one.
        """
        payload = result or {}
        if extract_tool_calls(payload):
            return False
        return not extract_completion_content(payload).strip()

    async def _orchestrate(
        self, method: str, messages: list[dict[str, str]], *args: Any, **kwargs: Any
    ) -> dict:
        """Call the orchestrator, retrying once when it produces nothing.

        A reasoning model can burn the whole output budget on reasoning and emit
        no content. Retrying is much cheaper than falling back to a slow local
        model, so the first empty reply is retried instead of immediately
        switching endpoints.
        """
        last_error: VLLMServiceError | None = None
        for _ in range(ORCHESTRATOR_ATTEMPTS):
            try:
                result = await getattr(self.orchestrator, method)(
                    messages, *args, **kwargs
                )
                if not self._empty(result):
                    return result
                last_error = VLLMServiceError("编排模型没有返回内容")
            except VLLMServiceError as exc:
                last_error = exc
        raise last_error or VLLMServiceError("编排模型调用失败")

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        try:
            result = await self._orchestrate("complete", messages, **kwargs)
        except VLLMServiceError as exc:
            if not self._can_fall_back():
                raise
            self._record_switch(str(exc))
            result = await self.solver.complete(messages, **kwargs)
            self._record(self.solver, True)
            return result
        self._record(self.orchestrator, False)
        return result

    async def complete_json(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict:
        try:
            result = await self._orchestrate("complete_json", messages, **kwargs)
        except VLLMServiceError as exc:
            if not self._can_fall_back():
                raise
            self._record_switch(str(exc))
            result = await self.solver.complete_json(messages, **kwargs)
            self._record(self.solver, True)
            return result
        self._record(self.orchestrator, False)
        return result

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict:
        try:
            result = await self._orchestrate(
                "complete_with_tools", messages, tools, **kwargs
            )
        except VLLMServiceError as exc:
            if not self._can_fall_back():
                raise
            self._record_switch(str(exc))
            result = await self.solver.complete_with_tools(messages, tools, **kwargs)
            self._record(self.solver, True)
            return result
        self._record(self.orchestrator, False)
        return result

    async def stream_raw(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[dict[str, Any]]:
        last_error: VLLMServiceError | None = None
        for _ in range(ORCHESTRATOR_ATTEMPTS):
            produced = False
            last_error = None
            try:
                async for event in self.orchestrator.stream_raw(messages):
                    if extract_stream_content(event):
                        produced = True
                    yield event
            except VLLMServiceError as exc:
                # Never restart on another model after usable output: that would
                # duplicate what the user already saw.
                if produced:
                    raise
                last_error = exc
            if produced:
                self._record(self.orchestrator, False)
                return
        if not self._can_fall_back():
            if last_error is not None:
                raise last_error
            self._record(self.orchestrator, False)
            return
        # A reasoning model can spend the whole budget thinking and stream no
        # content at all; that is a failure, so retry on the local solver.
        self._record_switch(str(last_error) if last_error else "编排模型没有返回内容")
        async for event in self.solver.stream_raw(messages):
            yield event
        self._record(self.solver, True)
