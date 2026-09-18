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
)
from app.services.vllm_client import VLLMClient, VLLMServiceError


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

    def _can_fall_back(self) -> bool:
        return self.fallback == "local" and self.solver is not self.orchestrator

    @staticmethod
    def _empty(result: Any) -> bool:
        """Reasoning models can spend the whole output budget thinking and
        return no content at all; that is a failure, not an answer."""
        return not extract_completion_content(result or {}).strip()

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        try:
            result = await self.orchestrator.complete(messages, **kwargs)
            if self._empty(result):
                raise VLLMServiceError("编排模型没有返回内容")
        except VLLMServiceError:
            if not self._can_fall_back():
                raise
            result = await self.solver.complete(messages, **kwargs)
            self._record(self.solver, True)
            return result
        self._record(self.orchestrator, False)
        return result

    async def complete_json(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict:
        try:
            result = await self.orchestrator.complete_json(messages, **kwargs)
            if self._empty(result):
                raise VLLMServiceError("编排模型没有返回内容")
        except VLLMServiceError:
            if not self._can_fall_back():
                raise
            result = await self.solver.complete_json(messages, **kwargs)
            self._record(self.solver, True)
            return result
        self._record(self.orchestrator, False)
        return result

    async def stream_raw(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[dict[str, Any]]:
        produced = False
        try:
            async for event in self.orchestrator.stream_raw(messages):
                if extract_stream_content(event):
                    produced = True
                yield event
        except VLLMServiceError:
            # Never restart on another model after usable output: that would
            # duplicate what the user already saw.
            if produced or not self._can_fall_back():
                raise
        if produced or not self._can_fall_back():
            self._record(self.orchestrator, False)
            return
        # A reasoning model can spend the whole budget thinking and stream no
        # content at all; that is a failure, so retry on the local solver.
        async for event in self.solver.stream_raw(messages):
            yield event
        self._record(self.solver, True)
