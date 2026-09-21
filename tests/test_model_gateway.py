from __future__ import annotations

import json
import unittest

from app.agent.schema import TokenUsage
from app.core.config import ModelConfig
from app.services.model_gateway import ModelGateway
from app.services.vllm_client import VLLMServiceError


def _config(role: str, model: str) -> ModelConfig:
    return ModelConfig(
        role=role,
        base_url="http://example.invalid/v1",
        model=model,
        api_key=None,
        connect_timeout=1.0,
        read_timeout=1.0,
        max_output_tokens=16,
    )


def _completion(content: str, usage: tuple[int, int] | None = None) -> dict:
    payload: dict = {"choices": [{"message": {"content": content}}]}
    if usage:
        payload["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return payload


def _tool_calls_payload(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ]
    }


class FakeClient:
    def __init__(
        self,
        model: str,
        *,
        role: str = "orchestrator",
        content: str = "ok",
        chunks: tuple[str, ...] = ("ok",),
        error: Exception | None = None,
        usage: tuple[int, int] | None = None,
        tool_call: tuple[str, dict] | None = None,
    ) -> None:
        self.config = _config(role, model)
        self.usage = TokenUsage()
        self._content = content
        self._chunks = chunks
        self._error = error
        self._tool_call = tool_call
        self.calls = 0
        if usage:
            self.usage.add(usage[0], usage[1], sum(usage))

    def reset_usage(self) -> None:
        self.usage = TokenUsage()

    def _check(self) -> None:
        if self._error is not None:
            raise self._error

    async def complete(self, messages, **kwargs) -> dict:
        self.calls += 1
        self._check()
        return _completion(self._content)

    async def complete_json(self, messages, **kwargs) -> dict:
        self.calls += 1
        self._check()
        return _completion(self._content)

    async def complete_with_tools(self, messages, tools, **kwargs) -> dict:
        self.calls += 1
        self._check()
        if self._tool_call is not None:
            return _tool_calls_payload(*self._tool_call)
        return _completion(self._content)

    async def stream_raw(self, messages):
        self.calls += 1
        self._check()
        for chunk in self._chunks:
            yield {"choices": [{"delta": {"content": chunk}}]}


class SequenceClient(FakeClient):
    """Returns a queued list of contents, one per call (last one repeats)."""

    def __init__(self, model: str, contents: list[str], **kwargs) -> None:
        super().__init__(model, content=contents[0], **kwargs)
        self._contents = contents

    async def complete_json(self, messages, **kwargs) -> dict:
        index = min(self.calls, len(self._contents) - 1)
        self.calls += 1
        return _completion(self._contents[index])


def _gateway(orchestrator: FakeClient, solver: FakeClient, fallback: str = "local"):
    return ModelGateway(orchestrator, solver, fallback)


class FallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_orchestrator_when_it_works(self) -> None:
        gateway = _gateway(FakeClient("cloud"), FakeClient("local", role="solver"))
        result = await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(gateway.last_model, "cloud")
        self.assertFalse(gateway.fell_back)

    async def test_falls_back_on_service_error(self) -> None:
        orchestrator = FakeClient("cloud", error=VLLMServiceError("down"))
        solver = FakeClient("local", role="solver", content="from-local")
        gateway = _gateway(orchestrator, solver)
        result = await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(result["choices"][0]["message"]["content"], "from-local")
        self.assertEqual(gateway.last_model, "local")
        self.assertTrue(gateway.fell_back)

    async def test_falls_back_when_orchestrator_returns_no_content(self) -> None:
        # A reasoning model can burn the whole budget thinking and emit nothing.
        orchestrator = FakeClient("cloud", content="")
        solver = FakeClient("local", role="solver", content="from-local")
        gateway = _gateway(orchestrator, solver)
        result = await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(result["choices"][0]["message"]["content"], "from-local")
        self.assertTrue(gateway.fell_back)

    async def test_fail_policy_does_not_fall_back(self) -> None:
        orchestrator = FakeClient("cloud", error=VLLMServiceError("down"))
        solver = FakeClient("local", role="solver")
        gateway = _gateway(orchestrator, solver, fallback="fail")
        with self.assertRaises(VLLMServiceError):
            await gateway.complete_json([{"role": "user", "content": "hi"}])

    async def test_same_endpoint_never_double_calls(self) -> None:
        shared = FakeClient("local", role="solver", error=VLLMServiceError("down"))
        gateway = _gateway(shared, shared)
        with self.assertRaises(VLLMServiceError):
            await gateway.complete_json([{"role": "user", "content": "hi"}])


class ToolCallTest(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_tool_call_without_falling_back(self) -> None:
        # A tool-call reply has no text content; it must not be treated as empty.
        orchestrator = FakeClient(
            "cloud", content="", tool_call=("route_question", {"intent": "math"})
        )
        solver = FakeClient("local", role="solver")
        gateway = _gateway(orchestrator, solver)
        result = await gateway.complete_with_tools(
            [{"role": "user", "content": "hi"}], [{"type": "function"}]
        )
        calls = result["choices"][0]["message"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "route_question")
        self.assertFalse(gateway.fell_back)
        self.assertEqual(gateway.last_model, "cloud")

    async def test_falls_back_to_solver_tool_call(self) -> None:
        orchestrator = FakeClient("cloud", error=VLLMServiceError("down"))
        solver = FakeClient(
            "local", role="solver", tool_call=("route_question", {"intent": "math"})
        )
        gateway = _gateway(orchestrator, solver)
        result = await gateway.complete_with_tools(
            [{"role": "user", "content": "hi"}], [{"type": "function"}]
        )
        self.assertTrue(result["choices"][0]["message"]["tool_calls"])
        self.assertTrue(gateway.fell_back)
        self.assertEqual(gateway.last_model, "local")


class RetryAndSwitchTest(unittest.IsolatedAsyncioTestCase):
    async def test_retries_empty_orchestrator_before_falling_back(self) -> None:
        orchestrator = SequenceClient("cloud", ["", "ok"])
        solver = FakeClient("local", role="solver", content="from-local")
        gateway = _gateway(orchestrator, solver)
        result = await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(orchestrator.calls, 2)
        self.assertFalse(gateway.fell_back)
        self.assertEqual(gateway.take_switches(), [])

    async def test_records_switch_after_exhausting_retries(self) -> None:
        orchestrator = SequenceClient("cloud", ["", ""])
        solver = FakeClient("local", role="solver", content="from-local")
        gateway = _gateway(orchestrator, solver)
        result = await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(result["choices"][0]["message"]["content"], "from-local")
        self.assertTrue(gateway.fell_back)
        switches = gateway.take_switches()
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0]["from_model"], "cloud")
        self.assertEqual(switches[0]["from_role"], "orchestrator")
        self.assertEqual(switches[0]["to_model"], "local")
        self.assertEqual(switches[0]["to_role"], "solver")
        self.assertEqual(gateway.take_switches(), [])

    async def test_reset_clears_recorded_switches(self) -> None:
        orchestrator = SequenceClient("cloud", [""])
        gateway = _gateway(orchestrator, FakeClient("local", role="solver"))
        await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertTrue(gateway.take_switches())
        gateway.reset_counters()
        self.assertEqual(gateway.take_switches(), [])


class StreamFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, gateway) -> list[str]:
        return [
            event["choices"][0]["delta"]["content"]
            async for event in gateway.stream_raw([{"role": "user", "content": "hi"}])
        ]

    async def test_streams_from_orchestrator(self) -> None:
        gateway = _gateway(
            FakeClient("cloud", chunks=("a", "b")),
            FakeClient("local", role="solver", chunks=("x",)),
        )
        self.assertEqual(await self._collect(gateway), ["a", "b"])
        self.assertFalse(gateway.fell_back)

    async def test_falls_back_when_no_content_streamed(self) -> None:
        # Reasoning-only output: the stream succeeded but produced no content.
        # Empty events are forwarded as they arrive (the caller filters them),
        # then the solver takes over.
        orchestrator = FakeClient("cloud", chunks=("", ""))
        solver = FakeClient("local", role="solver", chunks=("from", "-local"))
        gateway = _gateway(orchestrator, solver)
        collected = [chunk for chunk in await self._collect(gateway) if chunk]
        self.assertEqual(collected, ["from", "-local"])
        self.assertTrue(gateway.fell_back)

    async def test_falls_back_when_stream_raises_before_content(self) -> None:
        orchestrator = FakeClient("cloud", error=VLLMServiceError("down"))
        solver = FakeClient("local", role="solver", chunks=("from", "-local"))
        gateway = _gateway(orchestrator, solver)
        self.assertEqual(await self._collect(gateway), ["from", "-local"])
        self.assertTrue(gateway.fell_back)


class CounterResetTest(unittest.IsolatedAsyncioTestCase):
    async def test_reset_clears_fallbacks_and_usage(self) -> None:
        orchestrator = FakeClient("cloud", error=VLLMServiceError("down"))
        solver = FakeClient("local", role="solver", content="from-local", usage=(1, 1))
        gateway = _gateway(orchestrator, solver)
        await gateway.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(gateway.fallbacks, 1)
        self.assertEqual(gateway.usage.total_tokens, 2)
        gateway.reset_counters()
        self.assertEqual(gateway.fallbacks, 0)
        self.assertEqual(gateway.usage.calls, 0)
        self.assertEqual(gateway.usage.total_tokens, 0)


class UsageTest(unittest.TestCase):
    def test_sums_both_endpoints(self) -> None:
        orchestrator = FakeClient("cloud", usage=(10, 5))
        solver = FakeClient("local", role="solver", usage=(3, 2))
        gateway = _gateway(orchestrator, solver)
        self.assertEqual(gateway.usage.total_tokens, 20)
        self.assertEqual(gateway.usage.calls, 2)

    def test_does_not_double_count_a_shared_client(self) -> None:
        shared = FakeClient("local", role="solver", usage=(10, 5))
        gateway = _gateway(shared, shared)
        self.assertEqual(gateway.usage.total_tokens, 15)


if __name__ == "__main__":
    unittest.main()
