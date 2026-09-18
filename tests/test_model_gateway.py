from __future__ import annotations

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
    ) -> None:
        self.config = _config(role, model)
        self.usage = TokenUsage()
        self._content = content
        self._chunks = chunks
        self._error = error
        if usage:
            self.usage.add(usage[0], usage[1], sum(usage))

    def reset_usage(self) -> None:
        self.usage = TokenUsage()

    def _check(self) -> None:
        if self._error is not None:
            raise self._error

    async def complete(self, messages, **kwargs) -> dict:
        self._check()
        return _completion(self._content)

    async def complete_json(self, messages, **kwargs) -> dict:
        self._check()
        return _completion(self._content)

    async def stream_raw(self, messages):
        self._check()
        for chunk in self._chunks:
            yield {"choices": [{"delta": {"content": chunk}}]}


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
