from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.agent.loop import iter_agent_events
from app.agent.schema import (
    AgentRun,
    Observation,
    RouteDecision,
    TokenUsage,
    VerificationResult,
)
from app.agent.tools import ToolContext
from app.core.config import settings
from app.services import trace_service
from app.services.trace_service import (
    build_error_trace,
    build_trace,
    load_traces,
    record_trace,
    redact,
)


def _run(**overrides) -> AgentRun:
    base = dict(
        question="解方程 x^2-5x+6=0",
        decision=RouteDecision(
            intent="math", tool="solve_math_problem", query="x^2-5x+6=0"
        ),
        observations=[
            Observation(
                step=0,
                tool="solve_math_problem",
                arguments={"question": "x^2-5x+6=0"},
                success=True,
                summary='{"answer": "x=2 或 x=3"}',
                duration_ms=1200,
            )
        ],
        steps=1,
        stopped_reason="final",
        answer="x=2 或 x=3",
        verification=VerificationResult(
            status="verified", method="substitution", detail="全部满足"
        ),
        answer_model="mathllm-round7",
    )
    base.update(overrides)
    return AgentRun(**base)


class RedactTest(unittest.TestCase):
    def test_masks_api_keys_and_bearer_tokens(self) -> None:
        text = "failed with Authorization: Bearer sk-abcdef123456 and api_key=sk-zzz999"
        cleaned = redact(text)
        self.assertNotIn("sk-abcdef123456", cleaned)
        self.assertNotIn("sk-zzz999", cleaned)
        self.assertIn("***", cleaned)

    def test_leaves_ordinary_text_alone(self) -> None:
        self.assertEqual(redact("x=2 或 x=3"), "x=2 或 x=3")


class BuildTraceTest(unittest.TestCase):
    def test_carries_the_full_chain(self) -> None:
        trace = build_trace(
            _run(),
            started_at=datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc),
            duration_ms=1234,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, calls=2),
        )
        self.assertEqual(trace.stopped_reason, "final")
        self.assertEqual(trace.decision.tool, "solve_math_problem")
        self.assertEqual(len(trace.observations), 1)
        self.assertEqual(trace.verification.status, "verified")
        self.assertEqual(trace.usage.total_tokens, 30)
        self.assertEqual(trace.answer_model, "mathllm-round7")
        self.assertIn("2026-09-17", trace.started_at)

    def test_truncates_long_fields(self) -> None:
        trace = build_trace(
            _run(question="问" * 5000, answer="答" * 9000),
            started_at=datetime.now(timezone.utc),
            duration_ms=1,
        )
        self.assertLessEqual(len(trace.question), trace_service.MAX_QUESTION_CHARS)
        self.assertLessEqual(len(trace.answer), trace_service.MAX_ANSWER_CHARS)

    def test_redacts_secrets_in_the_question(self) -> None:
        trace = build_trace(
            _run(question="我的 key 是 sk-secret123456"),
            started_at=datetime.now(timezone.utc),
            duration_ms=1,
        )
        self.assertNotIn("sk-secret123456", trace.question)

    def test_error_trace_has_no_decision(self) -> None:
        trace = build_error_trace(
            "解方程",
            started_at=datetime.now(timezone.utc),
            duration_ms=5,
            error="VLLMServiceError: boom",
        )
        self.assertIsNone(trace.decision)
        self.assertEqual(trace.stopped_reason, "error")
        self.assertIn("boom", trace.error or "")


class RecordAndLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = trace_service.trace_path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_record_creates_directory_and_appends(self) -> None:
        nested = Path(self._tmp.name) / "a" / "b" / "runs.jsonl"
        record_trace(build_trace(_run(), started_at=datetime.now(timezone.utc), duration_ms=1), nested)
        record_trace(build_trace(_run(), started_at=datetime.now(timezone.utc), duration_ms=2), nested)
        lines = nested.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["duration_ms"], 1)

    def test_load_returns_recent_window(self) -> None:
        for index in range(5):
            record_trace(
                build_trace(
                    _run(question=f"q{index}"),
                    started_at=datetime.now(timezone.utc),
                    duration_ms=index,
                ),
                self.path,
            )
        traces = load_traces(self.path, limit=2)
        self.assertEqual([t.question for t in traces], ["q3", "q4"])

    def test_load_missing_file_is_empty(self) -> None:
        self.assertEqual(load_traces(Path(self._tmp.name) / "nope.jsonl"), [])

    def test_load_skips_broken_lines(self) -> None:
        self.path.write_text('{"not": "a trace"}\n', encoding="utf-8")
        self.assertEqual(load_traces(self.path), [])

    def test_record_never_raises(self) -> None:
        record_trace(_run_as_trace(), Path(self._tmp.name) / "\0bad")  # invalid path


def _run_as_trace():
    return build_trace(_run(), started_at=datetime.now(timezone.utc), duration_ms=1)


class LoopRecordsTraceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = trace_service.trace_path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_successful_run_is_recorded(self) -> None:
        client = _ScriptedClient()
        ctx = ToolContext(
            client=client,
            settings=replace(settings, trace_enabled=True, trace_dir=self._tmp.name),
        )
        async for _ in iter_agent_events(client, "解方程 x^2-5x+6=0", ctx, max_steps=2):
            pass
        traces = load_traces(self.path)
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace.decision.intent, "math")
        self.assertEqual(trace.stopped_reason, "final")
        self.assertIn("x=2", trace.answer)
        self.assertEqual(trace.answer_model, "fake-solver")

    async def test_failed_run_is_recorded_as_error(self) -> None:
        client = _ScriptedClient()
        ctx = ToolContext(
            client=client,
            settings=replace(settings, trace_enabled=True, trace_dir=self._tmp.name),
        )
        with self.assertRaises(ValueError):
            async for _ in iter_agent_events(client, "   ", ctx, max_steps=1):
                pass
        traces = load_traces(self.path)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].stopped_reason, "error")
        self.assertIsNone(traces[0].decision)

    async def test_disabled_tracing_writes_nothing(self) -> None:
        client = _ScriptedClient()
        ctx = ToolContext(
            client=client,
            settings=replace(settings, trace_enabled=False, trace_dir=self._tmp.name),
        )
        async for _ in iter_agent_events(client, "解方程", ctx, max_steps=1):
            pass
        self.assertEqual(load_traces(self.path), [])


class _ScriptedClient:
    """Minimal client: routes to math, solves with a fake local model."""

    def __init__(self) -> None:
        from app.agent.schema import TokenUsage

        self.usage = TokenUsage(prompt_tokens=7, completion_tokens=9, total_tokens=16, calls=2)
        self.config = type("Config", (), {"model": "fake-solver", "role": "solver"})()
        self._json = [
            json.dumps(
                {
                    "intent": "math",
                    "tool": "solve_math_problem",
                    "query": "x^2-5x+6=0",
                    "answer_style": "step_by_step",
                },
                ensure_ascii=False,
            ),
            '{"action": "final"}',
            json.dumps(
                {"applicable": True, "lhs": "x**2-5*x+6", "rhs": "0", "variables": ["x"]},
                ensure_ascii=False,
            ),
            json.dumps({"kind": "solution_set", "values": ["2", "3"]}, ensure_ascii=False),
        ]
        self._index = 0

    async def complete_json(self, messages, *, max_tokens=None, schema=None):
        index = min(self._index, len(self._json) - 1)
        self._index += 1
        return {"choices": [{"message": {"content": self._json[index]}}]}

    async def complete_with_tools(
        self, messages, tools, *, tool_choice=None, max_tokens=None
    ):
        index = min(self._index, len(self._json) - 1)
        self._index += 1
        raw = self._json[index]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"choices": [{"message": {"content": raw}}]}
        if isinstance(data, dict) and "intent" in data:
            name, arguments = "route_question", data
        elif isinstance(data, dict) and data.get("action") == "call_tool" and data.get("tool"):
            name, arguments = str(data["tool"]), data.get("arguments") or {}
        else:
            return {"choices": [{"message": {"content": raw}}]}
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

    async def complete(self, messages, **kwargs):
        return {"choices": [{"message": {"content": "x=2 或 x=3"}}]}

    async def stream_raw(self, messages):
        yield {"choices": [{"delta": {"content": "x=2 或 x=3"}}]}


if __name__ == "__main__":
    unittest.main()
