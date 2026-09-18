from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.agent.loop import iter_agent_events
from app.agent.tools import ToolContext
from app.core.config import settings
from app.services import rag_service
from app.services.rag_service import RetrievedChunk


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _route(intent: str, tool: str, query: str = "x^2-5x+6=0") -> str:
    return json.dumps(
        {
            "intent": intent,
            "tool": tool,
            "query": query,
            "answer_style": "step_by_step",
        },
        ensure_ascii=False,
    )


def _call_tool(tool: str, arguments: dict) -> str:
    return json.dumps(
        {"action": "call_tool", "tool": tool, "arguments": arguments},
        ensure_ascii=False,
    )


FINAL_JSON = '{"action": "final"}'


class ScriptedClient:
    def __init__(self, json_responses: list[str], chunks: list[str] | None = None) -> None:
        self._json = json_responses
        self.chunks = ["答案", "如下"] if chunks is None else chunks
        self.json_calls = 0
        self.complete_calls = 0
        self.stream_calls = 0
        self.reset_count = 0

    def reset_counters(self) -> None:
        self.reset_count += 1

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        schema: dict | None = None,
    ) -> dict:
        index = min(self.json_calls, len(self._json) - 1)
        self.json_calls += 1
        return _completion(self._json[index])

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.complete_calls += 1
        return _completion("工具结果")

    async def stream_raw(self, messages: list[dict[str, str]]):
        self.stream_calls += 1
        for chunk in self.chunks:
            yield {"choices": [{"delta": {"content": chunk}}]}


def make_ctx(client: ScriptedClient) -> ToolContext:
    return ToolContext(client=client, settings=replace(settings, trace_enabled=False))


async def collect(
    client: ScriptedClient, question: str = "求解 x^2-5x+6=0"
) -> list[dict]:
    return [
        event
        async for event in iter_agent_events(
            client, question, make_ctx(client), max_steps=2
        )
    ]


def _knowledge_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(content="判别式是 b^2-4ac", source="01-判别式.md", distance=0.4)
    ]


class IterAgentEventsTest(unittest.IsolatedAsyncioTestCase):
    async def test_event_sequence_for_single_tool(self) -> None:
        client = ScriptedClient([_route("math", "solve_math_problem"), FINAL_JSON])
        events = await collect(client)
        # The local solver streams while it writes, so the delta arrives before
        # ``tool_end`` and the answer phase does not emit it again.
        self.assertEqual(
            [event["type"] for event in events],
            [
                "thinking",
                "route",
                "tool_start",
                "answer_delta",
                "answer_delta",
                "tool_end",
                "thinking",
                "verify_start",
                "verify",
                "done",
            ],
        )
        self.assertEqual(events[0]["stage"], "classify")
        self.assertEqual(events[1]["decision"]["intent"], "math")
        self.assertEqual(events[2]["tool"], "solve_math_problem")
        self.assertTrue(events[5]["success"])
        self.assertEqual(events[7]["type"], "verify_start")
        self.assertEqual(events[9]["stopped_reason"], "final")
        # Per-run counters are reset once at the start of the run.
        self.assertEqual(client.reset_count, 1)

    async def test_math_answer_passes_through_solver(self) -> None:
        client = ScriptedClient(
            [_route("math", "solve_math_problem"), FINAL_JSON],
            chunks=["方程", "的解", "是 2 和 3"],
        )
        events = await collect(client)
        deltas = [event["content"] for event in events if event["type"] == "answer_delta"]
        # The solver's own stream is the answer: nothing is regenerated.
        self.assertEqual(deltas, ["方程", "的解", "是 2 和 3"])
        self.assertEqual(events[-1]["run"]["answer"], "方程的解是 2 和 3")
        self.assertEqual(client.complete_calls, 0)

    async def test_answer_deltas_reconstruct_answer(self) -> None:
        client = ScriptedClient(
            [_route("general", "none", query="你好"), FINAL_JSON],
            chunks=["方程", "的解", "是 2 和 3"],
        )
        events = await collect(client)
        streamed = "".join(
            event["content"] for event in events if event["type"] == "answer_delta"
        )
        done = events[-1]
        self.assertEqual(streamed, "方程的解是 2 和 3")
        self.assertEqual(done["run"]["answer"], streamed)

    async def test_empty_answer_gets_fallback_delta(self) -> None:
        client = ScriptedClient(
            [_route("general", "none", query="你好"), FINAL_JSON], chunks=[]
        )
        events = await collect(client)
        deltas = [event["content"] for event in events if event["type"] == "answer_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertIn("暂时无法生成回答", deltas[0])

    async def test_second_tool_emits_another_pair(self) -> None:
        client = ScriptedClient(
            [
                _route("math", "solve_math_problem"),
                _call_tool("calculate_expression", {"expression": "1+1"}),
                FINAL_JSON,
            ]
        )
        events = await collect(client)
        starts = [event for event in events if event["type"] == "tool_start"]
        self.assertEqual([event["tool"] for event in starts], [
            "solve_math_problem",
            "calculate_expression",
        ])
        self.assertEqual(events[-1]["steps"], 2)

    async def test_knowledge_answer_skips_verification(self) -> None:
        client = ScriptedClient(
            [_route("knowledge", "search_knowledge", query="什么是判别式"), FINAL_JSON]
        )
        with patch.object(rag_service, "search", return_value=_knowledge_chunks()):
            events = await collect(client, question="什么是判别式")
        types = [event["type"] for event in events]
        self.assertIn("verify_skipped", types)
        self.assertNotIn("verify_start", types)
        self.assertNotIn("verify", types)
        self.assertIsNone(events[-1]["verification"])
        self.assertIsNone(events[-1]["run"]["verification"])

    async def test_verify_rag_flag_keeps_grounding_check(self) -> None:
        client = ScriptedClient(
            [
                _route("knowledge", "search_knowledge", query="什么是判别式"),
                FINAL_JSON,
                json.dumps({"applicable": False, "lhs": "", "rhs": "0", "variables": []}),
                json.dumps({"kind": "text", "values": []}),
                json.dumps({"grounded": True, "unsupported": [], "reason": ""}),
            ]
        )
        ctx = ToolContext(
            client=client,
            settings=replace(settings, trace_enabled=False, verify_rag=True),
        )
        with patch.object(rag_service, "search", return_value=_knowledge_chunks()):
            events = [
                event
                async for event in iter_agent_events(
                    client, "什么是判别式", ctx, max_steps=2
                )
            ]
        types = [event["type"] for event in events]
        self.assertIn("verify_start", types)
        self.assertIn("verify", types)
        self.assertNotIn("verify_skipped", types)
        self.assertEqual(events[-1]["verification"]["status"], "verified")
        self.assertEqual(events[-1]["verification"]["method"], "grounding")

    async def test_empty_question_raises(self) -> None:
        client = ScriptedClient([_route("general", "none")])
        with self.assertRaises(ValueError):
            [
                event
                async for event in iter_agent_events(
                    client, "   ", make_ctx(client), max_steps=2
                )
            ]


if __name__ == "__main__":
    unittest.main()
