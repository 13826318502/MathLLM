from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.agent.loop import run_agent
from app.agent.tools import ToolContext
from app.core.config import settings
from app.services import rag_service
from app.services.rag_service import KnowledgeBaseError


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


class ScriptedClient:
    """Returns queued JSON replies for complete_json and one answer for complete."""

    def __init__(self, json_responses: list[str], answer: str = "最终答案") -> None:
        self._json = json_responses
        self.answer = answer
        self.json_calls = 0
        self.complete_calls = 0

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
        return _completion(self.answer)


def make_ctx(client: ScriptedClient) -> ToolContext:
    return ToolContext(client=client, settings=settings)


def call_tool_json(tool: str, arguments: dict) -> str:
    return json.dumps(
        {"action": "call_tool", "tool": tool, "arguments": arguments},
        ensure_ascii=False,
    )


FINAL_JSON = '{"action": "final"}'


class RunAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path_single_tool(self) -> None:
        client = ScriptedClient([_route("math", "solve_math_problem"), FINAL_JSON])
        result = await run_agent(client, "求解 x^2-5x+6=0", make_ctx(client))
        self.assertEqual(result.stopped_reason, "final")
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].tool, "solve_math_problem")
        self.assertTrue(result.observations[0].success)
        self.assertEqual(result.answer, "最终答案")
        self.assertEqual(client.complete_calls, 2)

    async def test_second_tool_call(self) -> None:
        client = ScriptedClient(
            [
                _route("math", "solve_math_problem"),
                call_tool_json("calculate_expression", {"expression": "2+3*4"}),
                FINAL_JSON,
            ]
        )
        result = await run_agent(client, "求解并计算", make_ctx(client))
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.stopped_reason, "final")
        self.assertEqual(result.observations[1].tool, "calculate_expression")
        self.assertTrue(result.observations[1].success)

    async def test_duplicate_call_stops(self) -> None:
        duplicate = call_tool_json(
            "solve_math_problem", {"question": "x^2-5x+6=0"}
        )
        client = ScriptedClient(
            [_route("math", "solve_math_problem"), duplicate, duplicate]
        )
        result = await run_agent(client, "求解 x^2-5x+6=0", make_ctx(client))
        self.assertEqual(result.stopped_reason, "duplicate")
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(client.complete_calls, 2)

    async def test_max_steps_stops(self) -> None:
        client = ScriptedClient(
            [
                _route("general", "none", query="随便算算"),
                call_tool_json("calculate_expression", {"expression": "1+1"}),
                call_tool_json("calculate_expression", {"expression": "2+2"}),
                call_tool_json("calculate_expression", {"expression": "3+3"}),
            ]
        )
        result = await run_agent(client, "随便算算", make_ctx(client), max_steps=2)
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.observations), 2)

    async def test_unknown_tool_stops(self) -> None:
        client = ScriptedClient(
            [_route("general", "none", query="你好"), call_tool_json("nope", {})]
        )
        result = await run_agent(client, "你好", make_ctx(client))
        self.assertEqual(result.stopped_reason, "unknown_tool")
        self.assertEqual(result.steps, 0)

    async def test_tool_failure_is_recorded(self) -> None:
        client = ScriptedClient([_route("knowledge", "search_knowledge"), FINAL_JSON])
        with patch.object(
            rag_service,
            "search",
            side_effect=KnowledgeBaseError("知识库尚未建立，请先运行索引"),
        ):
            result = await run_agent(client, "什么是判别式", make_ctx(client))
        self.assertEqual(result.stopped_reason, "final")
        self.assertFalse(result.observations[0].success)
        self.assertIn("知识库", result.observations[0].summary)

    async def test_unparsable_action_falls_back_to_final(self) -> None:
        client = ScriptedClient([_route("general", "none", query="你好"), "不是 JSON"])
        result = await run_agent(client, "你好", make_ctx(client))
        self.assertEqual(result.stopped_reason, "final")
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.answer, "最终答案")

    async def test_empty_question_rejected(self) -> None:
        client = ScriptedClient([_route("general", "none")])
        with self.assertRaises(ValueError):
            await run_agent(client, "   ", make_ctx(client))


if __name__ == "__main__":
    unittest.main()
