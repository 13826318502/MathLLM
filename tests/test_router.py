from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from app.agent.router import ROUTE_TOOL, classify, extract_json_object
from app.agent.schema import RouteDecision
from app.services.stream_service import extract_tool_calls, parse_tool_arguments


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _tool_completion(name: str, arguments: dict) -> dict:
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


def _as_tool_response(raw: str) -> dict:
    """Valid route JSON becomes a ``route_question`` call; anything else is text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _completion(raw)
    if isinstance(data, dict) and "intent" in data:
        return _tool_completion("route_question", data)
    return _completion(raw)


class FakeClient:
    """Returns preset contents and counts how many times it was called."""

    def __init__(self, contents: list[str]):
        self._contents = contents
        self.calls = 0

    def _next(self) -> str:
        index = min(self.calls, len(self._contents) - 1)
        self.calls += 1
        return self._contents[index]

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict:
        return _completion(self._next())

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        tool_choice=None,
        max_tokens: int | None = None,
    ) -> dict:
        return _as_tool_response(self._next())


class CapturingClient:
    """Records the messages it was called with."""

    def __init__(self, content: str):
        self.content = content
        self.messages: list[dict[str, str]] = []

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict:
        self.messages = messages
        return _completion(self.content)

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        tool_choice=None,
        max_tokens: int | None = None,
    ) -> dict:
        self.messages = messages
        return _as_tool_response(self.content)


class RecordingToolClient:
    """Returns a fixed tool call and records the request it was sent."""

    def __init__(self, payload: dict, fallback: str = '{"intent": "general"}'):
        self.payload = payload
        self.fallback = fallback
        self.tools: list[dict] = []
        self.tool_choice = None

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        tool_choice=None,
        max_tokens: int | None = None,
    ) -> dict:
        self.tools = tools
        self.tool_choice = tool_choice
        return self.payload

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict:
        return _completion(self.fallback)


class ExtractJsonObjectTest(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1}'), '{"a": 1}')

    def test_fenced_object(self) -> None:
        self.assertEqual(
            extract_json_object('```json\n{"a": 1}\n```'),
            '{"a": 1}',
        )

    def test_surrounding_prose(self) -> None:
        self.assertEqual(
            extract_json_object('好的，结果如下：{"a": 1} 请查收'),
            '{"a": 1}',
        )

    def test_non_json_passthrough(self) -> None:
        self.assertEqual(extract_json_object("这不是 JSON"), "这不是 JSON")


class RouteDecisionTest(unittest.TestCase):
    def test_valid_default_style(self) -> None:
        decision = RouteDecision.model_validate(
            {"intent": "math", "tool": "solve_math_problem", "query": "解方程"}
        )
        self.assertEqual(decision.intent, "math")
        self.assertEqual(decision.answer_style, "step_by_step")

    def test_tool_defaults_to_none(self) -> None:
        decision = RouteDecision.model_validate(
            {"intent": "general", "query": "你好"}
        )
        self.assertEqual(decision.tool, "none")

    def test_invalid_intent_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RouteDecision.model_validate(
                {"intent": "maths", "tool": "solve_math_problem", "query": "解方程"}
            )

    def test_invalid_tool_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RouteDecision.model_validate(
                {"intent": "math", "tool": "math_solver", "query": "解方程"}
            )

    def test_extra_field_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RouteDecision.model_validate(
                {
                    "intent": "math",
                    "tool": "solve_math_problem",
                    "query": "解方程",
                    "reason": "看起来像数学题",
                }
            )


class ToolCallParsingTest(unittest.TestCase):
    def test_arguments_are_json_string(self) -> None:
        call = {"function": {"name": "x", "arguments": '{"a": 1}'}}
        self.assertEqual(parse_tool_arguments(call), {"a": 1})

    def test_arguments_already_dict(self) -> None:
        call = {"function": {"name": "x", "arguments": {"a": 1}}}
        self.assertEqual(parse_tool_arguments(call), {"a": 1})

    def test_malformed_arguments_become_empty(self) -> None:
        call = {"function": {"name": "x", "arguments": "not json"}}
        self.assertEqual(parse_tool_arguments(call), {})

    def test_extract_tool_calls_empty_when_answered(self) -> None:
        self.assertEqual(extract_tool_calls(_completion("hi")), [])


class ClassifyTest(unittest.IsolatedAsyncioTestCase):
    async def test_valid_response_needs_one_call(self) -> None:
        client = FakeClient(
            ['{"intent": "math", "tool": "solve_math_problem", "query": "解 x^2-5x+6=0"}']
        )
        decision = await classify(client, "求解 x^2 - 5x + 6 = 0")
        self.assertEqual(decision.intent, "math")
        self.assertEqual(decision.tool, "solve_math_problem")
        self.assertEqual(client.calls, 1)

    async def test_retry_then_valid(self) -> None:
        client = FakeClient(
            [
                "这不是 JSON",
                '{"intent": "knowledge", "tool": "search_knowledge", "query": "判别式"}',
            ]
        )
        decision = await classify(client, "什么是判别式")
        self.assertEqual(decision.intent, "knowledge")
        self.assertEqual(client.calls, 2)

    async def test_fallback_after_failures(self) -> None:
        client = FakeClient(["nope", "still nope", "nope again"])
        decision = await classify(client, "你好", max_retries=2)
        self.assertEqual(decision.intent, "general")
        self.assertEqual(decision.tool, "none")
        self.assertEqual(decision.query, "你好")
        # One function-calling attempt, then three prompted-JSON attempts.
        self.assertEqual(client.calls, 4)

    async def test_empty_question_rejected(self) -> None:
        client = FakeClient(['{"intent": "general", "tool": "none", "query": "x"}'])
        with self.assertRaises(ValueError):
            await classify(client, "   ")

    async def test_history_and_summary_are_sent(self) -> None:
        client = CapturingClient(
            '{"intent": "math", "tool": "solve_math_problem", "query": "验算 x=2 和 x=3"}'
        )
        history = [
            {"role": "user", "content": "解方程 x^2-5x+6=0"},
            {"role": "assistant", "content": "x=2 或 x=3"},
        ]
        await classify(client, "那验算一下", history=history, summary="用户在解一元二次方程")
        contents = [message["content"] for message in client.messages]
        self.assertTrue(any("用户在解一元二次方程" in text for text in contents))
        self.assertIn({"role": "user", "content": "解方程 x^2-5x+6=0"}, client.messages)
        self.assertEqual(client.messages[-1], {"role": "user", "content": "那验算一下"})


class FunctionCallingRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_forces_route_question_tool(self) -> None:
        client = RecordingToolClient(
            _tool_completion(
                "route_question",
                {
                    "intent": "math",
                    "tool": "solve_math_problem",
                    "query": "解方程",
                    "answer_style": "step_by_step",
                },
            )
        )
        decision = await classify(client, "解方程")
        self.assertEqual(decision.intent, "math")
        self.assertEqual(client.tools, ROUTE_TOOL)
        self.assertEqual(
            client.tool_choice,
            {"type": "function", "function": {"name": "route_question"}},
        )

    async def test_invalid_arguments_fall_back_to_json(self) -> None:
        client = RecordingToolClient(
            _tool_completion("route_question", {"intent": "maths", "query": "x"}),
            fallback='{"intent": "general", "tool": "none", "query": "x"}',
        )
        decision = await classify(client, "x")
        self.assertEqual(decision.intent, "general")

    async def test_no_tool_call_falls_back_to_json(self) -> None:
        client = RecordingToolClient(
            _completion("随便答一句"),
            fallback='{"intent": "knowledge", "tool": "search_knowledge", "query": "x"}',
        )
        decision = await classify(client, "x")
        self.assertEqual(decision.intent, "knowledge")


if __name__ == "__main__":
    unittest.main()
