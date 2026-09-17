from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.agent.router import classify, extract_json_object
from app.agent.schema import RouteDecision


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class FakeClient:
    """Returns preset contents and counts how many times it was called."""

    def __init__(self, contents: list[str]):
        self._contents = contents
        self.calls = 0

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict:
        self.calls += 1
        index = min(self.calls - 1, len(self._contents) - 1)
        return _completion(self._contents[index])


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
        self.assertEqual(client.calls, 3)

    async def test_empty_question_rejected(self) -> None:
        client = FakeClient(['{"intent": "general", "tool": "none", "query": "x"}'])
        with self.assertRaises(ValueError):
            await classify(client, "   ")


if __name__ == "__main__":
    unittest.main()
