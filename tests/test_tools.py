from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

from app.agent.schema import ToolResult
from app.agent.tools import ToolContext, ToolSpec, call_tool, get_tool, run_tool
from app.core.config import settings
from app.services import rag_service
from app.services.rag_service import KnowledgeBaseError


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class FakeClient:
    def __init__(self, *, content: str = "", json_content: str = "") -> None:
        self.content = content
        self.json_content = json_content
        self.complete_calls = 0
        self.json_calls = 0

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.complete_calls += 1
        return _completion(self.content)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        schema: dict | None = None,
    ) -> dict:
        self.json_calls += 1
        return _completion(self.json_content)


def make_ctx(client: FakeClient | None = None) -> ToolContext:
    return ToolContext(client=client or FakeClient(), settings=settings)


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegistryTest(unittest.TestCase):
    def test_all_tools_registered(self) -> None:
        for name in (
            "solve_math_problem",
            "calculate_expression",
            "search_knowledge",
            "check_math_answer",
        ):
            self.assertIsNotNone(get_tool(name))

    def test_catalog_has_schemas(self) -> None:
        from app.agent.tools import tool_catalog

        catalog = tool_catalog()
        self.assertEqual(len(catalog), 4)
        for entry in catalog:
            self.assertIn("parameters", entry)
            self.assertEqual(entry["parameters"]["type"], "object")


class RunToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_arguments_do_not_raise(self) -> None:
        result = await call_tool("calculate_expression", {}, make_ctx())
        self.assertFalse(result.success)
        self.assertIn("参数", result.error or "")

    async def test_unknown_tool(self) -> None:
        result = await call_tool("does_not_exist", {}, make_ctx())
        self.assertFalse(result.success)
        self.assertIn("未知工具", result.error or "")

    async def test_exception_becomes_failure(self) -> None:
        async def boom(args: EmptyInput, ctx: ToolContext) -> ToolResult:
            raise RuntimeError("kaboom")

        spec = ToolSpec(
            name="boom",
            description="",
            input_model=EmptyInput,
            handler=boom,
        )
        result = await run_tool(spec, {}, make_ctx())
        self.assertFalse(result.success)
        self.assertIn("执行失败", result.error or "")

    async def test_timeout_becomes_failure(self) -> None:
        async def slow(args: EmptyInput, ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0.2)
            return ToolResult(success=True)

        spec = ToolSpec(
            name="slow",
            description="",
            input_model=EmptyInput,
            handler=slow,
            timeout=0.01,
        )
        result = await run_tool(spec, {}, make_ctx())
        self.assertFalse(result.success)
        self.assertIn("超时", result.error or "")


class CalculatorTest(unittest.IsolatedAsyncioTestCase):
    async def _calc(self, expression: str) -> ToolResult:
        return await call_tool(
            "calculate_expression",
            {"expression": expression},
            make_ctx(),
        )

    async def test_arithmetic(self) -> None:
        result = await self._calc("2 + 3 * 4")
        self.assertTrue(result.success)
        self.assertEqual(result.data["value"], 14)

    async def test_functions_and_constants(self) -> None:
        result = await self._calc("sqrt(16) + pi")
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.data["value"], 4 + 3.14159265, places=5)

    async def test_rejects_code_execution(self) -> None:
        for expression in (
            "__import__('os').system('echo hi')",
            "open('/etc/passwd')",
            "a.b",
            "(1).__class__",
        ):
            with self.subTest(expression=expression):
                result = await self._calc(expression)
                self.assertFalse(result.success)

    async def test_rejects_huge_exponent(self) -> None:
        result = await self._calc("9 ** 9 ** 9")
        self.assertFalse(result.success)

    async def test_division_by_zero(self) -> None:
        result = await self._calc("1 / 0")
        self.assertFalse(result.success)


class MathSolverTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_model_answer(self) -> None:
        client = FakeClient(content="因式分解得 x=2 或 x=3")
        result = await call_tool(
            "solve_math_problem",
            {"question": "解 x^2-5x+6=0"},
            make_ctx(client),
        )
        self.assertTrue(result.success)
        self.assertIn("x=2", result.data["answer"])
        self.assertEqual(client.complete_calls, 1)

    async def test_empty_model_answer_is_failure(self) -> None:
        result = await call_tool(
            "solve_math_problem",
            {"question": "解 x^2-5x+6=0"},
            make_ctx(FakeClient(content="")),
        )
        self.assertFalse(result.success)


class KnowledgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_reports_unavailable(self) -> None:
        with patch.object(
            rag_service,
            "search",
            side_effect=KnowledgeBaseError("知识库尚未建立，请先运行索引"),
        ):
            result = await call_tool("search_knowledge", {"query": "判别式"}, make_ctx())
        self.assertFalse(result.success)
        self.assertIn("知识库", result.error or "")


class CheckerTest(unittest.IsolatedAsyncioTestCase):
    async def test_valid_verdict(self) -> None:
        client = FakeClient(json_content='{"verdict": "correct", "reason": "结果正确"}')
        result = await call_tool(
            "check_math_answer",
            {"question": "1+1=?", "answer": "2"},
            make_ctx(client),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data["verdict"], "correct")

    async def test_invalid_verdict_is_failure(self) -> None:
        client = FakeClient(json_content="不是 JSON")
        result = await call_tool(
            "check_math_answer",
            {"question": "1+1=?", "answer": "2"},
            make_ctx(client),
        )
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
