from __future__ import annotations

import json
import unittest

from app.agent.schema import GroundingVerdict, Observation
from app.core.config import settings
from app.services import verify_service
from app.services.verify_service import (
    check_by_substitution,
    check_expression_value,
    citations_from_verdict,
    documents_from_observations,
    verify_answer,
)


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class ScriptedClient:
    """Returns queued JSON replies; the last reply repeats when exhausted."""

    def __init__(self, json_responses: list[str]) -> None:
        self._json = json_responses
        self.calls = 0

    async def complete_json(self, messages, *, max_tokens=None, schema=None) -> dict:
        index = min(self.calls, len(self._json) - 1)
        self.calls += 1
        return _completion(self._json[index])

    async def complete(self, messages, **kwargs) -> dict:
        return _completion("")


def _search_observation(documents: list[str]) -> Observation:
    return Observation(
        step=0,
        tool="search_knowledge",
        arguments={"query": "判别式"},
        success=True,
        summary=json.dumps(
            {"documents": [{"content": text, "source": "x.md"} for text in documents]},
            ensure_ascii=False,
        ),
        duration_ms=1,
    )


class SubstitutionTest(unittest.TestCase):
    def test_correct_roots_verified(self) -> None:
        result = check_by_substitution("x**2 - 5*x + 6", "0", ["2", "3"], ["x"])
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.method, "substitution")

    def test_wrong_root_refuted_with_counterexample(self) -> None:
        result = check_by_substitution("x**2 - 5*x + 6", "0", ["2", "4"], ["x"])
        self.assertEqual(result.status, "refuted")
        self.assertEqual(result.counterexample, "4")
        self.assertIn("不成立", result.detail)

    def test_assignment_prefix_is_stripped(self) -> None:
        result = check_by_substitution("x**2 - 5*x + 6", "0", ["x=2", "x=3"], ["x"])
        self.assertEqual(result.status, "verified")

    def test_equation_with_rhs(self) -> None:
        result = check_by_substitution("x + 1", "3", ["2"], ["x"])
        self.assertEqual(result.status, "verified")

    def test_multiple_variables_unknown(self) -> None:
        result = check_by_substitution("x + y", "3", ["1"], ["x", "y"])
        self.assertEqual(result.status, "unknown")

    def test_variable_absent_unknown(self) -> None:
        result = check_by_substitution("a + 1", "3", ["2"], ["x"])
        self.assertEqual(result.status, "unknown")

    def test_non_numeric_expression_refuted(self) -> None:
        result = check_by_substitution("x**2", "0", ["2"], ["x"])
        self.assertEqual(result.status, "refuted")

    def test_unsafe_expressions_rejected(self) -> None:
        for expression in (
            "__import__('os')",
            "x.__class__",
            "x.__class__.__mro__",
            "x.real",
            "open('/etc/passwd')",
        ):
            with self.subTest(expression=expression):
                result = check_by_substitution(expression, "0", ["2"], ["x"])
                self.assertEqual(result.status, "unknown")

    def test_no_values_unknown(self) -> None:
        result = check_by_substitution("x**2", "0", [], ["x"])
        self.assertEqual(result.status, "unknown")

    def test_degenerate_translation_is_unknown(self) -> None:
        # 模型把「解方程 x^2-5x+6=0」退化成 "x = 0" 时，不能据此判错。
        result = check_by_substitution("x", "0", ["2", "3"], ["x"])
        self.assertEqual(result.status, "unknown")
        self.assertIn("不可信", result.detail)


class ExpressionValueTest(unittest.TestCase):
    def test_fraction_matches(self) -> None:
        result = check_expression_value("1/3 + 1/6", ["1/2"])
        self.assertEqual(result.status, "verified")

    def test_decimal_matches_within_tolerance(self) -> None:
        result = check_expression_value("1/3", ["0.3333333333"])
        self.assertEqual(result.status, "verified")

    def test_wrong_value_refuted(self) -> None:
        result = check_expression_value("1/3 + 1/6", ["1/3"])
        self.assertEqual(result.status, "refuted")
        self.assertEqual(result.counterexample, "1/3")

    def test_symbolic_expression_unknown(self) -> None:
        result = check_expression_value("x + 1", ["2"])
        self.assertEqual(result.status, "unknown")


class DocumentsTest(unittest.TestCase):
    def test_collects_chunks(self) -> None:
        observations = [_search_observation(["判别式 Δ = b^2 - 4ac"])]
        self.assertEqual(len(documents_from_observations(observations)), 1)

    def test_ignores_failed_observations(self) -> None:
        observation = _search_observation(["x"])
        observation = observation.model_copy(update={"success": False})
        self.assertEqual(documents_from_observations([observation]), [])

    def test_sources_carry_line_ranges(self) -> None:
        observation = Observation(
            step=0,
            tool="search_knowledge",
            arguments={"query": "判别式"},
            success=True,
            summary=json.dumps(
                {
                    "documents": [
                        {
                            "content": "判别式 Δ = b^2 - 4ac",
                            "source": "01.md",
                            "start_line": 3,
                            "end_line": 7,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            duration_ms=1,
        )
        sources = verify_service.sources_from_observations([observation])
        self.assertEqual(sources[0]["start_line"], 3)
        self.assertEqual(sources[0]["end_line"], 7)


class CitationsTest(unittest.TestCase):
    def test_maps_used_chunks_to_sources(self) -> None:
        verdict = GroundingVerdict(grounded=True, used_chunks=[1, 2, 2, 9])
        sources = [
            {"source": "a.md", "start_line": 1, "end_line": 5, "content": "x"},
            {"source": "b.md", "start_line": 10, "end_line": 12, "content": "y"},
        ]
        citations = citations_from_verdict(verdict, sources)
        # 9 is out of range and the duplicate 2 collapses.
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].source, "a.md")
        self.assertEqual(citations[1].start_line, 10)


def _form(
    applicable: bool,
    lhs: str = "",
    rhs: str = "0",
    variables: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "applicable": applicable,
            "lhs": lhs,
            "rhs": rhs,
            "variables": variables or [],
        },
        ensure_ascii=False,
    )


def _extracted(kind: str, values: list[str] | None = None) -> str:
    return json.dumps({"kind": kind, "values": values or []}, ensure_ascii=False)


class VerifyAnswerTest(unittest.IsolatedAsyncioTestCase):
    def _client(self, *responses: str) -> ScriptedClient:
        return ScriptedClient(list(responses))

    async def test_substitution_path(self) -> None:
        client = self._client(
            _form(True, "x**2 - 5*x + 6", "0", ["x"]),
            _extracted("solution_set", ["2", "3"]),
        )
        result = await verify_answer(client, "解 x^2-5x+6=0", "最终答案 x=2 或 x=3", [])
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.method, "substitution")

    async def test_expression_path(self) -> None:
        client = self._client(
            _form(True, "1/3 + 1/6"),
            _extracted("numeric", ["1/2"]),
        )
        result = await verify_answer(client, "计算 1/3+1/6", "答案是 1/2", [])
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.method, "expression")

    async def test_refuted_path(self) -> None:
        client = self._client(
            _form(True, "x**2 - 5*x + 6", "0", ["x"]),
            _extracted("solution_set", ["5"]),
        )
        result = await verify_answer(client, "解 x^2-5x+6=0", "最终答案 x=5", [])
        self.assertEqual(result.status, "refuted")
        self.assertEqual(result.counterexample, "5")

    async def test_grounding_path(self) -> None:
        verdict = json.dumps({"grounded": True, "unsupported": [], "reason": "有依据"})
        observations = [_search_observation(["判别式大于零有两个实根"])]
        client = self._client(
            _form(False),
            _extracted("text"),
            verdict,
        )
        result = await verify_answer(
            client,
            "判别式怎么判断根的个数",
            "判别式大于零时有两个实根。",
            observations,
        )
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.method, "grounding")

    async def test_ungrounded_is_unknown_not_refuted(self) -> None:
        verdict = json.dumps(
            {"grounded": False, "unsupported": ["某个结论"], "reason": "无依据"}
        )
        observations = [_search_observation(["判别式大于零有两个实根"])]
        client = self._client(_form(False), _extracted("text"), verdict)
        result = await verify_answer(client, "q", "a", observations)
        self.assertEqual(result.status, "unknown")
        self.assertIn("某个结论", result.detail)

    async def test_text_answer_without_documents_is_unknown(self) -> None:
        client = self._client(_form(False), _extracted("text"))
        result = await verify_answer(client, "q", "a", [])
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.method, "none")

    async def test_unparsable_responses_are_unknown(self) -> None:
        result = await verify_answer(self._client("不是 JSON"), "q", "a", [])
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.method, "none")


if __name__ == "__main__":
    unittest.main()
