from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent.loop import iter_agent_events
from app.agent.schema import (
    Observation,
    RagGrounding,
    RouteDecision,
    RunTrace,
)
from app.agent.tools import ToolContext
from app.api.dependencies import get_orchestrator, get_settings
from app.api.main import app
from app.core.config import settings
from app.services import rag_attribution_service, rag_service, trace_service
from app.services.rag_attribution_service import (
    build_attribution,
    load_checks,
    save_check,
    summarize,
)
from app.services.rag_service import RetrievedChunk


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
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _completion(raw)
    if not isinstance(data, dict):
        return _completion(raw)
    if "intent" in data:
        return _tool_completion("route_question", data)
    if data.get("action") == "call_tool" and data.get("tool"):
        return _tool_completion(str(data["tool"]), data.get("arguments") or {})
    return _completion(raw)


def _decision(intent: str = "knowledge", tool: str = "search_knowledge", query: str = "判别式"):
    return RouteDecision(
        intent=intent, tool=tool, query=query, answer_style="step_by_step"
    )


def _search_observation(
    documents: list[dict], *, query: str = "判别式", success: bool = True
) -> Observation:
    return Observation(
        step=0,
        tool="search_knowledge",
        arguments={"query": query},
        success=success,
        summary=json.dumps({"documents": documents}, ensure_ascii=False),
        duration_ms=7,
    )


def _documents() -> list[dict]:
    return [
        {
            "content": "判别式 Δ = b^2 - 4ac",
            "source": "01-二次方程与判别式.md",
            "distance": 0.42,
            "rank": 1,
        }
    ]


class BuildAttributionTest(unittest.TestCase):
    def test_no_knowledge_use(self) -> None:
        attribution = build_attribution(_decision("general", "none", "你好"), [], "你好")
        self.assertFalse(attribution.used_knowledge)
        self.assertEqual(attribution.retrieved, [])

    def test_parses_retrieved_chunks(self) -> None:
        observations = [_search_observation(_documents())]
        attribution = build_attribution(_decision(), observations, "根据 01-二次方程与判别式.md，判别式是 b^2-4ac。")
        self.assertTrue(attribution.used_knowledge)
        self.assertEqual(attribution.query, "判别式")
        self.assertEqual(len(attribution.retrieved), 1)
        source = attribution.retrieved[0]
        self.assertEqual(source.source, "01-二次方程与判别式.md")
        self.assertEqual(source.rank, 1)
        self.assertAlmostEqual(source.distance, 0.42)
        self.assertIn("b^2", source.snippet)
        self.assertEqual(attribution.cited_sources, ["01-二次方程与判别式.md"])

    def test_failed_search_is_not_counted(self) -> None:
        observations = [_search_observation(_documents(), success=False)]
        attribution = build_attribution(_decision(), observations, "答案")
        self.assertEqual(attribution.retrieved, [])
        # The route still says knowledge, so the run counts as a knowledge run.
        self.assertTrue(attribution.used_knowledge)

    def test_uncited_source(self) -> None:
        observations = [_search_observation(_documents())]
        attribution = build_attribution(_decision(), observations, "答案没有提到来源。")
        self.assertEqual(attribution.cited_sources, [])


class SummarizeTest(unittest.TestCase):
    def _trace(self, run_id: str, grounding: RagGrounding | None) -> RunTrace:
        attribution = build_attribution(_decision(), [_search_observation(_documents())], "答案")
        attribution.grounding = grounding
        return RunTrace(
            run_id=run_id,
            question="什么是判别式",
            started_at="2026-09-19T00:00:00+00:00",
            duration_ms=100,
            decision=_decision(),
            observations=[],
            rag=attribution,
            answer="答案",
        )

    def test_counts_grounding_status(self) -> None:
        traces = [
            self._trace("a", RagGrounding(grounded=True)),
            self._trace("b", RagGrounding(grounded=False, unsupported=["x"])),
            self._trace("c", None),
        ]
        summary = summarize(traces, {}, window_days=7)
        self.assertEqual(summary.runs, 3)
        self.assertEqual(summary.knowledge_runs, 3)
        self.assertEqual(summary.grounding["verified"], 1)
        self.assertEqual(summary.grounding["unknown"], 1)
        self.assertEqual(summary.grounding["none"], 1)
        self.assertEqual(summary.ungrounded_runs, 1)
        self.assertAlmostEqual(summary.grounded_rate, 0.5)

    def test_empty_window(self) -> None:
        summary = summarize([], {}, window_days=7)
        self.assertEqual(summary.runs, 0)
        self.assertIsNone(summary.knowledge_rate)
        self.assertIsNone(summary.grounded_rate)


class ChecksStoreTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = rag_attribution_service.checks_path(tmp)
            self.assertEqual(load_checks(path), {})
            save_check(path, "run1", RagGrounding(grounded=False, unsupported=["y"]))
            save_check(path, "run1", RagGrounding(grounded=True))
            checks = load_checks(path)
            self.assertTrue(checks["run1"].grounded)


class _ScriptedClient:
    def __init__(self, json_responses: list[str], chunks: list[str] | None = None) -> None:
        self._json = json_responses
        self.chunks = ["判别式", "是 b^2-4ac"] if chunks is None else chunks

    def reset_counters(self) -> None:
        pass

    def take_switches(self) -> list[dict]:
        return []

    def _next(self) -> str:
        index = min(getattr(self, "_calls", 0), len(self._json) - 1)
        self._calls = getattr(self, "_calls", 0) + 1
        return self._json[index]

    async def complete_json(self, messages, *, max_tokens=None, schema=None) -> dict:
        return _completion(self._next())

    async def complete_with_tools(
        self, messages, tools, *, tool_choice=None, max_tokens=None
    ) -> dict:
        return _as_tool_response(self._next())

    async def complete(self, messages, **kwargs) -> dict:
        return _completion("")

    async def stream_raw(self, messages):
        for chunk in self.chunks:
            yield {"choices": [{"delta": {"content": chunk}}]}


def _route_json(intent: str, tool: str, query: str = "判别式") -> str:
    return json.dumps(
        {"intent": intent, "tool": tool, "query": query, "answer_style": "step_by_step"},
        ensure_ascii=False,
    )


class LoopGroundingTest(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_run_records_grounding(self) -> None:
        client = _ScriptedClient(
            [
                _route_json("knowledge", "search_knowledge"),
                '{"action": "final"}',
                json.dumps({"grounded": True, "unsupported": [], "reason": "有依据"}),
            ]
        )
        ctx = ToolContext(
            client=client, settings=replace(settings, trace_enabled=False)
        )
        chunks = [RetrievedChunk(content="判别式是 b^2-4ac", source="01.md", distance=0.4)]
        with patch.object(rag_service, "search", return_value=chunks):
            events = [
                event
                async for event in iter_agent_events(
                    client, "什么是判别式", ctx, max_steps=2
                )
            ]
        types = [event["type"] for event in events]
        self.assertIn("rag_grounding", types)
        run = events[-1]["run"]
        self.assertTrue(run["rag"]["used_knowledge"])
        self.assertTrue(run["rag"]["grounding"]["grounded"])


class RagAttributionRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        attribution = build_attribution(
            _decision(), [_search_observation(_documents())], "答案"
        )
        trace = RunTrace(
            run_id="run-1",
            question="什么是判别式",
            started_at="2026-09-19T00:00:00+00:00",
            duration_ms=120,
            decision=_decision(),
            observations=[_search_observation(_documents())],
            rag=attribution,
            answer="判别式是 b^2-4ac",
        )
        trace_service.record_trace(trace, trace_service.trace_path(self._tmp.name))
        self.settings = replace(settings, trace_dir=self._tmp.name, trace_enabled=False)
        self.client = _ScriptedClient(
            [json.dumps({"grounded": False, "unsupported": ["缺少推导"], "reason": "无依据"})]
        )
        app.dependency_overrides[get_settings] = lambda: self.settings
        app.dependency_overrides[get_orchestrator] = lambda: self.client
        self.http = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self._tmp.cleanup()

    def test_list_returns_summary_and_runs(self) -> None:
        response = self.http.get("/api/rag/attribution")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["runs"], 1)
        self.assertEqual(payload["runs"][0]["run_id"], "run-1")
        self.assertTrue(payload["runs"][0]["used_knowledge"])
        self.assertEqual(payload["runs"][0]["retrieved_count"], 1)

    def test_detail_returns_attribution(self) -> None:
        response = self.http.get("/api/rag/attribution/run-1")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rag"]["retrieved"][0]["source"], "01-二次方程与判别式.md")

    def test_detail_missing_run_is_404(self) -> None:
        response = self.http.get("/api/rag/attribution/nope")
        self.assertEqual(response.status_code, 404)

    def test_check_runs_grounding_and_persists(self) -> None:
        response = self.http.post("/api/rag/attribution/run-1/check")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["grounded"])
        self.assertEqual(payload["unsupported"], ["缺少推导"])
        checks = load_checks(rag_attribution_service.checks_path(self._tmp.name))
        self.assertIn("run-1", checks)
        # The stored check overrides the trace's own verdict on the detail view.
        detail = self.http.get("/api/rag/attribution/run-1").json()
        self.assertFalse(detail["rag"]["grounding"]["grounded"])

    def test_clear_traces_removes_runs_and_checks(self) -> None:
        self.http.post("/api/rag/attribution/run-1/check")
        checks_path = rag_attribution_service.checks_path(self._tmp.name)
        self.assertIn("run-1", load_checks(checks_path))
        response = self.http.delete("/api/traces")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["cleared_runs"], 1)
        self.assertEqual(payload["cleared_checks"], 1)
        self.assertEqual(load_checks(checks_path), {})
        self.assertEqual(self.http.get("/api/rag/attribution").json()["summary"]["runs"], 0)


class RetrievalStatusTest(unittest.TestCase):
    def _trace(self, observations: list[Observation]) -> RunTrace:
        return RunTrace(
            run_id="r1",
            question="q",
            started_at="2026-09-19T00:00:00+00:00",
            duration_ms=10,
            decision=_decision(),
            observations=observations,
            answer="答案",
        )

    def test_ok(self) -> None:
        item = rag_attribution_service.to_item(self._trace([_search_observation(_documents())]))
        self.assertEqual(item.retrieval, "ok")

    def test_empty(self) -> None:
        item = rag_attribution_service.to_item(self._trace([_search_observation([])]))
        self.assertEqual(item.retrieval, "empty")

    def test_failed_carries_error(self) -> None:
        observation = _search_observation(_documents(), success=False).model_copy(
            update={"error": "工具 search_knowledge 超时（30.0s）"}
        )
        item = rag_attribution_service.to_item(self._trace([observation]))
        self.assertEqual(item.retrieval, "failed")
        self.assertIn("超时", item.retrieval_error or "")

    def test_summary_counts_failures(self) -> None:
        traces = [
            self._trace([_search_observation(_documents())]),
            self._trace([_search_observation([])]),
            self._trace(
                [
                    _search_observation(_documents(), success=False).model_copy(
                        update={"error": "超时"}
                    )
                ]
            ),
        ]
        summary = summarize(traces, {}, window_days=7)
        self.assertEqual(summary.retrieval.get("ok"), 1)
        self.assertEqual(summary.retrieval.get("empty"), 1)
        self.assertEqual(summary.retrieval.get("failed"), 1)
        self.assertEqual(summary.retrieval_failures, 1)


if __name__ == "__main__":
    unittest.main()
