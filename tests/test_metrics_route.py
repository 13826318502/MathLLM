from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.schema import (
    AgentRun,
    Observation,
    RouteDecision,
    TokenUsage,
    VerificationResult,
)
from app.api.dependencies import get_settings
from app.api.main import app
from app.core.config import settings
from app.services import trace_service
from app.services.trace_service import build_trace, record_trace


def _trace():
    run = AgentRun(
        question="解方程 x^2-5x+6=0",
        decision=RouteDecision(
            intent="math", tool="solve_math_problem", query="x^2-5x+6=0"
        ),
        observations=[
            Observation(
                step=0,
                tool="solve_math_problem",
                arguments={},
                success=True,
                summary="{}",
                duration_ms=900,
            )
        ],
        steps=1,
        stopped_reason="final",
        answer="x=2 或 x=3",
        verification=VerificationResult(
            status="verified", method="substitution", detail="ok"
        ),
        answer_model="mathllm-round7",
    )
    return build_trace(
        run,
        started_at=datetime.now(timezone.utc),
        duration_ms=1234,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=2),
    )


class MetricsRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        record_trace(_trace(), trace_service.trace_path(self._tmp.name))
        app.dependency_overrides[get_settings] = lambda: replace(
            settings, trace_dir=self._tmp.name
        )
        self.http = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self._tmp.cleanup()

    def test_metrics_aggregates_stored_traces(self) -> None:
        response = self.http.get("/api/metrics?days=0")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["runs"], 1)
        self.assertEqual(payload["routes"], {"solve_math_problem": 1})
        self.assertEqual(payload["verification"], {"verified": 1})
        self.assertEqual(payload["tokens"]["total"], 15)
        self.assertEqual(payload["failure_rate"], 0.0)
        self.assertEqual(payload["completion_rate"], 1.0)

    def test_metrics_window_excludes_old_runs(self) -> None:
        response = self.http.get("/api/metrics?days=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["runs"], 1)

    def test_traces_lists_recent_runs(self) -> None:
        response = self.http.get("/api/traces?limit=5")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["decision"]["tool"], "solve_math_problem")
        self.assertEqual(payload[0]["answer"], "x=2 或 x=3")

    def test_clear_traces_removes_history(self) -> None:
        response = self.http.delete("/api/traces")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cleared_runs"], 1)
        self.assertEqual(self.http.get("/api/traces?limit=5").json(), [])
        self.assertEqual(self.http.get("/api/metrics?days=0").json()["runs"], 0)

    def test_clear_traces_is_idempotent(self) -> None:
        self.http.delete("/api/traces")
        response = self.http.delete("/api/traces")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cleared_runs"], 0)

    def test_metrics_with_no_traces(self) -> None:
        empty = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        app.dependency_overrides[get_settings] = lambda: replace(
            settings, trace_dir=empty.name
        )
        response = TestClient(app).get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["runs"], 0)
        self.assertIsNone(response.json()["failure_rate"])
        empty.cleanup()


if __name__ == "__main__":
    unittest.main()
