from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from app.api.dependencies import get_vllm_client
from app.api.main import app


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class FakeClient:
    def __init__(self) -> None:
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
        ]
        self.json_calls = 0

    async def complete_json(self, messages, *, max_tokens=None, schema=None) -> dict:
        index = min(self.json_calls, len(self._json) - 1)
        self.json_calls += 1
        return _completion(self._json[index])

    async def complete(self, messages, **kwargs) -> dict:
        return _completion("方程的解为 x=2 或 x=3。")


class AgentRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        app.dependency_overrides[get_vllm_client] = lambda: self.client
        self.http = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_agent_run_returns_trace(self) -> None:
        response = self.http.post(
            "/api/agent/run",
            json={"question": "求解 x^2-5x+6=0", "max_steps": 2},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stopped_reason"], "final")
        self.assertEqual(payload["steps"], 1)
        self.assertEqual(payload["decision"]["intent"], "math")
        self.assertEqual(payload["observations"][0]["tool"], "solve_math_problem")
        self.assertIn("x=2", payload["answer"])

    def test_empty_question_rejected(self) -> None:
        response = self.http.post("/api/agent/run", json={"question": ""})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
