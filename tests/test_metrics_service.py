from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.agent.schema import (
    Observation,
    RouteDecision,
    RunTrace,
    TokenUsage,
    VerificationResult,
)
from app.services.metrics_service import filter_since, summarize

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _trace(
    *,
    intent: str = "math",
    tool: str = "solve_math_problem",
    stopped_reason: str = "final",
    answer: str = "x=2",
    error: str | None = None,
    tools: list[tuple[str, bool]] | None = None,
    verification: VerificationResult | None = None,
    duration_ms: int = 1000,
    tokens: tuple[int, int] = (10, 5),
    answer_model: str = "mathllm-round7",
    fallbacks: int = 0,
    started_at: datetime | None = None,
) -> RunTrace:
    return RunTrace(
        run_id="r",
        question="q",
        started_at=(started_at or NOW).isoformat(timespec="seconds"),
        duration_ms=duration_ms,
        decision=RouteDecision(intent=intent, tool=tool, query="q"),
        observations=[
            Observation(
                step=index,
                tool=name,
                arguments={},
                success=ok,
                summary="{}",
                duration_ms=1,
            )
            for index, (name, ok) in enumerate(tools or [])
        ],
        verification=verification,
        answer=answer,
        answer_model=answer_model,
        stopped_reason=stopped_reason,
        error=error,
        usage=TokenUsage(
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            total_tokens=sum(tokens),
            calls=2,
        ),
        fallbacks=fallbacks,
    )


class EmptyTest(unittest.TestCase):
    def test_empty_window_reports_no_rates(self) -> None:
        summary = summarize([])
        self.assertEqual(summary.runs, 0)
        self.assertIsNone(summary.failure_rate)
        self.assertIsNone(summary.hallucination_rate)
        self.assertEqual(summary.latency_ms, {})


class CountsTest(unittest.TestCase):
    def test_counts_routes_tools_and_verification(self) -> None:
        traces = [
            _trace(
                tools=[("solve_math_problem", True)],
                verification=VerificationResult(
                    status="verified", method="substitution", detail="ok"
                ),
            ),
            _trace(
                intent="knowledge",
                tool="search_knowledge",
                tools=[("search_knowledge", True), ("calculate_expression", False)],
                verification=VerificationResult(
                    status="unknown", method="grounding", detail="unsupported"
                ),
                answer_model="deepseek-flash",
            ),
            _trace(
                intent="general",
                tool="none",
                tools=[],
                answer="",
                stopped_reason="duplicate",
                answer_model="deepseek-flash",
            ),
        ]
        summary = summarize(traces)
        self.assertEqual(summary.runs, 3)
        self.assertEqual(summary.intents, {"math": 1, "knowledge": 1, "general": 1})
        self.assertEqual(
            summary.routes,
            {"solve_math_problem": 1, "search_knowledge": 1, "none": 1},
        )
        self.assertEqual(
            summary.tool_calls,
            {"search_knowledge": 1, "solve_math_problem": 1, "calculate_expression": 1},
        )
        self.assertEqual(summary.verification, {"verified": 1, "unknown": 1})
        self.assertEqual(summary.models, {"deepseek-flash": 2, "mathllm-round7": 1})
        self.assertEqual(summary.stopped_reasons, {"final": 2, "duplicate": 1})


class RatesTest(unittest.TestCase):
    def test_failure_completion_and_guard_stops(self) -> None:
        traces = [
            _trace(),
            _trace(answer=""),
            _trace(stopped_reason="max_steps"),
            _trace(error="VLLMServiceError: boom", stopped_reason="error", answer=""),
        ]
        summary = summarize(traces)
        self.assertEqual(summary.failure_rate, 0.25)
        self.assertEqual(summary.completion_rate, 0.5)
        self.assertEqual(summary.guard_stop_rate, 0.25)

    def test_tool_success_rate(self) -> None:
        traces = [
            _trace(tools=[("a", True), ("b", False)]),
            _trace(tools=[("c", True), ("d", True)]),
        ]
        self.assertEqual(summarize(traces).tool_success_rate, 0.75)
        self.assertEqual(summarize(traces).avg_tool_calls, 2.0)

    def test_hallucination_counts_refuted_and_ungrounded(self) -> None:
        traces = [
            _trace(
                verification=VerificationResult(
                    status="refuted", method="substitution", detail="bad"
                )
            ),
            _trace(
                verification=VerificationResult(
                    status="unknown", method="grounding", detail="unsupported"
                )
            ),
            _trace(
                verification=VerificationResult(
                    status="unknown", method="none", detail="cannot check"
                )
            ),
            _trace(verification=None),
        ]
        summary = summarize(traces)
        self.assertEqual(summary.refuted, 1)
        self.assertEqual(summary.ungrounded, 1)
        # 2 signals over 3 verified runs; an unverified run is not a signal.
        self.assertEqual(summary.hallucination_rate, 0.6667)

    def test_fallback_rate(self) -> None:
        traces = [_trace(), _trace(fallbacks=2), _trace(fallbacks=0), _trace()]
        summary = summarize(traces)
        self.assertEqual(summary.runs_with_fallback, 1)
        self.assertEqual(summary.fallback_rate, 0.25)


class LatencyAndTokensTest(unittest.TestCase):
    def test_percentiles_and_sums(self) -> None:
        traces = [
            _trace(duration_ms=value, tokens=(value, value)) for value in (10, 20, 30, 40, 50)
        ]
        summary = summarize(traces)
        self.assertEqual(summary.latency_ms["p50"], 30)
        self.assertEqual(summary.latency_ms["p95"], 50)
        self.assertEqual(summary.latency_ms["max"], 50)
        self.assertEqual(summary.latency_ms["avg"], 30)
        self.assertEqual(summary.tokens["prompt"], 150)
        self.assertEqual(summary.tokens["total"], 300)
        self.assertEqual(summary.tokens["avg_per_run"], 60)


class FilterSinceTest(unittest.TestCase):
    def test_keeps_only_recent_traces(self) -> None:
        traces = [
            _trace(started_at=NOW - timedelta(days=10)),
            _trace(started_at=NOW - timedelta(days=3)),
            _trace(started_at=NOW),
        ]
        kept = filter_since(traces, 7, now=NOW)
        self.assertEqual(len(kept), 2)

    def test_zero_days_keeps_everything(self) -> None:
        traces = [_trace(started_at=NOW - timedelta(days=100))]
        self.assertEqual(len(filter_since(traces, 0, now=NOW)), 1)

    def test_ignores_broken_timestamps(self) -> None:
        broken = _trace()
        broken.started_at = "not-a-date"
        self.assertEqual(filter_since([broken], 7, now=NOW), [])


if __name__ == "__main__":
    unittest.main()
