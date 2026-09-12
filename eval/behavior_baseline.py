"""Run a base-model baseline for context-aware clarification behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx


CLARIFICATION_RE = re.compile(
    r"请(?:提供|告诉|补充|说明|告知)|请把.+发|还需要(?:提供|补充|说明|知道|确认)|"
    r"缺少|无法确定|哪一个|哪个|能否.*说明|具体.*呢|请问您?(?:指的是|想要|需要)"
)


def has_clarification_signal(text: str) -> bool:
    return bool(CLARIFICATION_RE.search(text)) or "？" in text or "?" in text


def expected_action(behavior: str) -> str:
    if behavior in {"need_clarification", "ambiguous"}:
        return "clarify"
    return "answer"


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.input)
    if args.limit > 0:
        cases = cases[: args.limit]
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, case in enumerate(cases, 1):
            content = ""
            error = None
            try:
                response = await client.post(
                    args.endpoint,
                    json={
                        "model": args.model,
                        "messages": case["messages"],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                content = str(payload["choices"][0]["message"]["content"])
            except Exception as exc:  # keep one failed request from stopping the run
                error = f"{type(exc).__name__}: {exc}"

            observed_action = "clarify" if has_clarification_signal(content) else "answer"
            action_ok = error is None and observed_action == expected_action(case["behavior"])
            result = {
                "index": index,
                "id": case.get("id", f"case-{index}"),
                "expected_behavior": case["behavior"],
                "expected_action": expected_action(case["behavior"]),
                "observed_action": observed_action,
                "action_ok": action_ok,
                "content": content,
            }
            if error:
                result["error"] = error
            results.append(result)
            print(f"[{index}/{len(cases)}] {result['id']} action_ok={action_ok}", flush=True)

    passed = sum(bool(item["action_ok"]) for item in results)
    failed_requests = sum("error" in item for item in results)
    by_behavior: dict[str, dict[str, int]] = {}
    for item in results:
        stats = by_behavior.setdefault(item["expected_behavior"], {"total": 0, "passed": 0})
        stats["total"] += 1
        stats["passed"] += int(item["action_ok"])
    summary = {
        "model": args.model,
        "input": str(args.input),
        "total": len(results),
        "action_passed": passed,
        "action_accuracy": round(passed / len(results), 4) if results else 0.0,
        "failed_requests": failed_requests,
        "by_behavior": by_behavior,
        "note": "action_accuracy is a heuristic for answer-vs-clarify behavior; inspect details for semantic correctness.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "details.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if failed_requests == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1/chat/completions")
    parser.add_argument("--model", default="mathllm-base-cpu:latest")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
