"""Evaluate a model exposed through an OpenAI-compatible chat endpoint.

The final-answer metric is deterministic: extract the final-answer section,
normalize it, then compare text or numeric values with a tolerance.  Every
question, raw answer, extracted answer, timing, and error is retained.

The process score is a reproducible heuristic for smoke tests and regression
comparisons.  It is not a mathematical proof checker and does not replace
manual review of incorrect or suspicious answers.

Example:
    python eval/evaluate.py --model_endpoint http://localhost:8000/v1 \
        --eval_data data/processed/eval.json \
        --output eval/results/finetuned_model --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import httpx


DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的数学解题助手。请按照以下要求回答数学问题：\n"
    "1. 先分析题目要求\n"
    "2. 给出详细的解题步骤\n"
    "3. 用 LaTeX 格式书写数学公式\n"
    "4. 最后给出明确的答案"
)
FINAL_MARKERS = (
    "**最终答案**", "最终答案：", "最终答案:", "Final Answer:",
    "Final answer:", "####",
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
LATEX_FRACTION_PATTERN = re.compile(
    r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    process_score: float = 0.0
    latency_ms: float = 0.0
    first_token_ms: float = 0.0
    model_final_answer: str = ""
    judge_reason: str = ""
    error: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        # Failed requests count as incorrect, while the failure count is also
        # reported separately so an availability problem is not hidden.
        return sum(1 for result in self.results if result.is_correct) / len(self.results)

    @property
    def successful_results(self) -> list[EvalResult]:
        return [result for result in self.results if not result.error]

    @property
    def avg_process_score(self) -> float:
        results = self.successful_results
        return sum(result.process_score for result in results) / len(results) if results else 0.0

    @property
    def avg_latency_ms(self) -> float:
        results = self.successful_results
        return sum(result.latency_ms for result in results) / len(results) if results else 0.0

    @property
    def avg_first_token_ms(self) -> float:
        results = self.successful_results
        return sum(result.first_token_ms for result in results) / len(results) if results else 0.0

    def summary(self) -> dict[str, Any]:
        failed = sum(1 for result in self.results if result.error)
        correct = sum(1 for result in self.results if result.is_correct)
        return {
            "total": len(self.results),
            "successful": len(self.successful_results),
            "failed": failed,
            "correct": correct,
            "incorrect": len(self.results) - correct,
            "bad_cases": len(self.results) - correct,
            "accuracy": round(self.accuracy, 4),
            "avg_process_score": round(self.avg_process_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_first_token_ms": round(self.avg_first_token_ms, 2),
        }


def _endpoint_base(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    return value[: -len("/chat/completions")] if value.endswith("/chat/completions") else value


def _chat_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions" if value.endswith("/v1") else f"{value}/v1/chat/completions"


def _models_url(endpoint: str) -> str:
    base = _endpoint_base(endpoint)
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/models"


def discover_model_name(endpoint: str, timeout: float = 20.0) -> str | None:
    """Get the first model id advertised by an OpenAI-compatible server."""
    try:
        response = httpx.get(_models_url(endpoint), timeout=timeout)
        response.raise_for_status()
        models = response.json().get("data", [])
        if models and isinstance(models[0], dict) and models[0].get("id"):
            return str(models[0]["id"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
    return None


def _messages_for_item(item: dict[str, Any]) -> tuple[str, list[dict[str, str]], str]:
    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("评测样本缺少合法的 messages 字段")
    user = next((message for message in messages if message.get("role") == "user"), None)
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if not user or not user.get("content"):
        raise ValueError("评测样本缺少 user 题目")
    if not assistants or not assistants[-1].get("content"):
        raise ValueError("评测样本缺少 assistant 标准答案")
    prompt_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
        if message.get("role") in {"system", "user"}
    ]
    return str(user["content"]), prompt_messages, str(assistants[-1]["content"])


def call_model(
    endpoint: str,
    question: str | list[dict[str, str]],
    model_name: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> tuple[str, float, float]:
    """Call the endpoint and return ``(answer, total_ms, first_token_ms)``."""
    messages = (
        [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": question}]
        if isinstance(question, str)
        else question
    )
    payload: dict[str, Any] = {
        "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": True,
    }
    if model_name:
        payload["model"] = model_name

    started = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    with httpx.Client(timeout=timeout) as client:
        try:
            with client.stream("POST", _chat_url(endpoint), json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        continue
                    event = json.loads(data)
                    choices = event.get("choices", [])
                    delta = choices[0].get("delta", {}).get("content", "") if choices else ""
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(str(delta))
        except httpx.HTTPStatusError as error:
            # A few compatible servers support chat completions but reject SSE.
            if error.response.status_code < 400 or error.response.status_code >= 500:
                raise
            payload["stream"] = False
            response = client.post(_chat_url(endpoint), json=payload)
            response.raise_for_status()
            body = response.json()
            try:
                answer = str(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as parse_error:
                raise ValueError("模型非流式响应缺少 choices[0].message.content") from parse_error
            total_ms = (time.perf_counter() - started) * 1000
            return answer, total_ms, total_ms

    total_ms = (time.perf_counter() - started) * 1000
    first_token_ms = (first_token_at - started) * 1000 if first_token_at is not None else total_ms
    answer = "".join(chunks).strip()
    if not answer:
        raise ValueError("模型返回为空或流式响应格式不兼容")
    return answer, total_ms, first_token_ms


def _clean_answer(text: str) -> str:
    value = str(text or "").replace("\\boxed", "")
    value = value.replace("（", "(").replace("）", ")").replace("，", ",").replace("：", ":")
    value = WHITESPACE_PATTERN.sub("", value)
    return value.strip("$`。.!！;,；:")


def extract_final_answer(text: str) -> str:
    """Extract the final-answer section from a solution or model response."""
    value = str(text or "").strip()
    positions = [(value.rfind(marker), marker) for marker in FINAL_MARKERS]
    positions = [(position, marker) for position, marker in positions if position >= 0]
    if positions:
        position, marker = max(positions, key=lambda pair: pair[0])
        candidate = value[position + len(marker) :].strip(" \t\r\n:：")
        if candidate:
            return candidate
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    return paragraphs[-1] if paragraphs else value


def _numeric_values(text: str) -> list[Fraction | float]:
    """Extract integers, decimals, scientific notation, and simple LaTeX fractions."""
    value = str(text or "")
    values: list[Fraction | float] = []
    fraction_spans: list[tuple[int, int]] = []
    for match in LATEX_FRACTION_PATTERN.finditer(value):
        try:
            numerator = Fraction(match.group(1).strip())
            denominator = Fraction(match.group(2).strip())
            if denominator == 0:
                continue
            values.append(numerator / denominator)
            fraction_spans.append(match.span())
        except (ValueError, ZeroDivisionError):
            continue
    remaining = list(value)
    for start, end in fraction_spans:
        for index in range(start, end):
            remaining[index] = " "
    for match in NUMBER_PATTERN.finditer("".join(remaining)):
        token = match.group(0)
        try:
            values.append(Fraction(token))
        except (ValueError, ZeroDivisionError):
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def _numbers_equal(expected: Iterable[Fraction | float], actual: Iterable[Fraction | float]) -> bool:
    expected_values, actual_values = list(expected), list(actual)
    if len(expected_values) != len(actual_values) or not expected_values:
        return False
    remaining = list(actual_values)
    for expected_value in expected_values:
        match_index = None
        for index, actual_value in enumerate(remaining):
            difference = abs(float(expected_value) - float(actual_value))
            scale = max(1.0, abs(float(expected_value)), abs(float(actual_value)))
            if difference <= 1e-6 or difference / scale <= 1e-5:
                match_index = index
                break
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def judge_answer(question: str, expected: str, actual: str) -> tuple[bool, str]:
    """Judge final answers with normalized text or numeric tolerance."""
    del question  # Reserved for a future symbolic judge.
    expected_clean = _clean_answer(extract_final_answer(expected))
    actual_clean = _clean_answer(extract_final_answer(actual))
    if expected_clean and expected_clean == actual_clean:
        return True, "normalized exact match"
    if _numbers_equal(_numeric_values(expected_clean), _numeric_values(actual_clean)):
        return True, "numeric match within abs_tol=1e-6 or rel_tol=1e-5"
    return False, "normalized text and numeric comparison did not match"


def judge_process(question: str, response: str) -> float:
    """Return a reproducible completeness heuristic in ``[0, 1]``."""
    del question
    text = str(response or "").strip()
    if not text:
        return 0.0
    score = 0.0
    if len(text) >= 20:
        score += 0.2
    if len(text) >= 80:
        score += 0.1
    if re.search(r"\*\*最终答案\*\*|最终答案|Final\s+Answer|####", text, re.I):
        score += 0.25
    if re.search(r"分析|步骤|因为|因此|所以|令|设|由|得到|先|then|therefore|because|let|thus", text, re.I):
        score += 0.25
    if "$" in text or "\\(" in text or "\\[" in text or re.search(r"\d", text):
        score += 0.2
    return round(min(score, 1.0), 4)


def run_evaluation(
    endpoint: str,
    eval_data_path: str,
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 120.0,
    limit: int | None = None,
) -> EvalReport:
    """Run evaluation and retain a result for every input sample."""
    with open(eval_data_path, "r", encoding="utf-8") as handle:
        eval_data = json.load(handle)
    if not isinstance(eval_data, list):
        raise ValueError("评测文件必须是 JSON 数组")
    if limit is not None:
        eval_data = eval_data[:limit]
    resolved_model = model_name or discover_model_name(endpoint, timeout=min(timeout, 20.0))
    report = EvalReport()
    for index, item in enumerate(eval_data, 1):
        print(f"Evaluating {index}/{len(eval_data)}...")
        try:
            question, prompt_messages, expected = _messages_for_item(item)
            model_answer, latency_ms, first_token_ms = call_model(
                endpoint, prompt_messages, resolved_model,
                temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            )
            is_correct, reason = judge_answer(question, expected, model_answer)
            report.results.append(EvalResult(
                question=question,
                expected_answer=extract_final_answer(expected),
                model_answer=model_answer,
                model_final_answer=extract_final_answer(model_answer),
                is_correct=is_correct,
                judge_reason=reason,
                process_score=judge_process(question, model_answer),
                latency_ms=latency_ms,
                first_token_ms=first_token_ms,
            ))
        except Exception as error:
            question, expected_answer = "", ""
            if isinstance(item, dict):
                try:
                    question, _, expected = _messages_for_item(item)
                    expected_answer = extract_final_answer(expected)
                except Exception:
                    pass
            report.results.append(EvalResult(
                question=question, expected_answer=expected_answer, model_answer="",
                is_correct=False, judge_reason="request or sample parsing failed",
                error=f"{type(error).__name__}: {error}",
            ))
            print(f"  failed: {type(error).__name__}: {error}")
    return report


def generate_comparison_chart(reports: dict[str, EvalReport], output_dir: str) -> None:
    """Generate accuracy and latency comparison charts."""
    if not reports:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("生成图表需要 matplotlib") from error
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    names = list(reports)
    plt.figure(figsize=(8, 5))
    plt.bar(names, [reports[name].accuracy * 100 for name in names])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_comparison.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.bar(names, [reports[name].avg_latency_ms for name in names])
    plt.ylabel("Average latency (ms)")
    plt.title("Model Latency Comparison")
    plt.tight_layout()
    plt.savefig(output_path / "latency_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 OpenAI 兼容数学模型接口")
    parser.add_argument("--model_endpoint", required=True, help="例如 http://localhost:8000/v1")
    parser.add_argument("--model_name", default=None, help="不填则尝试从 /v1/models 自动获取")
    parser.add_argument("--eval_data", default="./data/processed/eval.json")
    parser.add_argument("--output", default="./eval/results/finetuned_model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条，用于接口冒烟测试")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_evaluation(
        args.model_endpoint, args.eval_data, model_name=args.model_name,
        temperature=args.temperature, max_tokens=args.max_tokens,
        timeout=args.timeout, limit=args.limit,
    )
    summary = report.summary()
    print("\nEvaluation Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "details.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in report.results], handle, ensure_ascii=False, indent=2)
    with (output_dir / "bad_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [asdict(result) for result in report.results if not result.is_correct],
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_endpoint": args.model_endpoint,
                "model_name": args.model_name or discover_model_name(args.model_endpoint),
                "eval_data": str(Path(args.eval_data)),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
                "limit": args.limit,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    try:
        chart_label = output_dir.name or "model"
        generate_comparison_chart({chart_label: report}, str(output_dir))
    except RuntimeError as error:
        print(f"Warning: charts were not generated: {error}")
    print(
        f"\nResults saved to {output_dir}/report.json, "
        f"{output_dir}/details.json and {output_dir}/bad_cases.json"
    )


if __name__ == "__main__":
    main()
