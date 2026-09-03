"""Prepare raw mathematics data for LoRA/QLoRA training.

The script keeps source files under data/raw unchanged. It reads each source,
normalizes it to question/solution/answer metadata, converts it to messages,
validates and de-duplicates the examples, and writes train/eval JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


RAW_DIR = Path("./data/raw")
OUTPUT_DIR = Path("./data/processed")
SYSTEM_PROMPT = """你是一个专业的数学解题助手。请按照以下要求回答数学问题：
1. 先分析题目要求
2. 给出详细的解题步骤
3. 用 LaTeX 格式书写数学公式
4. 最后给出明确的答案
5. 写出最终答案前，重新检查关键计算，并确认最终答案与解题过程一致
6. 遇到进制问题，先确认数字的位数，再按位权展开；n 位 b 进制数按从高位到低位使用 b^(n-1), ..., b^1, b^0，不能混淆位数"""

VALID_ROLES = {"system", "user", "assistant"}
SIMILARITY_THRESHOLD = 0.95
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv"}
LOGGER = logging.getLogger(__name__)
GSM8K_CALCULATION_RE = re.compile(
    r"(?:(?P<prefix>(?<![A-Za-z0-9$\\])\d[\d., \t*+/()\-]*=)[ \t]*)?"
    r"(?:\\\$)?<<(?P<expression>[^<>]+)>>\s*"
    r"(?:\$?[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))?"
)
GSM8K_BARE_ARITHMETIC_RE = re.compile(
    r"(?<![A-Za-z0-9$])"
    r"(?P<expression>[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))?"
    r"(?:\s*(?:\*|x|×|/|÷|\+|-)\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))+"
    r"\s*=\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"(?![A-Za-z0-9$])",
    re.IGNORECASE,
)
GSM8K_BARE_FRACTION_RE = re.compile(
    r"(?<![A-Za-z0-9$])(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)(?![A-Za-z0-9$])"
)
UNSUPPORTED_ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", re.IGNORECASE | re.DOTALL)


def _as_text(value: Any) -> str:
    """Return a trimmed text value without destroying internal newlines."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    """Normalize human text while preserving useful solution line breaks."""
    normalized = unicodedata.normalize("NFKC", _as_text(value))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    # A single backslash before a row's first cell (for example ``\0``) is
    # malformed LaTeX; a matrix row separator must contain two backslashes.
    normalized = re.sub(
        r"(?<!\\)\\[ \t]*(?=[-+]?\d|&)", lambda _: r"\\", normalized
    )
    # A space after a LaTeX escape is formatting noise (``\\ frac`` -> ``\\frac``).
    normalized = re.sub(r"\\[ \t]+(?=[A-Za-z])", r"\\", normalized)
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_gsm8k_expression(expression: str) -> str:
    """Convert GSM8K's calculator annotations into simple LaTeX math."""
    expression = expression.strip()
    expression = re.sub(
        r"(?<![\w}])([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*/\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))",
        lambda match: f"\\frac{{{match.group(1)}}}{{{match.group(2)}}}",
        expression,
    )
    expression = expression.replace("*", r"\times ").replace("x", r"\times ")
    expression = expression.replace("×", r"\times ").replace("÷", r"\div ")
    expression = expression.replace("/", r"\div ")
    expression = re.sub(r"(?<![\w.])\.(?=\d)", "0.", expression)
    return expression


def _convert_gsm8k_annotations(text: str) -> str:
    """Remove ``<<...>>`` annotations and retain each calculation as LaTeX."""
    # GSM8K uses dollar signs for currency, not math delimiters.
    text = re.sub(r"(?<!\\)\$", lambda _: r"\$", text)
    text = GSM8K_CALCULATION_RE.sub(
        lambda match: f"${_normalize_gsm8k_expression(match.group('expression'))}$", text
    )
    text = GSM8K_BARE_ARITHMETIC_RE.sub(
        lambda match: f"${_normalize_gsm8k_expression(match.group('expression'))}$", text
    )
    return GSM8K_BARE_FRACTION_RE.sub(
        lambda match: f"$\\frac{{{match.group('numerator')}}}{{{match.group('denominator')}}}$", text
    )


def _canonicalize_question(text: str) -> str:
    """Normalize a question for whitespace- and punctuation-insensitive matching."""
    normalized = _normalize_text(text).lower()
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    # These commands only add visual LaTeX spacing and should not affect identity.
    normalized = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|enskip|thinspace)", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _question_signatures(text: str) -> set[str]:
    """Return exact and template signatures used to reject near-duplicate questions."""
    canonical = _canonicalize_question(text)
    signatures = {canonical}
    # Catch the common case where only numerical values changed.
    signatures.add(re.sub(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?", "<num>", canonical))
    # Catch single-letter variable renaming without changing LaTeX command names.
    signatures.add(re.sub(r"(?<![\\a-z])([a-z])(?=[^a-z]|$)", "<var>", canonical))
    return signatures


def _records_from_json(value: Any) -> list[dict[str, Any]]:
    """Accept a JSON list, one JSON object, or a common records wrapper."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
        return [value]
    return []


def _read_raw_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _records_from_json(json.loads(path.read_text(encoding="utf-8")))

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSONL line %s:%s: %s", path, line_number, exc)
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                LOGGER.warning("Skipping non-object JSONL line %s:%s", path, line_number)
        return records

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _source_from_filename(path: Path) -> str:
    name = path.stem.lower()
    if "gsm8k" in name or name.startswith("gsm"):
        return "gsm8k"
    if any(token in name for token in ("manual", "test", "chinese", "college")):
        return "manual"
    if name.startswith("math") or "competition" in name or "hendrycks" in name:
        return "math"
    return "generic"


def _infer_source(item: dict[str, Any], path: Path) -> str:
    declared = _as_text(item.get("source")).lower()
    if "gsm8k" in declared or declared == "gsm":
        return "gsm8k"
    if declared in {"math", "math_dataset", "hendrycks_math", "competition_math"}:
        return "math"
    if declared:
        return "manual"
    if "problem" in item and "solution" in item and ("level" in item or "type" in item):
        return "math"
    if "question" in item and "answer" in item and "####" in _as_text(item.get("answer")):
        return "gsm8k"
    return _source_from_filename(path)


def _extract_boxed_answer(solution: str) -> str:
    """Extract the last balanced \\boxed{...} or \\fbox{...} expression."""
    candidates: list[str] = []
    for marker in (r"\boxed", r"\fbox"):
        search_from = 0
        while True:
            marker_start = solution.find(marker, search_from)
            if marker_start < 0:
                break
            open_brace = solution.find("{", marker_start + len(marker))
            if open_brace < 0:
                break

            depth = 0
            close_brace = -1
            for index in range(open_brace, len(solution)):
                if solution[index] == "{":
                    depth += 1
                elif solution[index] == "}":
                    depth -= 1
                    if depth == 0:
                        close_brace = index
                        break
            if close_brace >= 0:
                candidates.append(solution[open_brace + 1 : close_brace].strip())
                search_from = close_brace + 1
            else:
                break

    if not candidates:
        return ""
    answer = candidates[-1]
    if answer and "$" not in answer and r"\(" not in answer:
        answer = f"${answer}$"
    return answer


def _normalize_item(item: dict[str, Any], path: Path) -> dict[str, str] | None:
    source_kind = _infer_source(item, path)

    if any(UNSUPPORTED_ASY_RE.search(_as_text(item.get(field))) for field in ("question", "problem", "solution", "answer")):
        LOGGER.warning("Skipping record with unsupported Asymptote diagram markup: %s", path.name)
        return None

    if source_kind == "gsm8k":
        question = _as_text(item.get("question"))
        raw_answer = _as_text(item.get("answer"))
        if "####" not in raw_answer:
            LOGGER.warning("Skipping GSM8K record without ####: %s", path.name)
            return None
        solution, answer = raw_answer.rsplit("####", 1)
        solution = _convert_gsm8k_annotations(solution)
        answer = answer.strip()
        if answer and "$" not in answer and r"\(" not in answer:
            answer = f"${answer}$"
        source = "gsm8k"
        subject = _as_text(item.get("subject")) or "基础应用题"
        difficulty = _as_text(item.get("difficulty")) or "基础"
    elif source_kind == "math":
        question = _as_text(item.get("problem") or item.get("question"))
        solution = _as_text(item.get("solution") or item.get("reasoning"))
        answer = _as_text(item.get("answer") or item.get("final_answer"))
        if not answer:
            answer = _extract_boxed_answer(solution)
        if not answer:
            LOGGER.warning("Skipping MATH record without extractable answer: %s", path.name)
            return None
        source = "math"
        subject = _as_text(item.get("subject") or item.get("type")) or "MATH"
        difficulty = _as_text(item.get("difficulty") or item.get("level")) or "未标注"
    else:
        question = _as_text(item.get("question") or item.get("problem"))
        solution = _as_text(item.get("solution") or item.get("reasoning") or item.get("explanation"))
        answer = _as_text(item.get("answer") or item.get("final_answer") or item.get("target"))
        source = _as_text(item.get("source")) or source_kind
        subject = _as_text(item.get("subject")) or "未标注"
        difficulty = _as_text(item.get("difficulty")) or "未标注"

    if not question or not solution or not answer:
        LOGGER.warning("Skipping record with missing question/solution/answer: %s", path.name)
        return None

    return {
        "question": _normalize_text(question),
        "solution": _normalize_text(solution),
        "answer": _normalize_text(answer),
        "source": _normalize_text(source),
        "subject": _normalize_text(subject),
        "difficulty": _normalize_text(difficulty),
    }


def load_raw_data(raw_dir: Path = RAW_DIR) -> list[dict[str, str]]:
    """Read JSON, JSONL and CSV files and normalize source-specific schemas."""
    if not raw_dir.exists():
        LOGGER.warning("Raw data directory does not exist: %s", raw_dir)
        return []

    normalized: list[dict[str, str]] = []
    for path in sorted(raw_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            records = _read_raw_file(path)
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
            LOGGER.warning("Skipping unreadable file %s: %s", path, exc)
            continue

        before = len(normalized)
        for record in records:
            converted = _normalize_item(record, path)
            if converted is not None:
                normalized.append(converted)
        LOGGER.info("Loaded %s: %s records, %s normalized", path.name, len(records), len(normalized) - before)

    return normalized


def format_example(item: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    """Convert one normalized example into the single-turn messages format."""
    question = _as_text(item.get("question"))
    solution = _as_text(item.get("solution"))
    answer = _as_text(item.get("answer"))
    assistant = solution
    if answer:
        assistant = f"{solution}\n\n**最终答案**\n\n{answer}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
    }


def _token_count(example: dict[str, Any], tokenizer: Any | None) -> int | None:
    if tokenizer is None:
        return None
    try:
        token_ids = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=False
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(example["messages"], tokenize=True)
    # Recent transformers versions return BatchEncoding, whose ``len`` is the
    # number of fields rather than the number of tokens.
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict) and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def validate_data(
    data: list[dict[str, Any]],
    tokenizer: Any | None = None,
    max_tokens: int = 2048,
) -> list[dict[str, Any]]:
    """Validate message structure, length rules, exact duplicates and near duplicates."""
    cleaned: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_signatures: set[str] = set()
    accepted_questions: list[str] = []

    for index, item in enumerate(data, 1):
        messages = item.get("messages") if isinstance(item, dict) else None
        if not isinstance(messages, list) or len(messages) < 3:
            LOGGER.warning("Dropping item %s: messages is missing or too short", index)
            continue

        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if len(roles) != len(messages) or any(role not in VALID_ROLES for role in roles):
            LOGGER.warning("Dropping item %s: invalid role", index)
            continue
        if roles[0] != "system" or roles[1] != "user" or roles[-1] != "assistant":
            LOGGER.warning("Dropping item %s: invalid message order", index)
            continue
        if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
            LOGGER.warning("Dropping item %s: empty message content", index)
            continue

        question = messages[1]["content"].strip()
        assistant = messages[-1]["content"].strip()
        if len(question) < 5:
            LOGGER.warning("Dropping item %s: question shorter than 5 characters", index)
            continue
        if len(assistant) < 20:
            LOGGER.warning("Dropping item %s: assistant response shorter than 20 characters", index)
            continue

        canonical_question = _canonicalize_question(question)
        signatures = _question_signatures(question)
        if canonical_question in seen_questions:
            LOGGER.warning("Dropping item %s: exact duplicate question", index)
            continue
        if seen_signatures.intersection(signatures):
            LOGGER.warning("Dropping item %s: template/number/variable near-duplicate question", index)
            continue
        if any(
            SequenceMatcher(None, canonical_question, existing).ratio() >= SIMILARITY_THRESHOLD
            for existing in accepted_questions
        ):
            LOGGER.warning("Dropping item %s: near-duplicate question", index)
            continue

        token_count = _token_count(item, tokenizer)
        if token_count is not None and token_count > max_tokens:
            LOGGER.warning("Dropping item %s: %s tokens exceeds %s", index, token_count, max_tokens)
            continue

        seen_questions.add(canonical_question)
        seen_signatures.update(signatures)
        accepted_questions.append(canonical_question)
        cleaned.append(item)

    return cleaned


def split_dataset(
    data: list[dict[str, Any]], eval_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split deterministically, stratifying by source when metadata is available."""
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")

    if len(data) <= 1:
        return list(data), []

    rng = random.Random(seed)
    eval_count = max(1, round(len(data) * eval_ratio))
    eval_count = min(eval_count, len(data) - 1)

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in data:
        source = item.get("_metadata", {}).get("source", "unknown")
        groups.setdefault(source, []).append(item)

    for group in groups.values():
        rng.shuffle(group)

    # Put one example from each source into eval when the target is large enough.
    eval_data: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    group_items = list(groups.items())
    rng.shuffle(group_items)
    for _, group in group_items:
        if len(eval_data) < eval_count:
            eval_data.append(group.pop())
        remaining.extend(group)

    rng.shuffle(remaining)
    eval_data.extend(remaining[: eval_count - len(eval_data)])
    selected_ids = {id(item) for item in eval_data}
    train_data = [item for item in data if id(item) not in selected_ids]
    rng.shuffle(train_data)
    rng.shuffle(eval_data)
    return train_data, eval_data


def _load_tokenizer(model_name: str | None) -> Any | None:
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--tokenizer requires transformers to be installed") from exc
    LOGGER.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


def _print_split_stats(name: str, data: list[dict[str, Any]]) -> None:
    source_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    for item in data:
        metadata = item.get("_metadata", {})
        source_counts[metadata.get("source", "unknown")] += 1
        subject_counts[metadata.get("subject", "unknown")] += 1
        difficulty_counts[metadata.get("difficulty", "unknown")] += 1
    print(f"{name} sources: {dict(source_counts)}")
    print(f"{name} subjects: {dict(subject_counts)}")
    print(f"{name} difficulties: {dict(difficulty_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MathLLM training data")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer for exact token length checks")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    print("Loading raw data...")
    raw_data = load_raw_data(args.raw_dir)
    print(f"Loaded {len(raw_data)} normalized examples")

    print("Formatting examples...")
    formatted: list[dict[str, Any]] = []
    for item in raw_data:
        example = format_example(item)
        example["_metadata"] = {
            "source": item.get("source", "unknown"),
            "subject": item.get("subject", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
        }
        formatted.append(example)

    tokenizer = _load_tokenizer(args.tokenizer)
    if tokenizer is None:
        print("Token length check: skipped (no tokenizer supplied)")

    print("Validating data...")
    cleaned = validate_data(formatted, tokenizer=tokenizer, max_tokens=args.max_tokens)
    print(f"After cleaning: {len(cleaned)} examples")

    print("Splitting dataset...")
    train_data, eval_data = split_dataset(cleaned, eval_ratio=args.eval_ratio, seed=args.seed)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    _print_split_stats("Train", train_data)
    _print_split_stats("Eval", eval_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("train.json", train_data), ("eval.json", eval_data)):
        output_records = [{"messages": record["messages"]} for record in records]
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(output_records, handle, ensure_ascii=False, indent=2)

    print(f"Done! Data saved to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
