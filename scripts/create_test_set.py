"""Create a fixed, independent test set for MathLLM.

The test set is selected from the reviewed raw data with a fixed seed. It is
written in the same ChatML ``messages`` format used by evaluation, but it is
kept under ``data/eval`` so it is not used for training. Correction samples
are deliberately excluded: those belong to the training correction set.

Run from the project root:

    python scripts/create_test_set.py --test-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# When executed as ``python scripts/create_test_set.py``, Python puts the
# scripts directory (not the project root) on sys.path. Add the root so the
# sibling prepare_data module can be imported reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import (  # noqa: E402
    RAW_DIR,
    REGRESSION_DIR,
    TEST_FILE,
    _print_split_stats,
    format_example,
    load_raw_data,
    validate_data,
    _question_signatures,
)


def _with_metadata(raw_data: list[dict[str, str]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for item in raw_data:
        example = format_example(item)
        example["_metadata"] = {
            "source": item.get("source", "unknown"),
            "subject": item.get("subject", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
        }
        formatted.append(example)
    return formatted


def _select_stratified(
    data: list[dict[str, Any]], test_ratio: float, seed: int
) -> list[dict[str, Any]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    eligible = [
        item
        for item in data
        if not str(item.get("_metadata", {}).get("source", "")).startswith("correction-")
    ]
    if not eligible:
        return []

    target = max(1, round(len(eligible) * test_ratio))
    target = min(target, max(1, len(eligible) - 1))
    rng = random.Random(seed)

    # First reserve one item for each source/subject stratum. This keeps the
    # final test set representative without making the split dependent on
    # source file order.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in eligible:
        metadata = item.get("_metadata", {})
        key = (str(metadata.get("source", "unknown")), str(metadata.get("subject", "unknown")))
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        rng.shuffle(group)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for _, group in group_items:
        if len(selected) < target:
            selected.append(group.pop())
        remaining.extend(group)

    rng.shuffle(remaining)
    selected.extend(remaining[: target - len(selected)])
    rng.shuffle(selected)
    return selected


def _fast_candidate_clean(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove obvious duplicates before selection without an O(n²) scan.

    Full near-duplicate validation is still applied to the selected test
    records. The training pipeline performs its complete validation after the
    test questions have been protected.
    """
    accepted: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for item in data:
        messages = item.get("messages", [])
        if len(messages) < 3:
            continue
        question = str(messages[1].get("content", "")).strip()
        assistant = str(messages[-1].get("content", "")).strip()
        if len(question) < 5 or len(assistant) < 20:
            continue
        signatures = _question_signatures(question)
        if seen_signatures.intersection(signatures):
            continue
        seen_signatures.update(signatures)
        accepted.append(item)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an independent MathLLM test set")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--regression-dir", type=Path, default=REGRESSION_DIR)
    parser.add_argument("--output", type=Path, default=TEST_FILE)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    raw_data = load_raw_data(args.raw_dir, regression_dir=args.regression_dir)
    formatted = _with_metadata(raw_data)
    candidates = _fast_candidate_clean(formatted)
    selected = _select_stratified(candidates, args.test_ratio, args.seed)
    selected = validate_data(selected, tokenizer=None, max_tokens=args.max_tokens)

    output_records = [{"messages": item["messages"]} for item in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output_records, handle, ensure_ascii=False, indent=2)

    print(f"Cleaned candidates: {len(candidates)}")
    print(f"Independent test examples: {len(output_records)}")
    _print_split_stats("Test", selected)
    print(f"Saved test set to {args.output}")


if __name__ == "__main__":
    main()
