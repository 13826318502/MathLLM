"""Split a correction-round candidate set without leaking holdout questions.

The input uses the independent-field schema expected by ``prepare_data.py``.
The training output is the only Round 3 file that belongs below ``data/raw``;
the validation output uses ChatML ``messages`` and stays below ``data/eval``.
The complete candidate file is archived below ``data/candidates`` so it is
available for audit but is not discovered by the training-data loader.

Run from the project root::

    python scripts/split_correction_round.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data as prep  # noqa: E402


REQUIRED_FIELDS = {
    "question",
    "solution",
    "answer",
    "source",
    "subject",
    "difficulty",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        missing = sorted(REQUIRED_FIELDS - set(value))
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
        if any(not str(value[field]).strip() for field in REQUIRED_FIELDS):
            raise ValueError(f"{path}:{line_number} contains an empty required field")
        records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a correction candidate file into train and validation")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "candidates" / "correction-round-4-all.jsonl",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=ROOT / "data" / "raw" / "corrections" / "correction-round-4-training.jsonl",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=ROOT / "data" / "eval" / "correction-validation" / "round-4.json",
    )
    parser.add_argument(
        "--archive-output",
        type=Path,
        default=ROOT / "data" / "candidates" / "archive" / "correction-round-4-all.jsonl",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    records = read_jsonl(args.input)
    if len(records) < 2:
        raise ValueError(f"Expected at least 2 correction candidates, found {len(records)}")

    canonical_questions = [prep._canonicalize_question(str(record["question"])) for record in records]
    if len(set(canonical_questions)) != len(canonical_questions):
        raise ValueError("Correction candidates contain duplicate questions after normalization")

    blocked = set()
    if prep.TEST_FILE.exists():
        blocked.update(prep._load_test_questions(prep.TEST_FILE))
    if prep.REGRESSION_DIR.exists():
        blocked.update(prep._load_regression_questions(prep.REGRESSION_DIR))
    leaked = [record["question"] for record, question in zip(records, canonical_questions) if question in blocked]
    if leaked:
        raise ValueError(f"Correction candidates overlap protected test/regression data: {len(leaked)} questions")

    rng = random.Random(args.seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    validation_count = round(len(shuffled) * args.validation_ratio)
    validation_count = max(1, min(len(shuffled) - 1, validation_count))
    validation_records = shuffled[:validation_count]
    training_records = shuffled[validation_count:]

    # Keep the raw candidate schema in the training file.  prepare_data.py
    # discovers this file and converts it to ChatML during preprocessing.
    write_jsonl(args.train_output, training_records)
    validation_messages = [prep.format_example(record) for record in validation_records]
    write_json(args.validation_output, validation_messages)

    args.archive_output.parent.mkdir(parents=True, exist_ok=True)
    if args.archive_output.exists():
        args.archive_output.unlink()
    shutil.move(str(args.input), str(args.archive_output))

    source_counts = Counter(str(record["source"]) for record in records)
    train_sources = Counter(str(record["source"]) for record in training_records)
    validation_sources = Counter(str(record["source"]) for record in validation_records)
    print(f"Candidates: {len(records)}")
    print(f"Training candidates: {len(training_records)} -> {args.train_output}")
    print(f"Correction validation: {len(validation_records)} -> {args.validation_output}")
    print(f"Archived all candidates: {args.archive_output}")
    print(f"All sources: {dict(source_counts)}")
    print(f"Training sources: {dict(train_sources)}")
    print(f"Validation sources: {dict(validation_sources)}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
