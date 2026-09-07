"""Build the one-model Round 7 staging split.

The training mix is deliberately explicit: all 150 new Round 7 correction
items plus 65 deterministic original-data replay items (about 70/30).  The
internal eval split contains 40 ordinary holdout items and never contains a
Round 7 correction item.  The protected independent test and regression sets
are not touched.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import (
    _canonicalize_question,
    _load_test_questions,
    format_example,
    load_raw_data,
    validate_data,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
REGRESSION = ROOT / "data/eval/regression"
TEST_FILE = ROOT / "data/eval/test.json"
CORRECTION_SOURCE = "correction-round-7"
OUTPUT = ROOT / "data/processed/round-7-staging"
MANIFEST = ROOT / "data/processed/round-7-staging/manifest.json"
SEED = 20260905
REPLAY_COUNT = 65
EVAL_COUNT = 40


def with_metadata(item: dict[str, str]) -> dict:
    example = format_example(item)
    example["_metadata"] = {
        "source": item.get("source", "unknown"),
        "subject": item.get("subject", "unknown"),
        "difficulty": item.get("difficulty", "unknown"),
    }
    return example


def stats(records: list[dict]) -> dict:
    return {
        "count": len(records),
        "sources": dict(Counter(x.get("_metadata", {}).get("source", "unknown") for x in records)),
        "subjects": dict(Counter(x.get("_metadata", {}).get("subject", "unknown") for x in records)),
    }


def public(records: list[dict]) -> list[dict]:
    return [{"messages": record["messages"]} for record in records]


def stratified_replay(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Take a proportional, source-stratified replay sample instead of a
    random slice that could accidentally overrepresent one raw dataset."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["_metadata"].get("source", "unknown")].append(record)
    total = sum(len(values) for values in groups.values())
    if count > total:
        raise RuntimeError("Replay count exceeds ordinary data")
    for values in groups.values():
        rng.shuffle(values)
    quotas = {source: int(len(values) * count / total) for source, values in groups.items()}
    remaining = count - sum(quotas.values())
    ranked = sorted(
        groups,
        key=lambda source: (len(groups[source]) * count / total - quotas[source], source),
        reverse=True,
    )
    for source in ranked[:remaining]:
        quotas[source] += 1
    selected: list[dict] = []
    for source, values in groups.items():
        selected.extend(values[:quotas[source]])
    rng.shuffle(selected)
    return selected


def main() -> None:
    excluded = {
        "correction-round-4", "correction-round-5", "correction-round-6",
    }
    normalized = load_raw_data(RAW_DIR, regression_dir=REGRESSION, excluded_sources=excluded)
    formatted = [with_metadata(item) for item in normalized]
    cleaned = validate_data(formatted, tokenizer=None, max_tokens=2048)
    test_questions = _load_test_questions(TEST_FILE)
    cleaned = [
        item for item in cleaned
        if _canonicalize_question(item["messages"][1]["content"]) not in test_questions
    ]

    corrections = [x for x in cleaned if x["_metadata"]["source"] == CORRECTION_SOURCE]
    ordinary = [x for x in cleaned if x["_metadata"]["source"] != CORRECTION_SOURCE]
    if len(corrections) != 150:
        raise RuntimeError(f"Expected 150 Round 7 corrections, found {len(corrections)}")
    if len(ordinary) < REPLAY_COUNT + EVAL_COUNT:
        raise RuntimeError("Not enough ordinary records for replay and internal eval")

    rng = random.Random(SEED)
    rng.shuffle(corrections)
    replay = stratified_replay(ordinary, REPLAY_COUNT, rng)
    replay_ids = {id(item) for item in replay}
    remaining_ordinary = [item for item in ordinary if id(item) not in replay_ids]
    rng.shuffle(remaining_ordinary)
    internal_eval = remaining_ordinary[:EVAL_COUNT]
    train = corrections + replay
    rng.shuffle(train)

    train_questions = {_canonicalize_question(x["messages"][1]["content"]) for x in train}
    eval_questions = {_canonicalize_question(x["messages"][1]["content"]) for x in internal_eval}
    if train_questions & eval_questions:
        raise RuntimeError("Train/eval question leakage detected")
    if len(train_questions) != len(train) or len(eval_questions) != len(internal_eval):
        raise RuntimeError("Duplicate questions detected in staging split")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "train.json").write_text(json.dumps(public(train), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "eval.json").write_text(json.dumps(public(internal_eval), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "round": "correction-round-7",
        "seed": SEED,
        "policy": "150 new corrections + 65 ordinary replay; 40 ordinary internal eval",
        "train_correction_ratio": round(len(corrections) / len(train), 6),
        "train_replay_ratio": round(len(replay) / len(train), 6),
        "train": stats(train),
        "eval": stats(internal_eval),
        "excluded_sources": sorted(excluded),
        "protected_sets": ["data/eval/test.json", "data/eval/regression/regression.json"],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Round 7 staging data

- `train.json`: {len(train)} records = {len(corrections)} new `correction-round-7` records + {len(replay)} ordinary replay records.
- Correction ratio: {len(corrections) / len(train):.2%}; replay ratio: {len(replay) / len(train):.2%}.
- `eval.json`: {len(internal_eval)} ordinary holdout records; Round 7 corrections remain train-only.
- Historical `correction-round-4/5/6` sources are excluded.
- `data/eval/test.json` and `data/eval/regression/regression.json` are protected and never copied into this staging split.
- `manifest.json` records the counts, seed, source and subject distributions.

This staging set is for the single Round 7 run, whose base model is the Round 6
merged model. It is not a replacement for the independent test or regression
sets used after merging.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"staging_dir={OUTPUT}")


if __name__ == "__main__":
    main()
