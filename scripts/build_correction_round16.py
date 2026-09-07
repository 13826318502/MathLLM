"""Build Round 16: full clean base data plus a small verified correction set."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from build_correction_round15 import (
    make_algebra_number,
    make_application,
    make_geometry,
    make_linear,
    make_probability,
)
from prepare_data import _canonicalize_question, _load_test_questions, format_example, load_raw_data


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
REGRESSION_DIR = ROOT / "data/eval/regression"
TEST_FILE = ROOT / "data/eval/test.json"
CORRECTION_RAW = ROOT / "data/raw/corrections/correction-round-16-training.jsonl"
VALIDATION_OUT = ROOT / "data/eval/correction-validation/round-16.json"
STAGING = ROOT / "data/processed/round-16-staging"
SEED = 20260916


def new_corrections() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = (
        make_application(3000, 36)
        + make_probability(3100, 36)
        + make_linear(3200, 36)
        + make_algebra_number(3300, 36)
        + make_geometry(3400, 36)
    )
    validation = (
        make_application(4000, 12)
        + make_probability(4100, 12)
        + make_linear(4200, 12)
        + make_algebra_number(4300, 12)
        + make_geometry(4400, 12)
    )
    for item in train + validation:
        item["source"] = "correction-round-16"
        item["question"] = item["question"].replace("R15T-", "R16T-")
    return train, validation


def main() -> None:
    corrections, validation = new_corrections()
    test_keys = _load_test_questions(TEST_FILE)
    all_keys = [_canonicalize_question(x["question"]) for x in corrections + validation]
    if len(set(all_keys)) != len(all_keys) or any(key in test_keys for key in all_keys):
        raise RuntimeError("Round 16 correction duplicate or test leakage")

    raw = load_raw_data(RAW_DIR, regression_dir=REGRESSION_DIR)
    original: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw:
        source = str(item.get("source", ""))
        key = _canonicalize_question(str(item.get("question", "")))
        if source.startswith("correction-") or key in test_keys or key in seen:
            continue
        seen.add(key)
        original.append(item)
    if len(original) < 2000:
        raise RuntimeError(f"Expected the full clean base set, found only {len(original)} items")

    rng = random.Random(SEED)
    rng.shuffle(original)
    train_messages = [format_example(item) for item in original]
    train_messages.extend(format_example(item) for item in corrections)
    rng.shuffle(train_messages)
    eval_messages = [format_example(item) for item in validation]

    CORRECTION_RAW.parent.mkdir(parents=True, exist_ok=True)
    CORRECTION_RAW.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in corrections) + "\n",
        encoding="utf-8",
    )
    VALIDATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_OUT.write_text(json.dumps(eval_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "train.json").write_text(json.dumps(train_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "eval.json").write_text(json.dumps(eval_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "round": "correction-round-16",
        "base_model": "models/Qwen2.5-7B-Instruct-modelscope",
        "policy": "all clean original data + 180 new verified corrections",
        "original_count": len(original),
        "correction_count": len(corrections),
        "train_count": len(train_messages),
        "validation_count": len(validation),
        "correction_subjects": dict(Counter(item["subject"] for item in corrections)),
        "original_sources": dict(Counter(str(item.get("source", "unknown")) for item in original)),
        "protected_sets": ["data/eval/test.json", "data/eval/regression/regression.json"],
    }
    (STAGING / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "README.md").write_text(
        "# Round 16 full-base staging\n\n"
        f"- 训练：{len(original)} 条清洗后的原始题 + {len(corrections)} 条新纠错题。\n"
        f"- 独立纠错验证：{len(validation)} 条。\n"
        "- 基座：原始 Qwen，不叠加历史合并模型。\n"
        "- 原始测试集与历史回归集冻结，不参与训练。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
