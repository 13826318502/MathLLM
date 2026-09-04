"""Collect a deterministic, train-only correction set from official sources.

The collector deliberately selects records that are not already present in
``data/raw`` or in the protected evaluation/regression sets.  It writes the
normalised independent-field format expected by ``prepare_data.py``.

Run from the project root::

    python scripts/collect_official_corrections.py --target 100

The resulting file is a candidate correction set.  Human spot-checking is
still required before a formal training run.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

import prepare_data as prep
import download_formal_data as formal


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CORRECTION_DIR = RAW_DIR / "corrections"
TEST_FILE = ROOT / "data" / "eval" / "test.json"
REGRESSION_DIR = ROOT / "data" / "eval" / "regression"
SEED = 20260904
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "precalculus",
)


def canonical(text: str) -> str:
    return prep._canonicalize_question(text)


def question_from_record(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "").strip()
    return str(record.get("question") or record.get("problem") or "").strip()


def blocked_questions() -> set[str]:
    blocked: set[str] = set()
    for path in (TEST_FILE,):
        if path.exists():
            for record in prep._read_raw_file(path):
                question = question_from_record(record)
                if question:
                    blocked.add(canonical(question))
    if REGRESSION_DIR.exists():
        blocked.update(prep._load_regression_questions(REGRESSION_DIR))
    return blocked


def existing_raw_questions(output_path: Path) -> set[str]:
    existing = blocked_questions()
    for path in sorted(RAW_DIR.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in prep.SUPPORTED_SUFFIXES:
            continue
        if path.resolve() == output_path.resolve():
            continue
        try:
            records = prep._read_raw_file(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for record in records:
            question = question_from_record(record)
            if question:
                existing.add(canonical(question))
    return existing


def gsm8k_record(item: dict[str, Any]) -> dict[str, str] | None:
    question = str(item.get("question") or "").strip()
    raw_answer = str(item.get("answer") or "").strip()
    if not question or "####" not in raw_answer:
        return None
    solution, answer = raw_answer.rsplit("####", 1)
    solution = prep._convert_gsm8k_annotations(solution).strip()
    answer = answer.strip()
    if answer and "$" not in answer and r"\(" not in answer:
        answer = f"${answer}$"
    if not solution or not answer:
        return None
    return {
        "question": prep._normalize_text(question),
        "solution": prep._normalize_text(solution),
        "answer": prep._normalize_text(answer),
        "source": "correction-official-gsm8k-round-2",
        "subject": "应用题",
        "difficulty": "基础",
    }


def math_record(item: dict[str, Any], config: str) -> dict[str, str] | None:
    normalized = prep._normalize_item(
        {**item, "source": "math", "_config": config},
        Path(f"official-{config}.jsonl"),
    )
    if normalized is None:
        return None
    subject_map = {
        "counting_and_probability": "概率与组合",
        "algebra": "代数",
        "intermediate_algebra": "代数",
        "number_theory": "数论与模运算",
        "geometry": "几何",
        "precalculus": "函数与预备微积分",
    }
    normalized["source"] = "correction-official-math-round-2"
    normalized["subject"] = subject_map.get(config, config)
    return normalized


def cmid_subject(question: str) -> str:
    text = question.lower()
    if any(token in text for token in ("矩阵", "行列式", "特征值", "特征向量", "线性方程组", "向量组", "秩", "二次型")):
        return "线性代数"
    if any(token in text for token in ("概率", "随机变量", "期望", "方差", "分布", "贝叶斯", "协方差")):
        return "概率论与统计"
    if any(token in text for token in ("极限", "导数", "微分", "积分", "级数", "微分方程", "偏导", "泰勒", "极值", "连续")):
        return "高等数学"
    return ""


def cmid_record(item: dict[str, Any]) -> dict[str, str] | None:
    question = str(item.get("query") or "").strip()
    solution = str(item.get("response") or "").strip()
    subject = cmid_subject(question)
    if not question or not solution or not subject:
        return None
    answer = formal._extract_cmid_answer(solution)
    if not answer:
        return None
    return {
        "question": prep._normalize_text(question),
        "solution": prep._normalize_text(solution),
        "answer": prep._normalize_text(answer),
        "source": "correction-official-cmid-round-2",
        "subject": subject,
        "difficulty": "未标注",
    }


def add_unique(
    candidates: Iterable[dict[str, str] | None],
    selected: list[dict[str, str]],
    seen: set[str],
    quota: int,
) -> None:
    for candidate in candidates:
        if len(selected) >= quota:
            return
        if candidate is None or len(candidate["question"]) < 5:
            continue
        key = canonical(candidate["question"])
        if key in seen or not candidate["solution"] or not candidate["answer"]:
            continue
        seen.add(key)
        selected.append(candidate)


def collect(target: int, output_path: Path) -> list[dict[str, str]]:
    if target != 100:
        raise ValueError("This curated round is defined for exactly 100 records")
    existing = existing_raw_questions(output_path)
    seen = set(existing)
    selected: list[dict[str, str]] = []
    rng = random.Random(SEED)

    quotas = {
        "gsm8k": 35,
        "counting_and_probability": 20,
        "algebra": 10,
        "intermediate_algebra": 5,
        "number_theory": 15,
        "geometry": 3,
        "precalculus": 2,
        "cmid": 10,
    }

    gsm = [dict(item) for item in load_dataset("openai/gsm8k", "main", split="train")]
    rng.shuffle(gsm)
    add_unique((gsm8k_record(item) for item in gsm), selected, seen, quotas["gsm8k"])

    for config in MATH_CONFIGS:
        records = [dict(item) for item in load_dataset("EleutherAI/hendrycks_math", config, split="train")]
        rng.shuffle(records)
        quota = quotas.get(config, 0)
        add_unique(
            (math_record(item, config) for item in records),
            selected,
            seen,
            len(selected) + quota,
        )

    cmid_target = quotas["cmid"]
    cmid_selected: list[dict[str, str]] = []
    cmid_seen = set(seen)
    stream = load_dataset("Mxode/CMID-Chinese_Math_Instruct_Dataset", split="train", streaming=True)
    for item in stream:
        if len(cmid_selected) >= cmid_target:
            break
        candidate = cmid_record(dict(item))
        if candidate is None:
            continue
        key = canonical(candidate["question"])
        if key in cmid_seen:
            continue
        cmid_seen.add(key)
        cmid_selected.append(candidate)
    selected.extend(cmid_selected)

    if len(selected) < target:
        raise RuntimeError(f"Only collected {len(selected)}/{target} unique records")
    return selected[:target]


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=CORRECTION_DIR / "correction-official-round-2.jsonl",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    records = collect(args.target, output)
    write_jsonl(output, records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    print(f"Saved {len(records)} records to {output}")
    print(f"Sources: {counts}")
    print("These are official-source candidates; human mathematical spot-checking is still required.")


if __name__ == "__main__":
    main()
