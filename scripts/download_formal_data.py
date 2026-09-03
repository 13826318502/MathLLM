"""Download a curated first formal dataset for MathLLM.

The project documents recommend a small, mixed dataset rather than blindly
feeding a very large corpus into SFT.  This script keeps downloaded source
files in ``data/raw`` and leaves the normalization, validation, de-duplication
and train/eval split to ``scripts/prepare_data.py``.

Sources:
    - openai/gsm8k, train split, 500 examples
    - EleutherAI/hendrycks_math, all subject configs, train split, 500 examples
    - Mxode/CMID-Chinese_Math_Instruct_Dataset, 250 Chinese examples selected
      by topic buckets

Run from the project root::

    python scripts/download_formal_data.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SEED = 42
GSM8K_TARGET = 500
MATH_TARGET = 500
CMID_TARGET = 250

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _sample(records: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < target:
        raise ValueError(f"Expected at least {target} records, found {len(records)}")
    indices = random.Random(seed).sample(range(len(records)), target)
    return [records[index] for index in sorted(indices)]


def download_gsm8k() -> int:
    print("Loading GSM8K train split...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    records = [dict(item) for item in dataset]
    selected = _sample(records, GSM8K_TARGET, SEED)
    path = RAW_DIR / "gsm8k_train_curated_500.jsonl"
    count = _write_jsonl(path, selected)
    print(f"Saved {count} GSM8K records to {path}")
    return count


def download_math() -> int:
    print("Loading Hendrycks MATH train splits...")
    records: list[dict[str, Any]] = []
    for config_name in MATH_CONFIGS:
        dataset = load_dataset("EleutherAI/hendrycks_math", config_name, split="train")
        for item in dataset:
            record = dict(item)
            record["_config"] = config_name
            records.append(record)
        print(f"  {config_name}: {len(dataset)} records")

    selected = _sample(records, MATH_TARGET, SEED + 1)
    path = RAW_DIR / "math_train_curated_500.jsonl"
    count = _write_jsonl(path, selected)
    print(f"Saved {count} MATH records to {path}")
    return count


def _cmid_subject(question: str) -> str:
    text = question.lower()
    if any(
        token in text
        for token in (
            "极限",
            "导数",
            "微分",
            "积分",
            "级数",
            "微分方程",
            "偏导",
            "泰勒",
            "极值",
            "连续",
        )
    ):
        return "高等数学"
    if any(
        token in text
        for token in (
            "矩阵",
            "行列式",
            "特征值",
            "特征向量",
            "线性方程组",
            "向量组",
            "秩",
            "二次型",
        )
    ):
        return "线性代数"
    if any(
        token in text
        for token in (
            "概率",
            "随机变量",
            "期望",
            "方差",
            "分布",
            "贝叶斯",
            "协方差",
            "数理统计",
        )
    ):
        return "概率论与统计"
    return "代数与其他"


def _last_boxed(text: str) -> str:
    """Extract the last balanced ``\\boxed{...}`` expression."""
    marker = r"\boxed"
    candidates: list[str] = []
    start = 0
    while True:
        marker_start = text.find(marker, start)
        if marker_start < 0:
            break
        open_brace = text.find("{", marker_start + len(marker))
        if open_brace < 0:
            break
        depth = 0
        for index in range(open_brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[open_brace + 1 : index].strip())
                    start = index + 1
                    break
        else:
            break
    if not candidates:
        return ""
    answer = candidates[-1]
    return answer if "$" in answer else f"${answer}$"


def _extract_cmid_answer(response: str) -> str:
    boxed = _last_boxed(response)
    if boxed:
        return boxed
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.search(r"(?:最终答案|答案|答)[：:]\s*(.+)$", line)
        if match:
            return match.group(1).strip()
    return ""


def download_cmid() -> int:
    print("Streaming CMID Chinese math data and selecting topic buckets...")
    dataset = load_dataset(
        "Mxode/CMID-Chinese_Math_Instruct_Dataset",
        split="train",
        streaming=True,
    )
    targets = {
        "高等数学": 80,
        "线性代数": 70,
        "概率论与统计": 70,
        "代数与其他": 30,
    }
    selected: dict[str, list[dict[str, Any]]] = {key: [] for key in targets}
    scanned = 0
    for item in dataset:
        scanned += 1
        question = str(item.get("query") or "").strip()
        response = str(item.get("response") or "").strip()
        subject = _cmid_subject(question)
        if not question or not response or len(selected[subject]) >= targets[subject]:
            if all(len(selected[key]) >= targets[key] for key in targets):
                break
            continue
        answer = _extract_cmid_answer(response)
        if not answer:
            continue
        selected[subject].append(
            {
                "question": question,
                "solution": response,
                "answer": answer,
                "source": "cmid_chinese_math",
                "subject": subject,
                "difficulty": "未标注",
            }
        )
        if all(len(selected[key]) >= targets[key] for key in targets):
            break
        if scanned >= 200000:
            break

    records = [record for bucket in selected.values() for record in bucket]
    if len(records) < CMID_TARGET:
        counts = {key: len(value) for key, value in selected.items()}
        raise RuntimeError(
            f"Only selected {len(records)}/{CMID_TARGET} CMID records after scanning "
            f"{scanned}; bucket counts: {counts}"
        )
    random.Random(SEED + 2).shuffle(records)
    path = RAW_DIR / "cmid_chinese_math_curated_250.jsonl"
    count = _write_jsonl(path, records[:CMID_TARGET])
    print(f"Saved {count} CMID records to {path}; scanned {scanned}")
    return count


def write_source_manifest(counts: dict[str, int]) -> None:
    manifest = RAW_DIR / "SOURCES.md"
    content = (
        "# Formal dataset sources\n\n"
        "This directory contains locally curated subsets. The files are intentionally\n"
        "ignored by Git; keep this manifest with any external dataset backup.\n\n"
        "| File | Source | Selection | License |\n"
        "|---|---|---|---|\n"
        "| `gsm8k_train_curated_500.jsonl` | `openai/gsm8k` train | seed 42, 500 records | MIT |\n"
        "| `math_train_curated_500.jsonl` | `EleutherAI/hendrycks_math` train | all subject configs, seed 43, 500 records | MIT |\n"
        "| `cmid_chinese_math_curated_250.jsonl` | `Mxode/CMID-Chinese_Math_Instruct_Dataset` train | topic buckets, seed 44, 250 records | CC BY-SA 4.0 |\n\n"
        "These sources still require the project's validation and at least 10% human\n"
        "spot-checking before treating the data as mathematically verified.\n\n"
        f"Downloaded counts: {counts}\n"
    )
    manifest.write_text(content, encoding="utf-8")


def main() -> None:
    counts = {
        "gsm8k": download_gsm8k(),
        "math": download_math(),
        "cmid": download_cmid(),
    }
    write_source_manifest(counts)
    print(f"Done. Curated raw records: {sum(counts.values())}")


if __name__ == "__main__":
    main()
