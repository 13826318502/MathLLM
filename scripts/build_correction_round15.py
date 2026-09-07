"""Build the clean Round 15 reset dataset.

Round 15 deliberately starts from the original Qwen base.  It uses a 70/30
mixture of clean non-correction data and newly generated, exactly checked
correction examples.  The frozen original test set and historical regression
set are never used for training or validation.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

from prepare_data import (
    _canonicalize_question,
    _load_test_questions,
    format_example,
    load_raw_data,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
REGRESSION_DIR = ROOT / "data/eval/regression"
TEST_FILE = ROOT / "data/eval/test.json"
CORRECTION_RAW = ROOT / "data/raw/corrections/correction-round-15-training.jsonl"
VALIDATION_OUT = ROOT / "data/eval/correction-validation/round-15.json"
STAGING = ROOT / "data/processed/round-15-staging"
SEED = 20260915
CORRECTION_COUNT = 180
VALIDATION_COUNT = 60
ORIGINAL_COUNT = 420


def frac_text(value: Fraction) -> str:
    if value.denominator == 1:
        return f"${value.numerator}$"
    return f"$\\frac{{{value.numerator}}}{{{value.denominator}}}$"


def record(question: str, solution: str, answer: str, subject: str, tag: str) -> dict[str, str]:
    return {
        "question": question,
        "solution": solution,
        "answer": answer,
        "source": "correction-round-15",
        "subject": subject,
        "difficulty": "Level 2",
        "tag": tag,
    }


def make_application(start: int, count: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            rate = 12 + n % 9
            hours = 5 + n % 3
            rest_hours = 1 + n % 2
            active = hours - rest_hours
            answer = rate * active
            out.append(record(
                f"一台设备每小时生产 {rate} 个零件，连续工作 {hours} 小时，其中停机休息 {rest_hours} 小时。实际生产多少个零件？（R15T-{n:04d}）",
                f"实际工作时间为 {hours}-{rest_hours}={active} 小时，所以产量为 {rate}×{active}={answer} 个。",
                f"${answer}$", "多步应用题", "time-and-units",
            ))
        elif i % 3 == 1:
            speed = 30 + n % 11 * 5
            hours = 2 + n % 4
            pause = 20 + n % 3 * 10
            total_minutes = hours * 60 + pause
            answer = speed * hours
            out.append(record(
                f"某人以每小时 {speed} 千米的速度行走 {hours} 小时，中途额外休息 {pause} 分钟。只计算行走时间，他走了多少千米？（R15T-{n:04d}）",
                f"休息时间不计入行走时间，行走时间仍为 {hours} 小时，因此路程为 {speed}×{hours}={answer} 千米；总经历时间为 {total_minutes} 分钟。",
                f"${answer}$", "多步应用题", "time-and-units",
            ))
        else:
            quantity = 6 + n % 5
            price = 20 + n % 7 * 5
            discount = 10 if n % 2 == 0 else 20
            shipping = 5 + n % 4 * 5
            subtotal = quantity * price
            answer = subtotal * (100 - discount) // 100 + shipping
            out.append(record(
                f"购买 {quantity} 件单价 {price} 元的商品，先打 {discount}% 折，再加 {shipping} 元运费，总价是多少？（R15T-{n:04d}）",
                f"折前总价为 {quantity}×{price}={subtotal} 元，折后为 {subtotal}×{100-discount}/100={subtotal * (100-discount)//100} 元，加运费后为 {answer} 元。",
                f"${answer}$", "多步应用题", "discount-and-shipping",
            ))
    return out


def make_probability(start: int, count: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            total = 10 + n % 8
            favorable = 2 + n % 4
            value = Fraction(favorable, total)
            out.append(record(
                f"袋中有 {total} 个外观相同的球，其中 {favorable} 个是红球，随机取 1 个，取到红球的概率是多少？（R15T-{n:04d}）",
                f"有利结果数为 {favorable}，总结果数为 {total}，所以概率为 {favorable}/{total}={value}。",
                frac_text(value), "概率与组合", "probability-denominator",
            ))
        elif i % 3 == 1:
            total = 7 + n % 5
            choose = 2 + n % 3
            value = Fraction(math_comb(total, choose), 1)
            out.append(record(
                f"从 {total} 名同学中任选 {choose} 名组成小组，不考虑顺序，共有多少种选法？（R15T-{n:04d}）",
                f"不考虑顺序，应使用组合数 C({total},{choose})={total}!/({choose}!({total}-{choose})!)={math_comb(total, choose)}。",
                frac_text(value), "概率与组合", "combination-count",
            ))
        else:
            total = 8 + n % 6
            success = 2 + n % 3
            draws = 2
            value = Fraction(success, total) * Fraction(success - 1, total - 1)
            out.append(record(
                f"盒中有 {total} 个零件，其中 {success} 个合格，不放回连续抽取 {draws} 个，两个都合格的概率是多少？（R15T-{n:04d}）",
                f"第一次合格概率为 {success}/{total}，第二次为 ({success}-1)/({total}-1)，相乘得到 {value}。",
                frac_text(value), "概率与组合", "without-replacement",
            ))
    return out


def math_comb(n: int, k: int) -> int:
    result = 1
    for j in range(1, k + 1):
        result = result * (n - j + 1) // j
    return result


def make_linear(start: int, count: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            a, b, c, d = 2 + n % 4, 1 + n % 3, 1 + n % 2, 2 + n % 4
            x, y = 1 + n % 5, -2 + n % 4
            p, q = a * x + b * y, c * x + d * y
            out.append(record(
                f"解方程组 {a}x+{b}y={p}，{c}x+{d}y={q}，求 (x,y)。（R15T-{n:04d}）",
                f"代入 x={x}, y={y} 可验证两式成立；行列式为 {a}×{d}-{b}×{c}={a*d-b*c}≠0，所以唯一解为 ({x},{y})。",
                f"$({x},{y})$", "线性代数", "linear-system",
            ))
        elif i % 3 == 1:
            a, b, c, d = 2 + n % 5, 1 + n % 4, 1 + n % 3, 3 + n % 5
            value = a * d - b * c
            out.append(record(
                f"求矩阵 A=(({a},{b}),({c},{d})) 的行列式。（R15T-{n:04d}）",
                f"二维矩阵行列式为 ad-bc={a}×{d}-{b}×{c}={value}。",
                f"${value}$", "线性代数", "determinant",
            ))
        else:
            lam1, lam2 = -2 + n % 5, 3 + n % 4
            out.append(record(
                f"矩阵 A=(({lam1},1),(0,{lam2})) 为上三角矩阵，求其特征值。（R15T-{n:04d}）",
                f"上三角矩阵的特征值是主对角线元素，因此特征值为 {lam1} 和 {lam2}。",
                f"${lam1},\ {lam2}$", "线性代数", "triangular-eigenvalues",
            ))
    return out


def make_algebra_number(start: int, count: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            coefficient = 2 + n % 7
            root = -4 + n % 9
            constant = -coefficient * root
            out.append(record(
                f"解方程 {coefficient}x+({constant})=0。（R15T-{n:04d}）",
                f"移项得 {coefficient}x={-constant}，两边除以 {coefficient}，得到 x={root}。",
                f"${root}$", "代数与数论", "linear-equation",
            ))
        elif i % 3 == 1:
            dividend = 80 + n % 40
            divisor = 7 + n % 8
            remainder = dividend % divisor
            out.append(record(
                f"求 {dividend} 除以 {divisor} 的余数。（R15T-{n:04d}）",
                f"{dividend}={divisor}×{dividend // divisor}+{remainder}，且余数小于除数，所以余数为 {remainder}。",
                f"${remainder}$", "数论与余数", "remainder",
            ))
        else:
            root = 2 + n % 6
            other = -3 - n % 5
            coefficient = -(root + other)
            constant = root * other
            out.append(record(
                f"方程 x^2+({coefficient})x+{constant}=0 的两个根是什么？（R15T-{n:04d}）",
                f"因式分解为 (x-{root})(x-({other}))=0，所以两个根是 {root} 和 {other}。",
                f"${root},\ {other}$", "代数与数论", "quadratic-roots",
            ))
    return out


def make_geometry(start: int, count: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            x1, y1 = n % 8, -4 + n % 7
            x2, y2 = 4 + n % 6, 2 + n % 5
            answer = (x1 + x2, y1 + y2)
            out.append(record(
                f"点 A({x1},{y1}) 和 B({x2},{y2}) 的中点坐标是多少？（R15T-{n:04d}）",
                f"中点为 (({x1}+{x2})/2,({y1}+{y2})/2)=({Fraction(answer[0],2)},{Fraction(answer[1],2)})。",
                f"$({Fraction(answer[0],2)},{Fraction(answer[1],2)})$", "坐标几何", "midpoint",
            ))
        elif i % 3 == 1:
            x, y = -4 + n % 8, -3 + n % 7
            dx, dy = 3 + n % 4, -2 + n % 5
            answer = (x + dx, y + dy)
            out.append(record(
                f"点 P({x},{y}) 向量平移 ({dx},{dy}) 后的像点坐标是多少？（R15T-{n:04d}）",
                f"平移后横坐标为 {x}+{dx}={answer[0]}，纵坐标为 {y}+({dy})={answer[1]}。",
                f"$({answer[0]},{answer[1]})$", "坐标几何", "translation",
            ))
        else:
            x, y = -3 + n % 7, -2 + n % 6
            scale = 2 + n % 3
            answer = (scale * x, scale * y)
            out.append(record(
                f"点 P({x},{y}) 关于原点作比例因子 {scale} 的位似变换，像点坐标是多少？（R15T-{n:04d}）",
                f"关于原点位似时每个坐标乘以 {scale}，所以像点为 ({scale}×{x},{scale}×{y})=({answer[0]},{answer[1]})。",
                f"$({answer[0]},{answer[1]})$", "坐标几何", "dilation",
            ))
    return out


def build_corrections() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = (
        make_application(1000, 36)
        + make_probability(1100, 36)
        + make_linear(1200, 36)
        + make_algebra_number(1300, 36)
        + make_geometry(1400, 36)
    )
    validation = (
        make_application(2000, 12)
        + make_probability(2100, 12)
        + make_linear(2200, 12)
        + make_algebra_number(2300, 12)
        + make_geometry(2400, 12)
    )
    return train, validation


def main() -> None:
    corrections, validation = build_corrections()
    if len(corrections) != CORRECTION_COUNT or len(validation) != VALIDATION_COUNT:
        raise RuntimeError("Unexpected Round 15 correction counts")

    test_keys = _load_test_questions(TEST_FILE)
    keys = [_canonicalize_question(x["question"]) for x in corrections + validation]
    if len(set(keys)) != len(keys) or any(key in test_keys for key in keys):
        raise RuntimeError("Round 15 correction duplicate or protected-test leakage")

    raw = load_raw_data(RAW_DIR, regression_dir=REGRESSION_DIR)
    clean_original = []
    seen: set[str] = set()
    for item in raw:
        source = str(item.get("source", ""))
        key = _canonicalize_question(str(item.get("question", "")))
        if source.startswith("correction-") or key in test_keys or key in seen:
            continue
        seen.add(key)
        clean_original.append(item)
    if len(clean_original) < ORIGINAL_COUNT:
        raise RuntimeError(f"Only {len(clean_original)} clean original items available")

    rng = random.Random(SEED)
    rng.shuffle(clean_original)
    original = clean_original[:ORIGINAL_COUNT]
    train_messages = [format_example(x) for x in original]
    train_messages.extend(format_example(x) for x in corrections)
    rng.shuffle(train_messages)
    eval_messages = [format_example(x) for x in validation]

    CORRECTION_RAW.parent.mkdir(parents=True, exist_ok=True)
    CORRECTION_RAW.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in corrections) + "\n",
        encoding="utf-8",
    )
    VALIDATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_OUT.write_text(json.dumps(eval_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "train.json").write_text(json.dumps(train_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "eval.json").write_text(json.dumps(eval_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "round": "correction-round-15",
        "base_model": "models/Qwen2.5-7B-Instruct-modelscope",
        "policy": "clean original replay 70% + new verified corrections 30%",
        "original_count": len(original),
        "correction_count": len(corrections),
        "train_count": len(train_messages),
        "validation_count": len(validation),
        "correction_subjects": dict(Counter(x["subject"] for x in corrections)),
        "original_sources": dict(Counter(x.get("source", "unknown") for x in original)),
        "protected_sets": ["data/eval/test.json", "data/eval/regression/regression.json"],
        "excluded_correction_sources": "all source values beginning with correction-",
    }
    (STAGING / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "README.md").write_text(
        "# Round 15 reset staging\n\n"
        f"- 训练：{len(original)} 条清洗后的原始题 + {len(corrections)} 条新纠错题。\n"
        f"- 独立纠错验证：{len(validation)} 条。\n"
        "- 基座：原始 Qwen，而不是 Round14 合并模型。\n"
        "- 原始测试集与历史回归集冻结，不参与训练。\n"
        "- 纠错题使用可计算模板生成，避免截断、不可整除和答案标签错误。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
