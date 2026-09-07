"""Build targeted, automatically checked Round 14 correction data.

The examples are generated from exact arithmetic templates targeting the
failure families observed in the regression set.  Training and validation use
disjoint parameter ranges and unique markers.  The script also builds the
70/30 correction/replay staging split.
"""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from pathlib import Path

from prepare_data import _canonicalize_question, _load_test_questions, format_example


ROOT = Path(__file__).resolve().parents[1]
RAW_OUT = ROOT / "data/raw/corrections/correction-round-14-training.jsonl"
VAL_OUT = ROOT / "data/eval/correction-validation/round-14.json"
STAGING = ROOT / "data/processed/round-14-staging"
REPLAY_SOURCE = ROOT / "data/processed/round-10-staging/train.json"
TEST_FILE = ROOT / "data/eval/test.json"
SEED = 20260914


def rec(question: str, solution: str, answer: str, subject: str, tag: str) -> dict[str, str]:
    return {
        "question": question,
        "solution": solution,
        "answer": answer,
        "source": "correction-round-14",
        "subject": subject,
        "difficulty": "Level 2",
        "tag": tag,
    }


def make_application(start: int, count: int) -> list[dict[str, str]]:
    out = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            rate, hours, breaks, pause = 10 + n % 8, 4 + n % 3, 1 + n % 2, 15
            active = hours * 60 - breaks * pause
            answer = rate * active // 60
            out.append(rec(
                f"某设备每小时生产 {rate} 个零件，工作 {hours} 小时，中途休息 {breaks} 次，每次 {pause} 分钟。它实际生产多少个零件？（R14T-{n:04d}）",
                f"总工作时间为 {hours}×60-{breaks}×{pause}={active} 分钟。实际生产量为 {rate}×{active}/60={answer} 个。",
                f"${answer}$", "多步应用题", "time-and-units",
            ))
        elif i % 3 == 1:
            qty, price, discount, shipping = 6 + n % 7, 12 + n % 9, 10 + n % 4 * 5, 3 + n % 6
            subtotal = qty * price
            answer = Fraction(subtotal * (100 - discount), 100) + shipping
            out.append(rec(
                f"商店购买 {qty} 件单价 {price} 元的物品，先按 {discount}% 打折，再加 {shipping} 元运费。总价是多少？（R14T-{n:04d}）",
                f"折前总价为 {qty}×{price}={subtotal}。折后为 {subtotal}×{100-discount}/100={answer-shipping}。加运费后总价为 {answer} 元。",
                f"${answer}$", "多步应用题", "discount-and-shipping",
            ))
        else:
            total, a, b, extra = 84 + n % 9 * 6, 2 + n % 4, 5 + n % 3, 6 + n % 5
            first = total * a // (a + b)
            second = total - first + extra
            out.append(rec(
                f"一笔 {total} 元先按 {a}:{b} 分给两组，第二组随后增加 {extra} 元。第二组最后得到多少元？（R14T-{n:04d}）",
                f"比例总份数为 {a+b}，第一组得到 {total}×{a}/{a+b}={first} 元，第二组原得 {total-first} 元；增加后为 {total-first}+{extra}={second} 元。",
                f"${second}$", "多步应用题", "ratio-and-update",
            ))
    return out


def make_probability(start: int, count: int) -> list[dict[str, str]]:
    out = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            red, blue = 4 + n % 5, 6 + n % 4
            numerator = math.comb(red, 1) * math.comb(blue, 2)
            denominator = math.comb(red + blue, 3)
            answer = Fraction(numerator, denominator)
            out.append(rec(
                f"盒中有 {red} 个红球和 {blue} 个蓝球，不放回抽取 3 个，恰好抽到 1 个红球的概率是多少？（R14T-{n:04d}）",
                f"有利情况数为 C({red},1)C({blue},2)={numerator}，总情况数为 C({red+blue},3)={denominator}，概率为 {answer}。",
                f"${answer}$", "概率与组合", "hypergeometric",
            ))
        elif i % 3 == 1:
            total, choose = 8 + n % 5, 2 + n % 3
            answer = math.comb(total, choose)
            out.append(rec(
                f"从 {total} 名学生中选出 {choose} 名组成委员会，不考虑顺序，共有多少种选法？（R14T-{n:04d}）",
                f"因为不考虑顺序，使用组合数 C({total},{choose})={answer}。",
                f"${answer}$", "概率与组合", "combination-count",
            ))
        else:
            favorable, total = 2 + n % 4, 5 + n % 6
            answer = Fraction(favorable, total)
            out.append(rec(
                f"一个试验有 {total} 个等可能结果，其中 {favorable} 个结果满足事件 A。事件 A 的概率是多少？（R14T-{n:04d}）",
                f"概率等于有利结果数除以总结果数，即 {favorable}/{total}={answer}。",
                f"${answer}$", "概率与组合", "probability-denominator",
            ))
    return out


def make_linear_algebra(start: int, count: int) -> list[dict[str, str]]:
    out = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            x, y = 2 + n % 5, -3 + n % 6
            a, b, c, d = 2 + n % 4, 1 + n % 3, 1 + n % 2, 2 + n % 4
            p, q = a*x+b*y, c*x+d*y
            out.append(rec(
                f"解方程组 {a}x+{b}y={p}，{c}x+{d}y={q}，求 (x,y)。（R14T-{n:04d}）",
                f"将 x={x}, y={y} 代入两式分别得到 {p} 和 {q}，且行列式 {a}×{d}-{b}×{c}≠0，所以方程组唯一解为 (x,y)=({x},{y})。",
                f"$({x},{y})$", "线性代数", "linear-system",
            ))
        elif i % 3 == 1:
            a, b, c, d = 2 + n % 6, 1 + n % 5, 1 + n % 4, 3 + n % 5
            answer = a * d - b * c
            out.append(rec(
                f"求矩阵 A=(({a},{b}),({c},{d})) 的行列式。（R14T-{n:04d}）",
                f"二维矩阵行列式为 ad-bc={a}×{d}-{b}×{c}={answer}。",
                f"${answer}$", "线性代数", "determinant",
            ))
        else:
            lam1, lam2 = -2 + n % 5, 3 + n % 4
            upper = 1 + n % 7
            out.append(rec(
                f"矩阵 A=(({lam1},{upper}),(0,{lam2})) 为上三角矩阵，求其特征值。（R14T-{n:04d}）",
                f"三角矩阵的特征值等于主对角线元素，因此特征值为 {lam1} 和 {lam2}。",
                f"${lam1},\ {lam2}$", "线性代数", "triangular-eigenvalues",
            ))
    return out


def make_algebra_number(start: int, count: int) -> list[dict[str, str]]:
    out = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            root = -4 + n % 9
            coefficient = 3 + n % 6
            constant = -coefficient * root
            out.append(rec(
                f"解方程 {coefficient}x+({constant})=0。（R14T-{n:04d}）",
                f"移项得 {coefficient}x={-constant}，两边除以 {coefficient}，所以 x={root}。",
                f"${root}$", "代数与数论", "linear-equation",
            ))
        elif i % 3 == 1:
            base, exponent, modulus = 2 + n % 4, 3 + n % 5, 5 + n % 6
            answer = pow(base, exponent, modulus)
            out.append(rec(
                f"求 {base}^{exponent} 除以 {modulus} 的余数。（R14T-{n:04d}）",
                f"计算 {base}^{exponent}={base**exponent}，而 {base**exponent}={base**exponent//modulus}×{modulus}+{answer}，所以余数为 {answer}。",
                f"${answer}$", "数论与余数", "remainder",
            ))
        else:
            a, b, c = 2 + n % 5, 3 + n % 4, 4 + n % 6
            answer = a * a + b * b + c * c - 2 * (a * b + a * c + b * c)
            out.append(rec(
                f"计算 ({a}+{b}+{c})^2-4({a}{b}+{a}{c}+{b}{c})。（R14T-{n:04d}）",
                f"先算和的平方为 {(a+b+c)**2}，括号内为 {a*b+a*c+b*c}，所以结果为 {(a+b+c)**2}-4×{a*b+a*c+b*c}={answer}。",
                f"${answer}$", "代数与数论", "identity-check",
            ))
    return out


def make_geometry(start: int, count: int) -> list[dict[str, str]]:
    out = []
    for i in range(count):
        n = start + i
        if i % 2 == 0:
            x, y, scale = -4 + n % 9, -5 + n % 8, 2 + n % 3
            answer = (scale * x, scale * y)
            out.append(rec(
                f"点 P({x},{y}) 关于原点作比例因子为 {scale} 的位似变换，求像点坐标。（R14T-{n:04d}）",
                f"关于原点的位似变换把每个坐标乘以 {scale}，所以 P' = ({scale}×{x},{scale}×{y})=({answer[0]},{answer[1]})。",
                f"$({answer[0]},{answer[1]})$", "坐标几何", "dilation",
            ))
        else:
            x1, y1, x2, y2 = n % 7, -3 + n % 5, 4 + n % 6, 2 + n % 4
            dx, dy = x2 - x1, y2 - y1
            answer = dx * dx + dy * dy
            out.append(rec(
                f"点 A({x1},{y1}) 和 B({x2},{y2}) 的距离平方是多少？（R14T-{n:04d}）",
                f"坐标差为 ({dx},{dy})，距离平方为 {dx}^2+{dy}^2={answer}。",
                f"${answer}$", "坐标几何", "distance-squared",
            ))
    return out


def build_records() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = (
        make_application(1000, 60)
        + make_probability(1100, 50)
        + make_linear_algebra(1200, 50)
        + make_algebra_number(1300, 70)
        + make_geometry(1400, 40)
        + make_probability(1500, 30)
    )
    validation = (
        make_application(2000, 12)
        + make_probability(2100, 12)
        + make_linear_algebra(2200, 12)
        + make_algebra_number(2300, 12)
        + make_geometry(2400, 12)
    )
    return train, validation


def main() -> None:
    train, validation = build_records()
    test_keys = _load_test_questions(TEST_FILE)
    all_records = train + validation
    keys = [_canonicalize_question(item["question"]) for item in all_records]
    if len(set(keys)) != len(keys) or any(key in test_keys for key in keys):
        raise RuntimeError("Round 14 duplicate or protected-test leakage detected")
    train_keys = set(keys[: len(train)])
    val_keys = set(keys[len(train):])
    if train_keys & val_keys:
        raise RuntimeError("Round 14 train/validation leakage detected")

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train) + "\n", encoding="utf-8")
    VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    VAL_OUT.write_text(json.dumps([format_example(x) for x in validation], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ordinary = json.loads(REPLAY_SOURCE.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    rng.shuffle(ordinary)
    replay_count = round(len(train) * 0.3 / 0.7)
    replay = ordinary[:replay_count]
    staged = [format_example(x) for x in train] + replay
    rng.shuffle(staged)
    STAGING.mkdir(parents=True, exist_ok=True)
    # ``staged`` is already in the public messages-only format.  Keep this
    # script self-contained instead of importing a removed round-specific
    # helper.
    (STAGING / "train.json").write_text(json.dumps(staged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "eval.json").write_text(json.dumps([format_example(x) for x in validation], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "round": "correction-round-14",
        "policy": f"{len(train)} targeted corrections + {len(replay)} ordinary replay (70/30)",
        "train_correction_count": len(train),
        "train_replay_count": len(replay),
        "train_count": len(staged),
        "validation_count": len(validation),
        "subjects": {subject: sum(x["subject"] == subject for x in train) for subject in sorted({x["subject"] for x in train})},
        "base_model": "outputs/correction-round-10-merged",
        "protected_sets": ["data/eval/test.json", "data/eval/regression/regression.json"],
    }
    (STAGING / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "README.md").write_text(
        "# Round 14 targeted staging\n\n"
        f"- 训练：{len(train)} 条针对回归错误类型的纠错题 + {len(replay)} 条普通题回放。\n"
        f"- 独立纠错验证：{len(validation)} 条。\n"
        "- 基座：`outputs/correction-round-10-merged`。\n"
        "- 原始测试集和历史回归集不参与训练。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
