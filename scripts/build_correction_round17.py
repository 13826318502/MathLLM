"""Build the next isolated correction-focused training round.

Round 17 contains genuinely new, programmatically checked correction examples
and a disjoint correction-validation set.  It also includes a small replay
sample of clean original data so the next incremental run does not train on
corrections alone.  Frozen test, regression, and prior validation questions
are protected from leakage.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

from prepare_data import (
    _canonicalize_question,
    _load_regression_questions,
    _load_test_questions,
    format_example,
    load_raw_data,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
REGRESSION_DIR = ROOT / "data/eval/regression"
TEST_FILE = ROOT / "data/eval/test.json"
RAW_OUT = RAW_DIR / "corrections/correction-round-17-training.jsonl"
VALIDATION_OUT = ROOT / "data/eval/correction-validation/round-17.json"
STAGING = ROOT / "data/processed/round-17-staging"
SEED = 20260917
CORRECTION_COUNT = 600
VALIDATION_COUNT = 120
REPLAY_COUNT = 257


def frac(value: Fraction | int) -> str:
    value = Fraction(value)
    return f"${value.numerator}$" if value.denominator == 1 else f"$\\frac{{{value.numerator}}}{{{value.denominator}}}$"


def rec(question: str, solution: str, answer: str, subject: str, tag: str) -> dict[str, str]:
    return {
        "question": question,
        "solution": solution,
        "answer": answer,
        "source": "correction-round-17",
        "subject": subject,
        "difficulty": "Level 2",
        "tag": tag,
    }


def make_application(start: int, count: int, marker: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 4 == 0:
            rate = 18 + (n * 7) % 23
            hours = 3 + (n % 5)
            rest = 15 + (n * 11) % 46
            active = hours * 60 - rest
            answer = Fraction(rate * active, 60)
            out.append(rec(
                f"某设备每小时加工 {rate} 件产品，计划连续工作 {hours} 小时，期间停机 {rest} 分钟。实际加工多少件？（{marker}T-{n:04d}）",
                f"总计划时间为 {hours}×60={hours * 60} 分钟，实际工作时间为 {hours * 60}-{rest}={active} 分钟。产量为 {rate}×{active}/60={answer} 件。",
                frac(answer), "多步应用题", "time-and-units",
            ))
        elif i % 4 == 1:
            speed = 36 + (n * 5) % 31
            travel_hours = 2 + n % 4
            pause = 20 + (n * 7) % 41
            answer = speed * travel_hours
            total_minutes = travel_hours * 60 + pause
            out.append(rec(
                f"一辆车以每小时 {speed} 千米行驶 {travel_hours} 小时，中途停车 {pause} 分钟。只计算行驶路程，车辆行驶了多少千米？（{marker}T-{n:04d}）",
                f"停车时间不计入行驶时间，行驶路程为 {speed}×{travel_hours}={answer} 千米；包含停车的总经历时间为 {total_minutes} 分钟。",
                frac(answer), "多步应用题", "time-and-units",
            ))
        elif i % 4 == 2:
            qty = 4 + n % 9
            price = 15 + (n * 3) % 28
            discount = 5 + (n % 4) * 5
            shipping = 6 + (n * 2) % 13
            subtotal = qty * price
            answer = Fraction(subtotal * (100 - discount), 100) + shipping
            out.append(rec(
                f"购买 {qty} 件单价 {price} 元的商品，先打 {discount}% 折，再支付 {shipping} 元运费，总价是多少？（{marker}T-{n:04d}）",
                f"折前总价为 {qty}×{price}={subtotal} 元，折后为 {subtotal}×{100 - discount}/100={Fraction(subtotal * (100 - discount), 100)} 元，加运费后为 {answer} 元。",
                frac(answer), "多步应用题", "discount-and-shipping",
            ))
        else:
            total = 72 + (n * 5) % 67
            first_minutes = 25 + n % 6 * 5
            second_minutes = 35 + (n * 3) % 7 * 5
            rate = 2 + n % 5
            answer = rate * (first_minutes + second_minutes)
            out.append(rec(
                f"一项工作第一阶段用时 {first_minutes} 分钟，第二阶段用时 {second_minutes} 分钟，每分钟完成 {rate} 个单位。两阶段共完成多少个单位？（已知总任务上限为 {total * rate} 个）（{marker}T-{n:04d}）",
                f"总工作时间为 {first_minutes}+{second_minutes}={first_minutes + second_minutes} 分钟，完成量为 {rate}×{first_minutes + second_minutes}={answer} 个单位。总任务上限不改变实际完成量。",
                frac(answer), "多步应用题", "multi-stage-calculation",
            ))
    return out


def comb(n: int, k: int) -> int:
    value = 1
    for j in range(1, k + 1):
        value = value * (n - j + 1) // j
    return value


def make_probability(start: int, count: int, marker: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 4 == 0:
            total = 11 + n % 13
            favorable = 2 + (n * 3) % min(8, total - 1)
            value = Fraction(favorable, total)
            out.append(rec(
                f"盒中有 {total} 个外观相同的球，其中 {favorable} 个为红球，随机取出 1 个，取到红球的概率是多少？（{marker}T-{n:04d}）",
                f"有利结果数为 {favorable}，样本空间总数为 {total}，所以概率为 {favorable}/{total}={value}。",
                frac(value), "概率与组合", "probability-denominator",
            ))
        elif i % 4 == 1:
            total = 9 + n % 10
            choose = 2 + n % 4
            value = comb(total, choose)
            out.append(rec(
                f"从 {total} 名学生中任选 {choose} 名组成小组，不考虑顺序，共有多少种选法？（{marker}T-{n:04d}）",
                f"不考虑顺序，使用组合数 C({total},{choose})={total}!/({choose}!({total}-{choose})!)={value}。",
                frac(value), "概率与组合", "combination-count",
            ))
        elif i % 4 == 2:
            total = 12 + n % 9
            good = 4 + n % 5
            draws = 3
            value = Fraction(good, total) * Fraction(good - 1, total - 1) * Fraction(good - 2, total - 2)
            out.append(rec(
                f"袋中有 {total} 个零件，其中 {good} 个合格。不放回连续抽取 {draws} 个，三个都合格的概率是多少？（{marker}T-{n:04d}）",
                f"三次抽取分别为 {good}/{total}、({good}-1)/({total}-1)、({good}-2)/({total}-2)，相乘得 {value}。",
                frac(value), "概率与组合", "without-replacement",
            ))
        else:
            red = 3 + n % 6
            blue = 5 + (n * 2) % 7
            total = red + blue
            value = Fraction(red, total) + Fraction(blue, total)
            out.append(rec(
                f"一个袋子中有 {red} 个红球和 {blue} 个蓝球，随机取 1 个，取到红球或蓝球的概率是多少？（{marker}T-{n:04d}）",
                f"红球和蓝球涵盖全部 {total} 个球，两个互斥事件概率相加为 {red}/{total}+{blue}/{total}={value}。",
                frac(value), "概率与组合", "disjoint-events",
            ))
    return out


def make_linear(start: int, count: int, marker: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 4 == 0:
            a, b, c, d = 2 + n % 5, 1 + (n * 2) % 6, 1 + n % 4, 3 + (n * 3) % 5
            x, y = -2 + n % 8, -3 + (n * 2) % 9
            det = a * d - b * c
            if det == 0:
                d += 1
            p, q = a * x + b * y, c * x + d * y
            out.append(rec(
                f"解方程组 {a}x+{b}y={p}，{c}x+{d}y={q}，求 (x,y)。（{marker}T-{n:04d}）",
                f"行列式为 {a}×{d}-{b}×{c}={a * d - b * c}≠0，方程组有唯一解；代回可得 x={x}，y={y}。",
                f"$({x},{y})$", "线性代数", "linear-system",
            ))
        elif i % 4 == 1:
            a, b, c, d = 2 + n % 7, 1 + n % 5, 1 + (n * 2) % 6, 4 + n % 7
            value = a * d - b * c
            out.append(rec(
                f"求矩阵 A=(({a},{b}),({c},{d})) 的行列式。（{marker}T-{n:04d}）",
                f"二维矩阵行列式为 ad-bc={a}×{d}-{b}×{c}={value}。",
                frac(value), "线性代数", "determinant",
            ))
        elif i % 4 == 2:
            lam1 = -4 + n % 9
            lam2 = 2 + (n * 3) % 8
            out.append(rec(
                f"矩阵 A=(({lam1},1),(0,{lam2})) 是上三角矩阵，求它的特征值。（{marker}T-{n:04d}）",
                f"上三角矩阵的特征值等于主对角线元素，因此特征值为 {lam1} 和 {lam2}。",
                f"${lam1},\\ {lam2}$", "线性代数", "triangular-eigenvalues",
            ))
        else:
            out.append(rec(
                f"若两个矩阵具有一个共同特征向量，是否必然说明它们可交换？请判断并说明。（{marker}T-{n:04d}）",
                "不必然。共同特征向量只说明存在一个非零向量同时被两个矩阵映射到其倍数，不能推出两个矩阵的乘积相等，因此不能推出可交换。",
                "$\\text{不一定}$", "线性代数", "common-eigenvector-vs-commutativity",
            ))
    return out


def make_algebra(start: int, count: int, marker: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 3 == 0:
            coefficient = 3 + n % 8
            root = -5 + (n * 2) % 13
            constant = -coefficient * root
            out.append(rec(
                f"解方程 {coefficient}x+({constant})=0。（{marker}T-{n:04d}）",
                f"移项得 {coefficient}x={-constant}，两边除以 {coefficient}，得到 x={root}。",
                frac(root), "代数推导", "missing-coefficient-check",
            ))
        elif i % 3 == 1:
            root1 = -4 + n % 10
            root2 = 2 + (n * 3) % 9
            coefficient = -(root1 + root2)
            constant = root1 * root2
            out.append(rec(
                f"方程 x^2+({coefficient})x+{constant}=0 的两个根是什么？（{marker}T-{n:04d}）",
                f"由根与系数关系或因式分解可得 (x-{root1})(x-{root2})=0，所以两个根为 {root1} 和 {root2}。",
                f"${root1},\\ {root2}$", "代数推导", "quadratic-roots",
            ))
        else:
            a = 2 + n % 7
            b = -3 + (n * 2) % 11
            c = 4 + n % 6
            value = a * c + b
            out.append(rec(
                f"化简表达式 {a}(x+{b})-{a}x+{c}。（{marker}T-{n:04d}）",
                f"展开得 {a}x+{a * b}-{a}x+{c}，x 项抵消，常数项为 {a * b}+{c}={value}。",
                frac(value), "代数推导", "cancellation-and-constant",
            ))
    return out


def make_number_theory(start: int, count: int, marker: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 2 == 0:
            divisor = 7 + n % 12
            quotient = 8 + (n * 3) % 19
            remainder = 1 + n % (divisor - 1)
            dividend = divisor * quotient + remainder
            out.append(rec(
                f"求 {dividend} 除以 {divisor} 的余数。（{marker}T-{n:04d}）",
                f"{dividend}={divisor}×{quotient}+{remainder}，且 {remainder}<{divisor}，所以余数为 {remainder}。",
                frac(remainder), "数论与余数", "remainder",
            ))
        else:
            base = 3 + n % 8
            digits = [1 + n % (base - 1), 1 + (n * 2) % (base - 1), n % base]
            value = digits[0] * base * base + digits[1] * base + digits[2]
            digit_text = "".join(str(d) for d in digits)
            out.append(rec(
                f"将 {digit_text}（{base} 进制）转换为十进制。（{marker}T-{n:04d}）",
                f"这是 3 位 {base} 进制数，按位权展开为 {digits[0]}×{base}^2+{digits[1]}×{base}+{digits[2]}={value}。",
                frac(value), "数论与余数", "base-conversion-place-value",
            ))
    return out


def make_geometry(start: int, count: int, marker: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(count):
        n = start + i
        if i % 4 == 0:
            x1, y1 = -6 + n % 13, -5 + (n * 2) % 11
            x2, y2 = 3 + (n * 3) % 12, -4 + n % 12
            answer = (Fraction(x1 + x2, 2), Fraction(y1 + y2, 2))
            out.append(rec(
                f"点 A({x1},{y1}) 与点 B({x2},{y2}) 的中点坐标是多少？（{marker}T-{n:04d}）",
                f"中点坐标为 (({x1}+{x2})/2,({y1}+{y2})/2)=({answer[0]},{answer[1]})。",
                f"$({answer[0]},{answer[1]})$", "坐标几何", "midpoint",
            ))
        elif i % 4 == 1:
            x, y = -7 + n % 14, -6 + (n * 2) % 13
            dx, dy = -4 + n % 9, -3 + (n * 3) % 8
            answer = (x + dx, y + dy)
            out.append(rec(
                f"点 P({x},{y}) 向量平移 ({dx},{dy}) 后的像点坐标是多少？（{marker}T-{n:04d}）",
                f"平移后坐标为 ({x}+{dx},{y}+({dy}))=({answer[0]},{answer[1]})。",
                f"$({answer[0]},{answer[1]})$", "坐标几何", "translation",
            ))
        elif i % 4 == 2:
            x, y = -5 + n % 12, -4 + (n * 2) % 10
            scale = 2 + n % 4
            answer = (scale * x, scale * y)
            out.append(rec(
                f"点 P({x},{y}) 关于原点作比例因子 {scale} 的位似变换，像点坐标是多少？（{marker}T-{n:04d}）",
                f"关于原点位似时每个坐标乘以 {scale}，像点为 ({scale}×{x},{scale}×{y})=({answer[0]},{answer[1]})。",
                f"$({answer[0]},{answer[1]})$", "坐标几何", "dilation",
            ))
        else:
            x, y = -6 + n % 13, -5 + (n * 2) % 11
            dx, dy = 3 + n % 5, 4 + (n * 3) % 6
            squared = dx * dx + dy * dy
            out.append(rec(
                f"点 A({x},{y}) 与点 B({x + dx},{y + dy}) 的距离平方是多少？（{marker}T-{n:04d}）",
                f"横纵坐标差分别为 {dx} 和 {dy}，距离平方为 {dx}^2+{dy}^2={squared}。",
                frac(squared), "坐标几何", "distance-squared",
            ))
    return out


def read_questions(path: Path) -> set[str]:
    if not path.exists() or not path.is_file():
        return set()
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
        records = value if isinstance(value, list) else [value]
    except (OSError, UnicodeError, json.JSONDecodeError):
        records = []
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError):
            return set()
    questions: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("messages"), list):
            user = next((m.get("content", "") for m in item["messages"] if m.get("role") == "user"), "")
            if user:
                questions.add(_canonicalize_question(str(user)))
        elif item.get("question"):
            questions.add(_canonicalize_question(str(item["question"])))
    return questions


def main() -> None:
    corrections = (
        make_application(5000, 100, "R17")
        + make_probability(5100, 100, "R17")
        + make_linear(5200, 100, "R17")
        + make_algebra(5300, 100, "R17")
        + make_number_theory(5400, 100, "R17")
        + make_geometry(5500, 100, "R17")
    )
    validation = (
        make_application(6000, 20, "R17V")
        + make_probability(6100, 20, "R17V")
        + make_linear(6200, 20, "R17V")
        + make_algebra(6300, 20, "R17V")
        + make_number_theory(6400, 20, "R17V")
        + make_geometry(6500, 20, "R17V")
    )
    if len(corrections) != CORRECTION_COUNT or len(validation) != VALIDATION_COUNT:
        raise RuntimeError("Unexpected Round 17 correction counts")

    current_round17 = read_questions(RAW_OUT) | read_questions(VALIDATION_OUT)
    protected = _load_test_questions(TEST_FILE) | _load_regression_questions(REGRESSION_DIR)
    protected |= set().union(*(read_questions(p) for p in (ROOT / "data/eval/correction-validation").glob("*.json")))
    protected -= read_questions(VALIDATION_OUT)
    existing_raw = {
        _canonicalize_question(item["question"])
        for item in load_raw_data(RAW_DIR, regression_dir=REGRESSION_DIR)
        if item.get("question")
    }
    existing_raw -= current_round17
    new_keys = [_canonicalize_question(item["question"]) for item in corrections + validation]
    if len(set(new_keys)) != len(new_keys):
        raise RuntimeError("Round 17 contains duplicate questions")
    overlap = set(new_keys) & (protected | existing_raw)
    if overlap:
        raise RuntimeError(f"Round 17 leaks {len(overlap)} existing/protected questions")

    raw = load_raw_data(RAW_DIR, regression_dir=REGRESSION_DIR)
    clean_original: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        source = str(item.get("source", ""))
        key = _canonicalize_question(item.get("question", ""))
        if source.startswith("correction-") or key in protected or key in seen:
            continue
        seen.add(key)
        clean_original.append(item)
    if len(clean_original) < REPLAY_COUNT:
        raise RuntimeError(f"Only {len(clean_original)} clean original items available")

    rng = random.Random(SEED)
    rng.shuffle(clean_original)
    replay = clean_original[:REPLAY_COUNT]
    train_messages = [format_example(item) for item in replay]
    train_messages.extend(format_example(item) for item in corrections)
    rng.shuffle(train_messages)
    eval_messages = [format_example(item) for item in validation]

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in corrections) + "\n", encoding="utf-8")
    VALIDATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_OUT.write_text(json.dumps(eval_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "train.json").write_text(json.dumps(train_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "eval.json").write_text(json.dumps(eval_messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "round": "correction-round-17",
        "base_model": "outputs/correction-round-16-merged",
        "policy": "70% new corrections + 30% clean original replay",
        "correction_count": len(corrections),
        "replay_count": len(replay),
        "train_count": len(train_messages),
        "validation_count": len(validation),
        "correction_subjects": dict(Counter(item["subject"] for item in corrections)),
        "replay_sources": dict(Counter(str(item.get("source", "unknown")) for item in replay)),
        "protected_sets": [
            "data/eval/test.json",
            "data/eval/regression/regression.json",
            "data/eval/correction-validation/*.json",
        ],
        "checks": {
            "new_question_duplicates": 0,
            "protected_or_existing_overlap": 0,
        },
    }
    (STAGING / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGING / "README.md").write_text(
        "# Round 17 correction-focused staging\n\n"
        f"- 训练：{len(corrections)} 条新纠错题 + {len(replay)} 条清洗原始题回放，共 {len(train_messages)} 条。\n"
        f"- 独立纠错验证：{len(validation)} 条，仅用于训练过程和纠错能力检查。\n"
        "- 建议基座：`outputs/correction-round-16-merged`。\n"
        "- 原始测试集、历史回归集和所有历史纠错验证集均冻结，未参与训练。\n"
        "- 新纠错题覆盖多步应用、概率组合、线性代数、代数推导、数论余数/进制、坐标几何。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
