"""Build the reviewed Round 7 correction set.

This is intentionally deterministic and self-contained: 150 new training
examples (30 per subject) plus 30 separate validation examples (6 per
subject).  The validation questions use different contexts and parameters
and are never copied into the training file.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

from prepare_data import SYSTEM_PROMPT, _canonicalize_question


ROOT = Path(__file__).resolve().parents[1]
RAW_OUT = ROOT / "data/raw/corrections/correction-round-7-training.jsonl"
VAL_OUT = ROOT / "data/eval/correction-validation/round-7.json"
TEST = ROOT / "data/eval/test.json"
REGRESSION = ROOT / "data/eval/regression/regression.json"
SUBJECTS = ("应用题", "代数与数论", "概率与组合", "线性代数", "高等数学")
TRAIN_CONTEXTS = (
    "a community kitchen", "a solar farm", "a museum archive", "a ferry terminal",
    "a robotics classroom", "a public clinic", "a weather office", "a design studio",
    "a city garden", "a research workshop",
)
VALIDATION_CONTEXTS = (
    "a validation bakery", "a validation observatory", "a validation depot",
    "a validation lab", "a validation bookshop", "a validation clinic",
)


def alpha_code(value: int) -> str:
    letters = []
    for _ in range(5):
        letters.append(chr(ord("a") + value % 26))
        value //= 26
    return "".join(reversed(letters))


def frac(value: Fraction | int) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def make_record(question: str, solution: str, answer: str, subject: str, label: str) -> dict[str, str]:
    return {
        "question": question.strip(),
        "solution": solution.strip(),
        "answer": answer.strip(),
        "source": "correction-round-7",
        "subject": subject,
        "difficulty": "Level 3" if int(label[-2:]) % 3 == 0 else "Level 2",
    }


def messages(item: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": f"{item['solution']}\n\n**最终答案**\n\n{item['answer']}"},
    ]}


def item(index: int, validation: bool = False) -> dict[str, str]:
    category = index % 5
    kind = (index // 5) % 6
    n = index + (1000 if validation else 1)
    context = (VALIDATION_CONTEXTS if validation else TRAIN_CONTEXTS)[index % (6 if validation else 10)]
    label = f"R7{'V' if validation else 'T'}-{alpha_code(index)}-{index + 1:02d}"
    case_code = alpha_code(index)
    c = f"{context}, case marker {case_code} (exercise {label})"

    if category == 0:  # multi-step applications and unit/time relations
        if kind == 0:
            u, price, discount, shipping = 4 + n % 7, 9 + n % 8, 10 + n % 4 * 5, 3 + n % 6
            subtotal = u * price
            answer = subtotal * (100 - discount) // 100 + shipping
            q = f"At {c}, {u} kits cost ${price} each. A {discount}% discount is applied to the subtotal, then ${shipping} shipping is added. What is the final cost?"
            s = f"The subtotal is {u} times {price}, or ${subtotal}. The discounted subtotal is {subtotal} times {100 - discount}/100 = ${subtotal * (100 - discount) // 100}. Adding shipping gives ${answer}."
            return make_record(q, s, f"${answer}", SUBJECTS[category], label)
        if kind == 1:
            rate, hours, pause = 15 + n % 9, 3 + n % 5, 15 + n % 4 * 15
            productive = hours * 60 - pause
            answer = Fraction(rate * productive, 60)
            q = f"At {c}, a printer makes {rate} labels per hour for a {hours}-hour shift and pauses for {pause} minutes. How many labels are printed?"
            s = f"Productive time is {productive}/60 = {frac(Fraction(productive, 60))} hours. Therefore the number printed is {rate} times {frac(Fraction(productive, 60))} = {frac(answer)}."
            return make_record(q, s, frac(answer), SUBJECTS[category], label)
        if kind == 2:
            a, b, part = 3 + n % 5, 4 + n % 6, 7 + n % 8
            total, answer = (a + b) * part, b * part
            q = f"At {c}, ${total} is divided between two teams in the ratio {a}:{b}. How much does the second team receive?"
            s = f"There are {a + b} total ratio parts, so one part is ${total}/{a + b} = ${part}. The second team receives {b} times ${part} = ${answer}."
            return make_record(q, s, f"${answer}", SUBJECTS[category], label)
        if kind == 3:
            speed, duration, stop = 18 + n % 8, 2 + n % 4, 10 + n % 5 * 5
            minutes = duration * 60 + stop
            h, m = divmod(minutes, 60)
            q = f"At {c}, a vehicle travels {speed * duration} km at {speed} km/h and stops for {stop} minutes. How long is the complete trip?"
            s = f"Driving takes {speed * duration}/{speed} = {duration} hours. Total time is {duration} times 60 + {stop} = {minutes} minutes, which is {h} hours {m} minutes."
            return make_record(q, s, f"{h} hours {m} minutes", SUBJECTS[category], label)
        if kind == 4:
            workers, hours, days = 2 + n % 5, 3 + n % 4, 2 + n % 3
            answer = workers * hours * days
            q = f"At {c}, {workers} workers each process {hours} units per day for {days} days. How many units are processed in total?"
            s = f"Multiply workers, daily units, and days: {workers} times {hours} times {days} = {answer}."
            return make_record(q, s, str(answer), SUBJECTS[category], label)
        base, add, days = 120 + n % 8 * 10, 15 + n % 5 * 5, 3 + n % 4
        answer = base + add * days
        q = f"At {c}, a tank starts with {base} liters and receives {add} liters each day for {days} days. How many liters are in it afterward?"
        s = f"The added amount is {add} times {days} = {add * days} liters. Therefore the final amount is {base} + {add * days} = {answer} liters."
        return make_record(q, s, str(answer), SUBJECTS[category], label)

    if category == 1:  # algebra and number theory
        if kind == 0:
            a, x, b = 3 + n % 7, 2 + n % 9, 4 + n % 8
            rhs = a * x + b
            q = f"For {c}, solve {a}x + {b} = {rhs}."
            s = f"Subtract {b}: {a}x = {rhs - b}. Divide by {a}: x = {x}."
            return make_record(q, s, str(x), SUBJECTS[category], label)
        if kind == 1:
            r1, r2 = 2 + n % 8, 5 + n % 7
            answer = f"x^2-{r1 + r2}x+{r1 * r2}=0"
            q = f"For {c}, find the monic quadratic equation whose roots are {r1} and {r2}."
            s = f"Use (x - {r1})(x - {r2}) = 0. Expanding gives x^2 - {r1 + r2}x + {r1 * r2} = 0."
            return make_record(q, s, answer, SUBJECTS[category], label)
        if kind == 2:
            first, d, term = 4 + n % 7, 3 + n % 5, 6 + n % 8
            answer = first + (term - 1) * d
            q = f"For {c}, an arithmetic sequence has first term {first} and common difference {d}. Find its term {term}."
            s = f"Use a_n = a_1 + (n - 1)d. Thus a_{term} = {first} + ({term} - 1) times {d} = {answer}."
            return make_record(q, s, str(answer), SUBJECTS[category], label)
        if kind == 3:
            a, r, terms = 2 + n % 6, 3, 3 + n % 5
            answer = a * (r ** terms - 1) // (r - 1)
            q = f"For {c}, find the sum of the first {terms} terms of a geometric sequence with first term {a} and ratio {r}."
            s = f"S_n = a(r^n - 1)/(r - 1), so S_{terms} = {a}({r}^{terms} - 1)/({r} - 1) = {answer}."
            return make_record(q, s, str(answer), SUBJECTS[category], label)
        if kind == 4:
            base, exponent, modulus = 2 + n % 5, 3 + n % 6, 11
            answer = pow(base, exponent, modulus)
            q = f"For {c}, find the remainder when {base}^{exponent} is divided by {modulus}."
            s = f"{base}^{exponent} = {base ** exponent} = {modulus} times {(base ** exponent - answer) // modulus} + {answer}, so the remainder is {answer}."
            return make_record(q, s, str(answer), SUBJECTS[category], label)
        a, b = 6 + n % 7, 4 + n % 6
        answer = math.gcd(a, b)
        q = f"For {c}, determine gcd({a}, {b}) using the Euclidean algorithm."
        s = f"Repeated division gives gcd({a}, {b}) = {answer}."
        return make_record(q, s, str(answer), SUBJECTS[category], label)

    if category == 2:  # probability and combinatorics
        if kind == 0:
            trials, successes = 5 + n % 4, 2 + n % 3
            ans = Fraction(math.comb(trials, successes), 2 ** trials)
            q = f"In {c}, {trials} fair coins are tossed. What is the probability of exactly {successes} heads?"
            s = f"There are C({trials},{successes}) = {math.comb(trials, successes)} favorable outcomes out of 2^{trials} = {2 ** trials}. Thus the probability is {math.comb(trials, successes)}/{2 ** trials} = {frac(ans)}."
            return make_record(q, s, frac(ans), SUBJECTS[category], label)
        if kind == 1:
            people, committee = 8 + n % 6, 3 + n % 4
            ans = math.comb(people, committee)
            q = f"In {c}, how many committees of {committee} people can be selected from {people} distinct people?"
            s = f"Order does not matter, so the count is C({people},{committee}) = {ans}."
            return make_record(q, s, str(ans), SUBJECTS[category], label)
        if kind == 2:
            target = 5 + n % 7
            favorable = sum(a + b == target for a in range(1, 7) for b in range(1, 7))
            ans = Fraction(favorable, 36)
            q = f"In {c}, two fair dice are rolled. What is the probability that their sum is {target}?"
            s = f"There are {favorable} ordered pairs with this sum among 36 equally likely outcomes, so the probability is {favorable}/36 = {frac(ans)}."
            return make_record(q, s, frac(ans), SUBJECTS[category], label)
        if kind == 3:
            trials = 4 + n % 5
            ans = Fraction(2 ** trials - 1, 2 ** trials)
            q = f"In {c}, each of {trials} independent trials succeeds with probability 1/2. What is the probability of at least one success?"
            s = f"Use the complement: P(no success) = (1/2)^{trials} = {frac(Fraction(1, 2 ** trials))}. Therefore P(at least one) = 1 - that value = {frac(ans)}."
            return make_record(q, s, frac(ans), SUBJECTS[category], label)
        if kind == 4:
            objects = 4 + n % 5
            ans = math.factorial(objects)
            q = f"In {c}, in how many orders can {objects} distinct exhibits be arranged?"
            s = f"All objects are distinct, so the number of orders is {objects}! = {ans}."
            return make_record(q, s, str(ans), SUBJECTS[category], label)
        red, blue, draw = 3 + n % 5, 5 + n % 4, 2 + n % 3
        favorable = math.comb(red, 1) * math.comb(blue, draw - 1)
        total = math.comb(red + blue, draw)
        ans = Fraction(favorable, total)
        q = f"In {c}, a box has {red} red and {blue} blue tokens. If {draw} are drawn without replacement, what is the probability of exactly one red token?"
        s = f"Favorable selections are C({red},1)C({blue},{draw - 1}) = {favorable}; all selections are C({red + blue},{draw}) = {total}. The probability is {favorable}/{total} = {frac(ans)}."
        return make_record(q, s, frac(ans), SUBJECTS[category], label)

    if category == 3:  # linear algebra
        if kind == 0:
            a, b, d = 2 + n % 7, 1 + n % 5, 3 + n % 6
            ans = a * d
            q = f"For {c}, find the determinant of the diagonal matrix [[{a},0],[0,{d}]]."
            s = f"The determinant of a diagonal 2 by 2 matrix is the product of its diagonal entries: {a} times {d} = {ans}."
            return make_record(q, s, str(ans), SUBJECTS[category], label)
        if kind == 1:
            a, b, c2, d, x, y = 2 + n % 5, 1 + n % 4, 1 + n % 3, 3 + n % 5, 2 + n % 4, 1 + n % 5
            u, v = a * x + b * y, c2 * x + d * y
            q = f"For {c}, compute [[{a},{b}],[{c2},{d}]] times [{x},{y}]^T."
            s = f"Multiply each row by the column: first entry {a} times {x} + {b} times {y} = {u}; second entry {c2} times {x} + {d} times {y} = {v}."
            return make_record(q, s, f"({u},{v})^T", SUBJECTS[category], label)
        if kind == 2:
            x, y = 2 + n % 5, 3 + n % 4
            s1, s2 = 2 * x + y, x - y
            q = f"For {c}, solve 2x + y = {s1} and x - y = {s2}."
            s = f"From x - y = {s2}, x = y + {s2}. Substitution gives 3y = {s1 - 2 * s2}, so y = {y} and x = {x}."
            return make_record(q, s, f"(x,y)=({x},{y})", SUBJECTS[category], label)
        if kind == 3:
            p, qv = 2 + n % 8, 4 + n % 7
            q = f"For {c}, find the eigenvalues of the diagonal matrix [[{p},0],[0,{qv}]]."
            s = f"For a diagonal matrix, the diagonal entries are the eigenvalues. They are {p} and {qv}."
            return make_record(q, s, f"{p}, {qv}", SUBJECTS[category], label)
        if kind == 4:
            ax, ay, bx, by = 2 + n % 5, 1 + n % 4, 1 + n % 6, 3 + n % 5
            ans = abs(ax * by - ay * bx)
            q = f"For {c}, find the area of the parallelogram spanned by vectors ({ax},{ay}) and ({bx},{by})."
            s = f"The area is the absolute value of the determinant: |{ax} times {by} - {ay} times {bx}| = {ans}."
            return make_record(q, s, str(ans), SUBJECTS[category], label)
        x1, y1, x2, y2 = 1 + n % 5, 2 + n % 4, 5 + n % 6, 7 + n % 5
        ans = Fraction(y2 - y1, x2 - x1)
        q = f"For {c}, find the slope through ({x1},{y1}) and ({x2},{y2})."
        s = f"The slope is (y2-y1)/(x2-x1) = ({y2}-{y1})/({x2}-{x1}) = {frac(ans)}."
        return make_record(q, s, frac(ans), SUBJECTS[category], label)

    # calculus and multi-step symbolic reasoning
    if kind == 0:
        a, power, x0 = 1 + n % 5, 2 + n % 4, 1 + n % 6
        ans = a * power * x0 ** (power - 1) + 3
        q = f"For {c}, if f(x) = {a}x^{power} + 3x - 2, find f'({x0})."
        s = f"f'(x) = {a * power}x^({power - 1}) + 3, so f'({x0}) = {a * power} times {x0}^{power - 1} + 3 = {ans}."
        return make_record(q, s, str(ans), SUBJECTS[category], label)
    if kind == 1:
        power, upper = 1 + n % 4, 2 + n % 5
        ans = Fraction(upper ** (power + 1), power + 1)
        q = f"For {c}, evaluate the integral from 0 to {upper} of x^{power} dx."
        s = f"An antiderivative is x^({power + 1})/({power + 1}); evaluating gives {upper}^{power + 1}/({power + 1}) = {frac(ans)}."
        return make_record(q, s, frac(ans), SUBJECTS[category], label)
    if kind == 2:
        a = 2 + n % 8
        q = f"For {c}, evaluate the limit as x approaches {a} of (x^2 - {a * a})/(x - {a})."
        s = f"Factor the numerator: (x^2 - {a * a})/(x - {a}) = x + {a} for x not equal to {a}. The limit is {2 * a}."
        return make_record(q, s, str(2 * a), SUBJECTS[category], label)
    if kind == 3:
        first = 2 + n % 5
        ans = Fraction(first, 1 - Fraction(1, 2))
        q = f"For {c}, find the sum {first} + {first}/2 + {first}/4 + ... ."
        s = f"This is geometric with a = {first}, r = 1/2, and |r| < 1. Thus S = a/(1-r) = {first}/(1/2) = {frac(ans)}."
        return make_record(q, s, frac(ans), SUBJECTS[category], label)
    if kind == 4:
        k, y0 = 1 + n % 4, 2 + n % 5
        ans = f"{y0}e^({k}t)"
        q = f"For {c}, solve y' = {k}y with y(0) = {y0}, giving y(t)."
        s = f"Separating and integrating gives y = Ce^({k}t). Since y(0) = {y0}, C = {y0}; therefore y(t) = {ans}."
        return make_record(q, s, ans, SUBJECTS[category], label)
    terms = 4 + n % 6
    ans = terms * (terms + 1) // 2
    q = f"For {c}, find 1 + 2 + 3 + ... + {terms}."
    s = f"The sum of the first n positive integers is n(n+1)/2, so the result is {terms} times {terms + 1}/2 = {ans}."
    return make_record(q, s, str(ans), SUBJECTS[category], label)


def protected_questions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value if isinstance(value, list) else value.get("data", [])
    questions = set()
    for record in records:
        messages_value = record.get("messages", [])
        if isinstance(messages_value, list) and len(messages_value) > 1:
            questions.add(_canonicalize_question(messages_value[1].get("content", "")))
    return questions


def main() -> None:
    training = [item(i) for i in range(150)]
    validation = [messages(item(i, validation=True)) for i in range(30)]
    protected = protected_questions(TEST) | protected_questions(REGRESSION)
    questions = [_canonicalize_question(x["question"]) for x in training]
    questions += [_canonicalize_question(x["messages"][1]["content"]) for x in validation]
    if len(questions) != len(set(questions)):
        raise RuntimeError("Round 7 contains duplicate questions")
    if set(questions) & protected:
        raise RuntimeError("Round 7 leaks a protected test/regression question")
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in training) + "\n", encoding="utf-8")
    VAL_OUT.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"training={len(training)} validation={len(validation)} protected_leakage=0")
    print(f"training_file={RAW_OUT}")
    print(f"validation_file={VAL_OUT}")


if __name__ == "__main__":
    main()
