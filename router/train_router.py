"""Train the lightweight TF-IDF math/non-math router.

Example:
    python router/train_router.py \
      --input router/data/train.jsonl \
      --output outputs/router/tfidf_router.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        text = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).strip()
        if not text or label not in {"math", "non_math"}:
            raise ValueError(f"Invalid record at {path}:{line_number}")
        texts.append(text)
        labels.append(label)
    if len(set(labels)) != 2:
        raise ValueError("Training data must contain both math and non_math labels")
    return texts, labels


def build_pipeline() -> Pipeline:
    # Character n-grams work better for Chinese than whitespace word tokens.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    args = parser.parse_args()

    texts, labels = load_jsonl(args.input)
    train_x, eval_x, train_y, eval_y = train_test_split(
        texts,
        labels,
        test_size=args.eval_ratio,
        random_state=42,
        stratify=labels,
    )

    router = build_pipeline()
    router.fit(train_x, train_y)
    predictions = router.predict(eval_x)
    print(classification_report(eval_y, predictions, digits=4))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(router, args.output)
    print(f"saved router: {args.output}")
    print(f"samples: total={len(texts)}, train={len(train_x)}, eval={len(eval_x)}")


if __name__ == "__main__":
    main()
