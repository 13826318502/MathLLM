"""Inspect route decisions from the command line."""

from __future__ import annotations

import argparse
import json

from .classifier import route_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="用户输入")
    args = parser.parse_args()
    result = route_text(" ".join(args.text))
    print(
        json.dumps(
            {
                "route": result.route,
                "confidence": round(result.confidence, 4),
                "math_probability": round(result.math_probability, 4),
                "source": result.source,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
