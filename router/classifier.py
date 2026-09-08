"""TF-IDF based math/non-math router with a safe rule fallback.

The router is deliberately separate from the language model. It only chooses
which system prompt should be sent to the same local model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "router" / "tfidf_router.joblib"
)

MATH_HINTS = (
    "求解",
    "计算",
    "方程",
    "不等式",
    "概率",
    "组合",
    "排列",
    "面积",
    "体积",
    "导数",
    "积分",
    "极限",
    "矩阵",
    "向量",
    "特征值",
    "函数",
    "数列",
    "余数",
    "因式分解",
    "证明",
    "solve",
    "calculate",
    "equation",
    "probability",
    "derivative",
    "integral",
    "matrix",
    "vector",
    "algebra",
    "geometry",
)

NON_MATH_HINTS = (
    "请假邮件",
    "邮件",
    "天气",
    "人工智能",
    "机器学习",
    "云计算",
    "翻译",
    "简历",
    "会议纪要",
    "学习计划",
    "科幻电影",
    "睡前故事",
    "旅行",
    "工作效率",
    "网站首页",
    "请介绍",
    "请解释",
)

MATH_PATTERNS = (
    re.compile(r"\d\s*[+\-*/×÷=<>≤≥^]\s*\d"),
    re.compile(r"[xyzt]\s*(?:\^|²|³)"),
    re.compile(r"(?:sin|cos|tan|log|ln|sqrt|frac)\b", re.IGNORECASE),
    re.compile(r"\\(?:frac|sum|int|lim|sqrt|begin)\b"),
)


@dataclass(frozen=True)
class RouteResult:
    """A routing decision returned to the prompt assembly layer."""

    route: str
    confidence: float
    math_probability: float
    source: str


def _model_path() -> Path:
    configured = os.getenv("MATHLLM_ROUTER_MODEL")
    return Path(configured) if configured else DEFAULT_MODEL_PATH


def _load_model() -> Any | None:
    path = _model_path()
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except (ImportError, OSError, ValueError):
        # The application can still start before the optional classifier is
        # trained or when only the base requirements are installed.
        return None


def _rule_fallback(text: str) -> RouteResult:
    normalized = text.strip().lower()
    keyword_hits = sum(1 for hint in MATH_HINTS if hint.lower() in normalized)
    non_math_hits = sum(
        1 for hint in NON_MATH_HINTS if hint.lower() in normalized
    )
    pattern_hits = sum(1 for pattern in MATH_PATTERNS if pattern.search(normalized))
    score = keyword_hits * 0.22 + pattern_hits * 0.30

    if non_math_hits >= 1 and keyword_hits == 0 and pattern_hits == 0:
        probability = 0.06
        route = "non_math"
    elif pattern_hits >= 1 or keyword_hits >= 2:
        probability = min(0.96, 0.72 + score)
        route = "math" if probability >= 0.80 else "uncertain"
    elif keyword_hits == 1:
        probability = 0.58
        route = "uncertain"
    else:
        probability = 0.08
        route = "non_math"

    return RouteResult(
        route=route,
        confidence=max(probability, 1.0 - probability),
        math_probability=probability,
        source="rules",
    )


def _to_result(model: Any, text: str) -> RouteResult:
    probabilities = model.predict_proba([text])[0]
    classes = [str(value) for value in model.classes_]
    scores = dict(zip(classes, probabilities))
    math_probability = float(scores.get("math", 0.0))

    if math_probability >= 0.80:
        route = "math"
    elif math_probability <= 0.20:
        route = "non_math"
    else:
        route = "uncertain"

    return RouteResult(
        route=route,
        confidence=max(math_probability, 1.0 - math_probability),
        math_probability=math_probability,
        source="tfidf",
    )


_ROUTER = None
_ROUTER_LOADED = False


def route_text(text: str) -> RouteResult:
    """Classify text as math, non_math, or uncertain.

    A trained TF-IDF pipeline is preferred. Until it exists, the deterministic
    fallback keeps the application usable and makes the routing behavior
    observable during setup.
    """

    global _ROUTER, _ROUTER_LOADED
    if not _ROUTER_LOADED:
        _ROUTER = _load_model()
        _ROUTER_LOADED = True

    rule_result = _rule_fallback(text)
    if rule_result.route in {"math", "non_math"}:
        return rule_result

    if _ROUTER is None:
        return rule_result

    try:
        return _to_result(_ROUTER, text)
    except (AttributeError, TypeError, ValueError):
        return _rule_fallback(text)
