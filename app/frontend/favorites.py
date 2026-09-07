"""Browser-local favorite-question helpers for the frontend."""

from __future__ import annotations

from collections.abc import Sequence

import gradio as gr


MAX_FAVORITES = 50


def _clean_favorites(values: Sequence[str] | None) -> list[str]:
    """Normalize stored favorites while preserving their order."""
    result: list[str] = []
    for value in values or []:
        question = str(value).strip()
        if question and question not in result:
            result.append(question)
    return result[:MAX_FAVORITES]


def render_favorites(values: Sequence[str] | None) -> str:
    """Render the compact Markdown summary shown in the sidebar."""
    favorites = _clean_favorites(values)
    if not favorites:
        return "还没有收藏题目。\n\n输入题目后点击 **收藏当前题目**，方便之后继续练习。"

    lines = [f"已收藏 **{len(favorites)}** 道题", ""]
    for index, question in enumerate(favorites, start=1):
        one_line = " ".join(question.split())
        lines.append(f"{index}. {one_line}")
    return "\n".join(lines)


def _dropdown_update(values: Sequence[str] | None) -> dict:
    favorites = _clean_favorites(values)
    return gr.update(choices=favorites, value=None)


def sync_favorites(values: Sequence[str] | None) -> tuple[str, dict]:
    """Synchronize the browser state with the sidebar components."""
    favorites = _clean_favorites(values)
    return render_favorites(favorites), _dropdown_update(favorites)


def add_favorite(
    question: str | None,
    values: Sequence[str] | None,
) -> tuple[list[str], str, dict, str]:
    """Add the current question to browser-local favorites."""
    favorites = _clean_favorites(values)
    question = (question or "").strip()
    if not question:
        return favorites, render_favorites(favorites), _dropdown_update(favorites), "请先输入一道题目。"
    if question in favorites:
        return favorites, render_favorites(favorites), _dropdown_update(favorites), "这道题已经收藏过了。"

    updated = [question, *favorites][:MAX_FAVORITES]
    return (
        updated,
        render_favorites(updated),
        _dropdown_update(updated),
        "已收藏当前题目。",
    )


def remove_favorite(
    selected: str | None,
    values: Sequence[str] | None,
) -> tuple[list[str], str, dict, str]:
    """Remove the selected question from browser-local favorites."""
    favorites = _clean_favorites(values)
    if not selected or selected not in favorites:
        return favorites, render_favorites(favorites), _dropdown_update(favorites), "请选择要删除的收藏题目。"

    updated = [question for question in favorites if question != selected]
    return (
        updated,
        render_favorites(updated),
        _dropdown_update(updated),
        "已移除这道收藏题目。",
    )


def clear_favorites() -> tuple[list[str], str, dict, str]:
    """Clear all browser-local favorites."""
    return [], render_favorites([]), _dropdown_update([]), "收藏已清空。"


def load_favorite(selected: str | None) -> str:
    """Put a selected favorite back into the question input."""
    return (selected or "").strip()


def append_instruction(question: str | None, instruction: str) -> str:
    """Add a learning-mode instruction without replacing the question."""
    question = (question or "").strip()
    if not question:
        return instruction
    if instruction in question:
        return question
    return f"{question}\n\n{instruction}"
