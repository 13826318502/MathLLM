"""Chat-message validation and prompt assembly."""

from __future__ import annotations

from app.api.models import ChatMessage
from app.core.prompts import SYSTEM_PROMPT


def _clean_content(content: str) -> str:
    return " ".join(content.split()).strip()


def build_solve_messages(question: str, max_chars: int) -> list[dict[str, str]]:
    question = question.strip()
    if not question:
        raise ValueError("题目不能为空")
    if len(question) > max_chars:
        raise ValueError(f"题目长度不能超过 {max_chars} 个字符")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_chat_messages(
    messages: list[ChatMessage],
    max_messages: int,
    max_chars: int,
    summary: str | None = None,
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("对话消息不能为空")
    if len(messages) > max_messages:
        raise ValueError(f"对话最多保留 {max_messages} 条消息")
    if messages[-1].role != "user":
        raise ValueError("最后一条消息必须来自用户")

    system_content = SYSTEM_PROMPT
    total_chars = 0
    summary_content = _clean_content(summary or "")
    if summary_content:
        total_chars += len(summary_content)
        if total_chars > max_chars:
            raise ValueError(f"对话内容不能超过 {max_chars} 个字符")
        system_content += (
            "\n\n以下是此前对话的摘要，仅在有助于回答当前问题时参考：\n"
            f"{summary_content}"
        )
    result: list[dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]
    for message in messages:
        content = _clean_content(message.content)
        if not content:
            raise ValueError("消息内容不能为空")
        total_chars += len(content)
        if total_chars > max_chars:
            raise ValueError(f"对话内容不能超过 {max_chars} 个字符")
        result.append({"role": message.role, "content": content})
    return result
