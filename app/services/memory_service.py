"""Model-backed conversation summary prompts."""

from __future__ import annotations


SUMMARY_SYSTEM_PROMPT = """你是数学学习助手的对话记忆整理器。

请把旧的数学对话压缩成简洁、准确、可供后续解题使用的摘要。
必须保留：原始题目、已知条件、数字和变量、使用过的公式或方法、已经得到的结论、用户不理解的地方、尚未解决的问题。
数学公式、变量、数字和单位必须原样保留，不要自行改写或推测。
删除寒暄、重复解释和与数学问题无关的内容。
请使用以下结构输出，保持简洁：

主题：
原始题目：
已知条件：
解题方法：
关键公式和计算：
已得结论：
用户疑问：
待解决问题：
"""


def build_summary_messages(
    messages: list[dict[str, str]],
    existing_summary: str | None = None,
) -> list[dict[str, str]]:
    """Build a deterministic request for summarizing old messages."""
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    if existing_summary:
        transcript = (
            "已有摘要：\n"
            f"{existing_summary.strip()}\n\n"
            "以下是需要合并进摘要的较早对话：\n"
            f"{transcript}"
        )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
