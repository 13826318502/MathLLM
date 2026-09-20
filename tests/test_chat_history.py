from __future__ import annotations

import unittest

from app.api.models import ChatMessage
from app.services.chat_service import build_history


class BuildHistoryTest(unittest.TestCase):
    def test_empty_returns_empty(self) -> None:
        self.assertEqual(build_history([], 20, 24000), [])

    def test_returns_turns_without_system_prompt(self) -> None:
        messages = [
            ChatMessage(role="user", content="解方程 x^2-5x+6=0"),
            ChatMessage(role="assistant", content="x=2 或 x=3"),
        ]
        history = build_history(messages, 20, 24000)
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "解方程 x^2-5x+6=0"},
                {"role": "assistant", "content": "x=2 或 x=3"},
            ],
        )

    def test_too_many_messages_rejected(self) -> None:
        messages = [ChatMessage(role="user", content="a")] * 3
        with self.assertRaises(ValueError):
            build_history(messages, 2, 24000)

    def test_char_budget_rejected(self) -> None:
        messages = [ChatMessage(role="user", content="a" * 100)]
        with self.assertRaises(ValueError):
            build_history(messages, 20, 10)


if __name__ == "__main__":
    unittest.main()
