from __future__ import annotations

import unittest

from app.api.models import ChatMessage
from app.frontend.memory import estimate_tokens, should_summarize, split_history
from app.services.chat_service import build_chat_messages


class MemoryHelpersTest(unittest.TestCase):
    def test_split_keeps_recent_user_turn(self) -> None:
        messages = [
            {"role": role, "content": f"消息 {index}"}
            for index, role in enumerate(["user", "assistant"] * 4)
        ]
        old, recent = split_history(
            messages,
            max_recent_messages=4,
            max_recent_tokens=200,
        )
        self.assertEqual(len(old), 4)
        self.assertEqual(len(recent), 4)
        self.assertEqual(recent[0]["role"], "user")
        self.assertEqual(recent[-1]["role"], "assistant")

    def test_summary_trigger_requires_old_history(self) -> None:
        messages = [
            {"role": "user", "content": "x" * 600},
            {"role": "assistant", "content": "y" * 600},
            {"role": "user", "content": "z"},
            {"role": "assistant", "content": "answer"},
        ]
        self.assertGreater(estimate_tokens(messages[0]["content"]), 200)
        self.assertTrue(should_summarize("", messages, 300, 2))

    def test_summary_is_sent_as_system_context(self) -> None:
        result = build_chat_messages(
            [ChatMessage(role="user", content="继续解释第二步")],
            max_messages=6,
            max_chars=4000,
            summary="用户正在学习一元二次方程。",
        )
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertIn("一元二次方程", result[0]["content"])


if __name__ == "__main__":
    unittest.main()
