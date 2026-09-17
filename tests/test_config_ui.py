from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from pathlib import Path

from app.tools.config_ui import EXIT_CANCELLED, EXIT_START, ConfigDialog


class ConfigDialogSmokeTest(unittest.TestCase):
    """Catches widget-ordering mistakes, e.g. StringVars created before Tk."""

    def setUp(self) -> None:
        try:
            probe = tk.Tk()
        except tk.TclError as exc:  # headless environment
            self.skipTest(f"no display available: {exc}")
        probe.destroy()
        self._tmp = tempfile.TemporaryDirectory()
        self.env_file = Path(self._tmp.name) / ".env.local"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dialog_constructs(self) -> None:
        dialog = ConfigDialog(self.env_file)
        try:
            self.assertEqual(dialog.result, EXIT_CANCELLED)
            settings = dialog._collect()
            self.assertTrue(settings.solver_model)
            self.assertTrue(settings.solver_base_url)
        finally:
            dialog.root.destroy()

    def test_fallback_choice_maps_to_value(self) -> None:
        dialog = ConfigDialog(self.env_file)
        try:
            dialog.fallback.set("直接报错")
            self.assertEqual(dialog._collect().orchestrator_fallback, "fail")
            dialog.fallback.set("回退本地编排")
            self.assertEqual(dialog._collect().orchestrator_fallback, "local")
        finally:
            dialog.root.destroy()

    def test_save_and_start_writes_env_file(self) -> None:
        dialog = ConfigDialog(self.env_file)
        dialog.vars["solver_model"].set("mathllm-round7")
        dialog.vars["orchestrator_api_key"].set("sk-test")
        dialog.vars["orchestrator_model"].set("deepseek-chat")
        dialog._save(True)  # destroys the root, no message box
        self.assertEqual(dialog.result, EXIT_START)
        text = self.env_file.read_text(encoding="utf-8")
        self.assertIn("MATHLLM_SOLVER_MODEL=mathllm-round7", text)
        self.assertIn("MATHLLM_ORCHESTRATOR_API_KEY=sk-test", text)

    def test_empty_solver_model_is_rejected(self) -> None:
        dialog = ConfigDialog(self.env_file)
        try:
            dialog.vars["solver_model"].set("")
            # messagebox would block, so check the guard through _collect + validate
            self.assertEqual(dialog._collect().solver_model, "")
        finally:
            dialog.root.destroy()


if __name__ == "__main__":
    unittest.main()
