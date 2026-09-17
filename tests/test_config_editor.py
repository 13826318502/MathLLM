from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config_editor import (
    ModelSettings,
    read_model_settings,
    render_env,
    write_model_settings,
)


class ReadTest(unittest.TestCase):
    def test_missing_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = read_model_settings(Path(tmp) / "nope.env")
        self.assertEqual(settings.solver_model, "mathllm-round7")
        self.assertFalse(settings.orchestrator_enabled)

    def test_reads_new_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.local"
            path.write_text(
                "MATHLLM_ORCHESTRATOR_BASE_URL=https://api.deepseek.com/v1\n"
                "MATHLLM_ORCHESTRATOR_MODEL=deepseek-chat\n"
                "MATHLLM_ORCHESTRATOR_API_KEY=sk-test\n"
                "MATHLLM_SOLVER_MODEL=mathllm-round7\n",
                encoding="utf-8",
            )
            settings = read_model_settings(path)
        self.assertTrue(settings.orchestrator_enabled)
        self.assertEqual(settings.orchestrator_api_key, "sk-test")
        self.assertEqual(settings.solver_model, "mathllm-round7")

    def test_reads_legacy_keys_as_solver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.local"
            path.write_text(
                "MATHLLM_API_BASE_URL=http://127.0.0.1:11434/v1\n"
                "MATHLLM_MODEL_NAME=legacy-model\n"
                "MATHLLM_API_KEY=ollama\n",
                encoding="utf-8",
            )
            settings = read_model_settings(path)
        self.assertEqual(settings.solver_model, "legacy-model")
        self.assertEqual(settings.solver_api_key, "ollama")

    def test_quoted_values_are_unwrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.local"
            path.write_text('MATHLLM_SOLVER_API_KEY="ollama"\n', encoding="utf-8")
            settings = read_model_settings(path)
        self.assertEqual(settings.solver_api_key, "ollama")


class RenderTest(unittest.TestCase):
    def test_preserves_unrelated_lines(self) -> None:
        existing = (
            "# 端口\n"
            "MATHLLM_API_PORT=8080\n"
            "\n"
            "MATHLLM_SOLVER_MODEL=old-model\n"
        )
        rendered = render_env(existing, ModelSettings(solver_model="new-model"))
        self.assertIn("# 端口", rendered)
        self.assertIn("MATHLLM_API_PORT=8080", rendered)
        self.assertIn("MATHLLM_SOLVER_MODEL=new-model", rendered)
        self.assertNotIn("old-model", rendered)

    def test_appends_missing_keys(self) -> None:
        rendered = render_env("", ModelSettings(solver_model="m"))
        self.assertIn("MATHLLM_SOLVER_MODEL=m", rendered)
        self.assertIn("MATHLLM_ORCHESTRATOR_FALLBACK=local", rendered)
        self.assertIn("# 模型配置", rendered)

    def test_empty_api_key_disables_orchestrator(self) -> None:
        settings = ModelSettings(
            orchestrator_api_key="",
            orchestrator_base_url="https://api.deepseek.com/v1",
            orchestrator_model="deepseek-chat",
        )
        rendered = render_env("", settings)
        self.assertIn("MATHLLM_ORCHESTRATOR_BASE_URL=\n", rendered)
        self.assertIn("MATHLLM_ORCHESTRATOR_MODEL=\n", rendered)

    def test_enabled_api_key_writes_endpoint(self) -> None:
        settings = ModelSettings(
            orchestrator_api_key="sk-x",
            orchestrator_base_url="https://api.deepseek.com/v1",
            orchestrator_model="deepseek-chat",
        )
        rendered = render_env("", settings)
        self.assertIn("MATHLLM_ORCHESTRATOR_BASE_URL=https://api.deepseek.com/v1", rendered)
        self.assertIn("MATHLLM_ORCHESTRATOR_MODEL=deepseek-chat", rendered)


class RoundTripTest(unittest.TestCase):
    def test_write_then_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.local"
            path.write_text("MATHLLM_API_PORT=8080\n", encoding="utf-8")
            original = ModelSettings(
                orchestrator_api_key="sk-abc",
                orchestrator_model="deepseek-chat",
                solver_model="mathllm-round7",
                solver_api_key="ollama",
            )
            write_model_settings(path, original)
            reloaded = read_model_settings(path)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(reloaded.orchestrator_api_key, "sk-abc")
        self.assertEqual(reloaded.solver_model, "mathllm-round7")
        self.assertIn("MATHLLM_API_PORT=8080", text)
        self.assertFalse(text.startswith("\ufeff"), "must be written without BOM")

    def test_write_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.local"
            settings = ModelSettings(orchestrator_api_key="sk-abc")
            write_model_settings(path, settings)
            first = path.read_text(encoding="utf-8")
            write_model_settings(path, settings)
            second = path.read_text(encoding="utf-8")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
