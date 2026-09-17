"""Read and write the model configuration stored in ``.env.local``.

Deliberately free of any GUI so it can be unit tested; the tkinter dialog is a
thin shell on top of it. Only the model keys are touched, every other line of
the file (comments, ports, RAG settings) is preserved verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV_FILE = ".env.local"

ORCHESTRATOR_BASE_URL = "MATHLLM_ORCHESTRATOR_BASE_URL"
ORCHESTRATOR_MODEL = "MATHLLM_ORCHESTRATOR_MODEL"
ORCHESTRATOR_API_KEY = "MATHLLM_ORCHESTRATOR_API_KEY"
ORCHESTRATOR_FALLBACK = "MATHLLM_ORCHESTRATOR_FALLBACK"
SOLVER_BASE_URL = "MATHLLM_SOLVER_BASE_URL"
SOLVER_MODEL = "MATHLLM_SOLVER_MODEL"
SOLVER_API_KEY = "MATHLLM_SOLVER_API_KEY"

# Legacy names, still read so an existing file pre-fills correctly.
_LEGACY_SOLVER_BASE_URL = "MATHLLM_API_BASE_URL"
_LEGACY_SOLVER_MODEL = "MATHLLM_MODEL_NAME"
_LEGACY_SOLVER_API_KEY = "MATHLLM_API_KEY"

_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


@dataclass
class ModelSettings:
    """The editable subset of ``.env.local``."""

    orchestrator_base_url: str = "https://api.deepseek.com/v1"
    orchestrator_model: str = "deepseek-chat"
    orchestrator_api_key: str = ""
    orchestrator_fallback: str = "local"
    solver_base_url: str = "http://127.0.0.1:11434/v1"
    solver_model: str = "mathllm-round7"
    solver_api_key: str = "ollama"

    @property
    def orchestrator_enabled(self) -> bool:
        """An empty API key means 'no cloud orchestrator, stay fully local'."""
        return bool(self.orchestrator_api_key.strip())


def read_env(path: Path | str) -> dict[str, str]:
    """Parse simple ``KEY=value`` assignments, skipping comments."""
    file = Path(path)
    values: dict[str, str] = {}
    if not file.exists():
        return values
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT.match(stripped)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return values


def read_model_settings(path: Path | str = DEFAULT_ENV_FILE) -> ModelSettings:
    env = read_env(path)
    defaults = ModelSettings()
    return ModelSettings(
        orchestrator_base_url=env.get(
            ORCHESTRATOR_BASE_URL, defaults.orchestrator_base_url
        ),
        orchestrator_model=env.get(ORCHESTRATOR_MODEL, defaults.orchestrator_model),
        orchestrator_api_key=env.get(ORCHESTRATOR_API_KEY, ""),
        orchestrator_fallback=env.get(ORCHESTRATOR_FALLBACK, "local") or "local",
        solver_base_url=(
            env.get(SOLVER_BASE_URL)
            or env.get(_LEGACY_SOLVER_BASE_URL)
            or defaults.solver_base_url
        ),
        solver_model=(
            env.get(SOLVER_MODEL)
            or env.get(_LEGACY_SOLVER_MODEL)
            or defaults.solver_model
        ),
        solver_api_key=(
            env.get(SOLVER_API_KEY)
            or env.get(_LEGACY_SOLVER_API_KEY)
            or defaults.solver_api_key
        ),
    )


def _updates(settings: ModelSettings) -> dict[str, str]:
    enabled = settings.orchestrator_enabled
    return {
        ORCHESTRATOR_BASE_URL: settings.orchestrator_base_url if enabled else "",
        ORCHESTRATOR_MODEL: settings.orchestrator_model if enabled else "",
        ORCHESTRATOR_API_KEY: settings.orchestrator_api_key,
        ORCHESTRATOR_FALLBACK: settings.orchestrator_fallback,
        SOLVER_BASE_URL: settings.solver_base_url,
        SOLVER_MODEL: settings.solver_model,
        SOLVER_API_KEY: settings.solver_api_key,
    }


def render_env(existing: str, settings: ModelSettings) -> str:
    """Rewrite only the model keys, preserving every other line."""
    updates = _updates(settings)
    lines = existing.splitlines()
    seen: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        match = _ASSIGNMENT.match(line.strip())
        if match and match.group(1) in updates:
            key = match.group(1)
            rendered.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            rendered.append(line)
    missing = [key for key in updates if key not in seen]
    if missing:
        if rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append("# 模型配置（由启动器写入）")
        rendered.extend(f"{key}={updates[key]}" for key in missing)
    return "\n".join(rendered).rstrip("\n") + "\n"


def write_model_settings(
    path: Path | str,
    settings: ModelSettings,
) -> None:
    """Write the model keys back, UTF-8 without BOM."""
    file = Path(path)
    existing = file.read_text(encoding="utf-8") if file.exists() else ""
    file.write_text(render_env(existing, settings), encoding="utf-8")


def describe(settings: ModelSettings) -> str:
    """One-line summary used by the launcher and the dialog."""
    if settings.orchestrator_enabled:
        orchestrator = f"{settings.orchestrator_model} @ {settings.orchestrator_base_url}"
    else:
        orchestrator = "未启用（全部走本地）"
    return f"编排：{orchestrator}；解题：{settings.solver_model}"
