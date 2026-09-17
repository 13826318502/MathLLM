"""Tk dialog for editing the model configuration.

Exit codes let the launcher decide what to do next:

    0   saved, start the services
    10  saved only
    1   cancelled

Everything except the widget wiring lives in ``app.config_editor`` so the
read/write logic stays unit tested.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from app.config_editor import (
    DEFAULT_ENV_FILE,
    ModelSettings,
    read_model_settings,
    write_model_settings,
)
from app.core.config import ModelConfig
from app.services.vllm_client import VLLMClient

EXIT_START = 0
EXIT_SAVED = 10
EXIT_CANCELLED = 1

_FALLBACK_CHOICES = (("回退本地编排", "local"), ("直接报错", "fail"))
_TEST_TIMEOUT = (5.0, 20.0)


class ConfigDialog:
    def __init__(self, env_file: Path) -> None:
        # The Tk root must exist before any StringVar is created.
        self.root = tk.Tk()
        self.root.title("MathLLM 模型配置")
        self.root.resizable(False, False)

        self.env_file = env_file
        self.settings = read_model_settings(env_file)
        self.result = EXIT_CANCELLED
        self.vars: dict[str, tk.StringVar] = {}
        self.status: dict[str, tk.StringVar] = {}
        self.fallback = tk.StringVar(master=self.root)

        self._build()

    # ---------------------------------------------------------------- layout

    def _text_field(self, parent: tk.Widget, row: int, key: str, label: str, value: str, masked: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        variable = tk.StringVar(master=parent, value=value)
        self.vars[key] = variable
        ttk.Entry(parent, textvariable=variable, width=44, show="•" if masked else "").grid(
            row=row, column=1, sticky="we", pady=3
        )

    def _section(self, parent: tk.Widget, title: str, rows: list[tuple[str, str, bool]]) -> tk.Widget:
        frame = ttk.LabelFrame(parent, text=title, padding=12)
        frame.columnconfigure(1, weight=1)
        for index, (key, label, masked) in enumerate(rows):
            self._text_field(frame, index, key, label, getattr(self.settings, key), masked)
        return frame

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            outer,
            text="编排模型负责路由、规划和验证；解题模型只负责真正的数学求解。",
            wraplength=430,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        orchestrator = self._section(
            outer,
            "编排模型（云端 · 负责路由/规划/验证）",
            [
                ("orchestrator_base_url", "Base URL", False),
                ("orchestrator_model", "模型名", False),
                ("orchestrator_api_key", "API Key", True),
            ],
        )
        orchestrator.grid(row=1, column=0, columnspan=2, sticky="we")

        self.status["orchestrator"] = tk.StringVar(master=self.root, value="")
        ttk.Label(orchestrator, textvariable=self.status["orchestrator"], foreground="#555").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            orchestrator, text="测试连接", command=lambda: self._test("orchestrator")
        ).grid(row=4, column=0, sticky="w", pady=(4, 0))

        ttk.Label(
            outer,
            text="API Key 留空 = 关闭云端编排，全部走本地模型。",
            foreground="#777",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 10))

        solver = self._section(
            outer,
            "解题模型（本地 · 只负责数学求解）",
            [
                ("solver_base_url", "Base URL", False),
                ("solver_model", "模型名", False),
                ("solver_api_key", "API Key", True),
            ],
        )
        solver.grid(row=3, column=0, columnspan=2, sticky="we")

        self.status["solver"] = tk.StringVar(master=self.root, value="")
        ttk.Label(solver, textvariable=self.status["solver"], foreground="#555").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            solver, text="测试连接", command=lambda: self._test("solver")
        ).grid(row=4, column=0, sticky="w", pady=(4, 0))

        fallback_row = ttk.Frame(outer)
        fallback_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 4))
        ttk.Label(fallback_row, text="云端不可用时：").grid(row=0, column=0)
        choices = [label for label, _ in _FALLBACK_CHOICES]
        self.fallback.set(
            dict(_FALLBACK_CHOICES).get(self.settings.orchestrator_fallback, "回退本地编排")
        )
        ttk.Combobox(
            fallback_row,
            textvariable=self.fallback,
            values=choices,
            state="readonly",
            width=16,
        ).grid(row=0, column=1, padx=(4, 0))

        ttk.Label(outer, text=f"保存到：{self.env_file}", foreground="#777").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="取消", command=self._cancel).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="仅保存", command=lambda: self._save(False)).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(buttons, text="保存并启动", command=lambda: self._save(True)).grid(
            row=0, column=2, padx=4
        )

        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

    # --------------------------------------------------------------- actions

    def _collect(self) -> ModelSettings:
        def value(key: str) -> str:
            return self.vars[key].get().strip()

        return ModelSettings(
            orchestrator_base_url=value("orchestrator_base_url"),
            orchestrator_model=value("orchestrator_model"),
            orchestrator_api_key=value("orchestrator_api_key"),
            orchestrator_fallback=dict(_FALLBACK_CHOICES).get(
                self.fallback.get(), "local"
            ),
            solver_base_url=value("solver_base_url"),
            solver_model=value("solver_model"),
            solver_api_key=value("solver_api_key"),
        )

    def _test(self, role: str) -> None:
        prefix = "orchestrator" if role == "orchestrator" else "solver"
        base_url = self.vars[f"{prefix}_base_url"].get().strip()
        model = self.vars[f"{prefix}_model"].get().strip()
        api_key = self.vars[f"{prefix}_api_key"].get().strip()
        if not base_url or not model:
            self.status[role].set("请先填写 Base URL 和模型名")
            return
        self.status[role].set("测试中…")
        config = ModelConfig(
            role=role,
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key or None,
            connect_timeout=_TEST_TIMEOUT[0],
            read_timeout=_TEST_TIMEOUT[1],
            max_output_tokens=64,
        )
        threading.Thread(
            target=self._test_worker, args=(role, config), daemon=True
        ).start()

    def _test_worker(self, role: str, config: ModelConfig) -> None:
        try:
            models = asyncio.run(VLLMClient(config).list_models())
        except Exception as exc:  # surfaced to the user, not swallowed
            message = f"连接失败：{exc}"
        else:
            preview = "、".join(models[:3]) if models else "未返回模型列表"
            message = f"连接正常（{len(models)} 个模型：{preview}）"
        self.root.after(0, lambda: self.status[role].set(message))

    def _save(self, start: bool) -> None:
        settings = self._collect()
        if not settings.solver_base_url or not settings.solver_model:
            messagebox.showerror("配置不完整", "解题模型的 Base URL 和模型名不能为空。")
            return
        if settings.orchestrator_enabled and (
            not settings.orchestrator_base_url or not settings.orchestrator_model
        ):
            messagebox.showerror(
                "配置不完整",
                "填写了编排模型 API Key 时，Base URL 和模型名也不能为空。",
            )
            return
        write_model_settings(self.env_file, settings)
        if start:
            self.result = EXIT_START
        else:
            self.result = EXIT_SAVED
            messagebox.showinfo("已保存", f"配置已写入\n{self.env_file}")
        self.root.destroy()

    def _cancel(self) -> None:
        self.result = EXIT_CANCELLED
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return self.result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MathLLM 模型配置")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    args = parser.parse_args(argv)
    return ConfigDialog(Path(args.env_file)).run()


if __name__ == "__main__":
    sys.exit(main())
