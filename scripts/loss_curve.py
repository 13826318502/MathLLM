"""Export and plot train/eval loss history from a Trainer state file.

The same utility is called automatically by ``scripts/train.py`` after a
training run. It can also recover metrics from an existing cloud output:

    python scripts/loss_curve.py \
        --trainer_state outputs/smoke-89/trainer_state.json \
        --output_dir eval/results/smoke-89
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HISTORY_FIELDS = ("step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm")


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def extract_history(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep train/eval loss records while preserving useful training fields."""
    history: list[dict[str, Any]] = []
    for item in log_history:
        if not isinstance(item, dict):
            continue
        if _number(item.get("loss")) is None and _number(item.get("eval_loss")) is None:
            continue
        record = {field: item.get(field) for field in HISTORY_FIELDS}
        history.append(record)
    return history


def export_loss_history(
    trainer_state_path: str | Path,
    output_dir: str | Path,
    train_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write JSON/CSV/PNG loss artifacts and return a compact summary."""
    state_path = Path(trainer_state_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json does not exist: {state_path}")
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    history = extract_history(state.get("log_history", []))

    with (output_path / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_path / "loss_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    train_points = [item for item in history if _number(item.get("loss")) is not None]
    eval_points = [item for item in history if _number(item.get("eval_loss")) is not None]
    train_loss = train_metrics.get("train_loss") if train_metrics else None
    if _number(train_loss) is None and train_points:
        train_loss = train_points[-1]["loss"]
    eval_loss = eval_points[-1]["eval_loss"] if eval_points else None
    summary = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "best_eval_loss": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "train_log_points": len(train_points),
        "eval_log_points": len(eval_points),
    }
    with (output_path / "loss_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    _plot_loss(history, output_path / "loss_curve.png")
    return summary


def _plot_loss(history: list[dict[str, Any]], output_path: Path) -> None:
    train_points = [item for item in history if _number(item.get("loss")) is not None]
    eval_points = [item for item in history if _number(item.get("eval_loss")) is not None]
    if not train_points and not eval_points:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; JSON/CSV loss history was still saved.")
        return

    figure, axis = plt.subplots(figsize=(9, 5))
    if train_points:
        train_steps = [
            item["step"] if _number(item.get("step")) is not None else index
            for index, item in enumerate(train_points, 1)
        ]
        axis.plot(
            train_steps,
            [item["loss"] for item in train_points],
            marker="o",
            label="Train loss",
        )
    if eval_points:
        eval_steps = [
            item["step"] if _number(item.get("step")) is not None else index
            for index, item in enumerate(eval_points, 1)
        ]
        axis.plot(
            eval_steps,
            [item["eval_loss"] for item in eval_points],
            marker="s",
            label="Eval loss",
        )
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Evaluation Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Trainer 的 train/eval loss 记录和曲线")
    parser.add_argument("--trainer_state", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = export_loss_history(args.trainer_state, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
