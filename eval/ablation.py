"""Run controlled LoRA ablation experiments with LLaMA-Factory.

The runner keeps the evaluation set fixed, samples training subsets with a
fixed seed, creates one isolated output directory per experiment, and records
training metrics without treating loss as mathematical accuracy.

Typical usage on the cloud instance::

    python eval/ablation.py --config configs/ablation.yaml --dry-run
    python eval/ablation.py --config configs/ablation.yaml --groups rank
    python eval/ablation.py --config configs/ablation.yaml --groups all

The optional evaluator command can use these placeholders:
``{model_dir}``, ``{eval_file}``, ``{output_dir}``, and ``{experiment_id}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_ROLE_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
}

RESULT_FIELDS = [
    "experiment_id",
    "group",
    "parameter",
    "value",
    "status",
    "train_samples",
    "duration_seconds",
    "train_loss",
    "eval_loss",
    "accuracy",
    "best_metric",
    "config_path",
    "output_dir",
    "log_path",
    "error",
]


def load_ablation_config(config_path: str) -> dict[str, Any]:
    """Load and validate the ablation configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Ablation config must be a YAML mapping")
    if "baseline" not in config:
        raise ValueError("Ablation config is missing 'baseline'")
    if "experiments" not in config or not config["experiments"]:
        raise ValueError("Ablation config must define at least one experiment group")

    for index, group in enumerate(config["experiments"]):
        if not isinstance(group, dict):
            raise ValueError(f"experiments[{index}] must be a mapping")
        for key in ("name", "parameter", "values"):
            if key not in group:
                raise ValueError(f"experiments[{index}] is missing '{key}'")
        if not group["values"]:
            raise ValueError(f"experiments[{index}].values cannot be empty")
    return config


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return records


def _validate_messages(records: Iterable[dict[str, Any]], path: Path) -> None:
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}: item {index} has no non-empty messages list")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles[:1] != ["system"] or roles[-1:] != ["assistant"]:
            raise ValueError(f"{path}: item {index} has invalid message order")
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            for message in messages
        ):
            raise ValueError(f"{path}: item {index} contains empty or invalid message content")


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace(".", "p").replace("-", "m").replace("/", "-")


def _experiment_id(group: str, parameter: str, value: Any) -> str:
    parameter_name = parameter.removeprefix("lora_")
    return f"exp-{group}-{parameter_name}{_safe_value(value)}"


def _normalize_groups(
    config: dict[str, Any], requested_groups: list[str] | None
) -> list[dict[str, Any]]:
    groups = config["experiments"]
    if not requested_groups or "all" in requested_groups:
        return groups
    requested = set(requested_groups)
    selected = [group for group in groups if group["name"] in requested]
    missing = requested - {group["name"] for group in selected}
    if missing:
        raise ValueError(f"Unknown ablation group(s): {', '.join(sorted(missing))}")
    return selected


def build_experiment_specs(
    config: dict[str, Any], requested_groups: list[str] | None = None
) -> list[dict[str, Any]]:
    """Build baseline plus one-variable-at-a-time experiment specifications."""
    baseline = dict(config["baseline"])
    specs: list[dict[str, Any]] = [
        {
            "experiment_id": "baseline",
            "group": "baseline",
            "parameter": "baseline",
            "value": "baseline",
            "params": baseline,
        }
    ]

    for group in _normalize_groups(config, requested_groups):
        parameter = group["parameter"]
        for value in group["values"]:
            params = dict(baseline)
            params[parameter] = value
            if params == baseline:
                continue
            specs.append(
                {
                    "experiment_id": _experiment_id(group["name"], parameter, value),
                    "group": group["name"],
                    "parameter": parameter,
                    "value": value,
                    "params": params,
                }
            )
    return specs


def _select_training_records(
    records: list[dict[str, Any]], data_size: int | str, seed: int
) -> list[dict[str, Any]]:
    if data_size in ("full", None):
        return list(records)
    try:
        requested = int(data_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data_size must be 'full' or an integer, got {data_size!r}") from exc
    if requested <= 0:
        raise ValueError(f"data_size must be positive, got {requested}")
    if requested > len(records):
        raise ValueError(
            f"Requested data_size={requested}, but only {len(records)} training records exist. "
            "Expand the dataset or choose a smaller value; do not silently reuse fewer samples."
        )
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:requested]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_dataset_bundle(
    run_dir: Path,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "train.json"
    eval_path = dataset_dir / "eval.json"
    _write_json(train_path, train_records)
    _write_json(eval_path, eval_records)
    _write_json(
        dataset_dir / "dataset_info.json",
        {
            "mathllm_train": {
                "file_name": train_path.name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": DEFAULT_ROLE_TAGS,
            },
            "mathllm_eval": {
                "file_name": eval_path.name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": DEFAULT_ROLE_TAGS,
            },
        },
    )
    return dataset_dir, eval_path


def _base_training_config(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path.cwd()
    base_path = _resolve_path(config.get("base_config", "configs/train_config.yaml"), project_root)
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Base training config must be a mapping: {base_path}")
    return base


def build_llamafactory_config(
    config: dict[str, Any],
    params: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    run_name: str,
    project_root: Path,
) -> dict[str, Any]:
    """Translate the project's nested config into LLaMA-Factory arguments."""
    base = _base_training_config(config)
    lora = dict(base.get("lora", {}))
    training = dict(base.get("training", {}))
    lora.update({key.removeprefix("lora_"): value for key, value in params.items() if key.startswith("lora_")})
    training.update(
        {
            key: value
            for key, value in params.items()
            if key
            in {
                "learning_rate",
                "num_train_epochs",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "warmup_ratio",
                "logging_steps",
                "save_steps",
                "save_total_limit",
                "bf16",
                "max_seq_length",
                "gradient_checkpointing",
                "per_device_eval_batch_size",
                "eval_steps",
            }
        }
    )

    target_modules = lora.get("target_modules", [])
    if isinstance(target_modules, list):
        target_modules = ",".join(target_modules)
    model_name = base.get("model_name_or_path")
    if not model_name:
        raise ValueError("Base training config is missing model_name_or_path")

    lf_config: dict[str, Any] = {
        "model_name_or_path": model_name,
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_rank": lora.get("r", 16),
        "lora_alpha": lora.get("lora_alpha", 32),
        "lora_dropout": lora.get("lora_dropout", 0.05),
        "lora_target": target_modules or "all",
        "dataset": "mathllm_train",
        "eval_dataset": "mathllm_eval",
        "dataset_dir": _relative_or_absolute(dataset_dir, project_root),
        "template": "qwen",
        "cutoff_len": training.get("max_seq_length", 2048),
        "output_dir": _relative_or_absolute(output_dir, project_root),
        "run_name": run_name,
        "overwrite_output_dir": True,
        "seed": config.get("seed", 42),
        "val_size": 0,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "packing": False,
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
        "learning_rate": training.get("learning_rate", 2e-4),
        "num_train_epochs": training.get("num_train_epochs", 3),
        "warmup_ratio": training.get("warmup_ratio", 0.03),
        "logging_steps": training.get("logging_steps", 5),
        "save_steps": training.get("save_steps", 5),
        "eval_steps": training.get("eval_steps", training.get("save_steps", 5)),
        "save_total_limit": training.get("save_total_limit", 3),
        "bf16": training.get("bf16", True),
        "fp16": False,
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
    }

    quantization_bit = params.get("quantization_bit", config.get("quantization_bit"))
    if quantization_bit is not None:
        lf_config["quantization_bit"] = quantization_bit
    return lf_config


def _read_trainer_state(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    history = state.get("log_history", [])
    train_losses = [item["loss"] for item in history if isinstance(item.get("loss"), (int, float))]
    eval_losses = [item["eval_loss"] for item in history if isinstance(item.get("eval_loss"), (int, float))]
    return {
        "train_loss": train_losses[-1] if train_losses else None,
        "eval_loss": eval_losses[-1] if eval_losses else None,
        "best_metric": state.get("best_metric"),
    }


def _read_optional_accuracy(run_dir: Path) -> float | None:
    for name in ("report.json", "evaluation.json", "metrics.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            value = report.get("accuracy") if isinstance(report, dict) else None
            if isinstance(value, (int, float)):
                return float(value)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _run_command(
    command: list[str], cwd: Path, log_path: Path, timeout: int | None
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        return completed.returncode, ""
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout} seconds"
    except OSError as exc:
        return 127, str(exc)


def run_single_experiment(
    spec: dict[str, Any],
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    output_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare and optionally run one isolated LLaMA-Factory experiment."""
    project_root = Path.cwd().resolve()
    experiment_id = spec["experiment_id"]
    run_dir = output_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    params = spec["params"]
    selected_train = _select_training_records(
        train_records, params.get("data_size", "full"), int(config.get("seed", 42))
    )
    dataset_dir, eval_path = _write_dataset_bundle(run_dir, selected_train, eval_records)
    model_output_dir = run_dir / "model"
    generated_config = build_llamafactory_config(
        config, params, dataset_dir, model_output_dir, experiment_id, project_root
    )
    config_path = run_dir / "llamafactory.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(generated_config, handle, allow_unicode=True, sort_keys=False)

    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "group": spec["group"],
        "parameter": spec["parameter"],
        "value": spec["value"],
        "status": "dry_run" if dry_run else "pending",
        "train_samples": len(selected_train),
        "duration_seconds": None,
        "train_loss": None,
        "eval_loss": None,
        "accuracy": None,
        "best_metric": None,
        "config_path": str(config_path),
        "output_dir": str(model_output_dir),
        "log_path": str(run_dir / "train.log"),
        "error": "",
    }
    if dry_run:
        return result

    runner = config.get("runner", {})
    command_value = runner.get("command", "llamafactory-cli")
    command = (
        [str(item) for item in command_value]
        if isinstance(command_value, list)
        else shlex.split(str(command_value), posix=sys.platform != "win32")
    )
    command.extend(["train", str(config_path)])
    timeout = runner.get("timeout_seconds")
    timeout = int(timeout) if timeout is not None else None
    started = time.perf_counter()
    return_code, error = _run_command(command, project_root, Path(result["log_path"]), timeout)
    result["duration_seconds"] = round(time.perf_counter() - started, 2)
    result["status"] = "completed" if return_code == 0 else "failed"
    result["error"] = error or (f"LLaMA-Factory exited with code {return_code}" if return_code else "")
    result.update(_read_trainer_state(model_output_dir))

    evaluation_command = runner.get("evaluation_command")
    if return_code == 0 and evaluation_command:
        rendered = str(evaluation_command).format(
            model_dir=str(model_output_dir),
            eval_file=str(eval_path),
            output_dir=str(run_dir),
            experiment_id=experiment_id,
        )
        eval_code, eval_error = _run_command(
            shlex.split(rendered, posix=sys.platform != "win32"),
            project_root,
            run_dir / "evaluation.log",
            timeout,
        )
        if eval_code != 0:
            result["status"] = "evaluation_failed"
            result["error"] = eval_error or f"Evaluator exited with code {eval_code}"
        result["accuracy"] = _read_optional_accuracy(run_dir)
    return result


def run_ablation_study(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    requested_groups: list[str] | None = None,
    dry_run: bool = False,
    max_experiments: int | None = None,
) -> list[dict[str, Any]]:
    """Run or prepare all selected one-variable-at-a-time experiments."""
    project_root = Path.cwd().resolve()
    data_config = config.get("data", {})
    train_path = _resolve_path(data_config.get("train_file", "data/processed/train.json"), project_root)
    eval_path = _resolve_path(data_config.get("eval_file", "data/processed/eval.json"), project_root)
    train_records = _load_json_records(train_path)
    eval_records = _load_json_records(eval_path)
    _validate_messages(train_records, train_path)
    _validate_messages(eval_records, eval_path)

    output_root = _resolve_path(
        output_dir or config.get("output_dir", "eval/results/ablation"), project_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    specs = build_experiment_specs(config, requested_groups)
    if max_experiments is not None:
        specs = specs[:max_experiments]

    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] {spec['experiment_id']} "
            f"({spec['parameter']}={spec['value']})"
        )
        try:
            result = run_single_experiment(
                spec, config, train_records, eval_records, output_root, dry_run=dry_run
            )
        except Exception as exc:  # keep independent experiments reportable
            result = {
                "experiment_id": spec["experiment_id"],
                "group": spec["group"],
                "parameter": spec["parameter"],
                "value": spec["value"],
                "status": "invalid",
                "train_samples": None,
                "duration_seconds": None,
                "train_loss": None,
                "eval_loss": None,
                "accuracy": None,
                "best_metric": None,
                "config_path": "",
                "output_dir": "",
                "log_path": "",
                "error": str(exc),
            }
        results.append(result)
        print(f"    status={result['status']} error={result['error'] or '-'}")

    _write_json(output_root / "ablation_results.json", results)
    _write_results_csv(output_root / "ablation_results.csv", results)
    plot_results(results, output_root)
    return results


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({field: result.get(field) for field in RESULT_FIELDS} for result in results)


def plot_results(results: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Plot accuracy when available, otherwise plot eval loss."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV/JSON results were still saved")
        return

    output_path = Path(output_dir)
    valid = [item for item in results if item.get("status") in {"completed", "dry_run"}]
    metric_name = "accuracy" if any(item.get("accuracy") is not None for item in valid) else "eval_loss"
    metric_label = "Accuracy" if metric_name == "accuracy" else "Eval loss"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in valid:
        if item["group"] == "baseline" or item.get(metric_name) is None:
            continue
        grouped.setdefault(item["group"], []).append(item)

    for group, items in grouped.items():
        items.sort(key=lambda item: str(item["value"]))
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(range(len(items)), [item[metric_name] for item in items], marker="o")
        axis.set_xticks(range(len(items)), [str(item["value"]) for item in items])
        axis.set_xlabel(items[0]["parameter"])
        axis.set_ylabel(metric_label)
        axis.set_title(f"Ablation: {group}")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path / f"{group}_ablation.png", dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaMA-Factory ablation experiments")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated groups, e.g. rank,learning_rate; default runs all",
    )
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs/datasets and validate sizes without training",
    )
    args = parser.parse_args()
    config = load_ablation_config(args.config)
    requested_groups = [item.strip() for item in args.groups.split(",")] if args.groups else None
    results = run_ablation_study(
        config,
        output_dir=args.output,
        requested_groups=requested_groups,
        dry_run=args.dry_run,
        max_experiments=args.max_experiments,
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(json.dumps({"total": len(results), "status_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
