"""使用 Transformers + PEFT + TRL 进行 LoRA SFT 训练。

用法::

    python scripts/train.py
    python scripts/train.py --config configs/training/correction-round-3-20260904/train_config.yaml

训练前确保基座模型和 ``data/processed/train.json``、``eval.json`` 已准备好。
默认使用标准 LoRA；显存不足时可以在训练配置中设置 ``use_4bit: true``，
切换为 QLoRA。训练完成后只保存 LoRA adapter，不会复制一份完整的基座模型。
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

try:
    from loss_curve import export_loss_history
except ModuleNotFoundError:  # supports ``import scripts.train`` from project root
    from scripts.loss_curve import export_loss_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ACTIVE_TRAIN_CONFIG = "configs/training/correction-round-10-20260908/train_config.yaml"


class AssistantOnlyDataCollator:
    """只对 Qwen ChatML 中 assistant 回复部分计算 causal-LM loss。

    Qwen 的对话模板通常包含 ``<|im_start|>assistant\\n`` 标记。该 collator
    会把该标记及其之前的 token 标签设为 ``-100``，因此 Trainer 只会对
    assistant 的答案内容（以及结尾标记）计算损失。padding 位置也会被屏蔽。
    """

    def __init__(self, tokenizer, response_template: str) -> None:
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.response_template_ids = tokenizer(
            response_template,
            add_special_tokens=False,
        )["input_ids"]
        if not self.response_template_ids:
            raise ValueError(
                "assistant_response_template tokenizes to an empty sequence: "
                f"{response_template!r}"
            )

    @staticmethod
    def _find_subsequence(sequence: list[int], pattern: list[int]) -> int:
        width = len(pattern)
        for start in range(len(sequence) - width + 1):
            if sequence[start : start + width] == pattern:
                return start
        return -1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Depending on the installed TRL version, SFTTrainer may pass already
        # tokenized examples or raw ``text`` examples to the collator.
        if features and "input_ids" not in features[0]:
            texts = [feature["text"] for feature in features]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            token_fields = {"input_ids", "attention_mask", "token_type_ids"}
            token_features = [
                {key: value for key, value in feature.items() if key in token_fields}
                for feature in features
            ]
            batch = self.tokenizer.pad(
                token_features,
                padding=True,
                return_tensors="pt",
            )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
            batch["attention_mask"] = attention_mask

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        template_ids = self.response_template_ids
        missing_indices: list[int] = []
        for row_index in range(input_ids.shape[0]):
            valid_length = int(attention_mask[row_index].sum().item())
            row = input_ids[row_index, :valid_length].tolist()
            marker_start = self._find_subsequence(row, template_ids)
            if marker_start < 0:
                missing_indices.append(row_index)
                continue
            # Do not train on system/user messages or the assistant header.
            marker_end = marker_start + len(template_ids)
            labels[row_index, :marker_end] = -100

        if missing_indices:
            raise ValueError(
                "Could not find assistant response marker in tokenized sample(s) "
                f"{missing_indices}; response_template={self.response_template!r}. "
                "Check that the tokenizer chat template and the configured marker match."
            )

        batch["labels"] = labels
        return batch


def load_config(config_path: str = ACTIVE_TRAIN_CONFIG) -> dict[str, Any]:
    """加载 YAML 配置，并检查顶层结构。"""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """将相对路径按项目根目录解析，避免从其他目录启动时找不到文件。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _supports_argument(callable_obj: Callable[..., Any], name: str) -> bool:
    """兼容不同 Transformers/TRL 版本的参数变化。"""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker()) if checker else False


def load_model_and_tokenizer(
    model_name: str,
    training_config: dict[str, Any] | None = None,
):
    """加载基座模型和 Tokenizer。

    默认使用标准 LoRA：基座以 BF16/FP16 加载并冻结。
    设置 ``use_4bit: true`` 或 ``quantization_bit: 4`` 时改用 QLoRA。
    """
    training_config = training_config or {}
    use_4bit = bool(
        training_config.get("use_4bit", False)
        or training_config.get("quantization_bit") == 4
    )
    requested_bf16 = bool(training_config.get("bf16", True))
    use_bf16 = requested_bf16 and _bf16_supported()
    if requested_bf16 and torch.cuda.is_available() and not use_bf16:
        print("Warning: current GPU does not support BF16; falling back to FP16.")

    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        model_kwargs: dict[str, Any] = {
            "torch_dtype": compute_dtype,
            "device_map": "auto",
        }
    else:
        print("Warning: CUDA is unavailable. Loading on CPU is for debug checks only.")
        model_kwargs = {"torch_dtype": torch.float32}

    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading requires a CUDA GPU and bitsandbytes.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, use_4bit, use_bf16


def create_lora_config(config: dict[str, Any]) -> LoraConfig:
    """从项目配置创建 PEFT LoRA 配置。"""
    lora = config.get("lora", {})
    if not isinstance(lora, dict):
        raise ValueError("'lora' must be a mapping in the training config")
    target_modules = lora.get("target_modules")
    if not target_modules:
        raise ValueError("lora.target_modules cannot be empty")
    if isinstance(target_modules, str):
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]
    rank = int(lora.get("r", 16))
    if rank <= 0:
        raise ValueError(f"lora.r must be positive, got {rank}")
    return LoraConfig(
        r=rank,
        lora_alpha=int(lora.get("lora_alpha", rank * 2)),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        bias=lora.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def create_chat_template_fn(tokenizer) -> Callable[[dict[str, Any]], str]:
    """返回把一条 messages 记录格式化为 Qwen 对话文本的函数。"""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; check the Qwen model directory")

    def format_example(example: dict[str, Any]) -> str:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Each example must contain a non-empty 'messages' list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError("Every message must have non-empty string role and content")
        if messages[0]["role"] != "system" or messages[-1]["role"] != "assistant":
            raise ValueError("Messages must start with system and end with assistant")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return format_example


def prepare_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """预先应用 ChatML 模板，生成 SFTTrainer 使用的 ``text`` 字段。"""
    formatter = create_chat_template_fn(tokenizer)

    def add_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": formatter(example)}

    columns_to_remove = [column for column in dataset.column_names if column != "text"]
    return dataset.map(add_text, remove_columns=columns_to_remove, desc="Applying chat template")


def _load_json_dataset(path_value: str | Path, split_name: str) -> Dataset:
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} dataset does not exist: {path}")
    dataset = load_dataset("json", data_files={split_name: str(path)})[split_name]
    if len(dataset) == 0:
        raise ValueError(f"{split_name} dataset is empty: {path}")
    return dataset


def _training_arguments(training: dict[str, Any], output_dir: Path, use_bf16: bool):
    """创建 TrainingArguments，并兼容 evaluation_strategy/eval_strategy。"""
    cuda_available = torch.cuda.is_available()
    use_gradient_checkpointing = bool(training.get("gradient_checkpointing", True))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "logging_steps": int(training.get("logging_steps", 5)),
        "save_steps": int(training.get("save_steps", 100)),
        "eval_steps": int(training.get("eval_steps", training.get("save_steps", 100))),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "logging_first_step": True,
        "gradient_checkpointing": use_gradient_checkpointing,
        "bf16": bool(use_bf16 and cuda_available),
        "fp16": bool(cuda_available and not use_bf16),
    }
    warmup_ratio = float(training.get("warmup_ratio", 0.03))
    if _supports_argument(TrainingArguments, "warmup_ratio"):
        kwargs["warmup_ratio"] = warmup_ratio
    elif _supports_argument(TrainingArguments, "warmup_steps"):
        # Transformers 5.x removed ``warmup_ratio`` from TrainingArguments.
        # Keep the run compatible; callers may provide an explicit step count
        # when using that API variant.
        kwargs["warmup_steps"] = int(training.get("warmup_steps", 0))
        print(
            "Warning: this Transformers version does not support warmup_ratio; "
            f"using warmup_steps={kwargs['warmup_steps']}."
        )
    strategy_name = (
        "eval_strategy"
        if _supports_argument(TrainingArguments, "eval_strategy")
        else "evaluation_strategy"
    )
    kwargs[strategy_name] = "steps"
    if use_gradient_checkpointing and _supports_argument(
        TrainingArguments, "gradient_checkpointing_kwargs"
    ):
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if _supports_argument(TrainingArguments, "optim"):
        kwargs["optim"] = training.get("optim", "adamw_torch")
    if _supports_argument(TrainingArguments, "lr_scheduler_type") and training.get(
        "lr_scheduler_type"
    ):
        kwargs["lr_scheduler_type"] = str(training["lr_scheduler_type"])
    return TrainingArguments(**kwargs)


def _build_sft_args(
    training: dict[str, Any],
    output_dir: Path,
    use_bf16: bool,
    max_seq_length: int,
):
    """在不同 TRL 版本中创建合适的 SFTConfig。"""
    base_args = _training_arguments(training, output_dir, use_bf16)
    kwargs: dict[str, Any] = {"output_dir": str(output_dir)}
    for key, value in vars(base_args).items():
        if _supports_argument(SFTConfig, key):
            kwargs[key] = value
    if _supports_argument(SFTConfig, "dataset_text_field"):
        kwargs["dataset_text_field"] = "text"
    if _supports_argument(SFTConfig, "packing"):
        kwargs["packing"] = False
    if _supports_argument(SFTConfig, "max_length"):
        kwargs["max_length"] = max_seq_length
    elif _supports_argument(SFTConfig, "max_seq_length"):
        kwargs["max_seq_length"] = max_seq_length
    return SFTConfig(**kwargs)


def _build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args,
    max_seq_length: int,
    data_collator=None,
):
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if _supports_argument(SFTTrainer, "processing_class"):
        kwargs["processing_class"] = tokenizer
    elif _supports_argument(SFTTrainer, "tokenizer"):
        kwargs["tokenizer"] = tokenizer
    if _supports_argument(SFTTrainer, "dataset_text_field"):
        kwargs["dataset_text_field"] = "text"
    if _supports_argument(SFTTrainer, "max_seq_length"):
        kwargs["max_seq_length"] = max_seq_length
    if _supports_argument(SFTTrainer, "packing"):
        kwargs["packing"] = False
    if data_collator is not None and _supports_argument(SFTTrainer, "data_collator"):
        kwargs["data_collator"] = data_collator
    return SFTTrainer(**kwargs)


def _build_data_collator(tokenizer, training: dict[str, Any]):
    """根据配置构造 loss collator；默认保持显式可审计的 assistant-only 训练。"""
    if not bool(training.get("assistant_only_loss", False)):
        return None
    response_template = str(
        training.get("assistant_response_template", "<|im_start|>assistant\n")
    )
    return AssistantOnlyDataCollator(tokenizer, response_template)


def _save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def train(config_path: str = ACTIVE_TRAIN_CONFIG) -> Path:
    config = load_config(config_path)
    data_config = config.get("data", {})
    training = config.get("training", {})
    if not isinstance(data_config, dict) or not isinstance(training, dict):
        raise ValueError("'data' and 'training' must be mappings in the training config")
    if "model_name_or_path" not in config:
        raise ValueError("Training config is missing model_name_or_path")
    if "train_file" not in data_config or "eval_file" not in data_config:
        raise ValueError("Training config must define data.train_file and data.eval_file")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    output_dir = resolve_path(training.get("output_dir", "outputs/math-lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, use_4bit, use_bf16 = load_model_and_tokenizer(
        config["model_name_or_path"], training
    )
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        print("Using QLoRA: base model is loaded in 4-bit NF4.")
    else:
        print("Using standard LoRA: base model is loaded in BF16/FP16.")

    print("Applying LoRA config...")
    model = get_peft_model(model, create_lora_config(config))
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset = _load_json_dataset(data_config["train_file"], "train")
    eval_dataset = _load_json_dataset(data_config["eval_file"], "eval")
    print(f"Train examples: {len(train_dataset)}; Eval examples: {len(eval_dataset)}")
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    max_seq_length = int(training.get("max_seq_length", 2048))
    if max_seq_length <= 0:
        raise ValueError("training.max_seq_length must be positive")
    sft_args = _build_sft_args(training, output_dir, use_bf16, max_seq_length)
    data_collator = _build_data_collator(tokenizer, training)
    if data_collator is not None:
        print(
            "Assistant-only loss enabled: system/user tokens and the assistant header "
            "will be masked with label=-100."
        )
    trainer = _build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        sft_args,
        max_seq_length,
        data_collator=data_collator,
    )

    print("Starting LoRA SFT training...")
    resume_from_checkpoint = training.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        resume_from_checkpoint = str(resolve_path(resume_from_checkpoint))
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存的是 LoRA adapter，而不是重新复制一份 7B 基座模型。
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()
    loss_summary = export_loss_history(
        output_dir / "trainer_state.json",
        output_dir,
        train_metrics=train_result.metrics,
    )
    train_metrics = dict(train_result.metrics)
    if loss_summary.get("eval_loss") is not None:
        train_metrics["eval_loss"] = loss_summary["eval_loss"]
    if loss_summary.get("best_eval_loss") is not None:
        train_metrics["best_eval_loss"] = loss_summary["best_eval_loss"]
    _save_json(output_dir / "train_results.json", train_metrics)
    _save_json(
        output_dir / "run_config.json",
        {
            "config_path": str(resolve_path(config_path)),
            "model_name_or_path": config["model_name_or_path"],
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "seed": seed,
            "use_4bit": use_4bit,
            "use_bf16": use_bf16,
            "max_seq_length": max_seq_length,
            "assistant_only_loss": bool(training.get("assistant_only_loss", False)),
            "assistant_response_template": training.get(
                "assistant_response_template", "<|im_start|>assistant\n"
            ),
        },
    )
    print(f"Training complete. LoRA adapter saved to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA adapter")
    parser.add_argument(
        "--config",
        default=ACTIVE_TRAIN_CONFIG,
        help="Path to the nested MathLLM training YAML config",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
