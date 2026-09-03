"""Merge a LoRA adapter into a full-precision base model.

The training run used QLoRA/4-bit loading, but merging must load the base
model in FP16 or BF16. The result is a standalone model that can be loaded by
Transformers or vLLM; it is not an AWQ/GPTQ quantized model.

Example::

    python scripts/merge_lora.py \
        --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
        --lora_path ./outputs/smoke-89-logging1/checkpoint-12 \
        --output_path ./outputs/smoke-89-merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name != "auto":
        raise ValueError("dtype must be one of: auto, float16, bfloat16")
    if not torch.cuda.is_available():
        print("CUDA unavailable; using float32 for CPU merge.")
        return torch.float32
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return torch.bfloat16 if checker and checker() else torch.float16


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    *,
    dtype: str = "auto",
    max_shard_size: str = "4GB",
    overwrite: bool = False,
) -> Path:
    """Load, merge, save, and verify a LoRA adapter."""
    base_path = _resolve_path(base_model_path)
    adapter_path = _resolve_path(lora_path)
    destination = _resolve_path(output_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base model does not exist: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter does not exist: {adapter_path}")
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"adapter_config.json does not exist: {adapter_config}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. "
            "Choose a new path or pass --overwrite explicitly."
        )
    destination.mkdir(parents=True, exist_ok=True)

    merge_dtype = _resolve_dtype(dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": merge_dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    print(f"Base model: {base_path}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Merge dtype: {merge_dtype}")
    print("Loading base model in full precision for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(str(base_path), **model_kwargs)

    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    peft_model.eval()

    print("Merging adapter weights...")
    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    merged_model.config.use_cache = True

    print(f"Saving merged model to: {destination}")
    merged_model.save_pretrained(
        str(destination),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(str(base_path), use_fast=True)
    tokenizer.save_pretrained(str(destination))

    _write_json(
        destination / "merge_config.json",
        {
            "base_model_path": str(base_path),
            "lora_adapter_path": str(adapter_path),
            "output_path": str(destination),
            "dtype": str(merge_dtype).replace("torch.", ""),
            "max_shard_size": max_shard_size,
            "quantization": None,
            "note": "Full-precision merged model; not an AWQ/GPTQ quantized model.",
        },
    )

    model_files = list(destination.glob("*.safetensors")) + list(destination.glob("*.bin"))
    if not (destination / "config.json").exists() or not model_files:
        raise RuntimeError(
            f"Merge finished but output verification failed: {destination} "
            "must contain config.json and model weight files."
        )
    print(f"Verified {len(model_files)} model weight file(s).")
    print("Done!")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone model")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="Merge precision; auto selects BF16 when supported, otherwise FP16",
    )
    parser.add_argument("--max_shard_size", default="4GB")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    args = parser.parse_args()
    merge_lora(
        args.base_model,
        args.lora_path,
        args.output_path,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
