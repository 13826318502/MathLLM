"""Quantize a merged MathLLM checkpoint to AWQ INT4 for vLLM.

The input must be a standalone model produced by ``merge_lora.py``.  This
script does not quantize a QLoRA/NF4 training checkpoint: QLoRA loading and an
AWQ deployment checkpoint are different formats and serve different purposes.

The implementation uses the vLLM project's ``llmcompressor`` package.  It
applies AWQ activation-aware scaling followed by W4A16 weight-only
quantization, using representative samples from the project's processed
training data.

Example::

    python scripts/quantize.py \
        --model_path ./outputs/math-lora-merged \
        --output_path ./outputs/math-lora-quantized \
        --calibration_data ./data/processed/train.json \
        --num_calibration_samples 256 \
        --max_seq_length 2048

Run this step on a CUDA GPU.  The local CPU-only environment is suitable for
checking the script, but not for practical AWQ calibration of a 7B model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_PATH = "./outputs/math-lora-merged"
DEFAULT_OUTPUT_PATH = "./outputs/math-lora-quantized"
DEFAULT_CALIBRATION_DATA = "./data/processed/train.json"
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_NUM_CALIBRATION_SAMPLES = 256
DEFAULT_SEED = 42


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_calibration_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration data does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Calibration data is not valid JSON: {path}") from exc
    if not isinstance(value, list):
        raise ValueError(f"Calibration data must be a JSON list: {path}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
            raise ValueError(f"Calibration record {index} has no messages list")
        messages = item["messages"]
        if not messages or any(
            not isinstance(message, dict)
            or not str(message.get("role", "")).strip()
            or not str(message.get("content", "")).strip()
            for message in messages
        ):
            raise ValueError(f"Calibration record {index} contains an empty message")
        records.append(item)
    if not records:
        raise ValueError(f"Calibration data contains no records: {path}")
    return records


def _build_calibration_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    num_samples: int,
    max_seq_length: int,
    seed: int,
) -> Any:
    from datasets import Dataset

    if num_samples <= 0:
        raise ValueError("num_calibration_samples must be positive")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")

    selected = list(records)
    random.Random(seed).shuffle(selected)
    selected = selected[: min(num_samples, len(selected))]

    texts: list[str] = []
    for index, item in enumerate(selected, 1):
        try:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        if not str(text).strip():
            raise ValueError(f"Calibration record {index} produced empty chat text")
        texts.append(str(text))

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(sample: dict[str, str]) -> dict[str, Any]:
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from compressed_tensors.offload import dispatch_model
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.awq import AWQModifier
    except ImportError as exc:
        raise RuntimeError(
            "AWQ quantization requires llmcompressor and compressed-tensors. "
            "Install the project dependencies, then run: pip install llmcompressor"
        ) from exc
    return dispatch_model, oneshot, QuantizationModifier, AWQModifier


def _load_model_and_tokenizer(model_path: Path, dtype_name: str) -> tuple[Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "AWQ calibration requires a CUDA GPU. The current PyTorch environment "
            "does not report torch.cuda.is_available() == True."
        )

    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, **model_kwargs
        )
    except TypeError:
        # Compatibility with older Transformers versions.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=dtype, **model_kwargs
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, dtype


def _verify_output(output_path: Path) -> dict[str, Any]:
    config_path = output_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Quantized output is missing config.json: {output_path}")
    weight_files = sorted(
        list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
    )
    if not weight_files:
        raise RuntimeError(f"Quantized output contains no model weight files: {output_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    has_quantization_metadata = any(
        key in config for key in ("quantization_config", "quantization_config_file")
    ) or (output_path / "quantization_config.json").exists()
    if not has_quantization_metadata:
        raise RuntimeError(
            "Output has model weights but no quantization metadata. Refusing to "
            "label it as an AWQ checkpoint."
        )
    return {
        "weight_files": [item.name for item in weight_files],
        "weight_file_count": len(weight_files),
        "has_quantization_metadata": has_quantization_metadata,
    }


def quantize_model(
    model_path: str,
    output_path: str,
    bits: int = 4,
    *,
    calibration_data: str = DEFAULT_CALIBRATION_DATA,
    num_calibration_samples: int = DEFAULT_NUM_CALIBRATION_SAMPLES,
    max_seq_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    seed: int = DEFAULT_SEED,
    dtype: str = "auto",
    scheme: str = "W4A16_ASYM",
    overwrite: bool = False,
) -> Path:
    """Create and verify a vLLM-compatible AWQ W4A16 checkpoint."""
    if bits != 4:
        raise ValueError(
            "This implementation intentionally supports AWQ INT4 only. "
            "bitsandbytes runtime 8-bit loading is not an offline quantization output."
        )

    source = _resolve_path(model_path)
    destination = _resolve_path(output_path)
    calibration_path = _resolve_path(calibration_data)
    if not source.exists():
        raise FileNotFoundError(f"Merged model does not exist: {source}")
    if not (source / "config.json").exists():
        raise ValueError(
            f"Input must be a Transformers model directory with config.json: {source}"
        )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Choose a new path or use --overwrite."
        )
    destination.mkdir(parents=True, exist_ok=True)

    records = _load_calibration_records(calibration_path)
    dispatch_model, oneshot, QuantizationModifier, AWQModifier = _load_dependencies()
    model, tokenizer, model_dtype = _load_model_and_tokenizer(source, dtype)
    calibration_dataset = _build_calibration_dataset(
        records,
        tokenizer,
        num_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        seed=seed,
    )

    recipe = [
        AWQModifier(),
        QuantizationModifier(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"],
        ),
    ]

    print(f"Input merged model: {source}")
    print(f"Calibration data: {calibration_path}")
    print(f"Calibration samples: {len(calibration_dataset)}")
    print(f"Calibration max sequence length: {max_seq_length}")
    print(f"Quantization scheme: {scheme}")
    print(f"Model dtype before quantization: {model_dtype}")
    print("Running AWQ calibration and W4A16 compression...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=len(calibration_dataset),
        pipeline="sequential",
    )

    # Let compressed-tensors place the model correctly before the optional
    # generation smoke check performed by a caller.
    dispatch_model(model)
    print(f"Saving compressed model to: {destination}")
    try:
        model.save_pretrained(str(destination), save_compressed=True)
    except TypeError as exc:
        raise RuntimeError(
            "The installed llmcompressor/transformers version does not support "
            "save_compressed=True; upgrade llmcompressor and compressed-tensors."
        ) from exc
    tokenizer.save_pretrained(str(destination))

    verification = _verify_output(destination)
    manifest = {
        "method": "AWQ",
        "scheme": scheme,
        "bits": bits,
        "activation_precision": "FP16",
        "base_input_model": str(source),
        "base_config_sha256": _file_sha256(source / "config.json"),
        "calibration_data": str(calibration_path),
        "calibration_data_sha256": _file_sha256(calibration_path),
        "calibration_records_available": len(records),
        "calibration_records_used": len(calibration_dataset),
        "max_seq_length": max_seq_length,
        "seed": seed,
        "dtype_before_quantization": str(model_dtype).replace("torch.", ""),
        "input_size_bytes": _directory_size(source),
        "output_size_bytes": _directory_size(destination),
        "python": sys.version,
        "platform": platform.platform(),
        "verification": verification,
        "note": "W4A16 AWQ checkpoint for vLLM; not a QLoRA/NF4 training checkpoint.",
    }
    (destination / "quantization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {verification['weight_file_count']} quantized weight file(s).")
    print(f"Input size: {manifest['input_size_bytes'] / 1024**3:.2f} GiB")
    print(f"Output size: {manifest['output_size_bytes'] / 1024**3:.2f} GiB")
    print("Done!")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a merged model to AWQ INT4/W4A16 for vLLM"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--calibration_data", default=DEFAULT_CALIBRATION_DATA)
    parser.add_argument("--num_calibration_samples", type=int, default=DEFAULT_NUM_CALIBRATION_SAMPLES)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--scheme", choices=("W4A16_ASYM", "W4A16"), default="W4A16_ASYM")
    parser.add_argument(
        "--bits",
        type=int,
        choices=(4,),
        default=4,
        help="AWQ INT4 only; kept for compatibility with the original skeleton",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    quantize_model(
        args.model_path,
        args.output_path,
        bits=args.bits,
        calibration_data=args.calibration_data,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        dtype=args.dtype,
        scheme=args.scheme,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
