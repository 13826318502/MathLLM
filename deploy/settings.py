"""Environment-backed vLLM deployment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class DeploymentSettings:
    model_path: str
    served_model_name: str
    host: str
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    dtype: str
    quantization: str | None
    # vLLM must be told which parser turns a model's raw output into tool calls.
    # Qwen2.5 uses the hermes format. Set the env var to an empty string to
    # disable function calling (the app then falls back to prompted JSON).
    tool_call_parser: str | None

    @classmethod
    def from_env(cls) -> "DeploymentSettings":
        quantization = os.getenv("MATHLLM_QUANTIZATION", "").strip() or None
        tool_call_parser = os.getenv("MATHLLM_TOOL_CALL_PARSER", "hermes").strip() or None
        return cls(
            model_path=os.getenv("MATHLLM_MODEL_PATH", "./model"),
            served_model_name=os.getenv("MATHLLM_MODEL_NAME", "mathllm-model"),
            host=os.getenv("MATHLLM_VLLM_HOST", "0.0.0.0"),
            port=_int("MATHLLM_VLLM_PORT", 8000),
            max_model_len=_int("MATHLLM_MAX_MODEL_LEN", 2048),
            gpu_memory_utilization=_float("MATHLLM_GPU_MEMORY_UTILIZATION", 0.9),
            dtype=os.getenv("MATHLLM_DTYPE", "float16"),
            quantization=quantization,
            tool_call_parser=tool_call_parser,
        )


settings = DeploymentSettings.from_env()
