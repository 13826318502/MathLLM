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

    @classmethod
    def from_env(cls) -> "DeploymentSettings":
        quantization = os.getenv("MATHLLM_QUANTIZATION", "").strip() or None
        return cls(
            model_path=os.getenv(
                "MATHLLM_MODEL_PATH", "./outputs/correction-round-5-merged"
            ),
            served_model_name=os.getenv("MATHLLM_MODEL_NAME", "mathllm-round5"),
            host=os.getenv("MATHLLM_VLLM_HOST", "0.0.0.0"),
            port=_int("MATHLLM_VLLM_PORT", 8000),
            max_model_len=_int("MATHLLM_MAX_MODEL_LEN", 2048),
            gpu_memory_utilization=_float("MATHLLM_GPU_MEMORY_UTILIZATION", 0.9),
            dtype=os.getenv("MATHLLM_DTYPE", "float16"),
            quantization=quantization,
        )


settings = DeploymentSettings.from_env()
