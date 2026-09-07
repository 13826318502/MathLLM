"""Start the vLLM OpenAI-compatible model service.

The launcher reads deployment settings from environment variables so it does
not depend on a round-specific configuration file.

Example:
    python deploy/server.py

Optional environment variables:
    MATHLLM_MODEL_PATH=./outputs/correction-round-5-merged
    MATHLLM_MODEL_NAME=mathllm-round5
    MATHLLM_VLLM_PORT=8000
"""

from __future__ import annotations

import subprocess

from deploy.settings import DeploymentSettings, settings


def build_command(config: DeploymentSettings) -> list[str]:
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_path,
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--dtype",
        config.dtype,
    ]
    if config.quantization:
        command.extend(["--quantization", config.quantization])
    return command


def start_vllm_server(config: DeploymentSettings = settings) -> None:
    command = build_command(config)
    print(f"Starting vLLM with model: {config.model_path}")
    print(f"Served model name: {config.served_model_name}")
    print(f"Command: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    start_vllm_server()
