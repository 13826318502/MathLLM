"""基于 vLLM 部署精调后的数学解题模型

提供 OpenAI 兼容的 API 接口，支持流式输出。

启动方式:
    python deploy/server.py

    # 或直接使用 vLLM CLI:
    python -m vllm.entrypoints.openai.api_server \
        --model ./outputs/math-lora-merged \
        --host 0.0.0.0 --port 8000 \
        --quantization awq

API 用法（与 OpenAI API 兼容）:
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "math-solver",
            "messages": [{"role": "user", "content": "求解方程 x^2 - 5x + 6 = 0"}],
            "stream": true
        }'
"""

import yaml
import subprocess
from pathlib import Path


def load_deploy_config(config_path: str = "configs/deploy_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def start_vllm_server(config: dict):
    """使用 vLLM 启动模型推理服务"""
    model_cfg = config["model"]
    server_cfg = config["server"]

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["model_path"],
        "--host", server_cfg["host"],
        "--port", str(server_cfg["port"]),
        "--max-model-len", str(model_cfg["max_model_len"]),
        "--gpu-memory-utilization", str(model_cfg["gpu_memory_utilization"]),
        "--dtype", model_cfg["dtype"],
    ]

    if model_cfg.get("quantization"):
        cmd.extend(["--quantization", model_cfg["quantization"]])

    print(f"Starting vLLM server...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    config = load_deploy_config()
    start_vllm_server(config)
