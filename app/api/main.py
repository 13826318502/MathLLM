"""FastAPI 后端服务：提供数学解题 API

功能:
    - /api/solve: 提交数学问题，返回解题过程（支持流式）
    - /api/chat: 多轮对话接口
    - /api/health: 健康检查

启动:
    python -m app.api.main
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import httpx
import json

app = FastAPI(title="MathLLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "math-solver"

SYSTEM_PROMPT = """你是一个专业的数学解题助手。请按照以下要求回答数学问题：
1. 先分析题目要求
2. 给出详细的解题步骤
3. 用 LaTeX 格式书写数学公式
4. 最后给出明确的答案"""


class SolveRequest(BaseModel):
    question: str
    stream: bool = True


class ChatRequest(BaseModel):
    messages: list[dict]
    stream: bool = True


@app.get("/api/health")
async def health():
    # TODO: 检查 vLLM 服务是否可用
    return {"status": "ok"}


@app.post("/api/solve")
async def solve(req: SolveRequest):
    """提交数学问题，流式返回解题过程"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.question},
    ]

    # TODO: 调用 vLLM API
    # - 如果 stream=True，返回 EventSourceResponse（SSE 流式）
    # - 如果 stream=False，返回完整响应
    pass


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """多轮对话接口"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.messages

    # TODO: 调用 vLLM API，支持流式和非流式
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
