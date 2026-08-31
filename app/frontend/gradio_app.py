"""Gradio 前端界面

简洁的 Web 界面，支持:
    - 输入数学问题，流式显示解题过程
    - LaTeX 公式渲染
    - 对话历史

启动:
    python app/frontend/gradio_app.py
"""

import gradio as gr
import requests
import json


def solve_math(question: str, history: list) -> tuple:
    """调用后端 API 解题，支持流式输出"""
    # TODO: 调用 FastAPI /api/solve 接口
    # - 使用 requests 发送请求
    # - 流式读取 SSE 响应
    # - 逐步 yield 更新界面
    pass


def clear_history():
    return [], ""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="MathLLM - 数学解题助手") as demo:
        gr.Markdown("# MathLLM 数学解题助手")
        gr.Markdown("基于精调 Qwen2.5-7B 的数学解题大模型")

        chatbot = gr.Chatbot(label="对话", height=500, render_markdown=True)
        question_input = gr.Textbox(
            label="输入数学问题",
            placeholder="例如：求解方程 x^2 - 5x + 6 = 0",
            lines=3,
        )

        with gr.Row():
            submit_btn = gr.Button("解题", variant="primary")
            clear_btn = gr.Button("清空")

        # TODO: 绑定按钮事件
        # submit_btn.click(solve_math, ...)
        # clear_btn.click(clear_history, ...)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
