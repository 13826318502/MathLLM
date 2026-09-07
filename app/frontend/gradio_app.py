"""Student-friendly Gradio page for MathLLM.

启动:
    python -m app.frontend.gradio_app
"""

from __future__ import annotations

import gradio as gr

from pathlib import Path

try:
    from .controller import clear_history, refresh_status, solve_math
    from .examples import EXAMPLE_QUESTIONS
    from .favorites import (
        add_favorite,
        append_instruction,
        clear_favorites,
        load_favorite,
        remove_favorite,
        sync_favorites,
    )
    from .ui_theme import APP_CSS, APP_THEME
except ImportError:  # Support running this file directly from the project root.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.frontend.controller import clear_history, refresh_status, solve_math
    from app.frontend.examples import EXAMPLE_QUESTIONS
    from app.frontend.favorites import (
        add_favorite,
        append_instruction,
        clear_favorites,
        load_favorite,
        remove_favorite,
        sync_favorites,
    )
    from app.frontend.ui_theme import APP_CSS, APP_THEME


LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]


def _load_ui_javascript() -> str:
    return Path(__file__).with_name("ui_behavior.js").read_text(encoding="utf-8")


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="MathLLM - 数学学习助手") as demo:
        favorites_state = gr.BrowserState(
            default_value=[],
            storage_key="mathllm_favorites",
        )
        memory_state = gr.State("")
        memory_cursor_state = gr.State(0)
        with gr.Column(elem_id="app-shell"):
            with gr.Row(elem_id="topbar"):
                with gr.Column(scale=5, min_width=360):
                    gr.Markdown(
                        "# MathLLM 数学学习助手",
                        elem_id="brand-title",
                    )
                    gr.Markdown(
                        "把不会的题讲明白，把解题思路一步一步说清楚。",
                        elem_id="brand-subtitle",
                    )
                with gr.Column(scale=1, min_width=150):
                    status = gr.Markdown(
                        "⏳ **正在检查服务**",
                        elem_id="model-badge",
                    )
                    refresh_btn = gr.Button("刷新状态", size="sm")

            with gr.Row(elem_id="main-layout"):
                with gr.Column(scale=7, elem_id="chat-panel"):
                    gr.Markdown("### 和助手一起解题")
                    chatbot = gr.Chatbot(
                        label="对话",
                        show_label=False,
                        height=530,
                        render_markdown=True,
                        latex_delimiters=LATEX_DELIMITERS,
                        layout="bubble",
                        buttons=["copy_all"],
                        placeholder="你可以从一道数学题开始，也可以继续追问“为什么”。",
                        elem_id="chatbot",
                    )
                    mode = gr.Radio(
                        choices=[
                            ("单题解答（不带历史）", "solve"),
                            ("连续追问（保留上下文）", "follow_up"),
                        ],
                        value="solve",
                        label="答题模式",
                        info="新题目建议使用单题解答；追问上一题时使用连续追问。",
                        elem_id="answer-mode",
                    )
                    question_input = gr.Textbox(
                        label="题目输入",
                        show_label=False,
                        placeholder="例如：求解方程 x² - 5x + 6 = 0",
                        lines=3,
                        max_lines=8,
                        elem_id="question-box",
                    )
                    with gr.Row():
                        submit_btn = gr.Button(
                            "开始解题  →",
                            variant="primary",
                            elem_id="submit-button",
                        )
                        clear_btn = gr.Button(
                            "清空对话",
                            elem_id="clear-button",
                        )
                    with gr.Row(elem_id="question-tools"):
                        favorite_btn = gr.Button(
                            "收藏当前题目",
                            size="sm",
                            elem_id="favorite-button",
                        )
                        hint_btn = gr.Button("只给我提示", size="sm")
                        simple_btn = gr.Button("讲简单一点", size="sm")
                    gr.Markdown(
                        "支持中文题目、Markdown 和 LaTeX。答案仅供学习参考，请检查关键步骤。",
                        elem_id="footer-note",
                    )

                with gr.Column(scale=3, elem_id="side-panel"):
                    gr.Markdown("### 你可以这样问", elem_classes=["side-heading"])
                    gr.Markdown(
                        "直接输入题目即可。想要更简单的解释，可以继续追问“能再讲简单一点吗？”",
                        elem_classes=["side-copy"],
                    )
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=question_input,
                        label="试试示例题",
                        example_labels=[
                            "一元二次方程",
                            "函数极值",
                            "矩阵行列式",
                            "概率计算",
                        ],
                        examples_per_page=4,
                        elem_id="examples-holder",
                    )
                    with gr.Accordion("我的收藏", open=True, elem_id="favorites-card"):
                        favorite_view = gr.Markdown(
                            "还没有收藏题目。\n\n输入题目后点击 **收藏当前题目**，方便之后继续练习。",
                            elem_id="favorites-view",
                        )
                        favorite_select = gr.Dropdown(
                            choices=[],
                            label="选择一道收藏题",
                            info="选择后会自动放回输入框",
                            interactive=True,
                            elem_id="favorite-select",
                        )
                        with gr.Row():
                            remove_favorite_btn = gr.Button("移除选中题", size="sm")
                            clear_favorites_btn = gr.Button("清空收藏", size="sm")
                        favorite_notice = gr.Markdown("", elem_id="favorite-notice")
                    with gr.Accordion("快捷学习工具", open=True):
                        gr.Markdown(
                            "先输入题目，再选择一种学习方式，最后点击 **开始解题**。",
                            elem_classes=["side-copy"],
                        )
                        check_btn = gr.Button("检查我的答案", size="sm")
                    with gr.Accordion("学习小贴士", open=True):
                        gr.Markdown(
                            "- 想看思路：输入“先给我提示”\\n"
                            "- 想换方法：输入“换一种方法解”\\n"
                            "- 想检查答案：输入你的答案并问“对吗？”\\n"
                            "- 想继续讨论：直接在同一个对话中追问"
                        )
                    with gr.Accordion("当前版本", open=False):
                        gr.Markdown(
                            "当前版本支持单题解答、多轮追问、流式回答、"
                            "Markdown/LaTeX 渲染和服务状态检查。"
                        )

            submit_btn.click(
                solve_math,
                inputs=[question_input, chatbot, mode, memory_state, memory_cursor_state],
                outputs=[chatbot, question_input, memory_state, memory_cursor_state],
            )
            question_input.submit(
                solve_math,
                inputs=[question_input, chatbot, mode, memory_state, memory_cursor_state],
                outputs=[chatbot, question_input, memory_state, memory_cursor_state],
            )
            clear_btn.click(
                clear_history,
                outputs=[chatbot, question_input, memory_state, memory_cursor_state],
            )
            refresh_btn.click(refresh_status, outputs=status)
            favorite_btn.click(
                add_favorite,
                inputs=[question_input, favorites_state],
                outputs=[favorites_state, favorite_view, favorite_select, favorite_notice],
            )
            remove_favorite_btn.click(
                remove_favorite,
                inputs=[favorite_select, favorites_state],
                outputs=[favorites_state, favorite_view, favorite_select, favorite_notice],
            )
            clear_favorites_btn.click(
                clear_favorites,
                outputs=[favorites_state, favorite_view, favorite_select, favorite_notice],
            )
            favorite_select.change(
                load_favorite,
                inputs=favorite_select,
                outputs=question_input,
            )
            hint_btn.click(
                lambda question: append_instruction(question, "请先给我解题思路和关键提示，不要直接给出最终答案。"),
                inputs=question_input,
                outputs=question_input,
            )
            simple_btn.click(
                lambda question: append_instruction(question, "请用适合初学者的简单语言解释，并说明每一步为什么这样做。"),
                inputs=question_input,
                outputs=question_input,
            )
            check_btn.click(
                lambda question: append_instruction(question, "请检查我写出的答案或思路，指出错误并给出改进建议。"),
                inputs=question_input,
                outputs=question_input,
            )
            demo.load(
                sync_favorites,
                inputs=favorites_state,
                outputs=[favorite_view, favorite_select],
            )
            demo.load(refresh_status, outputs=status)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=APP_THEME,
        css=APP_CSS,
        js=_load_ui_javascript(),
    )
