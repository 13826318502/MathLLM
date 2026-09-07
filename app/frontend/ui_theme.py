"""Visual theme and CSS for the Gradio interface."""

from __future__ import annotations

import gradio as gr


APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=("Inter", "Microsoft YaHei", "sans-serif"),
    font_mono=("JetBrains Mono", "Consolas", "monospace"),
).set(
    body_background_fill="#f6f8fc",
    body_background_fill_dark="#0f172a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#111827",
    block_border_width="0px",
    block_shadow="0 12px 36px rgba(15, 23, 42, 0.07)",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#1e293b",
)


APP_CSS = r"""
:root {
    --mathllm-indigo: #4f46e5;
    --mathllm-indigo-dark: #3730a3;
    --mathllm-ink: #172033;
    --mathllm-muted: #64748b;
    --mathllm-border: #e2e8f0;
    --mathllm-surface: rgba(255, 255, 255, 0.86);
}

body {
    background:
        radial-gradient(circle at 8% 0%, rgba(99, 102, 241, 0.16), transparent 30rem),
        radial-gradient(circle at 100% 20%, rgba(14, 165, 233, 0.10), transparent 26rem),
        #f6f8fc;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 28px 24px 38px !important;
}

#app-shell {
    gap: 18px;
}

#topbar {
    align-items: center;
    padding: 22px 26px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,.94), rgba(241,245,249,.84));
    box-shadow: 0 18px 50px rgba(30, 41, 59, 0.10);
}

#brand-title {
    margin: 0 !important;
    color: var(--mathllm-ink);
    letter-spacing: -0.03em;
}

#brand-subtitle {
    margin-top: 4px !important;
    color: var(--mathllm-muted);
    font-size: 0.95rem;
}

#model-badge {
    display: inline-block;
    padding: 9px 14px;
    border: 1px solid #c7d2fe;
    border-radius: 999px;
    background: #eef2ff;
    color: var(--mathllm-indigo-dark);
    font-size: 0.82rem;
    text-align: center;
}

#main-layout {
    align-items: stretch;
    gap: 18px;
}

#chat-panel,
#side-panel {
    border: 1px solid rgba(226, 232, 240, 0.9);
    border-radius: 22px;
    background: var(--mathllm-surface);
    box-shadow: 0 14px 38px rgba(15, 23, 42, 0.06);
}

#chat-panel {
    min-height: 650px;
    padding: 16px;
}

#side-panel {
    padding: 20px;
}

#chatbot {
    border: 0 !important;
    background: transparent !important;
}

#chatbot .message {
    border-radius: 16px !important;
}

#question-box textarea {
    min-height: 94px !important;
    border: 1px solid var(--mathllm-border) !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03);
    font-size: 1rem !important;
    line-height: 1.7 !important;
}

#question-box textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14) !important;
}

#question-tools {
    gap: 8px;
    margin-top: 2px;
}

#question-tools button,
#favorites-card button,
#side-panel button {
    border-radius: 11px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    color: #334155 !important;
    font-weight: 600 !important;
}

#question-tools button:hover,
#favorites-card button:hover,
#side-panel button:hover {
    border-color: #a5b4fc !important;
    background: #eef2ff !important;
    color: #3730a3 !important;
}

#favorite-button {
    border-color: #c7d2fe !important;
    background: #eef2ff !important;
    color: #3730a3 !important;
}

#favorites-card,
#favorites-card > div {
    border-radius: 16px !important;
}

#favorites-view {
    max-height: 180px;
    overflow-y: auto;
    color: var(--mathllm-muted);
    font-size: 0.86rem;
    line-height: 1.65;
}

#favorite-select {
    margin-top: 10px;
}

#favorite-notice {
    min-height: 24px;
    margin: 4px 0 0;
    color: var(--mathllm-indigo-dark);
    font-size: 0.82rem;
}

#chatbot .prose,
#chatbot .message-content {
    line-height: 1.75 !important;
}

#chatbot .prose pre {
    border: 1px solid #e0e7ff;
    border-radius: 12px;
    background: #f8faff;
}

#chatbot .prose code {
    border-radius: 5px;
    background: #eef2ff;
    color: #3730a3;
}

#submit-button,
#clear-button {
    min-height: 45px;
    border-radius: 13px !important;
    font-weight: 600 !important;
}

#submit-button {
    background: linear-gradient(135deg, #6366f1, #4338ca) !important;
}

#submit-button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

#clear-button {
    border: 1px solid var(--mathllm-border) !important;
    background: #ffffff !important;
}

.side-heading {
    color: var(--mathllm-ink);
    font-weight: 700;
}

.side-copy,
#footer-note {
    color: var(--mathllm-muted);
    font-size: 0.9rem;
    line-height: 1.75;
}

#examples-holder button {
    min-height: 42px !important;
    border-radius: 12px !important;
    border: 1px solid #e0e7ff !important;
    background: #f8faff !important;
    color: #3730a3 !important;
    text-align: left !important;
}

#examples-holder button:hover {
    border-color: #a5b4fc !important;
    background: #eef2ff !important;
}

@media (max-width: 850px) {
    .gradio-container {
        padding: 16px 10px 24px !important;
    }
    #topbar {
        padding: 18px;
        border-radius: 18px;
    }
    #chat-panel {
        min-height: 520px;
    }
    #side-panel {
        padding: 16px;
    }
}
"""
