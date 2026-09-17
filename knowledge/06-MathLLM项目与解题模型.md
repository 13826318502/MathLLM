# MathLLM 项目与解题模型介绍

MathLLM 是一个面向大学数学解题场景的端到端项目，同时包含模型微调流程和应用功能。

## 解题模型

- 基座模型：Qwen2.5-7B-Instruct。
- 微调方式：使用 Transformers、TRL 和 PEFT 进行 LoRA / QLoRA 监督微调。
- 当前版本：mathllm-round7。Round7 以 Round6 的合并模型为基座，用 QLoRA 做一轮纠错训练，
  数据为 150 条新纠错题加 65 条普通回放题，共 215 条；配置为 4-bit 量化、1 个 epoch、
  batch size 1、梯度累积 8、学习率 5e-6。
- 最优 checkpoint：outputs/correction-round-7/checkpoint-10，合并后得到独立模型
  outputs/correction-round-7-merged。
- 量化与部署：可做 AWQ W4A16 量化；本地 CPU 使用 GGUF Q4_K_M
  （models/cpu/correction-round-7-q4_k_m.gguf），通过 Ollama 以 OpenAI 兼容接口
  http://127.0.0.1:11434/v1 提供服务。

## 模型评测结果

- 原始测试集 116 条：程序准确率 66.38%，LLM Judge 77.19%。
- Round7 纠错验证集 30 条：程序准确率 80.00%，LLM Judge 96.67%。
- 历史回归集 42 条：程序准确率 38.10%，LLM Judge 43.59%。

## Agent 架构

Agent 层由三部分组成：

1. 结构化路由：把问题分成 math、knowledge、general 三类，并选择对应工具。
2. 有界 ReAct 循环：按需调用工具、观察结果、再决定下一步，最多 4 步，不会死循环。
3. 独立验证：答案返回前经过一次不依赖生成模型的校验。

Agent 可用的工具包括：solve_math_problem（求解具体数学题）、search_knowledge（知识库检索）、
calculate_expression（AST 白名单安全计算）、check_math_answer（答案校验）。

## 模型分工与模型路由

编排任务（路由、规划、答案验证、来源核对）走云端大模型，数学求解走本地微调模型
mathllm-round7。云端不可用时按 MATHLLM_ORCHESTRATOR_FALLBACK 处理，默认回退本地；
数学题的最终答案直接使用本地模型的输出，云端不改写。

## 答案独立验证

验证遵循「模型只翻译，SymPy 当裁判」的原则：模型负责把题目翻译成可代入的等式、
从解答中抽取最终答案值，判定由 SymPy 代入计算完成。方程与计算题使用代入检验或
独立求值；知识类问题使用检索片段做来源核对。被证伪时带反例自动重算一次，
无法独立验证时记为 unknown 且不重试。

## 知识库检索（RAG）

知识库检索基于本地 Chroma 向量库，数据源是 knowledge/ 目录下的 Markdown 文档。
文档按 500 字符切块、重叠 50 字符；embedding 使用中文模型 BAAI/bge-small-zh-v1.5；
默认检索条数为 3。索引写入 data/chroma，可用 python -m app.services.rag_service 重建。

## 可观测性

每次 Agent 运行都会把路由决策、工具调用、验证结论、耗时和 token 用量落盘为
data/traces/runs.jsonl，可通过 GET /api/metrics 查看运行指标、GET /api/traces 查看
逐次运行明细。Web 前端提供「Agent 模式」展示执行轨迹，「运行观测」页面展示整体指标与调用链。
