# Week 3 开发计划：FastAPI、Gradio 与端到端应用

## 1. 本周目标

第三周的目标是把第二周已经启动的 vLLM 模型服务，封装成可以实际使用的数学解题 Web 应用：

    用户输入题目
          ↓
    Gradio 前端
          ↓ HTTP/SSE
    FastAPI 后端
          ↓ OpenAI 兼容 API
    vLLM 推理服务
          ↓
    数学解题模型
          ↓
    流式返回解题过程

本周重点是完成应用层功能：

- FastAPI 后端接口；
- SSE 流式输出；
- 单题解题；
- 多轮对话；
- Gradio 网页界面；
- 前后端联调；
- 20–30 道题的手动测试和 Bad Case 记录。

本周不重新训练模型，也不做 RAG、监控和多模型对比等扩展功能。

## 2. 开始前的前置条件

第三周开始前，应确认第二周至少完成：

- vLLM 可以正常启动；
- GET /v1/models 可以返回模型信息；
- POST /v1/chat/completions 可以返回答案；
- 模型能够回答一条简单数学题；
- 已明确当前使用的是 FP16 还是 AWQ/INT4 模型。

如果 vLLM 还没有成功启动，应先处理第二周的部署问题，不要直接调试 Gradio。

当前项目需要重点核对：

- configs/deploy_config.yaml 中的 model_path 是否指向实际模型；
- 使用 AWQ 时，quantization 是否为 awq；
- app/api/main.py 中的 VLLM_BASE_URL 是否与实际服务地址一致；
- MODEL_NAME 是否与 vLLM 返回的模型名称一致。

## 3. 本周最终交付物

代码交付物：

- app/api/main.py 中的 /api/health、/api/solve、/api/chat 全部可用；
- app/frontend/gradio_app.py 可以启动 Web 界面；
- 支持流式输出；
- 支持 Markdown 和 LaTeX 渲染；
- 支持清空对话和多轮追问；
- 后端能够处理 vLLM 错误、超时和空响应。

测试交付物：

- vLLM、FastAPI、Gradio 的启动记录；
- API 请求示例和测试结果；
- 20–30 道手动测试结果；
- 至少 5 个 Bad Case；
- Bad Case 原因分类；
- 至少一张界面截图或一段演示视频。

## 4. 第一天：确认 vLLM 和应用通信链路

### 任务

1. 启动 vLLM；
2. 验证 vLLM API；
3. 确认 FastAPI 与 vLLM 的地址；
4. 确认项目依赖；
5. 建立第三周测试记录。

先直接测试 vLLM：

    curl http://localhost:8000/v1/models

再测试聊天接口：

    curl http://localhost:8000/v1/chat/completions
      -H "Content-Type: application/json"
      -d '{"model":"math-solver","messages":[{"role":"user","content":"求解 x^2-5x+6=0"}],"stream":false,"temperature":0.2,"max_tokens":512}'

确认 app/api/main.py 中的以下配置：

    VLLM_BASE_URL = "http://localhost:8000/v1"
    MODEL_NAME = "math-solver"

与实际服务完全一致。

建立测试记录，建议记录：

    日期
    GPU 型号
    vLLM 模型路径
    模型格式
    vLLM 地址
    模型名称
    测试题
    返回状态
    备注

## 5. 第二天：实现 FastAPI 健康检查和单题接口

修改文件：

    app/api/main.py

### /api/health

健康检查不能一直固定返回 status=ok，而应实际请求：

    GET http://localhost:8000/v1/models

正常时返回 vLLM 已连接和当前模型信息。vLLM 未启动时，返回明确的异常状态和原因。

需要测试：

- vLLM 正常启动；
- vLLM 停止；
- vLLM 地址配置错误；
- 请求超时。

### /api/solve

请求格式：

    {
      "question": "求解 x^2-5x+6=0",
      "stream": true
    }

后端处理顺序：

1. 检查 question 不能为空；
2. 添加固定 system prompt；
3. 组织 system 和 user 消息；
4. 调用 vLLM 的 chat completions 接口；
5. 根据 stream 参数返回流式或完整结果；
6. 处理错误、超时和空响应。

发送给 vLLM 的消息应包含：

    system：数学解题助手的行为要求
    user：用户输入的数学题

非流式接口应返回完整回答，并明确 finished=true。

使用 httpx 调用 vLLM。FastAPI 的异步接口中不要使用阻塞式 requests。

## 6. 第三天：实现 SSE 流式输出

### 目标

用户提交问题后，前端能够逐步看到模型回答，而不是等待整个回答生成完毕。

vLLM 流式响应一般包含多段 data 内容，最后以 [DONE] 结束。FastAPI 需要：

1. 使用 httpx 的异步流式请求；
2. 逐行读取 vLLM 响应；
3. 去除 data: 前缀；
4. 解析 JSON；
5. 提取 choices、delta、content；
6. 转换为前端使用的统一 SSE 数据；
7. 收到 [DONE] 后发送结束事件。

建议返回格式：

    data: {"content":"首先","finished":false}

    data: {"content":"分析","finished":false}

    data: {"content":"","finished":true}

必须注意：

- 响应类型设置为 text/event-stream；
- 每条事件使用正确的换行；
- 正确处理空 delta；
- 客户端断开时关闭后端请求；
- 设置连接和读取超时；
- 不把 Python 异常堆栈直接返回给用户。

使用 curl 测试：

    curl -N http://localhost:8080/api/solve
      -H "Content-Type: application/json"
      -d '{"question":"求解 x^2-5x+6=0","stream":true}'

确认回答是逐步输出的。

## 7. 第四天：实现多轮对话接口

修改 app/api/main.py 中的 /api/chat。

请求格式：

    {
      "messages": [
        {"role":"user","content":"求解 x^2-5x+6=0"},
        {"role":"assistant","content":"因式分解得 x=2 或 x=3。"},
        {"role":"user","content":"如果把 6 改成 4 呢？"}
      ],
      "stream": true
    }

后端处理规则：

- system prompt 只添加一次；
- 保留 user 和 assistant 历史消息；
- 消息顺序必须合法；
- 最后一条消息应为 user；
- 空消息和未知 role 应被拒绝；
- 限制历史长度，避免超过模型上下文窗口；
- 不要在每次请求中重复添加 system prompt。

至少测试：

1. 先让模型解一道方程；
2. 追问“换一种方法”；
3. 修改题目中的一个条件；
4. 追问上一步的计算原因；
5. 清空后确认新问题不携带旧历史。

## 8. 第五天：实现 Gradio 前端

修改文件：

    app/frontend/gradio_app.py

页面至少包含：

- 项目标题；
- 数学题目输入框；
- 对话显示区域；
- 解题按钮；
- 清空按钮；
- 流式输出；
- Markdown 和 LaTeX 渲染。

当前项目已经使用 gr.Chatbot 和 render_markdown=True，因此回答中的 LaTeX 可以直接展示。

### solve_math()

前端需要：

1. 接收题目和历史对话；
2. 调用 FastAPI 的 /api/solve 或 /api/chat；
3. 流式读取 SSE；
4. 每收到一段内容就更新 chatbot；
5. 回答结束后保存完整历史；
6. 请求失败时显示友好错误。

虽然当前前端可以使用 requests，但必须逐行读取流式响应，不能一次性读取完整内容。

### 事件绑定

完成解题按钮和清空按钮的绑定。检查：

- 空题目不会发送请求；
- 回答过程中界面持续更新；
- 不会重复提交；
- 点击清空后输入框和历史都清空；
- 网络错误不会让页面卡死。

启动：

    python app/frontend/gradio_app.py

浏览器访问：

    http://localhost:7860

## 9. 第六天：前后端联调

启动顺序：

    终端 1：vLLM，端口 8000
    终端 2：FastAPI，端口 8080
    终端 3：Gradio，端口 7860

启动 FastAPI：

    python -m app.api.main

启动 Gradio：

    python app/frontend/gradio_app.py

按照以下顺序联调：

1. 直接测试 vLLM；
2. 测试 FastAPI 的 /api/health；
3. 测试 /api/solve 非流式；
4. 测试 /api/solve 流式；
5. 测试 /api/chat；
6. 最后从浏览器测试 Gradio。

健康检查：

    curl http://localhost:8080/api/health

非流式解题：

    curl http://localhost:8080/api/solve
      -H "Content-Type: application/json"
      -d '{"question":"求函数 f(x)=x^3-3x+1 的极值点","stream":false}'

流式解题：

    curl -N http://localhost:8080/api/solve
      -H "Content-Type: application/json"
      -d '{"question":"求函数 f(x)=x^3-3x+1 的极值点","stream":true}'

### 错误场景测试

至少测试：

- vLLM 未启动；
- 题目为空；
- 题目过长；
- 请求格式错误；
- vLLM 返回 500；
- vLLM 响应超时；
- 用户中途关闭页面；
- 多次快速点击提交按钮。

## 10. 第七天：手动测试和 Bad Case 分析

按照开发指南手动测试 20–30 道题。

建议覆盖：

| 类型 | 数量 |
|---|---:|
| 高等数学 | 6–8 |
| 线性代数 | 5–6 |
| 概率论 | 4–5 |
| 基础代数 | 3–5 |
| 多轮追问 | 3–5 |

每道题记录：

    编号
    题目
    学科
    是否答对
    步骤是否完整
    公式是否正常
    是否流式返回
    总耗时
    是否报错
    Bad Case 分类

Bad Case 分类：

- 模型数学计算错误；
- 解题步骤缺失；
- 最终答案与过程不一致；
- LaTeX 格式错误；
- system prompt 没有生效；
- 多轮历史丢失；
- SSE 拼接错误；
- API 超时；
- 前端显示异常。

问题定位规则：

    直接调用 vLLM 就错误       → 模型或模型配置问题
    vLLM 正常，FastAPI 错误    → 后端接口问题
    FastAPI 正常，页面错误     → Gradio 或 SSE 解析问题

## 11. 本周不要做的事情

- 不要在第三周重新做 LoRA 训练；
- 不要只从浏览器测试而跳过 API 测试；
- 不要在异步 FastAPI 接口中使用阻塞式请求；
- 不要把 SSE 响应一次性读完后再显示；
- 不要让每轮对话重复添加 system prompt；
- 不要把完整异常堆栈直接暴露给用户；
- 不要在基础接口完成前增加 RAG、监控或多模型对比；
- 不要把 20–30 道题的测试结果直接写成正式准确率。

## 12. 第三周验收标准

- [ ] vLLM 服务可以稳定运行；
- [ ] /api/health 能真实检查 vLLM 状态；
- [ ] /api/solve 非流式接口可用；
- [ ] /api/solve SSE 流式接口可用；
- [ ] /api/chat 可以保留多轮上下文；
- [ ] system prompt 只添加一次；
- [ ] FastAPI 能处理超时、空输入和 vLLM 错误；
- [ ] Gradio 页面可以启动；
- [ ] 数学公式能够正常渲染；
- [ ] 回答能够逐步显示；
- [ ] 清空按钮和多轮对话可用；
- [ ] 已完成 20–30 道题的手动测试；
- [ ] 已记录 Bad Case 并区分模型问题与应用问题；
- [ ] 已保存至少一张界面截图或演示视频；
- [ ] 没有把未经验证的数据写入项目说明或简历。

## 13. 与第四周的衔接

第三周结束后，第四周继续完成：

- eval/evaluate.py 的准确率和延迟评测；
- 基座模型、精调模型和 GPT 模型对比；
- 消融实验；
- matplotlib 图表；
- Docker 部署；
- README 和项目展示材料整理。

第三周必须保存：

- API 测试结果；
- 手动测试记录；
- Bad Case；
- 模型版本；
- vLLM 参数；
- 前端截图。

## 14. 相关文件

- 开发指南：docs/development-guide.md
- 第二周计划：docs/development-plan/week2-training-and-deployment.md
- FastAPI 后端：app/api/main.py
- Gradio 前端：app/frontend/gradio_app.py
- vLLM 启动脚本：deploy/server.py
- 部署配置：configs/deploy_config.yaml
- 项目依赖：requirements.txt

