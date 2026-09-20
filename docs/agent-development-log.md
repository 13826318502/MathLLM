# MathLLM Agent 改造记录

> 项目：MathLLM-数学解答系统
> 时间范围：从「结构化输出地基」到「云端编排 + 本地解题」与「调用链可观测」
> 状态：P0 / P1 / P2 / P3 / Agent 前端接入 / P5 / 模型路由 / P7 块1-2（调用链日志 + 运行指标）/ 轨迹卡分块（路径 → 回答）完成，
> 136 个单元测试通过，全部经真机（本地 Ollama + Chroma + SymPy + 浏览器）验证

---

## 一、总览

在原有「FastAPI + vLLM/Ollama + SSE + 记忆摘要」的数学解题服务之上，新增了一层 **Agent 编排能力**，并按阶段推进：

| 阶段 | 主题 | 核心产出 | 验收 |
|---|---|---|---|
| P0 | 结构化输出地基 | 让模型稳定输出可校验的 JSON 路由结果 | 连续请求均能通过 Pydantic 校验 |
| P1 | 工具层 | 统一工具契约 + 4 个工具 + 注册表 | 工具失败永不抛异常，只返回结构化失败 |
| P2 | 单 Agent ReAct 循环 | `classify → 工具 → 观察 → 决策 → 答案` 有界循环 | 不死循环、不重复调同一工具 |
| P3 | 真实 RAG | Chroma 向量库 + 中文 embedding + 检索工具 | 检索全部命中正确文档 |
| 前端接入 | Agent 可视化 | SSE 事件流 + Web 端 Agent 模式与执行轨迹卡 | 浏览器实测：路由/工具/耗时/中断全部正常 |
| P5 | 答案独立验证 | 模型只翻译、SymPy 当裁判；被证伪则带反馈重生成 | 正确根 `verified`、错误根 `refuted` 且给反例、`unknown` 不重试 |
| 模型路由 | 云端编排 + 本地解题 | 路由/规划/验证走云端，数学求解走本地，附配置弹窗 | 未配云端时行为不变；事件带实际模型名 |
| 上线修复 | 流式 + 验证 | 解题工具自身流式；推理模型空输出回退；翻译补 few-shot | 首个分片 24.8s 出现；极值题 `verified` |
| P7 块1 | 调用链日志 | 每次运行落盘 JSONL：决策/工具/验证/耗时/token/错误 | 可按 run_id 回放；落盘时抓到流式空答案 bug |
| P7 块2 | 运行指标 | trace 聚合 + `GET /api/metrics` / `/api/traces` | 首跑就量化出 `fallback_rate=0.67` |

改造遵循「先地基、再工具、后循环、最后检索」的顺序，每一步都能独立跑通、独立测试，再往上叠加。

---

## 二、前置改动（Gradio 清理）

原项目同时存在 Gradio 前端和 Web 前端，但启动器只启动 Web。为聚焦 Web，做了清理：

| 动作 | 内容 |
|---|---|
| 删除 | `app/frontend/` 整个目录（`gradio_app.py`、`controller.py`、`api_client.py`、`favorites.py`、`ui_theme.py`、`ui_behavior.js`、`conversation.py`、`examples.py`、`config.py`） |
| 迁移 | `app/frontend/memory.py` → `app/services/history_memory.py`（去掉仅 Gradio 使用的 `normalize_history`） |
| 修改 | `tests/test_memory.py` 的 import 指向 `app.services.history_memory` |
| 移除 | `requirements.txt` 中的 `gradio>=4.20.0` |

`start_local.ps1` 无需改动，它本来就只启动 Web 前端。

---

## 三、P0 结构化输出地基

### 目标

让模型稳定输出一个能通过 Pydantic 校验的 `RouteDecision` JSON。这是后续路由、工具、循环的前提。

### 三层保险

| 层 | 手段 | 保证 |
|---|---|---|
| Prompt | 明确 schema + 只输出 JSON + 判定规则 | 模型知道要什么 |
| 引擎 | `response_format={"type": "json_object"}` | 输出是合法 JSON（语法层面） |
| 代码 | Pydantic 校验 + 错误反馈重试 + 兜底 | 字段/枚举正确（语义层面） |

关键认知：`json_object` 只保证语法合法，不保证字段和枚举正确，所以第三层必须自己做。

### 新增文件

- `app/agent/__init__.py`
- `app/agent/schema.py`
  - `RouteDecision`：`intent` / `tool` / `query` / `answer_style`，全部用 `Literal` 做枚举约束，`extra="forbid"` 拒绝多余字段
  - `ToolResult`：所有工具的统一返回信封 `{success, data, error}`
  - `MAX_QUERY_CHARS = 2000`
- `app/agent/router.py`
  - `ROUTER_SYSTEM_PROMPT`：含 schema、判定规则、防注入说明
  - `classify()`：调用结构化补全，失败则兜底为 `general/none`
  - 再导出 `extract_json_object`（保持 P0 测试兼容）
- `app/agent/structured.py`
  - `extract_json_object()`：剥离 ` ```json ` 围栏、前后散文，提取 JSON 子串
  - `complete_structured()`：通用「结构化补全」——校验失败时把 Pydantic 报错原样喂回模型重试，全部失败返回 `None`
- `tests/test_router.py`

### 修改文件

- `app/services/vllm_client.py`
  - `complete()` 新增 `response_format`、`temperature` 参数
  - 新增 `complete_json()`：默认 `json_object` 模式；传入 `schema` 时走 `json_schema` 模式（vLLM 支持，部分 OpenAI 兼容服务会忽略）

### 关键设计

- **剥离围栏**：7B 模型极爱输出 ` ```json `，必须在代码里剥离
- **重试带错误反馈**：不是简单重发，而是把校验报错告诉模型让它自我修正
- **兜底不崩**：重试用尽返回安全默认值，路由失败绝不能让请求失败
- **超长截断**：兜底路径把问题截到 2000 字符，避免兜底本身触发校验失败

### 真机验证

```
求解方程 x^2 - 5x + 6 = 0  -> math / solve_math_problem
什么是二次函数的判别式      -> knowledge / search_knowledge
你好，今天天气怎么样        -> general / none
```

---

## 四、P1 工具层

### 目标

定一个所有工具都必须遵守的契约，把参数校验、超时、异常收口到一处。Agent 永远拿不到异常，只会拿到 `ToolResult(success=False, ...)`。

### 新增文件

- `app/agent/tools/base.py`
  - `ToolContext`：工具需要的依赖（`client` + `settings`），由调用方注入
  - `ToolSpec`：`name` / `description` / `input_model` / `handler` / `timeout`
  - `run_tool()`：**参数校验 → `asyncio.wait_for` 超时 → 异常收口**，三个失败路径全部转成结构化 `ToolResult`
- `app/agent/tools/registry.py`
  - `_REGISTRY` 注册表
  - `call_tool()`：按名字执行；未知工具同样返回结构化失败
  - `tool_catalog()`：把工具清单渲染成 JSON Schema，供 P6 Planner 用
- `app/agent/tools/math_solver.py` — `solve_math_problem`：复用现有解题链路（`build_solve_messages` + `complete`），超时 180s
- `app/agent/tools/calculator.py` — `calculate_expression`：**AST 白名单**安全求值
- `app/agent/tools/knowledge.py` — `search_knowledge`：接口先定（含 `top_k`），P3 填实现
- `app/agent/tools/checker.py` — `check_math_answer`：结构化 LLM Judge，复用 `complete_json`
- `tests/test_tools.py`

### 修改文件

- `app/agent/schema.py`
  - `RouteDecision.tool` 枚举从占位名改为真实工具名：`solve_math_problem` / `search_knowledge` / `none`
  - 新增 `Verdict`：`correct` / `incorrect` / `uncertain` + `reason`
- `app/agent/router.py`：改用 `complete_structured`，逻辑更薄

### 计算器的安全设计

用 Python `ast` 白名单，**绝不用 `eval()`**：

- 允许：`+ - * / ** % //`、一元正负、`sqrt/log/log10/exp/sin/cos/tan/abs/round`、常量 `pi/e`
- 拒绝：属性访问（`a.b`、`(1).__class__`）、`__import__`、`open()` 等任意调用
- 防护：表达式长度 ≤ 200，指数上限 1000（防 `9**9**9` 卡死）

### 统一契约要点

- 工具**永不抛异常**：参数错误、超时、内部异常统一转成 `ToolResult(success=False, error=...)`
- 失败必须明确返回，**绝不能被当成正常答案**
- `search_knowledge` 此时故意返回失败，逼 P2 的循环从第一天就处理「工具失败」

---

## 五、P2 单 Agent ReAct 循环

### 目标

把 P0 的路由和 P1 的工具串成一个**有界循环**，并证明它不会死循环、不会重复调同一个工具。

### 新增文件

- `app/agent/loop.py`
  - `run_agent()`：主循环
  - `decide_next_action()`：把工具目录 + 已有观察塞进 prompt，用 `complete_structured` 拿结构化动作；解析失败直接当 `final`
  - `generate_answer()`：**普通 completion（非 JSON）**，按 `answer_style` 加不同指令生成最终答案
  - `_execute()`：调用工具、计时、把结果压成 `Observation`（截断到 4000 字符）
- `app/api/routes/agent.py` — `POST /api/agent/run`
- `tests/test_loop.py`、`tests/test_agent_route.py`

### 修改文件

- `app/agent/schema.py`：新增 `AgentAction`（`call_tool` / `final`）、`Observation`、`AgentRun`
- `app/api/models.py`：新增 `AgentRunRequest`（`question` + `max_steps`，上限 8）
- `app/api/main.py`：注册 agent router

### 循环流程

```
classify(P0) ──► 路由到工具则确定性执行一次
                        │
                        ▼
        ┌──► 模型决定下一步（结构化 AgentAction）
        │      ├─ final   ──► 结束
        │      └─ call_tool ──► 执行 ──► 记录 Observation ──┐
        └──────────────────────────────────────────────────┘
                        │
                        ▼
              generate_answer（普通 completion，非 JSON）
```

### 四道保险

| 保险 | 做法 | 防的问题 |
|---|---|---|
| 最大步数 | `max_steps=4`（默认），超出记 `max_steps` 并强制生成答案 | 无限循环 |
| 重复调用 | `(tool, 冻结后的 arguments)` 进 `seen` 集合，命中即停 | 反复调同一个工具 |
| 结果截断 | `Observation.summary` 限 4000 字符 | 上下文爆炸 |
| 失败不吞 | 工具失败照常记录并明确告诉模型 | 模型把错误当答案编造 |

`AgentRun.stopped_reason`（`final` / `max_steps` / `duplicate` / `unknown_tool`）能直接区分「正常结束」和「被保险掐断」。

### 真机验证

```
intent: math | tool: solve_math_problem
steps: 1 | stopped: final
  [0] solve_math_problem success=True 75357ms
answer: 方程的解为 x=3 或 x=2，最终答案 {2, 3}
```

（75 秒是 CPU 上 7B 推理的正常耗时。）

---

## 六、P3 真实 RAG（Chroma）

### 目标

把 `search_knowledge` 从占位变成真实检索，返回带来源的片段。

### 新增文件

- `app/services/rag_service.py`
  - `split_text()`：按段落打包 + 窗口切分，chunk 500 / overlap 50
  - `_load_documents()`：读取 `knowledge/` 下的 `.md` / `.txt`
  - `_normalize()`：L2 归一化，让 Chroma 默认 L2 距离等价于余弦排序
  - `FastEmbedEmbedder`：fastembed 中文模型，懒加载 + 进程级缓存
  - `index_knowledge()`：重建向量库，返回片段数
  - `search()`：返回 `RetrievedChunk(content, source, distance)`
  - `KnowledgeBaseError`：库缺失/不可读
  - 可直接 `python -m app.services.rag_service` 建立索引
- `knowledge/01-二次方程与判别式.md`
- `knowledge/02-函数与导数.md`
- `knowledge/03-概率基础.md`
- `knowledge/04-矩阵与行列式.md`
- `knowledge/05-常见公式速查.md`
- `tests/test_rag.py`

### 修改文件

- `app/agent/tools/knowledge.py`：接真实检索，用 `asyncio.to_thread` 避免阻塞事件循环
- `app/core/config.py`：新增 5 个配置项（见下）
- `requirements.txt`：新增 `chromadb>=0.5.0`、`fastembed>=0.4.0`
- `.gitignore`：忽略 `data/chroma/`
- `.env.local.example`：补 RAG 配置与 HF 镜像说明
- `tests/test_tools.py`、`tests/test_loop.py`：改为注入式，避免测试依赖网络

### 关键发现（值得写进面试）

**Chroma 自带的默认 embedding 是 all-MiniLM-L6-v2，纯英文模型，对中文几乎无效。**

实测（查询「判别式怎么判断根的个数」）：

```
vs 判别式文档  cosine = 0.4465
vs 导数文档    cosine = 0.8661   ← 完全排反
```

换成 `BAAI/bge-small-zh-v1.5`（fastembed，512 维，约 95MB）后，检索全部命中：

```
判别式怎么判断根的个数 -> 01-二次方程与判别式.md  d=0.620
怎么求函数的极值       -> 02-函数与导数.md       d=0.690
贝叶斯公式是什么       -> 03-概率基础.md         d=0.735
矩阵可逆的条件         -> 04-矩阵与行列式.md     d=0.576
等差数列前 n 项和      -> 05-常见公式速查.md     d=0.807
```

### 设计要点

- **显式传向量**：自己算 embedding 再传给 Chroma，绕开 Chroma 1.x 把 embedding function 写进 collection 配置的坑，同时让 embedder 可注入（测试离线跑）
- **不对称编码**：`Embedder` 协议区分 `embed_documents` / `embed_query`，为 bge 的查询前缀和后续 reranker 留位置
- **向量归一化**：服务层统一归一化，让默认 L2 距离等价于余弦排序
- **embedder 进程级缓存**：避免每次检索重载 ONNX 模型
- **失败分级**：库未建立 → `KnowledgeBaseError`；检索为空 → `success=False`；模型必须把失败当失败

---

## 七、Agent 接入 Web 前端与 SSE 流式

### 目标

让 Agent 的决策过程在界面上可见：用户能看到它判断了什么问题类型、调用了哪个工具、耗时多久、为什么结束，而不是对着一个几分钟没反应的页面等答案。

### 要解决的两个问题

- `/api/agent/run` 是**非流式**的。CPU 上一次完整运行约 3.5 分钟（classify + decide + 答案生成各一次 7B 推理），界面会像卡死。
- `web/app.js` 原本只调 `/api/solve` 和 `/api/chat`，**没有任何地方能显示路由和工具调用**。

### 后端改动

- `app/agent/loop.py`
  - 抽出 `iter_agent_events()` 异步生成器作为**唯一事实源**，边跑边 `yield` 事件
  - `run_agent()` 改为消费同一个生成器、取 `done` 事件里的 `AgentRun` 返回 —— 对外接口不变，P2 的 8 个测试成为回归网
  - 新增 `generate_answer_stream()`：复用已有的 `client.stream_raw()` + `extract_stream_content()`，让最终答案逐字吐出
  - 新增 `thinking` 事件，覆盖两个**不产出事件的长耗时模型调用**（classify / decide）
- `app/api/routes/agent.py`：新增 `POST /api/agent/stream`，返回 `EventSourceResponse`
- `tests/test_agent_stream.py`：新增 5 个用例，断言完整事件序列

### 事件协议

| 事件 | 载荷 | 前端行为 |
|---|---|---|
| `thinking` | `stage`: `classify` / `decide` / `answer` | 显示「正在分析问题类型…」「正在判断下一步…」「正在生成答案…」 |
| `route` | `decision` | 显示路由徽章 intent → tool |
| `tool_start` | `step`, `tool`, `arguments` | 该步显示「调用中…」 |
| `tool_end` | `step`, `tool`, `success`, `duration_ms`, `summary` | 该步显示成功/失败与耗时 |
| `answer_delta` | `content` | 追加到答案气泡 |
| `done` | `steps`, `stopped_reason`, `run` | 收起进度，显示结束原因 |
| `error` | `message` | 提示失败 |

### 前端改动

- `web/app.js`
  - `state.answerMode` 增加第三个值 `agent`，模式切换按钮变为三个
  - 新增 `renderTraceMarkup()`：渲染执行轨迹卡（路由徽章 + 每步工具 + 状态点 + 耗时 + 结束原因）
  - 新增 `handleAgentEvent()`：按 `type` 分发事件
  - 新增 `updateTraceCard()`：局部更新轨迹卡，不重绘整个页面
  - `sendQuestion()` 增加 `agent` 分支；复用现有 `AbortController`，所以「暂停输出」天然可用
- `web/styles.css`：轨迹卡样式（含深色模式）；模式切换改 `repeat(auto-fit, minmax(132px, 1fr))` 以容纳三个按钮
- `web/index.html`：缓存版本号

### 真机验证

概念题「判别式怎么判断二次方程根的个数」：

```
Agent 执行轨迹  已完成
路由  knowledge → search_knowledge
search_knowledge  成功 · 4.0s
结束原因：final
```

答案正确引用了检索到的判别式内容。另用「矩阵可逆的条件是什么」验证了：

- 轨迹卡在点击后**立即出现**，并显示「正在判断下一步…」
- 点「暂停输出」后，轨迹卡变为「已完成」、气泡显示「（输出已暂停，尚未生成回答）」、提示「已暂停模型输出」

### 踩坑

1. **轨迹卡不出现**：`render()` 在 `trace` 赋值之前调用，导致 `updateTraceCard()` 找不到卡片。改为先构造好消息对象再 `render()`，并让 `updateTraceCard` 在找不到卡片时补一次 `render()`。
2. **PowerShell 改文件编码把 `index.html` 写坏**：`Set-Content -Encoding UTF8` 在 PowerShell 5.1 下会先用 GBK 读取 UTF-8 文件，导致中文乱码、甚至吞字符（`∑</div>` 变成 `鈭?/div>`）。已用 .NET 的 `UTF8Encoding($false)` 无 BOM 还原，并确认 `web/` 三个文件均无 BOM、无乱码。

### 延迟说明

CPU 上跑一次完整 Agent 约 3.5 分钟：classify（约 20s）+ 工具（4s）+ decide（约 60s）+ 答案生成（约 60–120s）。流式让等待可见，但总时长受 CPU 推理限制；GPU 会快很多。

---

## 八、P5 答案独立验证与自动纠错

对应 Agent 学习框架第十节（Reflection 与结果验证）。框架明确要求：**不要只依靠模型自我评价**。

### 目标

让答案在被返回前经过一次**独立于模型**的验证；被证伪时带反馈重新生成，且有硬上限。

验收标准：

- 正确的根 → `verified`
- 故意给错的根 → `refuted`，并给出反例
- 无法独立验证的问题 → `unknown`，**不触发重试**
- 重试有硬上限，绝不无限循环

### 核心原则：模型只翻译，SymPy 当裁判

| 做法 | 问题 |
|---|---|
| 让模型自己检查答案 | 自我评价会重复同一个错误，这是框架的经典警告 |
| 让 SymPy 解一遍再比对 | 需要把题目翻译成方程，翻译错了就误判 |
| **让 SymPy 代入检验**（采用） | 只需翻译出等式，不需要 SymPy 会解；代入求值是最可靠的运算 |

因此分工是：

- **模型**负责两个"小任务"：把题目翻译成 `lhs = rhs`，把解答里的最终答案抽出来
- **SymPy** 负责判定：把答案值代入等式独立算出结果，做出裁决

### 验证策略（按强度降级）

| 策略 | 适用 | 独立性 | 判定 |
|---|---|---|---|
| **代入检验**（SymPy） | 方程 / 求根 / 解集 | 强：确定性运算 | 全部代入为 0 → `verified`；任一不为 0 → `refuted` + 反例 |
| **独立求值**（SymPy） | 纯计算题 | 强 | 独立算出的值与答案一致 → `verified`；不一致 → `refuted` |
| **来源核对** | 知识问答，有检索片段 | 中：约束在检索证据内的 LLM Judge | 关键结论都有依据 → `verified`；有未支持结论 → `unknown`（列出结论） |
| **LLM Judge** | 以上都不适用 | 弱 | 仅作兜底，结果标注 `method` |

关键原则：**只在有硬证据时才判 `refuted`**，其余一律 `unknown`，且 `unknown` 永不触发重试。

这直接回答框架的问题——「如何判断第一次回答真的错了？」答案是：只有当独立计算与答案矛盾时才算错。

### 为什么是两次调用，而不是一次

最初把「抽取答案」和「翻译题目」合并成一次结构化调用（省一次模型调用）。真机立刻暴露问题：

```
题目「求解 x^2 - 5x + 6 = 0」，解答「因式分解得 (x-2)(x-3)=0，最终答案 x=2 或 x=3」
模型翻译结果：lhs = "(x - 2)*(x - 3)", rhs = "0"
```

模型直接用了**解答里的因式分解**作为检验等式。这样即使解答推错，只要自洽就会被判 `verified` —— 等于拿答案验答案，验证失去意义。

改为两次调用后：

- `translate_question()`：**只看到题目**，看不到解答，物理上无法使用解答的推导
- `extract_answer()`：看到题目 + 解答，只负责抽值

代价是多一次模型调用，换来的是验证的独立性。

### 数据结构（`app/agent/schema.py`）

```python
class SympyForm(BaseModel):
    """把题目改写成 SymPy 可代入检验的形式（只依据题目生成）。"""
    applicable: bool = False
    lhs: str = ""
    rhs: str = "0"
    variables: list[str] = []

class ExtractedAnswer(BaseModel):
    """从自由文本解答中抽出的最终答案。"""
    kind: Literal["numeric", "expression", "solution_set", "text", "none"]
    values: list[str] = []

class GroundingVerdict(BaseModel):
    grounded: bool
    unsupported: list[str] = []
    reason: str = ""

class VerificationResult(BaseModel):
    status: Literal["verified", "refuted", "unknown"]
    method: Literal["substitution", "expression", "grounding", "llm_judge", "none"]
    detail: str
    counterexample: str | None = None
```

### 验证服务（`app/services/verify_service.py`）

| 函数 | 职责 |
|---|---|
| `translate_question()` | 题目 → `SympyForm`（只给题目） |
| `extract_answer()` | 题目 + 解答 → `ExtractedAnswer` |
| `check_by_substitution()` | 逐个代入 `lhs - rhs`，全部为 0 才 `verified` |
| `check_expression_value()` | 独立算出表达式值，与答案比对 |
| `check_grounding()` | 用检索片段约束 LLM Judge 核对来源 |
| `documents_from_observations()` | 从 `search_knowledge` 的观察结果里取出片段 |
| `verify_answer()` | 按上面的顺序分发 |

### 实现细节

**表达式安全**（防止模型输出被当成代码执行）：

- 字符白名单：只允许数字、字母、`_ + - * / % ^ ( ) . , =` 和空格
- 拒绝 `__`（挡住 `__import__`、`__class__`）
- 小数点必须夹在数字之间（挡住 `x.__class__`、`x.real` 这类属性访问）
- `sympify` 结果必须是 `sympy.Expr`，否则视为无法解析
- 所有解析失败一律 `unknown`，绝不猜

**退化翻译检测**（真机踩出来的第二个坑）：

模型把「解方程 x²-5x+6=0」翻译成 `lhs = "x"`、`rhs = "0"`，于是正确的根 x=2 被判 `refuted`。修法有两层：

1. prompt 加入三个 few-shot 示例，并明确「lhs 必须是方程左边的**完整表达式**，不是未知数本身」
2. 代码层加检测：若等式被翻译成「变量 = 常数」→ 判 `unknown`，不判 `refuted`

**数值容差**：`1/3` 与 `0.3333333333` 视为相等（容差 `1e-9`），避免浮点噪声造成误判。

### 接进 Agent 循环（`app/agent/loop.py`）

```
生成答案 ──► 翻译题目 ──► 抽取答案 ──► 代入/求值检验
                                          │
                          verified ───────┴───────► done
                          unknown  ──────────────► done（不重试）
                          refuted  ──► 带反例反馈重新生成（最多 1 次）
```

- 新增 `max_verify_retries = 1`；重试用尽后照常返回，`AgentRun` 里带上 `verification` 和 `verify_attempts`
- **流式与重试的冲突**：答案已经流式输出了，如果被证伪，发 `answer_reset` 事件让前端清空气泡，再重新流式输出修正版；否则用户会看到两段答案叠在一起
- 重试时把反例写进答案生成的 prompt：「上一版答案被独立验证判定为错误：<detail>，请重新计算」

### SSE 事件（新增）

| 事件 | 载荷 | 前端行为 |
|---|---|---|
| `verify_start` | `attempt` | 轨迹卡显示「正在独立验证答案…」 |
| `verify` | `verification` | 轨迹卡显示验证徽章 |
| `retry` | `attempt`, `reason` | 轨迹卡新增一行「第 N 次修正」 |
| `answer_reset` | — | 清空当前答案气泡，准备重新生成 |

前端轨迹卡效果：

```
Agent 执行轨迹  已完成
路由  knowledge → search_knowledge
search_knowledge  成功 · 2.4s
验证 · 已验证   来源核对
结束原因：final
```

### 和 `check_math_answer` 的关系

`check_math_answer` 原本是「模型自己判自己」的弱校验。P5 把它改成：

1. 先跑规则验证（SymPy）—— `verified` → `correct`，`refuted` → `incorrect`，并带上 `method`
2. 规则不适用（`unknown`）时才降级到 LLM Judge，结果标注 `method = "llm_judge"`

这样调用方能从 `method` 看出这个结论有多可信。

### 真机验证

```
解 x^2-5x+6=0，答 x=2 或 x=3  -> verified  substitution  「2 个答案全部满足等式」
解 x^2-5x+6=0，答 x=5 或 x=1  -> refuted   substitution  「代入 x=5 后等式不成立」反例=5
计算 1/3+1/6，答 1/2          -> verified  expression    「独立计算结果与答案一致（1/2）」
```

浏览器端知识问答：`验证 · 已验证 / 来源核对`。

### 成本

一次完整 Agent 运行增加 2–3 次模型调用（翻译 + 抽取 + 可能的来源核对）。CPU 上总计约 5–8 分钟，GPU 会快很多。

### 边界（不做什么）

- **不逐步验证每一步推导**：无法可靠自动化，容易变成自欺，只验证最终答案
- **来源核对不判错**：资料未支持记为 `unknown`，因为模型的常识可能超出知识库，判错会引发重试风暴
- **翻译不出来就如实说 `unknown`**：证明题、应用题、概念题本来就没有「根」的概念，不硬凑

---

## 九、模型路由：云端编排 + 本地解题

### 动机

到 P5 为止，所有模型调用都走本地 7B：路由、规划、验证、解题全用同一个模型。但 7B 的强项是数学求解（微调过），弱项恰恰是结构化决策和评判；云端大模型（DeepSeek）规划与判断更强，却不了解本项目微调后的解题风格。

于是按「让合适的模型做合适的事」拆开：

- **编排模型（云端 DeepSeek）**：任务路由、下一步决策、答案验证、来源核对
- **解题模型（本地 mathllm-round7）**：只负责真正的数学求解

这正是框架第十三节说的「模型路由」，也是 P6 里 Manager / Worker 的雏形。

### 调用分工

| 调用点 | 模型 |
|---|---|
| `classify()` 任务路由 | 编排（云端） |
| `decide_next_action()` ReAct 决策 | 编排 |
| `solve_math_problem` 工具 | **解题（本地）** |
| `generate_answer` 最终答案 | 数学题用本地、其余用编排（见下） |
| `verify_service.translate_question` / `extract_answer` | 编排 |
| `verify_service.check_grounding` | 编排 |
| `check_math_answer` 的 Judge 兜底 | 编排 |
| `search_knowledge` / `calculate_expression` | 无模型（Chroma / SymPy） |
| `/api/solve`、`/api/chat`、`/api/memory/summarize`（旧接口） | 解题（本地，保持离线可用） |

顺带的好处：**验证质量明显提升**——用 DeepSeek 当裁判比让 7B 自己判自己靠谱得多。

### 数学答案直通（云端不改写）

数学题的最终答案**直接使用本地模型的输出**，云端不改写。原因：

- 本地模型是微调过的解题模型，改写只会引入风险
- 省掉一次生成调用
- 云端只提供「要不要重解」的反馈，不碰数学内容

实现上，`intent == "math"` 且 `solve_math_problem` 成功时，直接把工具返回的解答作为最终答案流式输出；只有非数学问题才由编排模型生成答案。若验证判定答案错误，则由**本地模型带反例重新解题**，而不是让云端重写。

### 回退策略

云端不可用时（网络、超时、401），按 `MATHLLM_ORCHESTRATOR_FALLBACK` 处理：

| 取值 | 行为 |
|---|---|
| `local`（默认） | 同一次调用改用本地模型重试，轨迹里标注实际使用的模型 |
| `fail` | 直接报错 |

**流式回退的特殊处理**：如果已经吐出了部分内容才失败，**不会**换模型重来——那会让用户看到两段拼接的答案。只有在尚未产出任何内容时才回退。

### 向后兼容

没配 `MATHLLM_ORCHESTRATOR_*` 时，编排端点自动等于本地端点，行为与改造前完全一致。因此可以先做重构、保证测试全绿，再逐点切换调用模型。旧的 `MATHLLM_API_BASE_URL` / `MATHLLM_MODEL_NAME` / `MATHLLM_API_KEY` 仍作为解题模型的回退名读取，现有 `.env.local` 不用改。

### 配置

```text
MATHLLM_ORCHESTRATOR_BASE_URL=https://api.deepseek.com/v1
MATHLLM_ORCHESTRATOR_MODEL=deepseek-chat
MATHLLM_ORCHESTRATOR_API_KEY=sk-xxx        # 留空 = 关闭云端编排
MATHLLM_ORCHESTRATOR_FALLBACK=local

MATHLLM_SOLVER_BASE_URL=http://127.0.0.1:11434/v1
MATHLLM_SOLVER_MODEL=mathllm-round7
MATHLLM_SOLVER_API_KEY=ollama
```

代码里用 `ModelConfig` 表示一个 OpenAI 兼容端点，`Settings.solver` / `Settings.orchestrator` 两个属性产出它们；`ModelGateway` 暴露与 `VLLMClient` 相同的接口，因此所有编排调用点无需改动。

### 启动器与配置弹窗

`configure.bat`（只配置）或 `start_local.bat`（首次运行自动弹窗，`-Configure` 可强制）会打开 tkinter 窗口：

```
┌─ MathLLM 模型配置 ──────────────────────────────┐
│ 编排模型（云端 · 负责路由/规划/验证）             │
│   Base URL / 模型名 / API Key(掩码)              │
│   [测试连接]  → 连接正常（3 个模型：…）           │
│                                                  │
│ 解题模型（本地 · 只负责数学求解）                 │
│   Base URL / 模型名 / API Key(掩码)              │
│   [测试连接]                                     │
│                                                  │
│ 云端不可用时：[回退本地编排 ▾]                    │
│           [取消]  [仅保存]  [保存并启动]          │
└──────────────────────────────────────────────────┘
```

- 每个端点一个「测试连接」，直接调 `/models` 验证（在后台线程里跑，不冻结窗口）
- **API Key 留空即关闭云端编排**，全部走本地
- 退出码 `0` = 保存并启动、`10` = 仅保存、`1` = 取消；启动器据此决定是否继续
- 只更新 `.env.local` 里模型相关的键，端口、RAG 等设置原样保留

弹窗只是薄壳，读写逻辑在 `app/config_editor.py`，有 10 个单测覆盖（含「不留 BOM」与幂等性）。

### 前端展示

轨迹卡的每一步都显示实际使用的模型，让「模型路由」看得见：

```
Agent 执行轨迹  运行中
路由  general → none  mathllm-round7
```

事件里带 `model` / `role`（`route` / `tool_start` / `tool_end` / `verify` / `done`），前端按事件渲染；`done` 里额外带 `answer_model`，显示「答案由 … 生成」。

### 踩坑

**重构后 `/api/health` 返回 500**：`VLLMClient` 从接收整个 `Settings` 改为接收 `ModelConfig` 时，`list_models()` / `complete()` 里还残留 `config.vllm_base_url`、`config.model_name` 两个旧字段名，抛 `AttributeError`。原有测试没有覆盖健康检查接口，所以没被拦住。已修正，并补了 `test_health_uses_solver_client` 防止再犯。

### 权衡

| 得到 | 付出 |
|---|---|
| 路由 / 规划 / 验证质量显著提升 | 编排依赖网络，离线时回退本地 |
| 解题仍是本地微调模型，保真 | 用户题目会发送到云端 |
| 延迟可能更低（云端比 CPU 上的 7B 快） | 多一个 API Key 要管理 |
| 每步模型可见，便于定位问题 | 每次请求多一组编排 token 成本 |

### 修复一：数学答案恢复流式输出

**问题**：模型路由上线后，数学题变成「等 100 秒、然后整段答案一次性出现」，完全没有流式效果。

**原因**：为了「云端不改写数学答案」，数学题的答案改成直接取 `solve_math_problem` 工具的结果。而这个工具内部用的是 `ctx.solver.complete(...)` —— **非流式**。改造前之所以有流式，是因为答案由另一次 `generate_answer_stream()` 调用生成；去掉那次调用，流式也就没了。

| | 改造前 | 改造后（直通） |
|---|---|---|
| 模型调用 | 解题 1 次 + 流式生成 1 次 | 只有解题 1 次 |
| 总耗时（CPU） | 约 150s | 约 100s |
| 可见输出 | 后 75s 逐字出现 | 全程无输出，最后一次性出现 |

**修法**：让解题工具**本身支持流式**。这样既保留「云端不改写」，又恢复流式，而且仍然只有一次本地调用。

- `ToolSpec` 增加可选 `stream` / `stream_result` 字段
- `base.py` 增加 `run_tool_streaming()`：**参数校验、超时、异常收口与阻塞路径完全一致**，只是把结果边产出边推送
- `math_solver.py` 增加 `_solve_stream()`，用 `ctx.solver.stream_raw()` 逐字产出
- `loop.py` 的工具执行改走流式路径，并记住「这段已经流过了」，答案阶段不重复推送

**实测**：

```
[  24.8s] >>> 第一个字符出现（流式开始）
流式分片数: 498  首个分片: 24.8s
```

### 修复二：验证从「无法验证」恢复为「已验证」

**问题**：求极值点的问题，轨迹卡显示 `验证 · 无法验证`。

**诊断过程**（结论反直觉，值得记录）：

1. 先怀疑翻译环节。把编排模型设成本地模型时，翻译**正确**（`diff(x**3-3x+1, x) = 0`），验证通过。
2. 换成真实的云端编排模型 `deepseek-flash` 后，`translate_question` 返回 `None`。
3. 直接看原始响应，发现 `deepseek-flash` 是**推理模型**：输出分 `reasoning_content`（思考）和 `content`（答案）两部分。对「求极值点」这个翻译任务，它**陷入无休止思考**：

```
max_tokens=384   finish=length  content_len=0   reasoning_len=665
max_tokens=1024  finish=length  content_len=0   reasoning_len=3803
max_tokens=2048  finish=length  content_len=0   reasoning_len=7310
```

给到 2048 tokens 仍然全是思考，`content` 始终为空 → JSON 解析失败 → 重试 3 次 → 返回 `None` → 降级为「无法验证」。

**第一层修复：补 few-shot 示例**

给翻译 prompt 加上极值/最值的样例，并明确「极值点类先把函数求导，lhs 是导函数」：

```
题目「求函数 f(x)=x^3-3x+1 的极值点」
-> {"applicable": true, "lhs": "3*x**2 - 3", "rhs": "0", "variables": ["x"]}
```

效果（四类题各测两种模式）：

| 题目 | 思考开启 | 思考关闭 |
|---|---|---|
| 求 f(x)=x³-3x+1 的极值点 | 2.4s ✓ `3*x**2-3=0` | 1.4s ✓ 同上 |
| 求 f(x)=x²-4x 的最小值点 | 2.0s ✓ `2*x-4=0` | 1.4s ✓ 同上 |
| 求解方程 | 1.0s ✓ | 0.8s ✓ |
| 纯计算 | 1.1s ✓ | 1.2s ✓ |

顺带发现一个更危险的坑：**关闭思考（`reasoning_effort: "none"`）虽然更快，但在没有示例时会把极值题翻译成 `x**3-3x+1=0`（求零点而非极值），于是把正确答案判成 `refuted`** —— 这比「无法验证」更糟。所以最终选择「保留思考 + 补示例」，而不是关掉思考。

**第二层修复：编排返回空内容时回退本地**

推理模型把预算全花在思考上是常态，不能只靠 prompt。在 `ModelGateway` 里判断：编排返回的内容为空 → 视为失败 → 按回退策略改用本地模型，并在轨迹里标注实际使用的模型。

### 这两次修复的教训

1. **推理模型不适合「短 JSON 输出」这类机械任务**：它会过度思考，把输出预算耗尽在 `reasoning_content` 上，`content` 永远为空。
2. **能关掉思考就不要硬扛，但要看代价**：关掉思考更快，却可能给出「格式正确但语义错误」的翻译，反而把正确答案判错。这里用 few-shot 示例来补质量，比关思考更稳。
3. **不要只看「是否报错」判断成功**：HTTP 200 但 `content` 为空，是比报错更隐蔽的失败。客户端必须显式检查内容是否为空。

---

## 十、P7 可观测性：调用链日志与运行指标

对应框架第十二节（评测、监控与安全）。P7 分四块，本节是**块 1：调用链日志**与**块 2：运行指标**。

### 目标

每次 Agent 运行都完整落盘，出问题时能直接回放：路由对不对、哪个工具失败、验证为什么没通过、卡在哪一步、花了多少 token。

在此之前，SSE 事件推给前端就消失了，**没有任何持久化**——线上出问题只能靠复现。

### 设计

每次运行追加一行 JSONL 到 `data/traces/runs.jsonl`（已 gitignore）。

```
run_id        本次运行 ID（出问题时报这个）
question      用户问题（脱敏 + 截断）
started_at    开始时间（UTC ISO）
duration_ms   总耗时
decision      路由结果（intent / tool / query / answer_style）
observations  每次工具调用：step / tool / arguments / success / summary / error / duration_ms
verification  验证结论（status / method / detail / counterexample）
answer        最终答案（脱敏 + 截断）
answer_model  答案由哪个模型生成
stopped_reason final / max_steps / duplicate / unknown_tool / error
error         异常信息（若有）
usage         prompt / completion / total tokens 与模型调用次数
```

### 关键设计点

**1. 记录位置在循环外面，而不是里面**

`iter_agent_events` 被拆成两层：外层负责计时与落盘，内层 `_iter_agent_events` 是原来的循环。

```python
async def iter_agent_events(...):
    started_at = ...
    try:
        async for event in _iter_agent_events(...):
            if event["type"] == "done":
                run = AgentRun.model_validate(event["run"])
            yield event
    except Exception as exc:
        error = ...
        raise
    finally:
        # 正常结束和异常都落盘
        record_trace(...)
```

这样**半路失败也会留下记录**（`stopped_reason="error"`），而不是只记录成功的那部分。

**2. token 用量要主动采集**

原来响应里的 `usage` 字段被直接丢掉了。现在：

- 非流式：从响应的 `usage` 读取
- 流式：请求里带 `stream_options: {"include_usage": true}`，服务端在最后一个分片返回用量；不支持的服务器会忽略这个未知字段
- `VLLMClient` 自己累加；`ModelGateway.usage` 汇总两个端点（**同一个客户端对象不会重复计数**）

**3. 脱敏在写盘前完成**

`redact()` 会替换 `sk-***`、`Bearer ***`、`api_key=***` 这类模式。这样即使某个模型的报错信息里带出了凭据，也不会落到磁盘。

**4. 写入永不抛异常**

`record_trace` 内部用 `try/except` 包住——**观测不能影响请求本身**。

### 真机验证

```json
{
  "run_id": "b348d6b29371",
  "started_at": "2026-09-17T07:36:13+00:00",
  "duration_ms": 18088,
  "question": "判别式怎么判断二次方程根的个数",
  "decision": {"intent": "knowledge", "tool": "search_knowledge"},
  "observations": [{"step": 0, "tool": "search_knowledge", "success": true, "duration_ms": 2124, "summary": "..."}],
  "verification": {"status": "verified", "method": "grounding"},
  "answer_model": "deepseek-flash",
  "stopped_reason": "final",
  "error": null,
  "usage": {"prompt_tokens": 5049, "completion_tokens": 1900, "total_tokens": 6949, "calls": 6}
}
```

排查时的用法：

| 现象 | 看哪个字段 |
|---|---|
| 走了错误的工具 | `decision.tool` |
| 某个工具挂了 | `observations[].success` / `.error` |
| 答案没通过验证 | `verification.status` / `.counterexample` |
| 循环空转 | `usage.calls` 偏多、`stopped_reason=duplicate` |
| 卡在某一步 | `observations[].duration_ms` |
| 云端是否真的在用 | `answer_model`、`usage.calls` |

### 落盘时抓到的第二个 bug

第一次落盘就发现 `answer` 是兜底文案（`抱歉，暂时无法生成回答`），而 `done` 事件却显示正常。**如果没有 trace，这个 bug 会一直藏着**——前端只看到兜底文案，没有任何地方记录「这次生成其实是空的」。

- **原因**：`deepseek-flash` 流式生成答案时，512 的 `max_tokens` 被思考内容吃光，`content` 一个字符都没产出
- **更隐蔽**：原来的流式回退判断的是「是否产出过**事件**」，而思考分片也算事件，所以回退没触发
- **修复**：
  1. 回退判断改为「是否产出过**内容**」（`extract_stream_content` 非空）
  2. 给编排模型单独的输出预算 `MATHLLM_ORCHESTRATOR_MAX_OUTPUT_TOKENS=2048`（本地解题仍是 512）

修复后 `usage.calls` 正好 6 次（classify / decide / 答案 / 翻译 / 抽取 / 来源核对），答案是真实内容。

### 一个副作用

测试原本会往 `data/traces/runs.jsonl` 写 16 条垃圾数据。已让测试使用 `trace_enabled=False`，并清理了污染文件——现在跑测试不落盘。

### 块 2：运行指标

块 1 的 trace 是**单次运行**的明细，用于定位「这一次为什么错」；块 2 把它聚合成**多次运行的汇总**，用于看整体表现与趋势。两者互补。

#### 指标定义

全部从 trace 直接算出，**不需要人工标注**：

| 指标 | 定义 | 能看出什么 |
|---|---|---|
| `runs` | 窗口内运行总数 | 使用量 |
| `failure_rate` | `error` 非空的占比 | 稳定性 |
| `completion_rate` | 有非空答案的占比 | 任务完成率 |
| `guard_stop_rate` | `stopped_reason` 属于 `max_steps` / `duplicate` / `unknown_tool` 的占比 | 保险被触发的频率 |
| `stopped_reasons` | 结束原因分布 | — |
| `intents` / `routes` | 路由的 intent / tool 分布 | **路由是否偏斜**（全走 `none` 说明路由失效） |
| `tool_calls` | 各工具实际被调用的次数 | — |
| `tool_success_rate` | `observations[].success` 比例 | 工具可靠性 |
| `avg_tool_calls` | 平均工具调用次数 | 循环是否空转 |
| `verification` | `verified` / `refuted` / `unknown` 分布 | 验证覆盖与可信度 |
| `refuted` / `ungrounded` | 证伪数 / 来源未支持数 | — |
| `hallucination_rate` | `(refuted + ungrounded) / 有验证的运行数` | 幻觉率**代理指标** |
| `latency_ms` | p50 / p95 / max / avg | 尾延迟比平均值更有意义 |
| `tokens` | prompt / completion / total / avg_per_run | 成本 |
| `models` | 答案由哪个模型生成 | 云端是否真的在用 |
| `fallback_rate` | 发生过回退的运行占比 | 云端健康度 |

**比率在分母为 0 时返回 `null`**，而不是假装 0%——空窗口和「零失败」是两件事。

#### 接口

```
GET /api/metrics?days=7    聚合指标（days=0 表示全部）
GET /api/traces?limit=20   最近若干条明细，用于从指标异常跳到具体某次运行
```

#### 为指标补充的字段

块 1 的 trace 里没有「这次运行是否回退过」的信息，于是：

- `ModelGateway` 增加 `fallbacks` 计数器（每次回退 +1）
- `RunTrace` 增加 `fallbacks` 字段
- `loop.py` 落盘时写入

#### 真机结果

3 次真实运行（知识题 ×2、普通问答 ×1）：

```json
{
  "runs": 3,
  "failure_rate": 0.0,
  "completion_rate": 1.0,
  "guard_stop_rate": 0.0,
  "intents": {"knowledge": 2, "general": 1},
  "routes": {"search_knowledge": 2, "none": 1},
  "tool_calls": {"search_knowledge": 2},
  "tool_success_rate": 1.0,
  "avg_tool_calls": 0.67,
  "verification": {"verified": 2, "unknown": 1},
  "hallucination_rate": 0.0,
  "latency_ms": {"p50": 18088, "p95": 107834, "max": 107834, "avg": 44623},
  "tokens": {"prompt": 21555, "completion": 6524, "total": 28079, "avg_per_run": 9360},
  "models": {"deepseek-flash": 3},
  "fallback_rate": 0.6667
}
```

#### 指标立刻发挥了作用

**`fallback_rate: 0.6667`** —— 3 次运行里有 2 次发生过「云端返回空内容 → 回退本地」。

这解释了体感慢：虽然 `models` 显示答案都由 `deepseek-flash` 生成（流式那一步成功了），但编排的某几步（classify / decide / translate / extract）在回退本地，而本地一次结构化调用要几十秒。**P95 延迟 107 秒就是这么来的。**

没有这套指标，这个现象只能用「感觉慢」来描述，无法量化。有了数字之后方向就明确了：

- 在 trace 里记录**每次调用**用了哪个模型（目前只记了答案模型），定位到底哪一步在回退
- 或给机械任务关思考（`reasoning_effort`），配合已有的 few-shot 示例保质量

这也正好说明块 2 与块 3 的分工：**块 2 告诉你「哪里慢」，块 3 才能告诉你「改完有没有变好」。**

### 块 2 前端：运行观测页面

指标只通过 API 暴露的话，只有开发者会看。于是新增一个页面，把「整体指标」和「逐次运行」放在一起——**从指标异常直接点到具体某次运行**。

#### 页面结构

侧边栏「系统」下新增 **◫ 运行观测**（路由 `#observability`）。

**上半部分：整体指标**

```
运行总数 3      失败率 0.0%      完成率 100.0%
回退率 66.7%    延迟 P50 18.1s    Token 总量 28079
                P95 107.8s · 最大 107.8s    每次均 9360

路由分布 (decision.tool)   验证结论        答案模型
  search_knowledge  2      verified  2     deepseek-flash  3
  none              1      unknown   1
意图分布                   工具实际调用
  knowledge  2            search_knowledge  2
  general    1
幻觉率代理 0.0%（证伪 0 · 来源未支持 0）
```

**下半部分：逐次运行**（按任务区分，最新在最上面）

每条运行是一张可展开的卡片，折叠时只显示一行摘要：

```
▸ 09-17 07:55  矩阵可逆的条件是什么   [final]   107.8s · 9233 tok
```

展开后是完整调用链：

```
run_id       e3a56df24ba7
路由          knowledge → search_knowledge
验证          已验证 · 来源核对
答案模型      deepseek-flash
模型调用      8 次 · 回退 2 次
● search_knowledge  成功 · 2.0s
回答中的关键结论都能在检索资料中找到依据
（答案前 400 字）
```

#### 设计要点

| 决定 | 原因 |
|---|---|
| 展开/折叠用原生 `<details>` / `<summary>` | 不写一行 JS，浏览器自带行为；刷新时状态自然重置 |
| 分布用横向条形图而不是表格 | 一眼看出路由是否偏斜（全是 `none` 就说明路由失效了） |
| 时间窗下拉直接调 `?days=N` | 后端已支持窗口过滤，前端不重复计算 |
| 卡片复用 `trace-step` 样式 | 与 Agent 模式的轨迹卡视觉一致，用户不用重新学 |
| 比率显示 `—` 而不是 `0%` | 与后端一致：空窗口不是「零失败」 |
| 页面进入时才拉数据 | `setPage` / `hashchange` / 首次加载都会触发 `loadObservability()`，其余页面不付这个网络开销 |

#### 页面上直接印证了指标结论

展开那条 107.8 秒的运行，明细写着 **「模型调用 8 次 · 回退 2 次」**——这就是慢的原因：8 次调用里有 2 次云端返回空、回退到了本地，而本地一次结构化调用要几十秒。

**「指标异常 → 点进某次运行 → 看到具体哪一步」这条排查链路现在是通的。**

#### 一个失败记录

浏览器的截图工具（OpenChamber 的 `browser.capture`）连续多次报错（`UnknownVizError` / 20 秒超时），最终改用 `browser.snapshot` 的文本快照完成验证。**功能验证不受影响，但「截图留证」这条路在这台机器上不稳定。**

### 用户实测反馈的三个修复

上线「运行观测」页面后，实际使用中暴露了三个问题。**其中第一个最典型：看起来是前端渲染 bug，实际是后端配置问题。**

#### 修复一：LaTeX 没渲染 —— 其实是答案被截断

**现象**：数学答案里出现 `极小值点: $(1, -1`，公式没渲染出来。

**第一反应**：KaTeX 配置或 markdown 转换的问题。

**实际原因**：把 trace 里的原始答案打印出来看，结尾就是 `- 极小值点: $(1, -1` —— **闭合的 `)$` 根本没生成**。KaTeX 找不到配对的 `$`，只能原样显示。

```
本地解题模型的 max_tokens = 512
极值题的分步解答（含多个行间公式）超过 512 → 被硬截断
```

**教训**：**先看原始数据，再怀疑渲染层**。如果直接去调 KaTeX，会白花很多时间。

**修复**：

| 位置 | 改动 |
|---|---|
| `.env.local` | `MATHLLM_MAX_OUTPUT_TOKENS=512` → `1024` |
| `app/core/config.py` | 默认值 512 → 1024 |
| `.env.local.example` | 加注释说明为什么需要更大 |

**验证**：同一道题，答案从被截断变成完整的 874 字符，结尾是闭合的 `$(-1, 3)$`。

**代价**：本地生成耗时增加（该题 156 秒）。这是「完整答案」与「速度」的取舍，1024 是当前平衡点。

#### 修复二：运行过程中「运行观测」不能展开

**现象**：模型正在生成时切到「运行观测」页面，点开某条运行记录后立刻被收起。

**原因**：`updateTraceCard()` 在找不到轨迹卡时会调用 `render()`（这是早期修「轨迹卡不出现」时加的兜底）。生成过程中切页后，解题页的轨迹卡已不存在，于是这个兜底把**当前页面整个重建**，刚展开的 `<details>` 随之被销毁。

**修复**：加页面守卫——只在解题页更新轨迹卡。

```js
function updateTraceCard() {
  // 只有解题页有轨迹卡。若在别的页面调用 render()，
  // 会把「运行观测」里已展开的运行记录一起重建、收起。
  if (state.page !== "solve") return;
  ...
}
```

**教训**：**兜底式的全局重绘很危险**。一个为了修 A 页面而加的 `render()`，会在 B 页面产生「点了没反应」的诡异现象。

#### 修复三：极值题验证总是 `unknown`

**现象**：修完截断后，极值题的验证从 `verified` 退化成 `unknown`。

**排查**：看 trace 的 `verification.detail`，写的是「解答中没有可比较的数值或表达式」。

**原因**：答案写成**坐标对**：

```
- 极大值点为 $(x, y) = (-1, 3)$;
- 极小值点为 $(x, y) = (1, -1)$。
```

抽取器认为「没有可比较的数值」，返回空列表。但极值题真正能代入 `f'(x) = 0` 检验的是**自变量的值**（`-1` 和 `1`），不是坐标对。

**修复**：给抽取器 prompt 加一条规则和示例：

```
- 如果最终答案写成点或坐标对（例如「极大值点 $(-1, 3)$」），只提取自变量的值，
  多个点就逐个提取，例如 ["-1", "1"]

解答「极大值点 $(-1, 3)$，极小值点 $(1, -1)$」
-> {"kind": "solution_set", "values": ["-1", "1"]}
```

**验证**：

```
form     : {"lhs": "3*x**2 - 3", "rhs": "0", "variables": ["x"]}
extracted: {"kind": "solution_set", "values": ["-1", "1"]}
verdict  : verified | substitution
```

#### 三个修复的共同点

| 修复 | 表面现象 | 真实原因 | 定位手段 |
|---|---|---|---|
| 一 | 公式没渲染 | 答案被 token 上限截断 | 打印 trace 里的原始答案 |
| 二 | 卡片点不开 | 其他页面的兜底重绘 | 顺着「谁调用了 render()」找 |
| 三 | 验证变 unknown | 抽取器不认坐标对 | 看 trace 的 `verification.detail` |

**三次都靠 trace 定位**。这正好回答框架的问题「如何定位 Agent 出错的位置」——不是靠猜，而是靠落盘的决策、工具、验证结论和原始答案。

### 块 3–4（待做）

| 块 | 内容 |
|---|---|
| 块 3 评测 | 标注评测集 + 组件级评测脚本（路由准确率、检索命中率、答案准确率） |
| 块 4 安全 | 输入分隔与注入检测、越权面说明（日志脱敏已在块 1 完成） |

---

## 十一、目录结构变化

```
MathLLM-数学解答系统/
├── app/
│   ├── agent/                      # 新增：Agent 层
│   │   ├── __init__.py
│   │   ├── schema.py               # 结构化数据契约（含验证相关模型）
│   │   ├── structured.py           # 通用结构化补全 + 重试
│   │   ├── router.py               # 结构化路由
│   │   ├── loop.py                 # 有界 ReAct 循环 + SSE 事件 + 验证与重试
│   │   └── tools/                  # 新增：工具层
│   │       ├── __init__.py
│   │       ├── base.py             # 工具契约（校验/超时/异常收口）
│   │       ├── registry.py         # 注册表
│   │       ├── math_solver.py      # solve_math_problem
│   │       ├── calculator.py       # calculate_expression
│   │       ├── knowledge.py        # search_knowledge
│   │       └── checker.py          # check_math_answer（规则优先 + Judge 兜底）
│   ├── api/
│   │   ├── main.py                 # 注册 agent router
│   │   ├── models.py               # 新增 AgentRunRequest
│   │   └── routes/
│   │       └── agent.py            # 新增：/api/agent/run 与 /api/agent/stream
│   ├── config_editor.py            # 新增：.env.local 模型配置读写（可单测）
│   ├── core/config.py              # 新增 RAG 配置项 + ModelConfig 双端点
│   ├── tools/                      # 新增：独立小工具
│   │   └── config_ui.py            # 新增：tkinter 模型配置弹窗
│   └── services/
│       ├── vllm_client.py          # 改为接收 ModelConfig
│       ├── model_gateway.py        # 新增：云端优先、失败回退本地
│       ├── history_memory.py       # 新增（迁移自 app/frontend/memory.py）
│       ├── rag_service.py          # 新增：Chroma 检索
│       ├── verify_service.py       # 新增：题目翻译 / 答案抽取 / SymPy 检验
│       ├── trace_service.py        # 新增：运行 trace 落盘 / 读取 / 脱敏
│       └── metrics_service.py      # 新增：trace 聚合为运行指标
├── knowledge/                      # 新增：知识库文档
│   ├── 01-二次方程与判别式.md
│   ├── 02-函数与导数.md
│   ├── 03-概率基础.md
│   ├── 04-矩阵与行列式.md
│   └── 05-常见公式速查.md
├── tests/
│   ├── test_router.py              # 新增
│   ├── test_tools.py               # 新增
│   ├── test_loop.py                # 新增
│   ├── test_agent_route.py         # 新增
│   ├── test_agent_stream.py        # 新增：SSE 事件序列
│   ├── test_rag.py                 # 新增
│   ├── test_verify.py              # 新增：代入检验、独立求值、来源核对、分发
│   ├── test_config_editor.py       # 新增：.env.local 读写、BOM、幂等
│   ├── test_config_ui.py           # 新增：弹窗构造与保存
│   ├── test_model_gateway.py       # 新增：回退、空输出、token 汇总
│   ├── test_trace_service.py       # 新增：脱敏、截断、落盘、异常 trace
│   ├── test_metrics_service.py     # 新增：计数、比率、百分位、时间窗
│   ├── test_metrics_route.py       # 新增：/api/metrics 与 /api/traces
│   └── test_memory.py              # 修改 import
├── web/                            # Web 前端
│   ├── index.html                  # 导航项（含「运行观测」）与缓存版本号
│   ├── styles.css                  # 新增：轨迹卡、验证徽章、指标卡与运行卡片样式（含深色模式）
│   └── app.js                      # 新增：Agent 模式、轨迹渲染、事件分发、验证徽章、运行观测页面
├── configure.bat                   # 新增：只打开模型配置弹窗
├── data/chroma/                    # 新增（已 gitignore）：向量库持久化目录
└── data/traces/                    # 新增（已 gitignore）：运行 trace（runs.jsonl）
```

删除：`app/frontend/`（Gradio 全部文件）

---

## 十二、依赖与配置变化

### 依赖（requirements.txt）

| 变化 | 包 |
|---|---|
| 移除 | `gradio>=4.20.0` |
| 新增 | `chromadb>=0.5.0` |
| 新增 | `fastembed>=0.4.0` |
| 新增 | `sympy>=1.12`（P5 独立验证的裁判） |

计算器用标准库 `ast`，无新增依赖。P5 的表达式解析复用 `sympy`，不引入额外包。模型路由与配置弹窗也没有新增依赖：`ModelConfig` / `ModelGateway` 是纯标准库，弹窗用 Python 自带的 `tkinter`。

### 新增配置项（模型路由）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MATHLLM_ORCHESTRATOR_BASE_URL` | 空 | 云端编排端点；留空即回退本地 |
| `MATHLLM_ORCHESTRATOR_MODEL` | 空 | 编排模型名 |
| `MATHLLM_ORCHESTRATOR_API_KEY` | 空 | 留空 = 关闭云端编排 |
| `MATHLLM_ORCHESTRATOR_FALLBACK` | `local` | `local` 回退本地 / `fail` 直接报错 |
| `MATHLLM_SOLVER_BASE_URL` | `http://127.0.0.1:11434/v1` | 解题端点（兼容旧名 `MATHLLM_API_BASE_URL`） |
| `MATHLLM_SOLVER_MODEL` | `mathllm-round7` | 解题模型（兼容旧名 `MATHLLM_MODEL_NAME`） |
| `MATHLLM_SOLVER_API_KEY` | 空 | 解题 Key（兼容旧名 `MATHLLM_API_KEY`） |
| `MATHLLM_ORCHESTRATOR_MAX_OUTPUT_TOKENS` | `2048` | 编排模型的输出预算（推理模型需要更多思考空间） |
| `MATHLLM_TRACE_DIR` | `data/traces` | 运行 trace 目录 |
| `MATHLLM_TRACE_ENABLED` | `1` | 设为 `0` 关闭落盘 |

### 新增配置项（app/core/config.py）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MATHLLM_KNOWLEDGE_DIR` | `knowledge` | 知识库文档目录 |
| `MATHLLM_RAG_PERSIST_DIR` | `data/chroma` | Chroma 持久化目录 |
| `MATHLLM_RAG_COLLECTION` | `math_knowledge` | collection 名 |
| `MATHLLM_RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 中文 embedding 模型 |
| `MATHLLM_RAG_TOP_K` | `3` | 默认检索条数 |

网络受限时可用 `HF_ENDPOINT=https://hf-mirror.com` 下载 embedding 模型（已写入 `.env.local.example` 注释）。

---

## 十三、测试清单（共 136 个，全部离线通过）

| 测试文件 | 用例数 | 覆盖 |
|---|---:|---|
| `test_memory.py` | 3 | 历史切分、摘要触发、摘要注入 |
| `test_router.py` | 13 | JSON 剥离、枚举校验、多余字段拒绝、重试、兜底 |
| `test_tools.py` | 16 | 注册表、参数校验、超时、异常收口、计算器安全性、各工具成功/失败 |
| `test_loop.py` | 9 | 正常路径、数学答案直通、非数学走编排、重复检测、最大步数、未知工具、工具失败、动作非法、空输入 |
| `test_agent_route.py` | 3 | 路由接线（`dependency_overrides` 注入假客户端）、空问题 422、健康检查 |
| `test_agent_stream.py` | 6 | SSE 事件序列、数学答案直通、答案分片重组、空答案兜底、第二次工具调用、空问题 |
| `test_rag.py` | 11 | 切分、索引、检索命中、库缺失、空结果、工具映射 |
| `test_verify.py` | 23 | 代入检验（正确/错误/多变量/退化翻译）、独立求值、表达式安全、来源核对、策略分发 |
| `test_config_editor.py` | 10 | 新键/旧键读取、保留其他行、空 Key 关闭编排、无 BOM、幂等 |
| `test_config_ui.py` | 4 | 弹窗构造（防控件顺序错误）、下拉映射、保存并启动、校验 |
| `test_model_gateway.py` | 10 | 云端优先、异常回退、空内容回退、fail 策略、共享客户端不重复计数、token 汇总 |
| `test_trace_service.py` | 14 | 脱敏、截断、错误 trace、JSONL 追加与读取、循环落盘、关闭开关 |
| `test_metrics_service.py` | 10 | 计数、比率、幻觉代理、百分位、token 汇总、时间窗过滤 |
| `test_metrics_route.py` | 4 | `/api/metrics` 聚合、时间窗、`/api/traces` 明细、空窗口 |

### 怎么跑

```powershell
# 全部
& ".venv\Scripts\python.exe" -m unittest discover -s tests -v

# 单个文件
& ".venv\Scripts\python.exe" -m unittest tests.test_verify -v

# 单个用例
& ".venv\Scripts\python.exe" -m unittest tests.test_loop.RunAgentTest.test_max_steps_stops -v
```

### 测试约定

| 约定 | 原因 |
|---|---|
| **全部离线** | 模型用脚本化的假客户端注入，embedding 用确定性哈希替代，不联网、不加载模型 |
| **不产生副作用** | 涉及落盘的测试用 `trace_enabled=False` 或临时目录；否则会往 `data/traces/` 写垃圾（真发生过） |
| **假客户端按队列返回** | `ScriptedClient` 按调用顺序返回预设 JSON，用尽后重复最后一条，方便断言"第几次调用返回什么" |
| **GUI 测试会真的开窗口** | `test_config_ui` 会构造一次 tkinter 窗口再立刻销毁，用来拦截控件顺序错误（无显示环境自动 `skipTest`） |
| **时间用固定值** | 指标测试传入固定的 `now`，避免依赖真实时钟 |

### 每个文件守什么

| 文件 | 守住的回归 |
|---|---|
| `test_router.py` | 模型输出带 ```json 围栏 / 多余字段 / 枚举写错时不会崩，且能重试纠正 |
| `test_tools.py` | 工具永不抛异常；计算器不能执行任意代码；`__import__`、属性访问、超大指数全被拒 |
| `test_loop.py` | 不死循环、不重复调同一工具；数学答案直通、非数学走编排 |
| `test_agent_stream.py` | SSE 事件顺序；已流过的答案不重复推送 |
| `test_agent_route.py` | 路由接线正确（含 `/api/health`，这个洞曾导致 500） |
| `test_verify.py` | 代入检验不会把正确答案判错；退化翻译降级为 `unknown`；表达式白名单挡得住属性访问 |
| `test_rag.py` | 切分不产生超长块；检索命中期望来源；库缺失有明确错误 |
| `test_config_editor.py` | 只改模型键、保留其他行；UTF-8 无 BOM；幂等 |
| `test_config_ui.py` | 弹窗能构造（控件顺序错误会在这里暴露） |
| `test_model_gateway.py` | 云端失败/空输出时回退本地；`fail` 策略不回退；token 不重复计数 |
| `test_trace_service.py` | 写盘前脱敏；截断；异常也落盘；关闭开关后不写 |
| `test_metrics_service.py` | 计数、比率、百分位、时间窗；空窗口返回 `null` 而不是 0 |
| `test_metrics_route.py` | `/api/metrics` 与 `/api/traces` 的接线与聚合 |

---

## 十四、接口清单

| 方法 | 路径 | 走哪个模型 | 说明 |
|---|---|---|---|
| GET | `/api/health` | — | 服务与本地解题模型状态 |
| POST | `/api/solve` | 解题（本地） | 单题解答（原有） |
| POST | `/api/chat` | 解题（本地） | 多轮对话（原有） |
| POST | `/api/memory/summarize` | 解题（本地） | 对话摘要（原有） |
| **POST** | **`/api/agent/run`** | 编排 + 解题 | **新增：Agent 运行，非流式，返回完整轨迹** |
| **POST** | **`/api/agent/stream`** | 编排 + 解题 | **新增：Agent 运行，SSE 流式** |
| **GET** | **`/api/metrics?days=7`** | — | **新增：trace 聚合指标（`days=0` 表示全部）** |
| **GET** | **`/api/traces?limit=20`** | — | **新增：最近若干条运行明细** |

> 旧接口（`/api/solve`、`/api/chat`、`/api/memory/summarize`）刻意保持走**本地模型**，这样即使断网也仍然可用。Agent 接口才做模型分工。

### 各接口的请求与响应

**GET `/api/health`**

```json
{"status": "ok", "vllm": "connected", "model": "mathllm-round7",
 "available_models": ["mathllm-round7:latest", "mathllm-base-cpu:latest"], "detail": null}
```

模型不可达时返回 **503**，`status="error"`、`vllm="unavailable"`、`detail` 给出原因。

**POST `/api/solve`** — 单题解答

```json
{"question": "解方程 x^2-5x+6=0", "stream": true}
```

流式时逐条返回 SSE：`data: {"content": "因式分解得…", "finished": false}`，最后一条 `{"content": "", "finished": true}`。
非流式（`stream: false`）返回 `{"content": "…", "finished": true, "model": "mathllm-round7"}`。

**POST `/api/chat`** — 多轮对话

```json
{"messages": [{"role": "user", "content": "继续解释第二步"}],
 "summary": "用户正在学习一元二次方程", "stream": true}
```

`summary` 是可选的压缩记忆；超过 20 条消息或 24000 字符会被拒绝（422）。

**POST `/api/memory/summarize`** — 对话摘要

```json
{"messages": [{"role": "user", "content": "…"}], "existing_summary": null}
→ {"summary": "主题：…\n原始题目：…\n已得结论：…"}
```

**POST `/api/agent/run`** — 非流式，返回完整 `AgentRun`

```json
{
  "question": "求解 x^2-5x+6=0",
  "decision": {"intent": "math", "tool": "solve_math_problem", "query": "x^2-5x+6=0", "answer_style": "step_by_step"},
  "observations": [
    {"step": 0, "tool": "solve_math_problem", "arguments": {"question": "x^2-5x+6=0"},
     "success": true, "summary": "{\"answer\": \"…\", \"model\": \"mathllm-round7\"}", "duration_ms": 102000}
  ],
  "steps": 1,
  "stopped_reason": "final",
  "answer": "…",
  "verification": {"status": "verified", "method": "substitution", "detail": "2 个答案全部满足等式", "counterexample": null},
  "verify_attempts": 0,
  "answer_model": "mathllm-round7"
}
```

**POST `/api/agent/stream`** — SSE 事件序列

```text
data: {"type":"thinking","stage":"classify"}
data: {"type":"route","decision":{"intent":"knowledge","tool":"search_knowledge","query":"…"},"model":"deepseek-flash","role":"orchestrator"}
data: {"type":"tool_start","step":0,"tool":"search_knowledge","arguments":{"query":"…"},"model":"","role":""}
data: {"type":"tool_end","step":0,"tool":"search_knowledge","success":true,"duration_ms":2124,"summary":"…","model":"","role":""}
data: {"type":"thinking","stage":"decide"}
data: {"type":"thinking","stage":"answer"}
data: {"type":"answer_delta","content":"判别式"}
data: {"type":"answer_delta","content":"是用于…"}
data: {"type":"verify_start","attempt":0}
data: {"type":"verify","verification":{"status":"verified","method":"grounding","detail":"…"},"model":"deepseek-flash","role":"orchestrator"}
data: {"type":"done","steps":1,"stopped_reason":"final","verification":{…},"answer_model":"deepseek-flash","answer_role":"orchestrator","run":{…}}
```

被证伪时会额外出现 `retry` → `answer_reset`，然后重新流式输出修正版。

**GET `/api/metrics`** 与 **GET `/api/traces`** — 见第十节。

`/api/agent/run` 响应示例（`AgentRun`）：

```json
{
  "question": "求解 x^2-5x+6=0",
  "decision": {"intent": "math", "tool": "solve_math_problem", "query": "x^2-5x+6=0", "answer_style": "step_by_step"},
  "observations": [
    {"step": 0, "tool": "solve_math_problem", "success": true, "summary": "...", "duration_ms": 1200}
  ],
  "steps": 1,
  "stopped_reason": "final",
  "answer": "..."
}
```

---

## 十五、关键设计与踩坑

1. **JSON 模式只保证语法**：必须叠加 Pydantic 校验 + 错误反馈重试，否则枚举/字段仍会错。
2. **7B 原生 Function Calling 不稳**：改用「结构化输出选工具 + 程序执行」，把工具选择变成一次可校验的 JSON 决策。
3. **工具失败是常态**：把失败收口成 `ToolResult`，让循环从第一天就处理失败，而不是等接 RAG 时才发现。
4. **计算器绝不能 `eval()`**：AST 白名单 + 长度/指数上限。
5. **Chroma 1.x 的 embedding function 会写进 collection 配置**：改用「自己算向量、显式传入」绕开，同时获得可注入性。
6. **默认 embedding 是英文模型**：中文 RAG 必须换中文 embedding，否则排序完全失真。
7. **Windows 下 Chroma 占用文件**：临时目录清理要用 `ignore_cleanup_errors=True`。
8. **测试不能依赖网络**：embedder 与检索层都做注入，测试全离线。
9. **同步阻塞不能塞进事件循环**：Chroma 检索用 `asyncio.to_thread` 包装。
10. **循环必须有硬边界**：步数上限 + 重复检测 + 结果截断 + 失败上报，四者缺一不可。
11. **流式端点要覆盖「没有事件的等待」**：classify 和 decide 是两次长耗时模型调用，不发事件的话界面会在一两分钟内毫无反馈；用 `thinking` 事件补上。
12. **前后端共享同一套循环逻辑**：`run_agent` 和 SSE 端点都消费同一个 `iter_agent_events`，避免两条实现各自漂移；P2 的测试自动成为回归网。
13. **Windows 下不要用 `Set-Content -Encoding UTF8` 改 UTF-8 文件**：PowerShell 5.1 会先按 GBK 解码，导致中文乱码甚至吞字符；应改用 .NET 的 `UTF8Encoding($false)` 无 BOM 写入。
14. **验证器不能看到它要验证的推导**：把「翻译题目」和「抽取答案」合成一次调用时，模型会拿解答自己的因式分解去当检验等式，验证变成循环论证。必须拆成两次调用，翻译只看题目。
15. **只在有硬证据时判错**：`refuted` 必须来自 SymPy 的确定性矛盾；`unknown` 一律不重试。否则 7B 的翻译波动会引发误判和重试风暴。
16. **模型会把方程翻译退化**：把「解方程 x²-5x+6=0」译成 `x = 0` 是真机发生过的，会让正确的根被判错。既要用 few-shot 示例约束，也要在代码层检测「变量 = 常数」并降级为 `unknown`。
17. **流式输出与重试天然冲突**：答案已经吐给用户了才被证伪，必须显式发 `answer_reset` 让前端清空重画，而不是默默拼两段答案。
18. **让合适的模型做合适的事**：7B 微调模型擅长解题、不擅长规划与评判；云端大模型相反。把「编排」与「解题」拆到两个端点，比用一个模型硬扛所有任务更可靠。
19. **云端的流式回退不能重来**：编排流式调用若已产出内容才失败，绝不能换模型重新生成——用户会看到两段拼接的答案。只在尚未产出任何内容时才回退。
20. **用退出码连接 GUI 与启动器**：tkinter 弹窗与 PowerShell 启动器之间用 `0/10/1` 三个退出码表达「保存并启动 / 仅保存 / 取消」，比解析输出文本可靠。
21. **改了构造函数的参数类型要全局搜旧字段**：`VLLMClient` 从 `Settings` 换成 `ModelConfig` 后，残留的 `config.vllm_base_url` 让 `/api/health` 直接 500，而原有测试恰好没覆盖该接口。改签名后要搜一遍旧属性名，并补上缺失的接口测试。
22. **「去掉一次调用」会连带去掉流式**：为了省掉一次生成而让答案直通工具结果时，流式输出也跟着消失了。要恢复它，必须让**工具本身**支持流式，而不是在答案阶段再补一次生成。
23. **推理模型会把输出预算耗在思考上**：`reasoning_content` 吃满 `max_tokens` 后 `content` 为空，HTTP 却是 200。客户端必须显式检查内容是否为空，并把它当成失败处理（可回退到另一个模型）。
24. **关掉思考可能更危险**：`reasoning_effort: "none"` 让模型更快，但会让它给出「格式正确、语义错误」的翻译（把极值题译成求零点），从而把正确答案判成错误。用 few-shot 示例补质量，比关思考更稳。
25. **观测要包在循环外面**：把落盘放在 `try/finally` 的外层，半路失败也能留下记录；放在循环内部就只能记录成功的那部分。
26. **「是否产出过事件」不等于「是否产出过内容」**：推理模型会先吐思考分片，如果按「有事件」判断回退，就永远不会触发。判断依据必须是内容非空。
27. **没有 trace 就发现不了「静默失败」**：答案退化成兜底文案时，前端只显示兜底文案，`done` 事件看起来正常，只有把结果落盘才看得出这次生成是空的。
28. **观测不能影响主流程**：`record_trace` 内部吞掉所有异常，磁盘写不进去也绝不能让请求失败。
29. **测试要显式关掉副作用**：落盘功能默认开启，测试必须用 `trace_enabled=False`，否则会往真实目录写垃圾数据（这次就写了 16 条）。
30. **「不知道」和「零」必须区分开**：空窗口的比率返回 `null` 而不是 `0.0`，否则「没有数据」会被误读成「零失败」。
31. **先有观测再谈优化**：`fallback_rate=0.67` 这个数字一出来，「体感慢」立刻变成「云端有 2/3 的运行在回退本地」这个可执行的问题。没有指标时，连问题是什么都说不清。
32. **观测指标只能说明「哪里慢」，不能说明「改完对不对」**：前者是运行指标，后者需要标注评测集。两者缺一不可，别把「失败率下降」当成「答案更准」。
33. **先看原始数据，再怀疑渲染层**：公式没渲染出来，第一反应是 KaTeX 配置问题，实际是后端把答案截断了——闭合的 `$` 压根没生成。打印 trace 里的原始答案，两分钟就能定位。
34. **兜底式的全局重绘很危险**：一个为修 A 页面而加的 `render()`，会在 B 页面产生「点了没反应」的现象（展开的运行记录被重建、收起）。重绘要有页面边界。
35. **输出上限是真实约束，不是形式参数**：512 tokens 对分步解答来说太小，会被硬截断在公式中间。上限要按最长答案估，而不是按典型答案估。

---

## 十六、当前能力与后续

### 现在能做什么

- 一句话提问，自动判断是「解题 / 查概念 / 普通问答」
- 自动调用对应工具（MathLLM 求解 / 知识库检索 / 安全计算 / 答案校验）
- 有界循环，不会失控，能给出完整执行轨迹
- 中文知识库检索，返回带来源的片段
- 通过 `/api/agent/run` 以 JSON 返回全过程
- Web 端新增「Agent 模式」：实时显示路由结果、每步工具调用与耗时、结束原因；答案流式输出，可随时暂停
- 答案返回前经过独立验证：方程/计算题由 SymPy 代入或求值裁决，知识题由检索片段核对来源；被证伪时带反例自动重算一次
- 验证结论带 `method`（代入检验 / 独立求值 / 来源核对 / 模型评审），调用方能判断可信度
- 模型可分工：编排（路由 / 规划 / 验证）走云端大模型，数学求解走本地微调模型；云端不可用可回退本地
- 轨迹卡逐步显示实际使用的模型，配置可通过 `configure.bat` 的图形弹窗完成
- 数学答案由本地模型**边写边流式显示**（不再等全部生成完才出现）；编排模型返回空内容时自动回退本地
- 答案长度上限可配（默认 1024 tokens），避免分步解答被截断在公式中间
- 每次运行都落盘可回放：按 `run_id` 查决策、工具、验证、耗时、token、错误
- 通过 `GET /api/metrics` 查看运行指标：失败率、路由分布、验证分布、幻觉率代理、P50/P95 延迟、token、回退率
- Web 端「运行观测」页面：整体指标卡 + 逐次运行的可展开调用链，支持 1/7/30 天窗口切换

### 当前已知问题（下一步的输入）

这些不是猜测，都是指标和 trace 直接暴露出来的：

| 问题 | 证据 | 方向 |
|---|---|---|
| 云端编排回退率高 | `fallback_rate = 0.67`，某次运行「8 次调用 · 回退 2 次」，P95 107.8s | trace 里记录**每次调用**用了哪个模型，定位是哪一步回退；或对机械任务关闭思考 |
| 本地解题是主要瓶颈 | 解题单步约 100s（CPU），占一次数学请求的大部分 | GPU，或更小的本地模型 |
| 验证覆盖有限 | 积分、证明、应用题只能到 `unknown` | 扩充题型支持 |
| Agent 模式无多轮记忆 | 每次请求无状态，已有的摘要接口没接进 Agent | 把记忆接进 Agent |

### 这套东西怎么用在面试里

| 面试问题（框架第十六节） | 可引用的证据 |
|---|---|
| 如何评测一个 Agent？ | `GET /api/metrics` 的 12 项运行指标 + 计划中的标注评测集 |
| 如何定位 Agent 出错的位置？ | 每次运行落盘 trace，按 `run_id` 查决策/工具/验证；「运行观测」页面可视化 |
| 如何防止 Agent 无限循环？ | `max_steps` + 重复调用检测 + `max_verify_retries`，并有对应测试 |
| 如何保护工具和 API Key？ | 计算器 AST 白名单、验证器表达式白名单、日志写盘前脱敏、密钥只存 `.env.local` |
| Function Calling 如何工作？ | 7B 原生调用不稳，改用「结构化输出选工具 + 程序校验执行」，工具失败永不抛异常 |
| RAG 检索不到怎么办？ | 库缺失 → `KnowledgeBaseError`；检索为空 → `success=False`；两者都不会被当成答案 |
| 如何降低延迟和成本？ | 模型路由（编排走云端、解题走本地）+ 回退策略 + 结果可观测 |

### 后续阶段（对应 Agent 学习框架）

| 阶段 | 内容 |
|---|---|
| P4 | LangGraph 正式编排（条件边 + checkpoint + 断点恢复）；SSE 中间事件已在「前端接入」阶段提前完成 |
| P6 | Multi-Agent：Planner / Researcher / Solver / Critic / Summarizer |
| P7 | 块1 调用链日志 ✅ / 块2 运行指标 ✅ / 块3 标注评测集 / 块4 注入防护与越权面 |
| P8 | 部署与性能：并发、缓存、模型路由、Docker |

### 剩余待办（可勾选）

**功能**

- [ ] **P7 块 3：标注评测集** —— 目前只有运行指标，还不能回答「它到底对不对」。需要一份带标注的评测集（期望工具、期望来源、期望答案）+ 组件级评测脚本，跑出路由准确率 / 检索命中率 / 答案准确率
- [ ] **P7 块 4：安全加固** —— 用户输入分隔符与注入检测、越权面说明（日志脱敏已完成）
- [ ] **P4：LangGraph 编排** —— 把手写循环搬进 `StateGraph`，加 checkpoint 与断点恢复
- [ ] **P6：Multi-Agent** —— 拆成 Planner / Researcher / Solver / Critic；单 Agent 已稳定，可以对比「多 Agent 是否真的更好」
- [ ] **P8：部署与性能** —— 并发、缓存、模型路由降级、Docker

**优化（全部来自指标与实测，不是猜测）**

- [ ] 在 trace 里记录**每次调用**用了哪个模型，定位 `fallback_rate = 0.67` 具体是哪一步在回退
- [ ] 对机械任务尝试关闭思考（`reasoning_effort`），配合已有 few-shot 示例保质量
- [ ] 把 `/api/chat`、`/api/memory/summarize` 也切到编排模型（目前仍走本地，约 80s → 可降到 3s）
- [ ] Agent 模式接入多轮记忆（目前每次请求无状态，已有摘要接口未接入）
- [ ] 扩充验证覆盖：积分、证明、应用题目前只能到 `unknown`
- [ ] 本地解题是主要瓶颈（约 100s/题，CPU 推理），考虑 GPU 或更小的本地模型
- [ ] 答案上限（1024 tokens）与生成速度的平衡点可以再调

**工程收尾**

- [ ] 本地文件夹升级为仓库工作副本，避免每次推送都走一遍「克隆 → 覆盖 → 推送」流程
- [ ] README 增加「纯功能版」说明：不训练模型、把解题模型指向任意 OpenAI 兼容接口也能用
- [ ] 把本记录放进仓库 `docs/`，作为开发日志随代码一起维护

已完成：P0 结构化输出 → P1 工具层 → P2 有界 ReAct 循环 → P3 真实 RAG → Agent 前端接入 → P5 答案独立验证 → 模型路由（云端编排 + 本地解题）→ P7 块1 调用链日志 → **P7 块2 运行指标（含「运行观测」页面）** → **轨迹卡分块（路径 → 回答）**

---

## 十七、前端改造：Agent 执行轨迹按路径分块（路径 → 回答）

### 动机

轨迹卡和答案原本是两块分离的内容：上面一张轨迹卡（路由 → 工具 → 验证），下面一个独立的答案气泡。看轨迹时不知道答案是哪一步产生的，看答案时又要回头找它属于哪条路径。

目标是**把「路径选择」和「该路径的回答」绑在一起**：每选一条路径，紧跟着就是这条路径产出的内容。

改造前：

```
Agent 执行轨迹
  路由  math → solve_math_problem  deepseek-flash
  solve_math_problem  mathllm-round7  成功 · 82.5s
  验证 · 已验证  代入检验  deepseek-flash
  结束原因：final · 答案由 mathllm-round7 生成

（下面单独一个答案气泡：解题步骤 + 检查与验证 + 最终答案）
```

改造后：

```
① 解题路径
   路由  math → solve_math_problem  deepseek-flash
   ● solve_math_problem  mathllm-round7  成功 · 82.5s
   ── 解题回答 ──
   ### 题目类型与已知条件 …
   ### 解题步骤 …

② 验证路径
   ⚑ 已路由到验证路径：由独立模型重新检查答案是否成立，避免自证。
   ● 验证 · 已验证  代入检验  deepseek-flash
   2 个答案全部满足等式                      ← verification.detail
   ── 验证回答 ──
   ### 检查与验证 …
   ### 最终答案  x = 2 或 x = 3（高亮）
结束原因：final · 答案由 mathllm-round7 生成
[复制答案] [收藏上一道题]
```

### 先厘清「谁产生了什么」

改造前先确认了内容归属，避免把「模型自查」当成「系统验证」：

| 内容 | 产生者 | 来源 |
|---|---|---|
| 解题回答（题目类型 / 步骤） | 本地解题模型 `mathllm-round7` | `solve_math_problem` 工具流式输出 |
| 「检查与验证」段 | 本地解题模型（**自查**，不作为验证结论） | 同上，答案正文的一部分 |
| 验证结论（已验证 / 已证伪 / 无法验证） | **SymPy 独立计算** | `verify_service.check_by_substitution()` |
| 验证里的题目翻译 / 答案抽取 | 编排模型 `deepseek-flash`（失败回退本地） | `translate_question()` / `extract_answer()` |
| `2 个答案全部满足等式` | SymPy 代入结果 | `VerificationResult.detail` |

所以「验证回答」= 后端独立验证结论 + 模型自查段 + 最终答案，而不是把模型自查当成验证。

### 验证路径块逐行对照（界面上每条内容来自谁）

| 界面元素 | 产出者 | 代码位置 |
|---|---|---|
| ⚑ 已路由到验证路径：… | 前端固定文案 | `app.js` 的 `verifyNotice` |
| ● 验证 · 已验证 · 代入检验 | 状态/方法由 SymPy 判定，`代入检验` 是 `method=substitution` 的中文标签 | `verify_service.check_by_substitution()` / `app.js` 的 `VERIFY_METHOD` |
| `deepseek-flash` 标签 | 验证期间**最后一次**网关调用实际用的模型 | `loop.py` 的 `_model_label(client)`（读 `ModelGateway.last_model`） |
| 2 个答案全部满足等式 | SymPy 代入结果 | `VerificationResult.detail` |
| 验证回答 →「检查与验证 / 当 x=2 时…」 | 本地模型 `mathllm-round7` 的答案自查段 | `splitAnswer()` 从 `message.content` 切出 |
| 验证回答 →「最终答案 x=2 或 x=3」 | 同上，本地模型的答案正文 | 同上 |

一句话：**回答正文是本地模型写的，判定结论是 SymPy 算的，deepseek 只参与翻译与抽取。**

### deepseek 在验证里具体做什么（两个机械调用）

`verify_answer()` 只让编排模型做两件搬运工作，**都不下判断**：

1. `translate_question()` — 题目翻译（只给题目，不给解答）
   - 输入：`解方程 x^2-5x+6=0`
   - 输出（`SympyForm`）：
     ```json
     {"applicable": true, "lhs": "x**2 - 5*x + 6", "rhs": "0", "variables": ["x"]}
     ```
   - prompt 明确「绝对不要参考或使用任何解答过程」（`verify_service.py:41` / `:329`）
2. `extract_answer()` — 答案抽取（给题目 + 完整解答）
   - 输出（`ExtractedAnswer`）：
     ```json
     {"kind": "solution_set", "values": ["2", "3"]}
     ```
   - 只抠最终答案的值，不带 `x=` 前缀、不自己重算（`verify_service.py:79`）

之后才是裁判环节：

```python
check_by_substitution(lhs="x**2 - 5*x + 6", rhs="0",
                      values=["2", "3"], variables=["x"])
# x=2 → 0，x=3 → 0  → verified
# detail: "2 个答案全部满足等式"
```

模型输出里**没有任何判定**；`已验证` 完全来自 SymPy 代入。

#### 为什么要拆成两次调用

最初把「抽取答案」和「翻译题目」合成一次调用（省一次模型调用）时，模型会直接拿解答里的因式分解 `(x-2)(x-3)` 当检验等式——等于「拿答案验答案」，验证失去意义。拆开后翻译只看题目，物理上无法使用解答的推导（第九节 P5 已记录该踩坑）。

#### 两层兜底

- **表达式白名单**：`lhs` / `rhs` 先过 `_is_safe_expression()`（字符白名单 + 禁止 `__` + 禁止游离小数点）再 `sympify`，防止模型输出被当代码执行。
- **退化翻译检测**：译成「变量 = 常数」（`lhs` 是单个 Symbol）时判 `unknown` 而非 `refuted`（`verify_service.py:194`），避免翻译波动把正确答案判错。

知识类问题没有可代入等式，才走 `check_grounding()`：模型此时才真正参与判断，但被约束在检索片段内，且「资料未支持」只记 `unknown`、不判错。

> 补充：若云端返回空内容（推理模型常见），网关会回退本地，`deepseek-flash` 这个标签会随之变成实际使用的本地模型名（第九节「回退策略」）。

### 实现（纯前端）

`web/app.js`

| 函数 | 改动 |
|---|---|
| `isSectionLine(line, keywords)` / `splitAnswer(content)` | 新增。按标题把答案切成 `solution` / `verifySection` / `finalSection` 三段 |
| `renderTraceMarkup(trace, content)` | 改签名。块① = 路由 + 步骤 + 「解题回答」；块② = 提醒 + 验证行 + `verification.detail` + 重试 + 「验证回答」（自查段 + 高亮的最终答案） |
| `renderMessages()` | Agent 消息不再渲染独立 `.message-bubble`，答案进轨迹卡；动作按钮留在卡片底部 |
| `updateTraceCard()` | 加 rAF 节流 + `renderMath(card)`，并接收 `message.content` |
| `handleAgentEvent()` | `answer_delta` / `answer_reset` / `done` 不再直接写气泡，统一由末尾的 `updateTraceCard()` 重渲染 |
| `updateStreamingAnswer()` | 保留给非 Agent 模式（`/solve`、`/chat`） |

`web/styles.css`：新增 `.trace-block-label` / `.trace-answer` / `.trace-verify-answer` / `.trace-verify-detail` / `.trace-final-answer` 及深色模式。

`web/index.html`：缓存版本号 `20260918-08` → `20260918-09`。

### 标题切分的回退策略

模型输出的标题格式没有被 prompt 强制，所以切分必须能「优雅退化」：

- 识别规则：Markdown 标题（`#{1,6}`）以关键词开头，或整行就是关键词（可带 `：` / `**`）
  - 验证段关键词：`检查与验证` / `验证与检验` / `检验` / `验证`
  - 最终答案关键词：`最终答案` / `结论` / `答案`
- 只找到验证段 → 其后全部算验证回答
- 只找到最终答案 → 只切最终答案
- **一个都没找到 → 整段归到解题路径**（退化成「只穿插、不拆正文」，绝不丢内容）

刻意不用裸关键词「检查」，避免把「### 检查定义域」误判成验证段。

### 流式处理

答案边生成边切分：标题还没出现时，整段显示在「解题回答」下；标题出现后，后续内容自动归到「验证回答」。因为 `handleAgentEvent` 每个事件末尾都会重渲染轨迹卡，所以不需要额外的状态机。

### 验证

- `node --check web/app.js` 通过；三个前端文件均为无 BOM 的合法 UTF-8。
- 静态预览页复用**真实的** `renderTraceMarkup`（按函数名切片 eval，不是复制一份），覆盖四种状态：

  | 场景 | 结果 |
  |---|---|
  | 运行中（解答流式增长，验证块未出现） | 只有块①，`正在生成答案…` + 解题回答 |
  | 已完成 | 块① 解题回答；块② 独立结论 + 自查段 + 最终答案高亮 |
  | 证伪后修正 | 验证行下多一行「第 1 次修正…」，其余正常 |
  | 无标题回退 | 整段归解题回答，验证块只显示独立结论 |

- 真实页面 `index.html` 打开无脚本报错（仅后端未启动的 CORS 提示）。

### 踩坑

1. **PowerShell 改文件编码第二次把 `index.html` 写坏**：为改缓存版本号用了 `Set-Content`，PowerShell 5.1 默认按 GBK 写回，UTF-8 的图标字符 `⌁` 把后面的 `<` 字节吃掉，页面把 `</span>` 当文本显示、布局整体崩坏。这次改用 Edit 工具（内部 UTF-8）改版本号。第十五节第 13 条已记录过同类问题，**改静态文件时不要用 `Set-Content`**。
2. **静态预览不能整文件加载 `app.js`**：`app.js` 末尾有 `render()` 等顶层副作用，直接 eval 会因缺少 DOM 报错。改为按函数名切片、只 eval 需要的函数（含依赖 `escapeHtml` / `markdownToHtml` 等），保证预览用的是真实实现。
3. **浏览器面板跑不了真实页面**：面板里 `defer` 脚本在面板 DOM 就绪前执行，导致 `#mobile-menu` 为 null。这是面板环境问题，改用静态预览 + 文本快照完成验证。

### 权衡

| 得到 | 付出 |
|---|---|
| 每条路径后面紧跟它的产出，阅读顺序与执行顺序一致 | 依赖模型输出的标题格式；格式变了会退化成不拆分 |
| 验证结论（SymPy）与模型自查段并排展示，归属清晰 | 轨迹卡内容变长，卡片更高 |
| 流式期间也按路径分块，边写边归位 | 每次事件重渲染整张卡（已用 rAF 节流） |

---

# 第十七节：知识库归因（RAG Attribution）功能

> 目标：对**每一次模型的回答**回答三个问题——这次有没有走知识库？检索到了哪些片段（含来源、相似度、排名）？回答是不是基于这些片段？
>
> 交付：一个独立的「知识库归因」页面 + 后端归因服务 + 自动接地检查；旧运行记录也能派生归因。
>
> 状态：实现完成，`158` 个单元测试全部通过（新增 `12` 个），并已用真机（本地 Ollama + Chroma + 已有 traces + 浏览器）端到端验证。

---

## 一、需求与设计决策

用户最初想做「RAG 效果评估页」，澄清后明确：不要标注集跑分，而是**对每次回答做归因**——是否依据知识库、检索到了什么。据此确定的四个决策：

| 决策点 | 选择 | 理由 |
|---|---|---|
| 页面形态 | **独立新页面**（不是并进运行观测） | 归因是「内容质量」视角，观测是「运行性能」视角，分开更清晰 |
| 接地检查 | **自动**（每次知识库回答都跑） | 用户要求自动；每次多 1 次模型调用 |
| trace schema | **允许新增 `rag` 字段** | 旧数据无该字段仍可解析；新运行多一块结构化归因 |
| 相似度 | **展示 `distance`** | 需要 `search_knowledge` 返回距离；旧 trace 无此字段则显示 `—` |

### 为什么不是「图片转向量」那类方案
归因要判断「回答是否依据检索内容」，这是**语义核对**，不是向量相似度。项目里已有的 `verify_service.check_grounding()` 正是做这件事的受约束裁判（只在检索片段内判断），直接复用即可，无需引入新的向量模型。

## 二、数据模型改动

`app/agent/schema.py` 新增三个模型：

```python
class RetrievedSource(BaseModel):   # 一个被检索到的知识片段
    source: str
    rank: int = 0
    distance: float | None = None
    snippet: str = ""

class RagGrounding(BaseModel):      # 回答是否被检索资料支持
    grounded: bool
    unsupported: list[str] = []     # 资料未支持的关键结论
    reason: str = ""

class RagAttribution(BaseModel):    # 一次回答的知识库归因
    used_knowledge: bool = False
    query: str | None = None
    retrieved: list[RetrievedSource] = []
    cited_sources: list[str] = []   # 回答中显式提到的来源
    grounding: RagGrounding | None = None
```

- `AgentRun` 与 `RunTrace` 各新增可选字段 `rag: RagAttribution | None = None`。
- **向后兼容**：旧 JSONL 里没有 `rag`，`RunTrace.model_validate` 依然通过，归因由服务在读取时**按需派生**（见下）。

`app/agent/tools/knowledge.py` 的 `search_knowledge` 返回体补上 `distance` 与 `rank`：

```python
"documents": [
    {"content": chunk.content, "source": chunk.source,
     "distance": chunk.distance, "rank": rank}
    for rank, chunk in enumerate(chunks, start=1)
]
```

## 三、后端实现

### 3.1 归因服务 `app/services/rag_attribution_service.py`（新增）

核心是「从 trace 派生」，因此对新旧记录一视同仁：

| 函数 | 作用 |
|---|---|
| `build_attribution(decision, observations, answer)` | 解析 `search_knowledge` 观察里的 `documents`，生成 `RagAttribution`；`used_knowledge` 由「有检索结果」或「路由为 knowledge」判定；`cited_sources` 用来源文件名/词干在回答中匹配 |
| `attribution_from_trace(trace)` | 有 `trace.rag` 直接返回，否则现场派生（旧记录兼容） |
| `to_item(trace, checks)` / `list_items(...)` | 拍平成列表项（含来源列表、引用、接地状态、未支持结论） |
| `summarize(traces, checks, window_days)` | 概览统计：走知识库比例、平均检索片段数、接地通过率、未接地数 |
| `load_checks(path)` / `save_check(path, run_id, grounding)` | 手动「重新核对」结果的 sidecar 存储 |
| `checks_path(trace_dir)` | `data/traces/rag_checks.jsonl` |

> 注意：`build_attribution` 的 `retrieved` 与 `cited_sources` 都在**服务层派生**，页面不依赖被截断的 `observation.summary` 之外的任何信息；`snippet` 截断到 300 字符，避免 trace 膨胀。

### 3.2 自动接地：只在「跳过验证」分支跑

`app/services/verify_service.py` 把原来的 `check_grounding` 拆出一个返回结构化结论的 `judge_grounding(client, answer, documents) -> GroundingVerdict | None`，`check_grounding` 改为调用它。这样归因层能拿到 `unsupported` 列表。

`app/agent/loop.py` 的改动：

1. 新增变量 `rag_grounding: RagGrounding | None = None`。
2. 在 `should_skip_verification()` 为真（知识库回答默认跳过 SymPy 验证）时：
   - 若 `settings.rag_grounding` 且确有检索片段，发 `rag_grounding_start` 事件；
   - 调 `verify_service.judge_grounding()`，成功后发 `rag_grounding` 事件（含 `grounded/unsupported/reason`）；
   - 之后照旧发 `verify_skipped`。
3. 若 `verify_rag=True`（走 `verify_answer`），当返回的 `verification.method == "grounding"` 时，把结论同步成 `RagGrounding`。
4. 答案定稿后 `build_attribution(...)` 组装 `rag`，把 `rag_grounding` 填进去，随 `AgentRun` 一起返回，并写进 `done` 事件。

**为什么只放在 skip 分支**：如果 `verify_rag=True`，`verify_answer` 已经调用过 `check_grounding`，再跑一次就是重复的模型调用；放在 skip 分支既满足「自动接地」，又不与既有验证路径打架，也不改变 `run.verification` 的语义（默认仍为 `None`）。这一点直接保住了既有的两个知识库流式测试。

`app/services/trace_service.py` 的 `build_trace()` 增加 `rag=run.rag`，归因随运行一起落盘到 `data/traces/runs.jsonl`。

### 3.3 配置 `app/core/config.py`

新增 `rag_grounding: bool`，读取 `MATHLLM_RAG_GROUNDING`（默认 `1`，即开启）。想省一次模型调用可设为 `0`。

### 3.4 路由 `app/api/routes/rag_attribution.py`（新增，已在 `main.py` 注册）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/rag/attribution?days=7&limit=50` | 概览 + 最近运行归因列表（新→旧） |
| GET | `/api/rag/attribution/{run_id}` | 单次详情：`RunTrace` + `rag`（含片段、引用、接地） |
| POST | `/api/rag/attribution/{run_id}/check` | 对某次回答**重新**跑接地裁判，结果写入 `rag_checks.jsonl`，并覆盖详情里的结论 |

- `days=0` 表示全部；沿用 `metrics_service.filter_since()` 做时间窗过滤。
- 详情接口对旧记录会现场派生 `rag`；`check` 接口在没有检索片段时返回 `400`。

## 四、前端实现

`web/index.html`：系统分组下新增导航 `data-page="rag-attribution"`「知识库归因」（图标 `◈`）。

`web/app.js`：

| 位置 | 改动 |
|---|---|
| `PAGE_META` | 新增 `"rag-attribution": ["系统 / 知识库归因", "看看每次回答有没有依据知识库"]` |
| `state` | 新增 `ragAttribution: { loading, days: 7, summary, runs, error }` |
| `setPage` / `render` / `hashchange` / 初始化 | 进入该页时调用 `loadRagAttribution()`；`render()` 增加页面分支 |
| 新增函数 | `ragAttributionPage()`、`loadRagAttribution()`、`ragAttributionRun(item)`、`ragDetailMarkup(trace)`、`loadRagDetail(runId)`、`checkRagGrounding(runId)`、`ragGroundingBadge(status)` |
| `bindPageEvents` | 绑定 `#rag-refresh`、`#rag-days`、`[data-rag-detail]`、`[data-rag-check]` |

页面结构：

1. **概览卡片**：运行总数、走知识库比例、平均检索片段数、接地通过率、未接地回答数。
2. **逐次回答列表**：每条 `<details>` 显示时间、问题、`知识库/未用知识库` 徽标、`已接地/未接地/未检查` 徽标、片段数与耗时；展开显示路由、检索查询、命中来源、回答引用、未支持结论。
3. **加载详情**：点按后请求详情接口，内联渲染每个检索片段的 `#rank + 来源 + 距离 + 片段文本`、接地结论、模型回答（截断 1200 字）。
4. **重新核对**：对知识库回答重跑接地裁判并刷新。

`web/styles.css`：新增 `.rag-detail` / `.rag-detail-block` / `.rag-chunk` / `.rag-chunk-head` / `.rag-rank` / `.rag-distance` / `.rag-chunk-body`，复用既有 `run-card` / `run-badge` / `obs-note` / `stat-card` 样式。

## 五、测试

新增 `tests/test_rag_attribution.py`（12 个用例，全部离线）：

- `BuildAttributionTest`：无知识库使用、片段解析（来源/rank/distance/snippet）、引用匹配、检索失败不计片段但算知识库运行。
- `SummarizeTest`：接地状态计数、通过率、空窗口返回 `None`。
- `ChecksStoreTest`：sidecar 读写与「后写覆盖」。
- `LoopGroundingTest`：知识库运行在默认设置下自动记录 `rag.grounding`。
- `RagAttributionRouteTest`：列表/详情/404/`check` 落盘并覆盖详情结论。

结果：`Ran 158 tests ... OK`（原 146 + 新 12）。`node --check web/app.js` 通过。

## 六、真机验证

1. 启动后端（8080）与静态前端（7860）。
2. `GET /api/rag/attribution?days=0` 返回 4 次既有运行：`knowledge_rate=0.5`、`avg_retrieved=1.5`，其中一次知识库运行派生出 `retrieved_count=3`，证明**旧 trace 也能派生归因**。
3. `GET /api/rag/attribution/{run_id}` 返回 3 个片段（来源、rank、snippet；`distance` 对旧 trace 为 `null`，新运行才有）。
4. 浏览器打开 `http://localhost:7860/#rag-attribution`：导航高亮、概览卡片、逐次回答徽标均正确；点「加载详情」后详情区展开（按钮位置随内容下移，说明片段与回答已插入）。

## 七、踩坑与权衡

1. **自动接地的调用顺序会打乱脚本化测试的响应队列**：最初把接地放在答案生成后、验证前，导致 `verify_rag=True` 的测试里 `form/extracted/verdict` 三次响应被提前消耗。改为**只在 skip 分支**执行后，验证路径的调用序列保持不变，测试零改动。
2. **`observation.summary` 会被截断到 4000 字符**：所以归因的片段解析必须容忍截断；`snippet` 再截到 300 字符，只用于展示，不用于判定。
3. **旧 trace 没有 `distance`/`rank`**：解析时用枚举序号兜底 `rank`，`distance` 缺失就置 `None`，页面显示 `—`，不报错。
4. **`data/traces/` 已被 gitignore**：`rag_checks.jsonl` 放在同目录，天然不入库；归因本身是运行时产物，不污染仓库。

| 得到 | 付出 |
|---|---|
| 每次回答自动有「是否用知识库 / 检索了什么 / 是否接地」三问答案 | 每次知识库回答多 1 次接地模型调用（可 `MATHLLM_RAG_GROUNDING=0` 关闭） |
| 旧运行记录也能派生归因，无需迁移 | 旧记录的 `distance` 为空，展示为 `—` |
| 归因结构化落盘，可回放、可统计 | `runs.jsonl` 每条记录略增（片段 snippet 最多 300 字/条） |

## 八、后续可选

- 概览加「走知识库但未接地」的筛选，直接定位疑似幻觉回答。
- 引用检测目前是「来源名/词干出现在回答里」，可升级为按片段内容做引用归因。
- 若接入多模态，图片类回答也可复用同一套归因结构（`used_knowledge` + `retrieved` + `grounding`）。

---

# 第十八节：知识库归因的可用性打磨（只显示知识库回答 + 无结果兜底）

> 反馈来自真机使用：① 归因页把「未走知识库」的运行也列出来了，噪声大；② 详情里看不出「检索到几段、引用了几段、准确率多少」；③ 知识库没检索到内容时，模型还在自由发挥，应该直接说「根据知识库的信息无法回答」。
>
> 状态：三项全部完成，`159` 个单元测试通过（新增 `1` 个），并用真机浏览器逐项验证。

---

## 一、列表只显示「走知识库」的回答

**改动位置**：`web/app.js` 的 `ragAttributionPage()`。

```js
const knowledgeRuns = view.runs.filter((item) => item.used_knowledge);
const runs = knowledgeRuns.length
  ? knowledgeRuns.map(ragAttributionRun).join("")
  : `<div class="card empty-state"><strong>窗口内没有走知识库的回答</strong>…</div>`;
```

- 概览统计口径不变（仍以全部运行做分母，才能看出「走知识库比例」），但列表只渲染 `used_knowledge === true` 的运行。
- 每行的徽标从「知识库 / 未用知识库」二选一简化为固定的「知识库」；「重新核对」按钮对所有列出的运行都显示（因为都走过知识库）。
- 概览卡片也做了语义微调：`走知识库` 卡片副标题改为「全部 N 次运行中 M 次」，新增 `未检查` 卡片（做了接地判定之外的知识库回答数）。

## 二、详情展示「检索到多少片段 / 引用了哪些 / 准确率多高」

**改动位置**：`web/app.js` 的 `ragDetailMarkup()` + `ragAttributionRun()` 的行内 meta。

详情顶部新增一行统计（`.rag-stat-row`）：

| 统计项 | 取值 |
|---|---|
| 相关片段 | `rag.retrieved.length` 个 |
| 引用来源 | `rag.cited_sources.length` 个 |
| 接地准确率 | 本次：已接地 `100%` / 未接地 `0%` / 未检查 `—`；并附「窗口 X%」（当前时间窗的 `summary.grounded_rate`） |

列表行的 meta 也从「N 片段 · 耗时」升级为「N 片段 · 引用 M · 耗时」，不展开就能看到引用数量。

`web/styles.css` 新增 `.rag-stat-row` / `.rag-stat`（自适应卡片式小格），深色模式沿用变量自动适配。

> 「准确率」采用**接地准确率**（回答被检索资料支持的比例），因为它是项目里可判定的量；检索相似度 `distance` 仍逐片段展示（旧记录为 `—`）。

## 三、知识库没检索到 → 直接回答「根据知识库的信息无法回答」

**问题**：`search_knowledge` 无结果或失败时，`observation.success = false`，此前仍会走模型生成，模型可能凭自身知识作答，违背「knowledge 意图只依据知识库」的约束。

**改动位置**：`app/agent/loop.py`。

1. 新增常量：

```python
KNOWLEDGE_UNAVAILABLE_ANSWER = "根据知识库的信息无法回答。"
```

2. 在答案生成分支里，`math` 直通之后、模型生成之前，插入知识库兜底：

```python
elif decision.intent == "knowledge" and not verify_service.documents_from_observations(
    observations
):
    # A knowledge intent may only be answered from the knowledge base.
    # With nothing retrieved, do not let the model improvise an answer.
    yield {"type": "thinking", "stage": "answer"}
    answer = KNOWLEDGE_UNAVAILABLE_ANSWER
    yield {"type": "answer_delta", "content": answer}
    answer_model = {"model": "", "role": ""}
```

- 触发条件：路由意图为 `knowledge` 且**没有任何检索片段**（无论是因为没命中、工具报错还是超时）。
- 此时不再调用模型生成答案（`stream_raw` 调用次数为 0），回答是确定性的固定文案。
- 该路径下 `documents` 为空，后续 `rag_grounding` 自然跳过，归因里 `grounding = null`、`retrieved = []`，页面显示「未检索到知识库片段 / 未做接地检查」。
- 非 knowledge 意图（math / general）不受影响。

## 四、测试

`tests/test_loop.py` 新增：

```python
async def test_knowledge_without_results_answers_unavailable(self) -> None:
    client = ScriptedClient([_route("knowledge", "search_knowledge"), FINAL_JSON])
    with patch.object(rag_service, "search", return_value=[]):
        result = await run_agent(client, "什么是判别式", make_ctx(client))
    self.assertEqual(result.stopped_reason, "final")
    self.assertEqual(result.answer, "根据知识库的信息无法回答。")
    self.assertEqual(client.stream_calls, 0)   # 不再让模型自由发挥
```

结果：`Ran 159 tests ... OK`（原 158 + 新 1）；`node --check web/app.js` 通过。

## 五、真机验证

1. `web/index.html` 缓存版本号从 `20260918-12` 升到 `20260920-01`，否则浏览器会继续用旧 `app.js`（踩坑见下）。
2. 启动后端 + 静态前端，打开 `/#rag-attribution`：
   - 概览：走知识库 50.0%（全部 6 次运行中 3 次）、接地准确率 100%、未检查 2。
   - 列表只剩 3 条知识库运行（两条数学运行不再出现）。
3. 点「加载详情」（3 片段那条）后，`browser.inspect` 读到元素文本：

   ```
   相关片段 3 个  引用来源 0 个  接地准确率 100% 窗口 100.0%
   检索片段 #1 06-MathLLM项目与解题模型.md 距离 —
   ```

   详情高度 1106px，证明片段与统计已插入。

## 六、踩坑

1. **浏览器缓存旧脚本**：只改了 `app.js` 但没改 `index.html` 里的 `?v=` 版本号，页面仍是旧行为（显示全部 6 条）。把 `styles.css` / `app.js` 的 `?v=` 统一升到 `20260920-01` 后恢复。**改前端静态资源务必同步升版本号。**
2. **浏览器面板的滚动不是窗口滚动**：`browser.scroll` 读到 `maxScrollY=0`，但详情其实已经渲染。改用 `browser.inspect` 读取元素文本/高度来确认，比依赖视口文本快照可靠。
3. **`browser.capture` 两次失败**（`UnknownVizError` / 超时），本轮验证改用 `snapshot` + `inspect` 完成。

## 七、权衡

| 得到 | 付出 |
|---|---|
| 列表聚焦知识库回答，噪声大幅减少 | 概览「走知识库比例」与列表条数不同口径，需要副标题解释 |
| 详情一眼看到片段数/引用数/接地准确率 | 旧记录 `distance` 为 `—`、引用多为 0（回答很少显式写来源名） |
| 知识库无结果时确定性兜底，杜绝模型自由发挥 | 工具超时也会被归为「无法回答」，真实故障被这条文案掩盖（后续可细分「无命中」与「检索失败」两种文案） |

---

# 第十九节：fastembed 缓存被 Temp 清理误删导致检索失败（含缓存目录迁移）

> 现象：Agent 模式问「mathllm 是什么模型」，路由到 `search_knowledge`，工具失败：
>
> ```
> 工具 search_knowledge 执行失败: [ONNXRuntimeError] : 3 : NO_SUCHFILE :
> Load model from C:\Users\zpb\AppData\Local\Temp\fastembed_cache\models--Qdrant--
> bge-small-zh-v1.5\snapshots\46fbe35f...\model\optimized.onnx failed:
> Load model ... optimized.onnx failed. File doesn't exist
> ```
>
> 状态：定位为缓存目录问题，已迁移到项目内持久目录并重建；`159` 个测试通过，检索真机验证恢复。

---

## 一、根因

1. `fastembed` 的默认缓存目录是 `os.path.join(tempfile.gettempdir(), "fastembed_cache")`，在 Windows 上即 `%TEMP%\fastembed_cache`。
2. 第十七节 RAG 建库时，embedding 模型（`BAAI/bge-small-zh-v1.5`）就下载到了这个 Temp 目录里；Chroma 索引里存的是**向量**，模型文件仍只在缓存目录。
3. 之前做 C 盘清理时，把 `C:\Users\zpb\AppData\Local\Temp` 的内容整体删除了，于是 `fastembed_cache` 被破坏：`blobs/`、`refs/`、`snapshots/` 目录结构还在，但 `model/optimized.onnx` 这个权重文件没了。
4. 之后查询时，`rag_service.search()` → `get_default_embedder()` → `FastEmbedEmbedder._load()` → `TextEmbedding(...)` 尝试从缓存加载，看到 snapshot 目录已存在就不再重新下载，但权重文件缺失 → `NO_SUCHFILE`。
5. 因为查询向量算不出来，`search_knowledge` 工具抛错，Agent 就返回「暂时无法完成请求 / 工具执行失败」。

> 关键点：**索引在、模型缓存不在**。删除 Temp 不会损坏 `data/chroma`（索引是持久目录），但会让查询时的 embedding 模型加载失败。

## 二、修复：把缓存迁出 Temp

### 1. 配置项（`app/core/config.py`）

新增字段 `rag_cache_dir`，读取 `MATHLLM_FASTEMBED_CACHE_DIR`，默认 `data/fastembed_cache`：

```python
rag_cache_dir=os.getenv("MATHLLM_FASTEMBED_CACHE_DIR", "data/fastembed_cache"),
```

### 2. Embedder 支持自定义缓存目录（`app/services/rag_service.py`）

```python
class FastEmbedEmbedder:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model
```

进程级缓存键从「仅模型名」改为「模型名@缓存目录」，避免不同目录互相串用：

```python
def get_default_embedder(settings):
    key = f"{settings.rag_embedding_model}@{settings.rag_cache_dir}"
    ...
```

### 3. 忽略与模板

- `.gitignore` 增加 `data/fastembed_cache/`（模型权重不入库）。
- `.env.local.example` 增加 `MATHLLM_FASTEMBED_CACHE_DIR=data/fastembed_cache` 及注释说明。

### 4. 重建缓存

```powershell
Remove-Item -Recurse -Force "$env:TEMP\fastembed_cache"
python -c "from app.core.config import settings; from app.services import rag_service; rag_service.get_default_embedder(settings).embed_query('判别式')"
```

新缓存落到 `data/fastembed_cache/fast-bge-small-zh-v1.5/`，其中 `model_optimized.onnx` 约 90MB（fastembed 0.8.0 的缓存布局与旧版不同，不再用 `models--Qdrant--*/snapshots/...`）。

## 三、验证

- `rag_service.search(settings, "二次方程判别式")` 返回 3 条，Top1 命中 `01-二次方程与判别式.md`（distance 0.436）。
- `embed_query("判别式")` 维度 512，缓存目录为 `data/fastembed_cache`。
- `Ran 159 tests ... OK`。

## 四、踩坑与教训

1. **清理系统 Temp 会误伤应用缓存**：fastembed（以及不少库）默认把模型/缓存放在 `%TEMP%`。清理 `AppData\Local\Temp` 前要意识到这一点；对需要在意的模型，应显式指定持久 `cache_dir`。本次已把 fastembed 迁到 `data/fastembed_cache`。
2. **残缺缓存比没有缓存更难排查**：`snapshots/` 目录存在会让 fastembed 误判「已下载」，直接去读缺失的权重文件，报 `NO_SUCHFILE`，而不是自动重下。遇到这种情况要**整目录删除**再重建。
3. **改后端配置后必须重启后端**：正在运行的旧进程仍持有旧的缓存路径，不重启不会生效。
4. **这类故障不会污染仓库**：`data/fastembed_cache/` 已 gitignore；`data/chroma` 索引未受影响，无需重新建库。

## 五、权衡

| 得到 | 付出 |
|---|---|
| 模型缓存落在项目内，清理系统 Temp 不再影响 RAG | 项目 `data/` 下多占约 90MB（已 gitignore，不影响仓库） |
| 缓存路径可配置，便于换模型/换机器 | 首次使用要重新下载一次模型（约 90MB） |
| 进程级缓存键含目录，多配置互不干扰 | 键格式变化，属内部实现细节 |

---

# 第二十节：知识库检索失败可诊断化（区分失败类型 + 启动预热 + 归因页检索状态）

> 背景：第十九节解决了缓存位置，但「超时/失败」和「知识库确实没有」在界面上仍然长得一样——都显示「根据知识库的信息无法回答」。用户要求：**能直接判断问题出在哪里，方便以后修改**。
>
> 采纳方案：① 区分失败类型（核心，可诊断）② 启动预热 embedding 模型（预防）③ 归因页展示检索状态（长期观测）。
>
> 状态：完成，`168` 个单元测试通过（新增 `9` 个），并用真机 API 验证。

---

## 一、问题：三类失败被压成一句话

| 真实情况 | 修复前表现 | 问题 |
|---|---|---|
| 检索超时 / 工具失败 | 「根据知识库的信息无法回答」 | 把**基础设施故障**说成了**没有答案**，误导 |
| 索引未建立 | 同上 | 看不出要建库 |
| embedding 加载失败 | 同上 | 看不出是模型/网络问题 |
| 检索正常但没有相关内容 | 「根据知识库的信息无法回答」 | 唯一正确的用法 |

目标：让前三种明确报出「检索失败 + 原因」，只有第四种才是「无法回答」。

## 二、实现

### 1. `rag_service.py`：异常分型 + 预热

新增两个 `KnowledgeBaseError` 子类：

```python
class KnowledgeIndexMissing(KnowledgeBaseError):   # 索引没建
class KnowledgeEmbeddingError(KnowledgeBaseError): # embedding 模型加载/计算失败
```

- `FastEmbedEmbedder._load()` 把 `TextEmbedding(...)` 的异常包成 `KnowledgeEmbeddingError`（含异常类型与信息）。
- `index_knowledge()` 的 `embed_documents`、`search()` 的 `embed_query`、`collection.query()` 分别包成 `KnowledgeEmbeddingError` / `KnowledgeBaseError`。
- `search()` 里「目录不存在」和「collection 不存在」改为抛 `KnowledgeIndexMissing`（原来是笼统的 `KnowledgeBaseError`）。
- 新增 `warm_up(settings, *, embedder=None) -> int`：主动 `embed_query("warmup")` 一次，把模型加载移出请求路径；失败抛 `KnowledgeEmbeddingError`。

CLI 增加 `--warm`：

```powershell
python -m app.services.rag_service --warm
# embedding 模型已预热（向量维度 512）
```

### 2. `knowledge.py`：工具层区分「失败」与「无命中」

| 情况 | 返回 |
|---|---|
| 索引缺失 | `success=False, error="知识库尚未建立，请先运行索引", data.reason="index_missing"` |
| embedding 失败 | `success=False, error="检索失败：<原因>", data.reason="embedding_error"` |
| 其他检索异常 | `success=False, error="<原因>", data.reason="retrieval_error"` |
| 检索成功但 0 命中 | **`success=True, data={"documents": [], "reason": "no_match"}`** |
| 超时 | 由 `run_tool` 的 `asyncio.wait_for` 产生，`success=False, error="工具 search_knowledge 超时（30.0s）"` |

> **语义变更**：0 命中从 `success=False` 改为 `success=True`（检索本身成功了，只是没有相关内容）。这样调用方才能把「失败」和「无命中」分开。

### 3. `loop.py`：兜底文案按原因分流

新增常量与辅助函数：

```python
KNOWLEDGE_UNAVAILABLE_ANSWER = "根据知识库的信息无法回答。"
KNOWLEDGE_FAILED_TEMPLATE = "知识库检索失败（{reason}），暂时无法回答。"

def _knowledge_failure(observations) -> str | None:
    for observation in observations:
        if observation.tool == "search_knowledge" and not observation.success:
            return observation.error or "未知错误"
    return None
```

知识库回答且没有片段时：

```python
failure = _knowledge_failure(observations)
answer = KNOWLEDGE_FAILED_TEMPLATE.format(reason=failure) if failure else KNOWLEDGE_UNAVAILABLE_ANSWER
```

于是：

| 看到 | 含义 |
|---|---|
| 知识库检索失败（工具 search_knowledge 超时（30.0s）） | 超时/基础设施问题 |
| 知识库检索失败（知识库尚未建立，请先运行索引） | 索引没建 |
| 知识库检索失败（检索失败：embedding 模型加载失败：…） | 模型/网络问题 |
| 根据知识库的信息无法回答。 | 检索正常，知识库确实没有 |

### 4. 启动预热：`start_local.ps1`

`--ensure` 之后增加：

```powershell
Write-Host "Preloading the embedding model (first run may download ~90MB)..." -ForegroundColor Cyan
& $pythonPath -m app.services.rag_service --warm
if ($LASTEXITCODE -ne 0) { Write-Host "Embedding model preload failed; ..." -ForegroundColor Yellow }
```

把首查的模型加载（首次含约 90MB 下载）挪到启动阶段，避免在 30s 工具超时里完成。

### 5. 归因页：新增「检索状态」

`rag_attribution_service.RagAttributionItem` 增加：

```python
retrieval: str = "none"          # ok / empty / failed / none
retrieval_error: str | None = None
```

`_retrieval_status()` 从观察里判定：有失败的 `search_knowledge` → `failed`（带 error）；有片段 → `ok`；used_knowledge 但无片段 → `empty`；否则 `none`。

`RagAttributionSummary` 增加 `retrieval`（分布）与 `retrieval_failures`（失败次数）。

前端（`web/app.js`）：
- 列表行新增检索状态徽标：`检索成功` / `无命中` / `检索失败`（失败时 hover 显示原因）。
- 详情统计行新增「检索状态」格（失败时附错误信息）。
- 概览新增「检索失败」卡片（副标题显示无命中次数）。

## 三、测试（新增 9 个，共 168）

- `test_rag.py`：0 命中改为成功且 `reason=no_match`；`KnowledgeIndexMissing` / `KnowledgeEmbeddingError` 的分型；`warm_up` 返回维度、失败包成 `KnowledgeEmbeddingError`。
- `test_loop.py`：检索失败时回答「知识库检索失败（…）」；无命中时回答「根据知识库的信息无法回答。」。
- `test_rag_attribution.py`：`retrieval` 取值为 ok / empty / failed（带 error）；summary 统计失败数。

`Ran 168 tests ... OK`；`node --check web/app.js` 通过。

## 四、真机验证

- `python -m app.services.rag_service --warm` → `embedding 模型已预热（向量维度 512）`。
- 另起后端（8090，避开用户正在运行的 8080 旧进程）后 `GET /api/rag/attribution`：
  - `summary.retrieval = {"failed":5,"ok":1}`，`retrieval_failures = 5`
  - 逐条 `retrieval`：failed / failed / failed / failed / none / ok / none / failed / none
  - 即历史上有 **5 次知识库运行检索失败**，现在能在页面上直接看出来。

## 五、踩坑与教训

1. **旧进程不重启看不到新字段**：第一次在 8080 上验证时字段为空——因为 8080 还是用户 14:17 启动的旧后端（新代码没加载）。改端口到 8090 才验证成功。**改后端代码后必须重启。**
2. **「0 命中」的语义要慎重**：从 `success=False` 改成 `success=True` 是有意为之——它是「检索成功、结果为空」，不是「失败」。这样调用方和页面才能区分。改动同时更新了对应的单元测试。
3. **超时是框架层抛的**：`asyncio.wait_for` 在 `run_tool` 里兜底，工具处理函数拿不到，所以超时只能通过 `observation.success=False` + `error` 文本识别，`_knowledge_failure()` 正是基于这一点。

## 六、权衡

| 得到 | 付出 |
|---|---|
| 一眼分清「检索失败/超时」「索引缺失」「模型问题」「确实没有」 | 回答文案变长，含错误详情 |
| 启动预热把冷启动移出请求路径 | 启动时多一步（首次含约 90MB 下载，有进度） |
| 归因页可长期统计检索失败次数 | 多了 `retrieval`/`retrieval_error` 字段与徽标 |
| 失败原因进 trace，可回放排查 | 无 |

---

# 第二十一节：路由到 none 时引导回数学/知识问题

> 反馈：Agent 轨迹里出现「路由 general → none」时，模型仍会自由回答（如闲聊）。用户要求：路由到 `none` 时直接返回引导语，把用户拉回数学题或知识类问题。
>
> 状态：完成，`169` 个单元测试通过（更新 4 个、新增 1 个）。

---

## 一、改动

`app/agent/loop.py` 新增常量：

```python
OUT_OF_SCOPE_ANSWER = "根据知识库的知识无法回答该问题，请提问数学问题或其他知识类问题。"
```

在答案分支中，`math` 直通之后、knowledge 兜底之前，新增：

```python
elif decision.tool == "none" and decision.intent != "math":
    # The router picked no tool: the question is outside the assistant's
    # scope (math and the knowledge base). Guide the user back instead of
    # letting the model answer anything.
    yield {"type": "thinking", "stage": "answer"}
    answer = OUT_OF_SCOPE_ANSWER
    yield {"type": "answer_delta", "content": answer}
    answer_model = {"model": "", "role": ""}
```

- 触发条件：`tool == "none"`（即路由判定为 `general`，或路由解析失败回落成 `general/none`）。
- 结果：确定性文案，不调用模型（`stream_raw` 调用为 0），`answer_model` 为空。
- 排除 `math`：若路由把数学题误判成 `math/none`，不走引导，仍走原生成路径，避免对数学题说「请提问数学问题」。

## 二、行为变化

| 输入 | 之前 | 之后 |
|---|---|---|
| 你好 / 今天天气怎么样 | 模型自由回答 | 「根据知识库的知识无法回答该问题，请提问数学问题或其他知识类问题。」 |
| 数学题 | 解题 | 不变 |
| 数学概念 / 项目问题 | 检索知识库 | 不变 |

> 这是有意的产品取舍：助手收敛到「数学 + 知识库」，不再承接通用闲聊。

## 三、测试

原本有 4 个用例用 `general/none` 来验证「模型生成答案」路径，因行为改变需要改用 `knowledge + search_knowledge`（该路径同样走 `generate_answer_stream`）：

- `test_loop.py`
  - `test_non_math_answer_uses_orchestrator_stream`：改用 `knowledge` 路由 + patch 检索命中。
  - `test_unparsable_action_falls_back_to_final`：断言答案改为 `OUT_OF_SCOPE_ANSWER`。
  - 新增 `test_general_route_returns_out_of_scope_guidance`：`general/none` → 固定引导语，`stream_calls == 0`。
- `test_agent_stream.py`
  - `test_answer_deltas_reconstruct_answer`、`test_empty_answer_gets_fallback_delta`：改用 `knowledge` 路由 + patch 检索命中。

`Ran 169 tests ... OK`；`node --check web/app.js` 通过。

## 四、权衡

| 得到 | 付出 |
|---|---|
| 助手聚焦数学/知识，闲聊不再自由发挥 | 失去通用闲聊能力（有意为之） |
| 路由异常回落成 general/none 时也有确定行为 | 引导语固定，不含具体建议 |
| 不调用模型，省一次请求 | 无 |

---

# 第二十二节：引用定位到「文档 + 行范围」+ 知识库文档浏览页

> 反馈：归因页的「引用来源」按文件名做字面匹配，模型几乎不在回答里写文件名，所以恒为 0，没意义。用户要求：让接地裁判直接输出「回答引用了哪个文档的哪几行（范围即可）」；另外新增一个页面浏览知识库里的 Markdown 文档。
>
> 状态：完成，`177` 个单元测试通过（新增 8 个），并用真机浏览器验证。

---

## 一、引用定位：从「字面匹配文件名」改为「裁判判定片段」

### 1. 建库时记录行号范围（`rag_service.py`）

片段原本按字符窗口切，没有行号。新增：

```python
def normalize_text(text): ...            # \r\n -> \n, strip
def _chunk_offsets(normalized, chunks): ...  # 用 forward find 定位每个片段的字符偏移
def _line_range(normalized, start, end): ...  # 偏移 -> 1-based 行号
```

`index_knowledge` 给每个片段写入 metadata `{"source", "start_line", "end_line"}`；`search` 把它们读进 `RetrievedChunk`（新增 `start_line`/`end_line` 字段）。`search_knowledge` 工具把行号一并返回。

> 行号是近似的（片段按字符切、可能从行中间开始），符合「给个范围就行」。

### 2. 裁判输出「用到了哪些片段」（`schema.py` + `verify_service.py`）

`GroundingVerdict` 增加 `used_chunks: list[int]`（资料里 `[片段 N]` 的编号）。裁判 prompt 现在给每个片段标注来源与行范围：

```
[片段 1]（来源：01-二次方程与判别式.md 第 1-16 行）
<片段内容>
```

并新增 `sources_from_observations()`（返回 content + source + 行范围），`judge_grounding()` 改为接收这些结构；`documents_from_observations()` 保留为它的文本投影。

### 3. 映射成引用（`loop.py` + `verify_service.py`）

新增 `verify_service.citations_from_verdict(verdict, sources)`：把 `used_chunks` 映射成 `RagCitation{source, start_line, end_line}`，越界索引忽略、重复去重。

`RagGrounding` 增加 `citations`；自动接地时写入：

```python
rag_grounding = RagGrounding(
    grounded=verdict.grounded,
    unsupported=list(verdict.unsupported),
    reason=verdict.reason,
    citations=verify_service.citations_from_verdict(verdict, sources),
)
```

`RagAttributionItem` 增加 `citations`，归因页据此展示。

### 4. 前端展示（`web/app.js`）

- 列表行：`引用位置：06-....md 第 10-25 行`（原来的「回答引用」）
- 详情统计：`引用来源 N 个` 改为按 `grounding.citations` 计数
- 详情新增「引用位置」块，逐片段显示 `#rank + 来源 + 第 X-Y 行 + 距离`

## 二、新增「知识库文档」浏览页

### 后端 `app/api/routes/knowledge.py`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/knowledge/documents` | 列出 `knowledge/` 下的 `.md`/`.txt`（名称、大小、修改时间） |
| GET | `/api/knowledge/documents/{name:path}` | 返回原始 Markdown 文本 |

`_safe_path()` 做路径穿越校验（解析后必须落在 `knowledge/` 内）、后缀白名单校验。已在 `main.py` 注册。

### 前端

- `index.html` 工作区新增导航「▤ 知识库文档」
- `web/app.js`：`PAGE_META` / `state.knowledge` / `knowledgePage()` / `loadKnowledge()` / `loadKnowledgeDoc()`，左列文档列表、右侧用 `markdownToHtml` + `renderMath` 渲染 Markdown（支持 KaTeX）
- `web/styles.css`：`.knowledge-layout` / `.knowledge-list` / `.knowledge-doc-item` / `.knowledge-view`（含窄屏单列）

## 三、测试（新增 8 个，共 177）

- `test_rag.py`：索引后片段带行号范围
- `test_verify.py`：`sources_from_observations` 带行号；`citations_from_verdict` 映射/越界/去重
- `test_knowledge_route.py`（新）：列出、取内容、404、后缀拒绝、路径穿越拒绝

`Ran 177 tests ... OK`；`node --check web/app.js` 通过。

## 四、真机验证

- 重建索引：`python -m app.services.rag_service` → `已索引 10 个片段`；检索返回 `01-...md L1-16` 等行号
- 另起后端（8090）：
  - `GET /api/knowledge/documents` 返回 6 篇文档；`GET .../01-二次方程与判别式.md` 返回内容
  - 归因条目新增 `citations` 字段
- 浏览器（前端临时指向 8090）：知识库文档页列出 6 篇并正确渲染 Markdown；归因页显示引用行范围

## 五、踩坑（重要）

1. **又用 PowerShell 改 `index.html` 写坏了文件**：为改缓存版本号，用了
   ```powershell
   (Get-Content -Raw) -replace ... | Set-Content -Encoding UTF8
   ```
   `Get-Content` 在中文 Windows 上按系统默认编码（GBK）读取 UTF-8 文件 → 得到乱码字符串 → `Set-Content` 再按 UTF-8 写回，**整个文件被双重编码写坏**；随后手动去 BOM 又误删了首字符 `<`，导致 `<!doctype>` 变成 `!doctype>`，页面进入怪异模式、显示乱码。
   **教训（第三次了）：改 `web/` 下的静态文件一律用 Edit 工具，绝不用 `Get-Content`/`Set-Content`。**
   恢复方式：`git checkout -- web/index.html` 还原后，用 Edit 工具重加导航与版本号。
2. **HTML 文档本身没有版本号，会被浏览器缓存**：改坏期间浏览器缓存了坏版本，即使磁盘已修好，页面仍显示乱码。需要 `Ctrl+Shift+R` 强制刷新（或用带新 query 的 URL）。
3. **控制台乱码会误导排查**：PowerShell 控制台按 GBK 显示，UTF-8 内容会显示成乱码；判断文件编码要用字节级校验（如 `UTF8Encoding(throwOnInvalidBytes)`），不能靠肉眼看输出。

## 六、权衡

| 得到 | 付出 |
|---|---|
| 引用落到「文档 + 行范围」，不再是恒为 0 的字面匹配 | 行号是近似的（字符切分，非行对齐） |
| 裁判一次调用同时给出 grounded / unsupported / citations | 裁判 prompt 变长，token 略增 |
| 新增文档浏览页，可直接对照引用阅读原文 | 需要重建一次索引才能拿到行号（约 1 秒） |
| 文档接口做了路径穿越与后缀校验 | 无 |

---

# 第二十三节：界面调整（归因去回答 / 解题页侧栏改为知识库文档）

> 反馈：① 知识库归因详情里的「模型回答」没必要展示；② 解题页右侧把「试试这些题」换成知识库文档列表（可点击跳转），示例题下移，删掉「快捷学习工具」。
>
> 状态：完成，纯前端改动，复用已有 `/api/knowledge/documents`，真机浏览器验证通过。

---

## 一、知识库归因详情去掉「模型回答」

`web/app.js` 的 `ragDetailMarkup()` 删除末尾整块：

```js
<div class="rag-detail-block"><strong>模型回答</strong><div class="run-answer">…</div></div>
```

详情现在只保留：统计行（相关片段 / 引用来源 / 检索状态 / 接地准确率）、检索片段（含行号与距离）、引用位置、接地结论。

## 二、解题页右侧改版

原来右侧是两张卡：「试试这些题」+「快捷学习工具」。改为：

| 位置 | 现在 | 说明 |
|---|---|---|
| 上 | **知识库文档** | 列出 `knowledge/` 文档（名称 + 大小），点击跳转到「知识库文档」页并打开该文档 |
| 下 | **试试这些题** | 原来的 4 个示例题，整体下移到原「快捷学习工具」位置 |
| — | 删除「快捷学习工具」 | 连同 `quick` 数组一起移除 |

实现：

- `solvePage()` 删除 `quick` 数组，侧栏改为：
  ```html
  <div class="side-stack">
    <div class="card side-card"><h3>知识库文档</h3><div class="knowledge-mini-list">${knowledgeMiniList()}</div></div>
    <div class="card side-card"><h3>试试这些题</h3><div class="example-list">…</div></div>
  </div>
  ```
- 新增 `state.solveDocs = { loading, loaded, docs }` 与 `loadSolveDocs()`：进入解题页时按需拉取一次文档列表（best-effort，失败静默），成功后重渲染。
- 新增 `knowledgeMiniList()` 渲染列表；`openKnowledgeDoc(name)` 设置 `state.knowledge.current` 后 `setPage("knowledge")`，复用知识库文档页的加载逻辑。
- `render()` 末尾：`if (state.page === "solve") loadSolveDocs();`（内部有 `loaded/loading` 守卫，不会循环）。
- `bindPageEvents()` 绑定 `[data-open-doc]`。
- `web/styles.css` 新增 `.knowledge-mini-list` / `.knowledge-mini-item`。

> 说明：`appendInstruction` 与 `[data-instruction]` 处理器保留（「学习工具」页的 `data-tool-instruction` 仍在用），只是解题页不再有快捷工具按钮。

## 三、验证

- `node --check web/app.js` 通过。
- 真机（前端临时指向带新接口的 8090 后端）：
  - 解题页右侧显示 6 篇知识库文档 + 4 个示例题，无「快捷学习工具」
  - 点「03-概率基础.md」→ 跳转到 `#knowledge` 并打开该文档（内容正确渲染）
  - 归因详情不再出现「模型回答」
- 缓存版本号 `20260920-03` → `20260920-04`（用 Edit 工具修改，**没有再用 PowerShell**）。

## 四、权衡

| 得到 | 付出 |
|---|---|
| 归因页聚焦「检索/引用/接地」，不再混入回答正文 | 想回看回答需去运行观测页 |
| 解题页侧栏直接暴露知识库文档，一键跳转阅读 | 文档列表需后端 `/api/knowledge/documents`，旧后端会显示「知识库暂无文档」 |
| 去掉快捷工具，界面更简洁 | 快捷学习入口（提示/讲简单/检查答案）不再提供 |

---

# 第二十四节：Agent 模式接入对话历史 + 输出截断修复

> 反馈：① 输出在 `$$ x_1 = 2,` 处被截断；② 想删掉「连续追问」模式，因为以为 Agent 模式已带追问——实际并没有，于是先把历史能力加进 Agent，再删掉该模式。
>
> 状态：完成，`183` 个单元测试通过（新增 6 个），前端真机验证。

---

## 一、输出截断：输出 token 上限

- 现象：math 运行的答案在公式中间断掉（trace 尾部 `$$ x_1 = 2,`）。
- 原因：`.env.local` 里 `MATHLLM_MAX_OUTPUT_TOKENS=512`，解题模型（`mathllm-round7`）的输出被硬截断。步进解答 + 行间公式很容易超过 512 token（`.env.local.example` 的默认值本就是 1024）。
- 修复：`.env.local` 改为 `MATHLLM_MAX_OUTPUT_TOKENS=1024`（该文件 gitignore，不入库）。需重启后端生效。
- 说明：`.env.local` 由 `start_local.ps1` 加载；直接用 `python -m app.api.main` 启动时若 shell 未设该变量，则用 `config.py` 默认值 1024。

## 二、Agent 模式接入对话历史

**前提纠正**：Agent 模式原本**不带上下文**——`sendQuestion` 只发 `{question, max_steps}`，`AgentRunRequest` 也只有这两个字段；带上下文的是「连续追问」（走 `/chat`）。

**后端**：

| 文件 | 改动 |
|---|---|
| `app/api/models.py` | `AgentRunRequest` 增加 `messages: list[ChatMessage]`（≤20）与 `summary: str | None`（≤4000） |
| `app/services/chat_service.py` | 新增 `build_history(messages, max_messages, max_chars)`：校验并返回不含 system 的 user/assistant 轮次 |
| `app/agent/router.py` | `classify(..., history=None, summary=None)`：把摘要作为 system、历史轮次拼在 system 与当前问题之间；路由提示词新增「结合历史把追问/省略/指代改写成完整自包含的 query」 |
| `app/agent/loop.py` | `run_agent` / `iter_agent_events` / `_iter_agent_events` / `decide_next_action` / `_answer_messages` / `generate_answer_stream` 全部透传 `history` + `summary` |
| `app/api/routes/agent.py` | 两个接口用 `build_history` 校验并传入；校验失败返回 422 |

**前端**（`web/app.js`）：

- 删除「连续追问」按钮与 `follow_up` 分支；模式只剩 `solve` / `agent`。
- Agent 模式的请求体改为 `{ question, max_steps, messages: 最近对话, summary: 压缩摘要 }`，复用原「连续追问」的 `splitHistory` + `shouldSummarize` + `requestConversationSummary` 逻辑。
- Agent 模式不再每次清空 `memorySummary`；模式按钮文案改为「带上下文，自动选工具并展示轨迹」。
- 状态栏文案：Agent 模式显示「带上下文追问，过长时自动压缩旧内容 / 已启用压缩摘要」。

> 效果：Agent 模式现在可以追问（「那第二步呢」），路由会把追问改写成自包含 query 再调工具；历史过长时自动摘要压缩。

## 三、测试（新增 6 个，共 183）

- `test_router.py`：`CapturingClient` 验证 `classify` 把 `summary` 与历史轮次放进 messages、当前问题在最后。
- `test_chat_history.py`（新）：`build_history` 空输入、去 system、超条数、超字符数。
- `test_agent_route.py`：`/api/agent/run` 接受 `messages` + `summary` 并正常返回。

`Ran 183 tests ... OK`；`node --check web/app.js` 通过。

## 四、真机验证

- 解题页模式切换只剩「单题解答 / Agent 模式」，Agent 按钮标注「带上下文」。
- 缓存版本号 `20260920-04` → `20260920-05`。

## 五、踩坑与教训

1. **删除功能前先确认前提**：用户以为「Agent 已带追问」，实际 Agent 只发当前问题、不带历史。若直接删除会丢掉多轮能力——先纠正、再加历史、最后删除。
2. **本地 `.env.local` 会覆盖代码默认值**：`config.py` 默认 1024，但 `.env.local` 里的 512 生效，导致截断。排查输出异常时先看 `.env.local`。

## 六、权衡

| 得到 | 付出 |
|---|---|
| Agent 模式可多轮追问，路由能消解指代 | 每次请求多带历史，输入 token 增加 |
| 少一个模式，界面更简单 | 原「连续追问」走 `/chat`（纯对话）的路径不再从界面进入 |
| 输出上限 1024，步进解答不再断在公式中间 | 输出越长，本地模型耗时越久 |

---

# 第二十五节：修复知识库回答被误切到「验证回答」

> 现象：知识库问题的轨迹卡里，第①块「解题回答」只剩一个标题，整段正文跑到了第②块「验证回答」下。
>
> 状态：完成，纯前端修复，用真实代码切片在 node 里验证。

---

## 一、原因

`web/app.js` 的 `splitAnswer()` 会按标题关键词把回答切成「解题回答 / 验证回答 / 最终答案」三段：

```js
const FINAL_SECTION_WORDS = ["最终答案", "结论", "答案"];
```

模型的回答里有一个标题「## 结论先行」，`isSectionLine()` 对标题用的是 `plain.startsWith(word)`，于是「结论先行」命中了「结论」→ 被当成「最终答案」段的起点。结果：

- `solution` = 「## 结论先行」之前的内容（只有 `# MathLLM 项目与解题模型介绍` 这个标题）
- `finalSection` = 「## 结论先行」到结尾的全部正文

渲染时 `finalSection` 落在第②块「验证回答」下，于是看起来「回答跑到了验证里」。

## 二、修复（两处）

1. **知识库回答不做拆分**：知识库回答会 `verify_skipped`（跳过独立验证），本就没有「验证段」，整段应放「解题回答」。

   ```js
   const verifySkipped = trace.verificationSkipped || "";
   const split = verifySkipped
     ? { solution: String(content || "").trim(), verifySection: "", finalSection: "" }
     : splitAnswer(content);
   ```

2. **标题匹配收紧**：标题里关键词之后必须紧跟分隔符或结束，才算命中。

   ```js
   return keywords.some((word) => {
     if (!plain.startsWith(word)) return false;
     if (plain === word) return true;
     // Only a separator may follow, so "结论先行" does not match "结论".
     return /^[\s:：,，.。\-—(（]/.test(plain.slice(word.length));
   });
   ```

   这样 `结论先行` 不再命中 `结论`，而 `最终答案：`、`检查与验证`、`## 最终答案` 仍正常命中。

## 三、验证

用真实代码切片（`const VERIFY_SECTION_WORDS` 到 `function renderTraceMarkup`）在 node 里 eval：

| 输入 | solution | verifySection | finalSection |
|---|---|---|---|
| 含「## 结论先行 / ## 分步解答」的知识库回答 | 整段 | 空 | 空 |
| 含「## 检查与验证 / ## 最终答案」的数学回答 | 解题正文 | 「检查与验证」段 | 「最终答案」段 |

`node --check web/app.js` 通过；缓存版本号 `20260920-05` → `20260920-06`。

## 四、权衡

| 得到 | 付出 |
|---|---|
| 知识库回答完整显示在「解题回答」 | 无 |
| 标题关键词不再被前缀误匹配 | 关键词后必须跟分隔符；极端写法（如「最终答案如下」无分隔）不会命中，退化为不拆分（内容仍完整） |








