# MathLLM 大模型知识文档

本文档以当前项目为主线，帮助理解“一个数学大模型项目是如何从数据走到服务的”，并扩展到项目之外常见的大模型技术。重点不是记忆命令，而是理解每个组件解决什么问题、输入输出是什么、什么时候会出错。

## 1. 先看懂整个项目

### 1.1 项目目标

本项目以 `Qwen2.5-7B-Instruct` 为基座模型，使用数学解题数据做监督微调，使模型更稳定地输出数学解题过程和最终答案，最后通过 vLLM 提供接口，再由 FastAPI 和 Gradio 包装成应用。

项目不是从零训练一个大模型，而是：

```text
通用基座模型
  + 数学领域示例
  → 参数高效微调后的模型
  → 合并模型
  → 低比特部署模型
  → 可调用的数学解题服务
```

### 1.2 当前代码对应关系

| 层次 | 项目文件 | 大模型知识重点 |
|---|---|---|
| 数据层 | `scripts/prepare_data.py`、`scripts/create_test_set.py` | 数据格式、去重、数据泄漏、token 长度 |
| 模型加载 | `scripts/train.py`、`configs/train_config.yaml` | Transformer、tokenizer、精度、显存 |
| 训练层 | `scripts/train.py` | Causal LM、SFT、LoRA、QLoRA、checkpoint |
| 模型转换 | `scripts/merge_lora.py` | adapter 与基座权重合并 |
| 部署压缩 | `scripts/quantize.py` | AWQ、W4A16、校准数据、量化误差 |
| 推理引擎 | `deploy/server.py`、`configs/deploy_config.yaml` | vLLM、KV Cache、批处理、吞吐 |
| 评测层 | `eval/evaluate.py`、`eval/llm_judge.py` | 答案抽取、数值等价、语义评测 |
| 应用层 | `app/api/main.py`、`app/frontend/gradio_app.py` | OpenAI 兼容 API、SSE、FastAPI、Gradio |

当前项目的训练主程序是自定义的 `Transformers + TRL + PEFT`，不是 LLaMA-Factory。Web 应用层仍有 TODO，需要在 vLLM 服务成功后再联调。

### 1.3 一次完整运行的边界

```text
data/raw/
    ↓  prepare_data.py
data/processed/train.json + eval.json
    ↓  train.py
outputs/math-lora/                       # LoRA adapter/checkpoints
    ↓  merge_lora.py + 完整精度基座模型
outputs/math-lora-merged/                # 独立 FP16/BF16 模型
    ↓  quantize.py + 训练集校准样本
outputs/math-lora-quantized/             # AWQ W4A16 部署模型
    ↓  deploy/server.py
vLLM OpenAI 兼容接口
    ↓
evaluate.py / FastAPI / Gradio
```

独立测试集 `data/eval/test.json` 只用于最终比较；回归集 `data/eval/regression/` 用来检查已发现的错误是否复发。

## 2. 基座模型：你下载的模型究竟是什么

### 2.1 模型不是一个单独文件

一个 Hugging Face 格式的大模型目录通常包含：

- `config.json`：网络结构和模型超参数，例如层数、隐藏维度、注意力头数；
- `model-*.safetensors`：神经网络权重；
- tokenizer 文件：词表、分词规则和特殊 token；
- chat template：把 `system/user/assistant` 消息转换成模型实际输入文本的规则；
- generation 配置：生成时的默认参数。

因此，只有权重而没有正确 tokenizer 或 config，通常不能正确运行模型。

### 2.2 Transformer 在做什么

对文本来说，模型的核心任务不是直接“理解答案”，而是根据已有 token 预测下一个 token：

```text
已知：小明有 3 个苹果，又买了 2 个
预测下一个 token：，/ 一 / 共 / 有 ...
```

Transformer 的主要组成是：

1. token embedding：把 token ID 变成向量；
2. self-attention：让当前位置关注上下文中相关的位置；
3. MLP/FFN：对每个位置做非线性变换；
4. residual connection 和 normalization：稳定深层网络训练；
5. language model head：把隐藏向量映射为词表上的概率。

Qwen2.5-7B 中的“7B”表示大约 70 亿个可训练参数，不等于模型文件一定只有 7GB。FP16 权重每个参数约 2 字节，实际还要加上分片、配置和 tokenizer；训练时还需要梯度、优化器状态和激活值。

### 2.3 预训练与指令微调

- 预训练：在海量文本上学习语言、知识和模式，目标是 next-token prediction；
- 指令微调（Instruction Tuning）：用“指令 → 高质量回答”示例，让模型学会按要求回答；
- 本项目的 SFT：属于监督指令微调，重点是让模型模仿数学解题样本的格式和解法风格。

微调不会凭空给模型安装一个数学定理库。它更像是在已有通用能力上调整概率分布：面对数学问题时，更倾向于采用项目数据中常见的解题表达和步骤。

## 3. Tokenizer、ChatML 与上下文长度

### 3.1 tokenizer 的作用

Tokenizer 把字符串变成 token ID：

```text
文本 → token 切分 → token ID → embedding 向量
```

token 不一定等于汉字、单词或字符。一个中文词、英文单词、数字、LaTeX 命令都可能被切成不同数量的 token。因此“字符数少”不代表“token 数少”。

训练和推理必须使用与基座模型匹配的 tokenizer。更换 tokenizer 会改变 token ID 的含义，不能只替换一个词表文件。

### 3.2 `messages` 不是模型直接看到的文本

项目数据保存为：

```json
{
  "messages": [
    {"role": "system", "content": "你是数学解题助手"},
    {"role": "user", "content": "求解 x^2-5x+6=0"},
    {"role": "assistant", "content": "因式分解后得到 x=2 或 x=3"}
  ]
}
```

`messages` 是结构化数据；Qwen tokenizer 的 `apply_chat_template()` 会将其转换为包含角色边界和特殊 token 的训练文本。项目中：

- `prepare_data.py` 负责生成标准 `messages`；
- `train.py` 负责按 Qwen chat template 生成 `text`；
- 推理时必须使用相同的模板和 system prompt。

不要手动把 `[im_start]`、`[im_end]` 写进数据，除非明确知道当前 tokenizer 的模板约定。

### 3.3 上下文长度不是输出长度

`max_seq_length=2048` 表示一次训练样本的输入序列上限，包含 system、user、assistant 以及特殊 token。`max_tokens` 通常表示推理时最多新生成多少 token。

如果解题过程过长被截断：

- 训练可能看不到完整答案；
- 最终答案可能被截掉；
- 量化校准样本可能不能代表真实请求。

项目已有 tokenizer 长度检查，但需要继续学习并验证：TRL 当前版本是否按预期构造 labels，以及是否只对 assistant completion 计算 loss，不能因为使用了 `messages` 就默认认为 loss mask 一定正确。

## 4. SFT：项目当前的训练方式

### 4.1 Causal Language Modeling 的 loss

对 token 序列 `x1, x2, ..., xn`，模型在位置 `i` 预测下一个 token `x(i+1)`。训练目标是让正确 token 的概率更高，常用交叉熵：

```text
loss = - log P(正确的下一个 token | 前面的 token)
```

一个样本的 loss 通常是多个 token loss 的平均值。它衡量“模仿参考文本”的难度，不直接等价于数学准确率。

### 4.2 SFT 的输入输出

本项目使用 `SFTTrainer`：

1. 从 JSON 加载训练和验证集；
2. 用 tokenizer chat template 生成 `text`；
3. tokenizer 将 `text` 转成 `input_ids`；
4. trainer 构造 batch、labels、attention mask；
5. 前向计算 loss；
6. 反向传播只更新 LoRA 参数；
7. 定期保存 checkpoint 和 eval loss。

需要重点理解：`eval_loss` 只是验证文本的 token 预测损失。最终应该同时看答案准确率、过程质量、错误类型和泛化能力。

### 4.3 训练配置中的关键量

在 `configs/train_config.yaml` 中：

- `num_train_epochs`：完整遍历数据的次数；
- `per_device_train_batch_size`：单张 GPU 每次放多少样本；
- `gradient_accumulation_steps`：累积多个小 batch 后再更新一次；
- `learning_rate`：参数更新步长；
- `warmup_ratio`：训练初期逐步增大学习率的比例；
- `gradient_checkpointing`：用额外计算换显存；
- `bf16`/`fp16`：训练计算精度；
- `save_steps`、`eval_steps`：保存和验证的频率；
- `max_seq_length`：输入序列上限。

有效 batch size 近似为：

```text
单卡 batch size × 梯度累积步数 × GPU 数量
```

增大学习率或 LoRA 容量不一定更好；小数据集上更容易过拟合。应观察 train loss、eval loss、固定验证集准确率和 bad cases。

## 5. LoRA 与 QLoRA

### 5.1 LoRA 的数学形式

原始权重矩阵为 `W`，LoRA 不直接更新它，而是学习低秩增量：

```text
W' = W + ΔW
ΔW = B A
```

其中：

- `W` 是冻结的基座权重；
- `A` 和 `B` 是可训练的小矩阵；
- `r` 是低秩维度，也就是中间维度；
- `lora_alpha` 控制增量缩放；
- `target_modules` 决定把 LoRA 加到哪些线性层。

如果原矩阵是 `d_out × d_in`，全量更新需要 `d_out*d_in` 个参数，而 LoRA 约需要 `r*(d_in+d_out)` 个参数。`r` 越大，能表达的更新空间通常越大，但显存、训练时间和过拟合风险也会增加。

### 5.2 为什么初始化不会破坏基座模型

常见实现让其中一个 LoRA 矩阵初始化为零，使训练开始时 `ΔW` 接近零。这样模型一开始接近原始基座，训练逐步学习任务相关的增量。

LoRA 不是一个与基座完全独立、可以脱离基座运行的完整模型。adapter 通常必须配合同一基座模型和匹配 tokenizer 使用。

### 5.3 QLoRA 的含义

QLoRA 通常是：

```text
4-bit 量化加载基座模型 + LoRA adapter 训练
```

项目中的 `use_4bit: true` 会使用 bitsandbytes 的 NF4 和 double quantization，以减少训练时基座模型的显存占用。它主要是训练阶段的内存方案。

必须区分：

| 名称 | 主要用途 | 项目中的位置 |
|---|---|---|
| LoRA | 参数高效微调 | `scripts/train.py` |
| QLoRA/NF4 | 低显存训练 LoRA | `scripts/train.py` 的 4-bit 加载 |
| AWQ W4A16 | 离线部署量化 | `scripts/quantize.py` |
| GPTQ | 另一种离线量化方法 | 当前项目没有实现 |

QLoRA 的 NF4 checkpoint 不能直接当成 AWQ 部署模型。项目的推荐链路是先得到完整精度合并模型，再做 AWQ。

### 5.4 adapter、checkpoint、合并模型

- adapter：LoRA 参数和配置，体积小，需要基座；
- checkpoint：训练过程中的 adapter、trainer state 和优化信息，可能有多个；
- merged model：基座权重与 LoRA 增量合并后的独立模型，体积接近基座；
- quantized model：对合并模型的权重做低比特压缩后的部署产物。

`merge_lora.py` 必须以完整精度重新加载基座。不能把训练时的 4-bit 加载状态误当作最终合并模型。

## 6. 量化：从 FP16 到 AWQ W4A16

### 6.1 为什么量化

量化用较少的 bit 表示权重或激活值，从而减少：

- 模型磁盘体积；
- GPU 显存；
- 内存带宽压力；
- 某些硬件上的推理成本。

代价是量化误差，可能表现为答案错误、推理不稳定、长文本质量下降或特定题型退化。

### 6.2 `W4A16` 如何读

- `W4`：权重以 4 bit 表示；
- `A16`：激活值通常以 16 bit 计算；
- 这是 weight-only 量化，不是所有张量都变成 4 bit。

AWQ 会使用代表性校准数据观察激活情况，尽量保护对输出敏感的重要权重。校准数据应覆盖真实数学问题的语言、长度和题型，但不能使用独立测试集。

### 6.3 当前量化脚本的边界

`scripts/quantize.py` 当前使用 `llmcompressor`，目标是生成 vLLM 可加载的 `compressed-tensors` AWQ W4A16 模型。它：

1. 检查输入是否是合并模型；
2. 从 `data/processed/train.json` 选校准样本；
3. 应用 tokenizer chat template；
4. 截断到最大校准长度；
5. 执行 AWQ scaling 和量化；
6. 保存量化模型、tokenizer 和 `quantization_manifest.json`；
7. 检查权重文件和量化元数据。

代码通过不代表量化完成。还必须在云 GPU 上实际运行，并用 vLLM 加载和评测。

### 6.4 量化正确性的判断

至少比较：

- 独立测试集数学准确率；
- 各题型准确率；
- 合并模型与量化模型的 bad cases；
- 首 token 延迟、总延迟和 tokens/s；
- GPU 显存和模型目录大小；
- vLLM 是否能稳定启动并处理多次请求。

不能只看到模型文件变小，就认为量化成功。

## 7. vLLM 与大模型推理

### 7.1 vLLM 是什么

vLLM 是面向大语言模型推理和服务的引擎。它不是模型本身，而是负责：

- 加载模型权重；
- 管理请求队列；
- 执行 prefill 和 decode；
- 管理 KV Cache；
- 对多个请求做连续批处理；
- 提供 OpenAI 兼容 HTTP API。

项目通过 `deploy/server.py` 根据 `configs/deploy_config.yaml` 启动 vLLM。

### 7.2 prefill、decode 与 KV Cache

- prefill：第一次处理用户输入，把整段 prompt 计算成隐藏状态；
- decode：逐 token 生成回答；
- KV Cache：缓存已经计算过的 attention key/value，避免每次生成都重复计算。

上下文越长、并发越高，KV Cache 占用越大。`max_model_len=4096`、`gpu_memory_utilization=0.9` 等参数必须结合实际 GPU 显存测试，不能照抄到所有云服务器。

### 7.3 需要区分的性能指标

- TTFT/首 token 延迟：从请求到第一个 token 的时间；
- total latency：完整回答返回的总时间；
- decode speed：生成阶段 tokens/s；
- throughput：单位时间处理的总 token 或请求数；
- p50/p95：典型请求和尾部慢请求的延迟。

单题平均延迟不能代表多人并发时的服务能力。

### 7.4 OpenAI 兼容 API

常见请求结构：

```json
{
  "model": "模型服务公布的 model id",
  "messages": [
    {"role": "system", "content": "你是数学助手"},
    {"role": "user", "content": "求解 ..."}
  ],
  "temperature": 0,
  "max_tokens": 512,
  "stream": false
}
```

`stream=false` 返回一个完整 JSON；`stream=true` 通常返回一系列 SSE `data:` 事件，客户端需要逐段拼接内容，并处理 `[DONE]`、超时和断开连接。

## 8. 评测：为什么“模型正确”不只是一个数字

### 8.1 当前评测流程

`eval/evaluate.py` 会：

1. 调用 OpenAI 兼容的 vLLM 接口；
2. 保存模型原始回答和时间；
3. 抽取最终答案；
4. 做归一化文本比较或数值容差比较；
5. 计算准确率、延迟和错误样本。

`eval/llm_judge.py` 再用独立模型判断数学语义，输出 `correct`、`incorrect` 或 `uncertain`。

### 8.2 常见评测陷阱

- 参考答案格式和模型答案格式不同，但数学上等价；
- 模型最终答案正确，但中间推理错误；
- 参考答案本身有错误；
- 数值抽取器把题目中的条件数字误当成答案；
- 多答案、方程组、区间、集合不能用单个数字比较；
- LLM Judge 可能误判、超时、返回非法 JSON；
- 测试集被用于调参后，准确率会虚高；
- 不同模型使用了不同 prompt、temperature 或 max tokens，比较不公平。

因此报告应区分：模型错误、参考数据错误、解析错误、接口错误和 judge 错误。

### 8.3 更成熟的数学评测

可以逐步增加：

- 题型/来源/难度分层指标；
- 符号等价检查；
- Python/SymPy 结果验证；
- 最终答案与解题过程分开评分；
- 人工复核抽样和置信区间；
- 固定 prompt、固定 seed、固定生成参数；
- 基座模型、LoRA 模型、合并模型和量化模型的成对比较。

自动指标用于发现问题，人工复核用于确认结论。

## 9. 当前项目之外值得扩展的知识

### 9.1 DPO、RLHF 与偏好优化

SFT 学的是“参考答案长什么样”；偏好优化学的是“两个回答中哪个更好”。

- RLHF：奖励模型 + 强化学习，流程复杂、成本高；
- DPO：直接使用 chosen/rejected 回答对优化偏好，工程上更简单；
- ORPO、KTO：其他偏好优化方法。

如果本项目以后收集了“正确解法 vs 错误解法”“简洁解法 vs 冗长解法”，可以考虑 DPO，但前提是 SFT 基线和评测体系已经可靠。

### 9.2 RAG 与工具调用

RAG 是检索外部知识后再生成回答，适合数学公式库、课程资料、学校规则等会变化或需要溯源的内容。它不能自动解决计算错误，检索文档质量也会影响结果。

工具调用可以让模型把计算交给：

- Python；
- SymPy；
- 计算器；
- 几何绘图或公式验证器。

对数学场景来说，“模型生成推理 + 程序验证结果”通常比单纯增加训练数据更有价值，但要防止工具结果和自然语言解释不一致。

### 9.3 MoE 与更大模型

MoE 用多个专家网络，每次只激活部分专家，可能在相似计算量下扩大参数规模。它会带来路由、负载均衡、显存和部署复杂度，当前 7B dense 模型项目不需要优先引入。

### 9.4 分布式训练

当单卡放不下模型或需要提高吞吐时，可以学习：

- DDP：数据并行；
- FSDP：分片参数、梯度和优化器状态；
- DeepSpeed ZeRO：分阶段切分训练状态；
- tensor parallel：拆分单层计算；
- pipeline parallel：拆分模型层。

当前项目先完成单卡/单机可复现训练，再考虑分布式。否则排错成本会显著增加。

### 9.5 推理优化进阶

可以继续学习：

- FlashAttention；
- paged KV cache；
- prefix caching；
- speculative decoding；
- quantization-aware serving；
- batching 与请求调度；
- CPU offload 和多 GPU tensor parallel。

这些优化都必须用基准测试证明收益，不能只根据概念名称判断速度更快。

### 9.6 MLOps 与模型生命周期

一个可长期维护的模型项目还需要：

- 数据版本管理和哈希；
- 模型 artifact registry；
- 实验追踪；
- 训练/评测 manifest；
- 模型回滚；
- 服务健康检查和监控；
- GPU 利用率、显存、延迟、错误率监控；
- 数据和模型的访问权限控制。

建议每次运行都记录：Git commit、配置文件、基座模型版本、tokenizer 版本、数据文件哈希、软件版本、GPU 信息和完整命令。

## 10. 推荐学习顺序

### 第一阶段：理解当前项目

1. 读 `docs/design.md`，画出自己的数据流和模型流；
2. 读 `scripts/prepare_data.py`，确认一条 JSON 如何变成 `messages`；
3. 用 tokenizer 打印一条 chat template 和 token 数；
4. 读 `scripts/train.py`，理解模型加载、LoRA 注入和 trainer 参数；
5. 读 `merge_lora.py` 和 `quantize.py`，区分 adapter、合并模型和部署量化模型；
6. 读 `deploy/server.py` 和 `eval/evaluate.py`，理解一次请求的完整路径。

### 第二阶段：完成最小闭环

```text
小数据训练
  → 保存 adapter
  → 合并
  → vLLM 启动
  → curl 请求
  → evaluate.py 冒烟评测
```

只有这条链路稳定后，才适合做完整训练和消融实验。

### 第三阶段：补齐工程能力

- 实现 FastAPI 的 `/api/health`、`/api/solve` 和 `/api/chat`；
- 实现 SSE 流式转发和 Gradio 事件绑定；
- 增加 pytest 和 mock vLLM 测试；
- 锁定依赖和运行环境；
- 增加数据、训练、量化和评测 manifest；
- 限制 CORS、端口和 API 访问权限。

### 第四阶段：扩展研究能力

- 按题型分析错误；
- 引入程序验证和工具调用；
- 构造偏好数据尝试 DPO；
- 比较不同量化方案；
- 做并发压测和成本分析；
- 学习分布式训练和更高级推理优化。

## 11. 最后应该能回答的问题

1. 为什么本项目选择 SFT，而不是直接做 RLHF？
2. LoRA 的 `r` 改变了什么，为什么不是越大越好？
3. QLoRA 的 NF4 和 AWQ 的 W4A16 分别解决哪个阶段的问题？
4. 为什么量化前需要先合并 LoRA？
5. `messages` 如何通过 chat template 变成 token？
6. `max_seq_length` 和 `max_tokens` 有什么区别？
7. eval loss 下降但准确率不升高，可能有哪些原因？
8. 为什么独立测试集不能用于选 checkpoint？
9. vLLM 如何通过 KV Cache 和连续批处理提高服务效率？
10. SSE 流式响应和普通 JSON 响应有什么不同？
11. 为什么 LLM Judge 不能直接作为唯一真值？
12. 如何证明一条实验结果使用了正确的模型、数据和配置？

能结合本项目的实际文件、命令和实验结果回答这些问题，才算真正理解了项目，而不只是会运行脚本。

## 12. 关联文档

- [项目 README](../../README.md)
- [项目设计](../design.md)
- [开发总指南](../development-guide.md)
- [ChatML 格式](chatml-format.md)
- [LoRA 原理](lora-principle.md)
- [QLoRA 与 LoRA 对比](qlora-vs-lora.md)
- [INT4 量化](int4-quantization.md)
- [Loss 曲线](loss-curve.md)
- [vLLM 与 PagedAttention](vllm-pagedattention.md)
- [消融实验](ablation-study.md)
