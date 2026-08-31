# ChatML 格式：大模型对话数据格式详解

> 本文档讲解 ChatML 格式的原理、为什么精调要用这个格式、以及如何正确构造训练数据。
> 阅读时间：约 15 分钟。

---

## 一、为什么数据格式这么重要？

### 1.1 大模型是怎么理解对话的？

大模型本质上是一个**文本续写机器**——给它一段文本，它预测接下来的文本。它并不天然理解"这是一个对话"、"这是用户说的"、"这是模型该回答的"。

为了让模型理解对话结构，我们需要一种**约定好的文本格式**来标记：

- 哪些是系统指令
- 哪些是用户输入
- 哪些是模型回答

这种格式就是 **Chat Template（对话模板）**。

### 1.2 不同模型有不同的对话模板

每个模型家族都有自己的对话模板格式：

| 模型 | 对话模板名称 | 特殊标记 |
|------|------------|---------|
| Qwen2.5 | ChatML | im_start / im_end 标记 |
| LLaMA 3 | LLaMA 3 Template | 特殊 token 标记 |
| ChatGPT | messages API | OpenAI 内部处理 |
| Mistral | Mistral Template | INST 标记 |

**关键认知**：每个模型在预训练时就是用特定格式训练的。如果你的精调数据格式和预训练时不一致，模型就会"困惑"，效果大打折扣。

---

## 二、ChatML 格式详解

### 2.1 ChatML 的全称

ChatML = **Chat Markup Language**（聊天标记语言）

最初由 OpenAI 提出，后来被 Qwen（通义千问）等多个模型采用并适配。

### 2.2 ChatML 的三个角色

ChatML 定义了三个角色，每个角色用特殊的开始和结束标记包裹：

**System 角色（系统指令）**：告诉模型它应该怎么行动。
- 例如："你是一个专业的数学解题助手"

**User 角色（用户输入）**：用户说的话。
- 例如："求解方程 x^2 - 5x + 6 = 0"

**Assistant 角色（模型回答）**：模型应该生成的回答。
- 包含完整的解题过程

### 2.3 ChatML 的实际文本格式

Qwen2.5 的 ChatML 使用两个特殊 token 来标记每段内容的边界：

- **im_start**：标记一段内容的开始，后面跟角色名
- **im_end**：标记一段内容的结束

一个完整的单轮对话结构如下：

`
[im_start]system
你是一个数学解题助手。[im_end]
[im_start]user
求解方程 x^2 - 5x + 6 = 0[im_end]
[im_start]assistant
对方程进行因式分解：(x-2)(x-3) = 0，因此 x=2 或 x=3。[im_end]
`

**注意**：im_start 和 im_end 是 Qwen2.5 tokenizer 中的特殊 token（有特殊 token ID），不是普通文本。模型在预训练时就已经学会识别这些标记了。

---

## 三、messages 格式（训练数据的存储格式）

### 3.1 JSON 中的 messages 格式

在实际存储训练数据时，我们不直接写 ChatML 文本，而是用结构化的 JSON（messages 格式）：

`json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个数学解题助手。"
    },
    {
      "role": "user",
      "content": "求解 x^2 - 5x + 6 = 0"
    },
    {
      "role": "assistant",
      "content": "对方程进行因式分解..."
    }
  ]
}
`

### 3.2 为什么用 messages 格式而不是直接写 ChatML 文本？

| 方式 | 优点 | 缺点 |
|------|------|------|
| 直接写 ChatML 文本 | 简单直接 | 容易写错格式，换模型要全部重写 |
| messages 格式 | 结构清晰，可复用 | 需要 tokenizer 转换 |

messages 格式是**模型无关**的——同样的 messages 数据，用不同的 tokenizer 转换就能适配不同的模型。这就是为什么 LLaMA-Factory 和 HuggingFace TRL 都推荐使用 messages 格式。

### 3.3 从 messages 到 ChatML 的转换

HuggingFace 的 tokenizer 有一个方法 pply_chat_template，自动把 messages 格式转换成模型需要的 ChatML 文本：

`python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "你是数学助手"},
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "1+1=2"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False)
print(text)
`

输出会自动变成 Qwen2.5 的 ChatML 格式文本。

---

## 四、训练数据构造规范

### 4.1 数学解题数据的 messages 格式

`json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个专业的数学解题助手。请按以下要求回答：1. 分析题目 2. 详细解题步骤 3. 用 LaTeX 格式书写公式 4. 给出明确答案"
    },
    {
      "role": "user",
      "content": "求函数 f(x) = x^3 - 3x + 1 的极值点"
    },
    {
      "role": "assistant",
      "content": "**分析题目**\n\n求 f(x) = x^3 - 3x + 1 的极值点。\n\n**求解过程**\n\n第一步，对 f(x) 求导：\nf'(x) = 3x^2 - 3\n\n第二步，令 f'(x) = 0：\n3x^2 - 3 = 0\nx^2 = 1\nx = \pm 1\n\n第三步，二阶导数检验：\nf''(x) = 6x\n- f''(1) = 6 > 0，故 x=1 为极小值点\n- f''(-1) = -6 < 0，故 x=-1 为极大值点\n\n**最终答案**\n\n极小值点 x=1，f(1)=-1；极大值点 x=-1，f(-1)=3。"
    }
  ]
}
`

### 4.2 多轮对话格式

数学解题也支持多轮对话（追问、换条件等）：

`json
{
  "messages": [
    {"role": "system", "content": "你是数学解题助手"},
    {"role": "user", "content": "求解 x^2 - 5x + 6 = 0"},
    {"role": "assistant", "content": "因式分解得 (x-2)(x-3)=0，因此 x=2 或 x=3"},
    {"role": "user", "content": "如果把 6 改成 4 呢？"},
    {"role": "assistant", "content": "方程变为 x^2 - 5x + 4 = 0，因式分解得 (x-1)(x-4)=0，因此 x=1 或 x=4"}
  ]
}
`

### 4.3 训练时的 Loss Mask（重要）

训练时**不是对所有 token 都计算 loss**。模型只应该在 **assistant 的回答** 上计算 loss：

`
训练时 Loss 计算示意：

[im_start]system
你是数学助手[im_end]            <- 不计算 loss（固定文本）

[im_start]user
求解 x^2-5x+6=0[im_end]        <- 不计算 loss（用户输入）

[im_start]assistant
因式分解得 (x-2)(x-3)=0[im_end] <- 计算 loss（模型需要学怎么回答）
`

为什么？
- system prompt 是固定的，不需要模型学习
- user 的输入是给定的，模型不需要学会生成用户的问题
- 只有 assistant 的回答是模型需要学习的目标

LLaMA-Factory 和 TRL 会自动处理 Loss Mask，你不需要手动实现。但你需要知道这个原理。

---

## 五、数据质量检查清单

### 5.1 格式检查

| 检查项 | 说明 | 常见问题 |
|--------|------|---------|
| role 字段是否正确 | 只能是 system/user/assistant | 拼写错误（如 Assisstant） |
| messages 顺序是否正确 | system 在最前，user/assistant 交替 | 两个 user 连续出现 |
| content 是否为空 | 不允许空字符串 | 解析原始数据时产生空字段 |
| 最后一条是否为 assistant | 训练数据必须以 assistant 结尾 | 数据截断导致 |

### 5.2 内容质量检查

| 检查项 | 说明 | 为什么重要 |
|--------|------|----------|
| 解题过程是否完整 | 不能只有最终答案 | 模型需要学会推理过程 |
| 数学公式格式是否正确 | LaTeX 语法是否合法 | 错误的 LaTeX 会被模型学到 |
| 答案是否正确 | 人工抽查 10% | 错误答案会教坏模型 |
| 是否有重复数据 | 相似度大于 0.95 去重 | 重复数据导致过拟合 |

### 5.3 长度检查

| 检查项 | 建议阈值 | 处理方式 |
|--------|---------|---------|
| assistant 回答太短 | 小于 20 字符 | 丢弃（信息太少） |
| 总 token 数太长 | 大于 2048 tokens | 截断或丢弃 |
| user 输入太短 | 小于 5 字符 | 丢弃（不是有效问题） |

---

## 六、不同数据来源的格式转换

### 6.1 GSM8K 数据转换

GSM8K 的原始格式：

`json
{
  "question": "Janet has 3 dogs. Each dog eats 2 bags of food per day. How many bags does she need for a week?",
  "answer": "3 dogs * 2 bags = 6 bags per day. 6 * 7 = 42 bags for a week.\n#### 42"
}
`

转换为 messages 格式的 Python 代码：

`python
def convert_gsm8k(item):
    parts = item["answer"].split("####")
    solution = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) > 1 else ""
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": f"{solution}\n\n**最终答案**\n\n{final_answer}"}
        ]
    }
`

### 6.2 MATH 数据转换

MATH 的原始格式：

`json
{
  "problem": "Find the value of x such that sqrt(x+5) = 3",
  "solution": "Squaring both sides: x + 5 = 9, so x = 4.",
  "level": "Level 1",
  "type": "Algebra"
}
`

转换为 messages 格式：

`python
def convert_math(item):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["problem"]},
            {"role": "assistant", "content": item["solution"]}
        ]
    }
`

---

## 七、面试高频问题

### Q: ChatML 格式是什么？

> ChatML 是一种对话标记语言，用特殊的开始和结束标记来区分系统指令、用户输入和模型回答。Qwen2.5 使用 im_start 和 im_end 这两个特殊 token 来标记每个角色的内容边界。模型在预训练时就是用这种格式训练的，所以精调时也必须使用相同格式，否则模型无法正确区分对话角色。

### Q: 为什么精调数据要用 messages 格式而不是直接写 ChatML 文本？

> messages 格式是模型无关的结构化格式，通过 tokenizer 的 apply_chat_template 方法可以自动转换为任何模型需要的 ChatML 格式。这样做的好处是：第一，数据结构清晰不容易出错；第二，同一份数据可以适配不同模型（换个 tokenizer 就行）；第三，训练框架（LLaMA-Factory、TRL）都原生支持 messages 格式。

### Q: 训练时 Loss Mask 是什么？为什么需要？

> Loss Mask 指训练时只在 assistant 的回答上计算 loss，不对 system prompt 和 user 输入计算 loss。因为训练的目标是让模型学会怎么回答，而不是学会怎么生成系统指令或用户问题。如果不做 Loss Mask，模型会把大量训练容量浪费在学习固定的 system prompt 上，影响学习效率。

### Q: 如果换了基座模型（比如从 Qwen 换成 LLaMA），数据格式要怎么改？

> messages 格式的 JSON 数据不需要改。只需要在训练时换用对应模型的 tokenizer，调用 apply_chat_template 就会自动转换成 LLaMA 需要的格式。这是 messages 格式最大的优势——数据格式和模型格式解耦。

---

## 参考资料

- [Qwen2.5 官方文档](https://qwen.readthedocs.io/) - ChatML 格式说明
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating) - apply_chat_template 用法
- [LLaMA-Factory 数据格式文档](https://github.com/hiyouga/LLaMA-Factory) - 训练数据格式要求
