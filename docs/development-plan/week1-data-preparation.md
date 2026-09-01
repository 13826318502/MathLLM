# Week 1 开发计划：数学训练数据准备

## 1. 本周目标

本周不进行正式模型训练，重点是建立一套可靠的数据准备流程，最终产出可以直接交给 LoRA/QLoRA 训练使用的数据集。

本周结束时应完成：

- 收集并整理第一版数学问答数据；
- 实现 `scripts/prepare_data.py` 的数据加载、格式转换、清洗和划分功能；
- 生成 `data/processed/train.json` 和 `data/processed/eval.json`；
- 对处理后的数据进行自动检查和人工抽查；
- 确认训练集与验证集之间没有重复或高度相似的题目。

完整流程如下：

```text
原始数据
  ↓
统一字段
  ↓
转换为 messages/ChatML 结构
  ↓
清洗、去重、长度检查
  ↓
人工抽查数学正确性
  ↓
按 90%/10% 划分
  ↓
train.json + eval.json
```

## 2. 本周最终交付物

目录结构应类似于：

```text
data/
├── raw/
│   ├── gsm8k_train.jsonl
│   ├── math_train.jsonl
│   └── college_math.jsonl
└── processed/
    ├── train.json
    └── eval.json
```

代码交付物：

- `load_raw_data()`：读取 JSON、JSONL、CSV 并统一字段；
- `format_example()`：将统一数据转换为 `messages` 格式；
- `validate_data()`：过滤无效、过短、重复和格式错误的数据；
- `split_dataset()`：固定随机种子并按 90%/10% 划分数据；
- `main()`：能够完整运行并输出数据统计信息。

## 3. 推荐数据规模和构成

先用 50–100 条数据跑通流程，再扩充到正式规模。不要一开始就下载和处理几十万条数据。

文档建议的第一版数据规模约为 1300 条：

| 来源 | 建议数量 | 主要作用 |
|---|---:|---|
| GSM8K | 500 | 基础计算和应用题 |
| MATH | 500 | 高中及竞赛数学 |
| GPT 生成并人工核验 | 200 | 补充指定题型 |
| 考研/大学数学 | 100 | 高数、线代、概率论领域适配 |
| 合计 | 约 1300 | 第一版训练集 |

由于本项目目标是中文大学数学助手，建议不要只依赖 GSM8K 和 MATH。最终数据中应有足够比例的中文高等数学、线性代数和概率论题目。

每条数据最好保留以下元信息，方便后续分析：

```text
source      数据来源
subject     学科类别
difficulty  难度
```

元信息可以保留在中间数据中；最终交给训练框架的数据至少需要 `messages` 字段。

## 4. 第一天：确定数据规范和 system prompt

### 任务

- 确定模型覆盖的数学范围；
- 固定 system prompt；
- 确定原始数据的统一字段；
- 手工编写 10 条标准样本作为测试数据；
- 明确哪些数据应该被删除。

### 原始数据统一格式

建议所有原始数据最终都转换为以下中间结构：

```json
{
  "question": "求函数 f(x)=x^3-3x+1 的极值点",
  "solution": "先求导：f'(x)=3x^2-3。令 f'(x)=0，得到 x=±1。再通过二阶导数判断极值类型。",
  "answer": "x=1 为极小值点，函数值为 -1；x=-1 为极大值点，函数值为 3。",
  "source": "college_math",
  "subject": "高等数学",
  "difficulty": "基础"
}
```

### system prompt

项目当前使用的 system prompt 是：

```text
你是一个专业的数学解题助手。请按照以下要求回答数学问题：
1. 先分析题目要求
2. 给出详细的解题步骤
3. 用 LaTeX 格式书写数学公式
4. 最后给出明确的答案
```

第一版建议保持固定，不要让不同数据使用完全不同的 system prompt，否则模型学习到的输出风格会不稳定。

## 5. 第二天：收集和保存原始数据

### 推荐数据来源

- GSM8K：基础数学应用题；
- MATH：高中和竞赛数学；
- 考研数学真题或大学教材习题：补充中文大学数学；
- GPT 生成数据：只用于补充缺失题型，必须人工验算。

原始文件统一放入：

```text
D:\MathLLM\data\raw\
```

原始数据不要直接覆盖或改写，保留原始文件，后续转换逻辑统一放在 `prepare_data.py` 中。

### GSM8K 的转换规则

GSM8K 通常类似于：

```json
{
  "question": "Janet's dogs eat 3 bags of food per day...",
  "answer": "3 * 14 = 42\n#### 42"
}
```

应转换为：

```text
question = 原来的 question
solution = answer 中 #### 之前的内容
answer   = answer 中 #### 之后的内容
```

如果一条 GSM8K 数据没有 `####`，应记录并人工检查，不要盲目把整段内容当成最终答案。

### MATH 的转换规则

MATH 通常类似于：

```json
{
  "problem": "Find the value of x...",
  "solution": "Squaring both sides...",
  "level": "Level 1",
  "type": "Algebra"
}
```

应转换为：

```text
problem → question
solution → solution
type → subject
level → difficulty
```

如果 MATH 原始记录中没有独立的最终答案字段，应从 solution 中提取，或者保留完整 solution 并在人工检查时确认结论明确。

## 6. 第三天：实现数据格式转换

`scripts/prepare_data.py` 目前只是代码骨架，四个核心函数都有 `TODO`，因此不能直接认为运行命令就能完成数据准备。

### `load_raw_data()`

需要完成：

- 遍历 `data/raw/` 下的 `.json`、`.jsonl`、`.csv` 文件；
- 根据文件名或字段判断数据来源；
- 将 GSM8K、MATH 和自建数据统一为 `question/solution/answer`；
- 对缺失字段的样本记录警告并跳过；
- 返回统一的字典列表。

建议返回的数据至少包含：

```python
{
    "question": "...",
    "solution": "...",
    "answer": "...",
    "source": "...",
    "subject": "..."
}
```

### `format_example()`

将中间结构转换成：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个专业的数学解题助手。..."
    },
    {
      "role": "user",
      "content": "题目内容"
    },
    {
      "role": "assistant",
      "content": "完整解题过程\n\n**最终答案**\n\n最终答案内容"
    }
  ]
}
```

格式要求：

- `system` 必须在第一条；
- `user` 必须在第二条；
- `assistant` 必须在最后一条；
- `content` 不能是空字符串；
- 公式使用 `$...$` 或 `$$...$$`；
- 不要手工添加 `[im_start]` 和 `[im_end]`；
- 由 Qwen tokenizer 或训练框架自动应用 chat template。

## 7. 第四天：清洗、去重和长度检查

### 必须过滤的情况

| 检查项 | 建议规则 |
|---|---|
| 题目为空 | 删除 |
| 题目过短 | 少于 5 个字符删除 |
| 解题过程为空 | 删除 |
| assistant 回复过短 | 少于 20 个字符删除 |
| role 错误 | 删除 |
| 消息顺序错误 | 删除 |
| 最后一条不是 assistant | 删除 |
| 总长度超过 2048 tokens | 截断或删除 |
| 数学答案明显错误 | 删除或人工修正 |
| 题目完全重复 | 删除重复项 |

### 去重策略

先做精确去重：

- 去除首尾空格；
- 合并连续空白；
- 统一全角和半角符号；
- 对标准化后的题目计算哈希值。

再做近似去重：

- 文档建议相似度超过 0.95 的样本去重；
- 特别检查只替换了数字、变量名或少量文字的题目；
- 高度相似的题目应全部放入同一数据集，不能一题放训练集、一题放验证集。

例如：

```text
求 x^2 - 5x + 6 = 0
求 x^2 - 5x + 4 = 0
```

这两道题结构高度相似，不应被简单随机分到 train 和 eval 两边。

### 数学正确性

数据清洗不能只检查 JSON 格式，还要检查数学内容：

- 题目条件是否完整；
- 解题过程中的公式是否正确；
- 中间计算是否有错误；
- 最终答案是否和过程一致；
- 是否满足定义域、边界条件和特殊情况；
- LaTeX 是否明显损坏。

至少人工抽查 10%。抽查时不要只看第一批数据，应该按来源和学科分层抽查。

## 8. 第五天：划分训练集和验证集

文档要求按 90%/10% 划分，并随机打乱。建议固定随机种子，例如 `42`，这样以后可以重复得到相同划分。

如果最终清洗后有 1300 条数据：

```text
train.json：约 1170 条
eval.json：约 130 条
```

划分时应注意：

- 先清洗和去重，再划分；
- 不要把重复题分到两边；
- 保证高数、线代、概率论等类别在 eval 中都有样本；
- 记录每个来源、学科和难度在 train/eval 中的数量；
- 如果使用公共数据集，可以优先使用官方 test split 作为验证集。

## 9. 第六天：自动验证输出文件

运行：

```powershell
python scripts/prepare_data.py
```

脚本至少应输出：

```text
Loaded xxx raw examples
After cleaning: xxx examples
Train: xxx, Eval: xxx
Done! Data saved to data/processed/
```

然后检查文件是否能正常读取：

```powershell
python -c "import json; x=json.load(open('data/processed/train.json',encoding='utf-8')); print('train:',len(x)); print(x[0])"
python -c "import json; x=json.load(open('data/processed/eval.json',encoding='utf-8')); print('eval:',len(x)); print(x[0])"
```

建议再写一个小型检查脚本或检查函数，验证：

```text
每条记录都有 messages
messages 不为空
role 只能是 system/user/assistant
第一条是 system
最后一条是 assistant
所有 content 非空
题目没有重复
```

如果已经下载 Qwen tokenizer，还应使用 `tokenizer.apply_chat_template()` 检查若干样本的 token 长度是否不超过 2048。

## 10. 第七天：人工验收和整理记录

### 抽查建议

如果有 1300 条数据，至少抽查约 130 条：

- GSM8K：40 条；
- MATH：40 条；
- 中文大学数学：30 条；
- GPT 生成数据：20 条。

每条抽查记录建议记录：

```text
样本编号
数据来源
学科
题目是否清楚
过程是否正确
答案是否正确
LaTeX 是否正常
是否保留/删除
删除原因
```

### 本周验收标准

- `train.json` 和 `eval.json` 均能被 JSON 正常读取；
- 每条数据都有合法的 `messages`；
- 每条数据最后一条都是 `assistant`；
- 没有空题目和空回答；
- assistant 回复包含过程和最终答案；
- 公式格式基本统一；
- 训练集和验证集没有重复题；
- 人工抽查 10% 后没有明显数学错误；
- 数据量达到计划规模，或明确记录未达到的原因；
- 记录各来源、学科和难度的样本数量。

## 11. 常见问题和处理方式

### 运行脚本后没有生成数据

当前 `scripts/prepare_data.py` 中的函数还是 `TODO`，需要先实现，不能只运行命令。

### GSM8K 的答案格式不统一

优先按照 `####` 拆分。如果缺少 `####`，加入异常列表并人工检查。

### MATH 没有独立 answer 字段

从 solution 中确认最终结论；如果不能可靠提取，就保留完整 solution，并在后续评测脚本中单独处理最终答案。

### 数据太长

优先删除重复和无关叙述，再考虑截断。不要直接从中间截断数学推导，否则可能留下不完整答案。

### GPT 生成数据看起来很完整但结果错误

必须人工验算。数学数据不能仅凭语言是否通顺来判断质量。

## 12. 与后续训练阶段的衔接

第一周结束后，第二周应先使用约 100 条数据、1 个 epoch 做小规模训练，确认：

- 数据格式能被 LLaMA-Factory/TRL 读取；
- chat template 没有报错；
- assistant 部分能够正确计算 loss；
- 模型输出没有乱码或重复；
- loss 能够正常下降。

确认流程正常后，再使用完整训练集进行 LoRA 或 QLoRA 训练。

另外，当前 `eval/evaluate.py` 仍有 `TODO`，后续需要明确如何从 assistant 回复中提取最终答案，不能简单把整段解题过程当成答案进行比较。

## 13. 相关文件

- [开发指南](../development-guide.md)
- [ChatML 格式说明](../learning/chatml-format.md)
- [数据预处理脚本](../../scripts/prepare_data.py)
- [训练配置](../../configs/train_config.yaml)
