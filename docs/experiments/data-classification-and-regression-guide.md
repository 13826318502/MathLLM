# MathLLM 数据分类与错误回归集说明

## 1. 文档目的

本文说明 MathLLM 当前数据目录中各类数据的职责、生命周期和使用边界，特别是
模型发现真实数学错误后，如何分别构造纠错训练样本和错误回归样本。

核心原则是：

```text
训练集负责学习
验证集负责选择 checkpoint 和调参
独立测试集负责最终比较
回归集负责检查已经发现的错误是否再次出现
```

不同用途的数据不能因为题型相似就混放，否则会导致训练数据泄漏，或者无法判断
模型是真正学会了方法，还是只记住了原题答案。

## 2. 数据类别总表

| 数据类别 | 目录/文件 | 主要用途 | 是否参与训练 | 是否用于选择 checkpoint |
|---|---|---|---:|---:|
| 原始数据 | `data/raw/` | 保存下载、收集和人工编写的原始样本 | 经过预处理后参与 | 否 |
| 纠错训练样本 | `data/raw/corrections/` | 学习已经发现的错误模式 | 是 | 间接通过 `eval.json` |
| 处理后训练集 | `data/processed/train.json` | 供 LoRA/QLoRA 学习 | 是 | 否 |
| 处理后验证集 | `data/processed/eval.json` | 监控泛化、选择最佳 checkpoint | 否 | 是 |
| 独立测试集 | `data/eval/test.json` | 对确定的模型做最终统一比较 | 否 | 否 |
| 错误回归集 | `data/eval/regression/regression.json` | 检查已知错误是否复发 | 否 | 否 |
| 评测输出 | `eval/results/` | 保存答案、指标、错题和评测报告 | 否 | 否 |

## 3. 各类数据的具体用途

### 3.1 `data/raw/`：原始数据

这里保存尚未统一格式的 JSON、JSONL、CSV 等来源数据，以及人工编写的数学样本。
原始字段可以是 `question`、`solution`、`answer`、`source`、`subject` 等，
不要求一开始就是 ChatML `messages` 格式。

统一处理由 `scripts/prepare_data.py` 完成，程序负责清洗、格式转换、去重、排除
测试集和回归集，并生成 `data/processed/train.json` 与
`data/processed/eval.json`。

### 3.2 `data/raw/corrections/`：纠错训练样本

这里放与已知错误同类型、但数字、变量、条件或题目表述不同的样本。每条样本都必须：

- 有完整、正确、人工验算过的解题过程；
- 有明确的最终答案；
- 使用原始独立字段格式，不直接手写 `messages`；
- 不直接复制原来的错误题；
- 最好使用 SymPy、手算或其他独立方法交叉验证。

这类样本的目的不是让模型背诵某一道题，而是让模型学习可迁移的方法，例如：

```text
已知错误：三根的平均值与乘积关系中漏掉 1/3
纠错训练样本：使用另一组三次方程系数，正确处理韦达定理和平均值
```

运行 `prepare_data.py` 时，`data/raw/` 会被递归扫描，因此 corrections 中的样本会
进入训练/验证集划分。加入前必须确认它们没有与 `test.json` 或回归集中的题目高度重复。

### 3.3 `data/processed/train.json`：处理后训练集

这是供训练框架读取的 ChatML 数据，通常包含：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

该文件由程序生成，不建议手工修改。纠错训练样本经过清洗后会和其他训练样本一起
进入这里。

### 3.4 `data/processed/eval.json`：处理后验证集

验证集不用于梯度更新，而是用于观察 `eval_loss`、选择最佳 checkpoint 和比较训练
参数。它可以参与训练过程中的定期评估，但不能作为最终测试成绩。

### 3.5 `data/eval/test.json`：独立测试集

这是固定的独立测试集，原则上不参与训练、checkpoint 选择和反复调参。确定模型后，
使用同一份测试集、相同的生成参数和相同的评测程序比较基座模型、LoRA 合并模型和
量化模型。

如果已经查看了测试集的错题，并根据这些错题设计了训练样本，那么这份测试集已经
参与了开发迭代，不应再被称为完全盲测。此时应继续保留它作为开发回归依据，并另外
冻结一份新的、未查看的测试集用于最终论文或正式报告。

### 3.6 `data/eval/regression/regression.json`：错误回归集

回归集只用于训练后的专项检查，不参与 `prepare_data.py` 的训练/验证划分，也不用于
选择最佳 checkpoint。

回归集建议包含两类样本：

1. 原来答错的原题：确认模型是否还会犯同一个错误；
2. 同类型但条件不同的新题：确认模型学会的是方法，而不是记住原题答案。

因此，用户发现模型在独立测试集中的真实错误后，推荐按以下方式分流：

| 样本 | 去向 | 原因 |
|---|---|---|
| 原来的错误题 | `data/eval/regression/regression.json` | 只用于复发检查，避免原题进入训练 |
| 同类型、不同条件且答案正确的训练样本 | `data/raw/corrections/` | 让模型学习可迁移的方法 |
| 同类型、不同条件的新验证题 | `data/eval/regression/regression.json` | 验证是否真正泛化 |

不要把原来的错误题复制到 `data/raw/corrections/`，也不要把整个回归目录复制到
`data/raw/`。当前仓库已经通过目录分离实现这一边界：`prepare_data.py` 扫描
`data/raw/`，不会读取 `data/eval/`。

## 4. 推荐的纠错迭代流程

```text
固定 test.json
      ↓
训练模型并评测
      ↓
LLM Judge + 人工复核，确认真实数学错误
      ↓
原错误题 ─────────────→ regression.json
同类不同条件的正确题 ─→ raw/corrections/
      ↓
prepare_data.py 重新清洗、去重、划分
      ↓
重新训练并选择 checkpoint
      ↓
先跑 regression.json，确认旧错误是否消失
      ↓
再用固定 test.json 或新的盲测集做统一比较
```

每次迭代建议单独保存结果目录，例如：

```text
eval/results/
├── baseline/
├── correction-round-1/
└── regression-round-1/
```

不要覆盖旧报告，否则无法比较改进前后的准确率、错误类型和延迟。

## 5. 推荐的文件内容

训练纠错样本可以使用 JSONL 独立字段，例如：

```json
{"question":"新的同类型题目","solution":"完整且验算过的解题过程","answer":"明确最终答案","source":"correction-round-1","subject":"线性代数","difficulty":"中等"}
```

回归集使用评测程序能够读取的 JSON 数组，并至少保留 `messages`、标准答案和来源
信息。推荐在不影响评测读取的前提下增加元数据，例如 `regression_id`、
`origin_eval_index`、`error_type` 和 `target_skill`，方便追踪错误来源。

## 6. 当前仓库快照

截至本说明编写时：

- `data/eval/test.json`：116 条独立测试题；
- `data/eval/regression/regression.json`：4 条回归样本；
- `data/raw/corrections/`：当前未发现 JSON/JSONL 纠错样本文件，仅有目录说明；
- `data/processed/`：处理后数据属于运行产物，通常不提交到 GitHub；
- `eval/results/`：保存模型推理和测评输出，不属于训练数据目录。

在下一轮纠错训练前，应先人工验算新增 corrections 样本，再执行数据预处理并检查
训练集、验证集和回归集之间没有重复或高相似题目。

## 7. 检查清单

- [ ] 原错误题没有复制到 `data/raw/corrections/`；
- [ ] corrections 样本与原错误题条件不同；
- [ ] corrections 的解题过程和答案已人工验算；
- [ ] regression 同时包含原错误题和同类新题；
- [ ] `data/eval/test.json` 未被训练或调参修改；
- [ ] 训练、验证、测试和回归结果使用不同输出目录；
- [ ] 重新训练后先运行回归评测，再进行统一测试集比较；
- [ ] 正式报告区分基础匹配、LLM Judge 和人工抽检结果。
