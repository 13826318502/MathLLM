# 纠错训练集

本目录中的样本用于训练模型改进已经发现的错误模式。

- 这里放的是与错误类型相同、但题目条件和数字不同的变体；
- 原始错误题保留在 `data/eval/regression/`，不放入训练集；
- 每条记录使用独立字段格式，不直接使用 `messages`；
- 加入训练前必须人工验算，最好再用 SymPy 或其他独立方法复核。

运行 `python scripts/prepare_data.py` 时，程序会递归扫描 `data/raw/`，因此本目录中的 JSONL 样本会自动参与训练/验证集划分。

## 官方来源纠错轮次

历史文件 `correction-training-all.jsonl` 已移动到
`data/archive/corrections/correction-training-all.jsonl`，不再被当前数据准备程序读取。

Round 4 候选题完整归档在
`data/candidates/archive/correction-round-4-all.jsonl`，经过固定种子切分后，只有
`correction-round-4-training.jsonl` 放在本目录并会被 `prepare_data.py` 自动读取；
留出的 20 条在 `data/eval/correction-validation/round-4.json`，只用于纠错能力验证。

Round 4 的 100 条题目筛选时排除了已有 `data/raw/`、`data/eval/test.json` 和
`data/eval/regression/` 中的题目，并转换为本目录要求的独立字段格式：

- GSM8K：35 条应用题，来源 `openai/gsm8k`；
- Hendrycks MATH：55 条，覆盖概率组合、代数、数论/模运算、几何和预备微积分，来源 `EleutherAI/hendrycks_math`；
- CMID：10 条中文高数、线代和概率题，来源 `Mxode/CMID-Chinese_Math_Instruct_Dataset`。

本轮已按固定种子完成收集和切分：

```bash
python scripts/collect_official_corrections.py --target 100 \\
  --round-label correction-round-4 \\
  --output data/candidates/correction-round-4-all.jsonl
python scripts/split_correction_round.py \\
  --input data/candidates/correction-round-4-all.jsonl \\
  --train-output data/raw/corrections/correction-round-4-training.jsonl \\
  --validation-output data/eval/correction-validation/round-4.json
```

Round 4 的 80 条训练题和 20 条验证题目前仍是官方筛选候选，不等于项目已完成全量人工验算。正式训练前仍需
按项目要求抽查并用独立方法复核；原错误题只保留在回归集，不复制到训练集。

## Round 5 与 Round 6

- `correction-round-5-training.jsonl`：Round 5 已完成训练使用的 120 条纠错训练原始数据，历史保留；
- `correction-round-6-training.jsonl`：Round 6 新增的 240 条纠错训练原始数据，source 固定为
  `correction-round-6`，进入 Round 6 前会再做纠错组内去重；
- Round 6 配置明确排除 Round 4、Round 5 两个旧纠错源，避免旧纠错题重复进入本轮。

## Round 7

- `correction-round-7-training.jsonl`：本轮新增 150 条纠错训练题，平均覆盖应用题、代数与数论、概率与组合、线性代数和高等数学五类，每类 30 条；
- source 统一为 `correction-round-7`；
- 这些题允许同类型、同能力点的相似变体保留，只禁止完全相同的题目；
- Round 7 staging 构建时明确排除 Round 4、Round 5、Round 6 纠错源，并从原始数据抽取 65 条回放样本。

Round 4 和 Round 5 的活动训练源现已归档到 `data/archive/round-4-5-20260905/`；
Round 7 的训练源现已归档到 `data/archive/corrections/round-7-20260905/`；

## Round 8

- `correction-round-8-training.jsonl`：Round 8 新增 240 条纠错训练题；
- source 统一为 `correction-round-8`；
- 与 Round 7 只做完全相同题目的排除，同能力点的不同数字、不同叙述和不同难度变体允许保留；
- Round 8 staging 另外抽取 130 条原始数据回放，避免重复训练完整基础训练集；
- Round 8 的 50 条纠错验证题放在 `data/eval/correction-validation/round-8.json`，不进入训练。

## Round 9

- `correction-round-9-training.jsonl`：Round 9 新增 180 条纠错训练题；
- source 统一为 `correction-round-9`，题目标记使用全新的 R9 范围；
- 使用新的参数、场景和题型组合，避免直接复制 Round 8；
- Round 9 staging 另外抽取 120 条分层原始数据回放；
- Round 9 的 40 条纠错验证题放在 `data/eval/correction-validation/round-9.json`，不进入训练。

Round 9 的活动纠错源已归档。当前活动纠错目录以 Round 10 为本轮训练入口，历史纠错训练源只在
`data/archive/` 中保存，Round 10 整理程序会排除 Round 4 至 Round 9 的纠错源。

## Round 10

- `correction-round-10-training.jsonl`：Round 10 新增 360 条纠错训练题；
- source 统一为 `correction-round-10`，题目标记使用新的 R10 范围；
- `data/raw/base-extra/`：从 GSM8K、Hendrycks MATH 和 CMID 各补充一批基础训练数据，原始数量共 1,250 条；
- 清洗后与原有基础数据合并，避免测试集、回归集和历史纠错题泄漏。
