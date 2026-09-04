# 纠错训练集

本目录中的样本用于训练模型改进已经发现的错误模式。

- 这里放的是与错误类型相同、但题目条件和数字不同的变体；
- 原始错误题保留在 `data/eval/regression/`，不放入训练集；
- 每条记录使用独立字段格式，不直接使用 `messages`；
- 加入训练前必须人工验算，最好再用 SymPy 或其他独立方法复核。

运行 `python scripts/prepare_data.py` 时，程序会递归扫描 `data/raw/`，因此本目录中的 JSONL 样本会自动参与训练/验证集划分。

## 官方来源纠错轮次

`correction-official-round-2.jsonl` 是从项目原先使用的官方数据集训练分片中
重新筛选的 100 条题目。筛选时排除了已有 `data/raw/`、`data/eval/test.json`
和 `data/eval/regression/` 中的题目，并转换为本目录要求的独立字段格式：

- GSM8K：35 条应用题，来源 `openai/gsm8k`；
- Hendrycks MATH：55 条，覆盖概率组合、代数、数论/模运算、几何和预备微积分，来源 `EleutherAI/hendrycks_math`；
- CMID：10 条中文高数、线代和概率题，来源 `Mxode/CMID-Chinese_Math_Instruct_Dataset`。

本轮可以用以下命令按固定种子重新收集：

```bash
python scripts/collect_official_corrections.py --target 100
```

这些题目是官方数据源筛选结果，不等于项目已完成人工验算。正式训练前仍需
按项目要求抽查并用独立方法复核；原错误题只保留在回归集，不复制到训练集。
