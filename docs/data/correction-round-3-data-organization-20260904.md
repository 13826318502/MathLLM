# 第三轮纠错数据整理记录（2026-09-04）

## 目的

本轮数据用于在 `outputs/correction-round-2-merged` 的基础上学习已发现错误的同类型解题能力。原错误题不进入训练，只保留在回归集，用于检查模型是否再次犯错。

## 已完成的整理

| 数据 | 数量 | 位置 | 用途 |
| --- | ---: | --- | --- |
| 第三轮完整候选 | 100 | `data/candidates/correction-round-3-all.jsonl` | 审计、人工验算，不会被训练脚本自动读取 |
| 第三轮训练候选 | 80 | `data/raw/corrections/correction-round-3-training.jsonl` | 人工确认后纳入训练 |
| 第三轮纠错验证集 | 20 | `data/eval/correction-validation/round-3.json` | 验证同类型能力是否提升，不参与训练 |
| 原始独立最终测试集 | 116 | `data/eval/test.json` | 保持不变，用于最终泛化对比 |
| 原错误回归集 | 42 | `data/eval/regression/regression.json` | 保持不变，检查旧错误是否复发 |

第三轮候选来源为 GSM8K 35 条、MATH 55 条、CMID 10 条。固定随机种子为 `20260904`，验证集比例为 20%，因此训练候选为 80 条、纠错验证集为 20 条。切分后的来源数量为：

- 训练候选：GSM8K 27、MATH 45、CMID 8；
- 纠错验证集：GSM8K 8、MATH 10、CMID 2。

## 预处理试运行

为避免覆盖当前可复现的数据，使用 `prepare_data.py` 生成了暂存版本：

- `data/processed/round-3-staging/train.json`：1118 条；
- `data/processed/round-3-staging/eval.json`：124 条。

该暂存版本包含原始训练数据、上一轮纠错训练数据和本轮 80 条训练候选；本轮 20 条纠错验证题不在其中。当前正式的 `data/processed/train.json`（1046 条）和 `data/processed/eval.json`（116 条）未被覆盖。

预处理检查结果：

- JSON 可正常读取；
- 训练和普通验证记录均为 `system → user → assistant`；
- 所有消息内容非空；
- 第三轮 100 条候选没有重复题目；
- 与现有原始数据、固定最终测试集和原错误回归集没有题目级重复；
- 候选解答中未发现 GSM8K 的 `<<...>>` 计算标记；
- 全部 100 条候选的独立字段齐全。

## 正式训练前必须确认

第三轮数据虽然来自官方数据集并已完成程序级检查，但仍不是“人工验算完成”的数据。正式训练前需要抽查并核对：

1. 题目条件、单位、数字和答案是否一致；
2. 解题过程是否能够推出最终答案；
3. 概率、组合、矩阵、代数和多步应用题是否漏情况或漏系数；
4. 数学公式是否使用正确的 LaTeX；
5. 验证题是否确实没有与训练题构成近似重复。

人工确认后，可将暂存训练文件用于本轮训练，例如把 `data/processed/round-3-staging/train.json` 复制为训练配置使用的输入，并把 `data/processed/round-3-staging/eval.json` 作为训练过程验证集。`data/eval/correction-validation/round-3.json`、`data/eval/test.json` 和 `data/eval/regression/regression.json` 应继续单独评测。

## 使用的整理程序

```powershell
& ".venv\Scripts\python.exe" scripts\split_correction_round.py
& ".venv\Scripts\python.exe" scripts\prepare_data.py `
  --raw-dir data\raw `
  --regression-dir data\eval\regression `
  --test-file data\eval\test.json `
  --output-dir data\processed\round-3-staging `
  --seed 42
```
