# Round 6 数据与配置准备记录（2026-09-05）

## 结论

Round 5 作为已完成实验保留，不覆盖、不删除。本轮 Round 6 从
`outputs/correction-round-5-merged` 继续训练，目标是针对上一轮暴露的错误
类型增加纠错能力，同时用原始数据回放降低灾难性遗忘风险。

## 数据安排

| 数据 | 路径 | 数量 | 用途 |
|---|---|---:|---|
| Round 6 纠错训练原始数据 | `data/raw/corrections/correction-round-6-training.jsonl` | 240 | 经过程序生成、公式计算和去重后并入训练 |
| Round 6 纠错验证集 | `data/eval/correction-validation/round-6.json` | 30 | 独立检查新纠错能力，不参与训练 |
| Round 6 训练快照 | `data/processed/round-6-staging/train.json` | 1155 | 实际训练输入 |
| Round 6 内部验证快照 | `data/processed/round-6-staging/eval.json` | 128 | 训练期间选 checkpoint |
| 原始测试集 | `data/eval/test.json` | 116 | 合并后独立评测 |
| 历史回归集 | `data/eval/regression/regression.json` | 42 | 合并后检查旧错误是否复发 |

Round 6 暂存集的 1155 条训练数据由 917 条原始数据回放和 238 条新纠错题
组成；另有 128 条原始数据进入内部验证集。新纠错题全部强制留在训练集，
避免纠错题被内部验证划走；原始纠错数据中的 2 条组内近似重复题被过滤。

## 质量与泄漏检查

- Round 6 训练纠错题：240 条，5 个学科各 48 条；
- Round 6 纠错验证题：30 条，5 个学科各 6 条；
- 训练题无完全重复，验证题无完全重复；
- 与原始测试集和历史回归集无直接题目泄漏；
- 训练快照与内部验证快照不重叠；
- Round 4、Round 5 旧纠错源未纳入本轮；
- `test.json` 和 `regression.json` 仅在合并模型后进行独立评测。

生成和重建命令：

```powershell
python scripts/build_correction_round6.py
python scripts/prepare_data.py `
  --raw-dir data/raw `
  --regression-dir data/eval/regression `
  --test-file data/eval/test.json `
  --output-dir data/processed/round-6-staging `
  --eval-ratio 0.1 `
  --seed 42 `
  --exclude-source correction-round-4 `
  --exclude-source correction-round-5
```

训练入口：

```powershell
python scripts/train.py --config configs/training/correction-round-6-20260905/train_config.yaml
```

训练后不能只看 `eval_loss`。应选择 checkpoint 后合并为
`outputs/correction-round-6-merged`，再分别跑原始测试集、Round 6 纠错验证集
和回归集，并与 Round 5 的三项结果对照。
