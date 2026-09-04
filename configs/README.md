# 配置文件说明

## 当前正式训练

`training/correction-round-2-20260904/train_config.yaml` 是本轮正式训练的
唯一推荐配置。本轮使用原始 Qwen 基座、最新的完整训练集和普通验证集，
输出到：

```text
outputs/correction-round-2/
```

正式训练应使用以下命令：

```bash
python scripts/train.py --config configs/training/correction-round-2-20260904/train_config.yaml
```

项目不再保留根目录兼容配置。以后新实验必须在
`configs/training/<实验名称>/` 中保存独立配置，并使用独立的 `outputs/`
目录。

当前数据文件为：

```text
data/processed/train.json   # 1046 条
data/processed/eval.json    # 116 条
```

## 其他配置

| 文件 | 用途 | 是否用于当前正式训练 |
|---|---|---:|
| `ablation.yaml` | 消融实验计划 | 否 |
| `deployment/correction-round-2-20260904-merged.yaml` | 本轮合并模型部署 | 训练合并后使用 |
| `judge.env.example` | LLM Judge 接口环境变量示例 | 否 |
| `archive/full-training-20260903/` | 第一次全量训练配置快照 | 否 |
| `archive/smoke/` | 历史冒烟测试配置，仅供追溯 | 否 |
| `archive/deployment/` | 历史量化部署配置，仅供追溯 | 否 |

每次新的正式实验应修改输出目录并保留独立配置或记录，避免覆盖以前的
adapter、checkpoint 和训练日志。`data/eval/test.json` 与
`data/eval/regression/regression.json` 不参与训练，只在训练完成后评测。
