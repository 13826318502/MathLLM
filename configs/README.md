# 配置文件说明

## 当前正式训练

`training/correction-round-3-20260904/train_config.yaml` 是当前轮次正式训练的
唯一推荐配置。本轮以 `outputs/correction-round-2-merged` 作为训练基座，
使用 Round 3 暂存训练集和普通验证集，
输出到：

```text
outputs/correction-round-3/
```

正式训练应使用以下命令：

```bash
python scripts/train.py --config configs/training/correction-round-3-20260904/train_config.yaml
```

训练过程中不需要启动 vLLM。训练完成后先从最优 checkpoint 合并出
`outputs/correction-round-3-merged/`，再使用
`deployment/correction-round-3-20260904-merged.yaml` 启动 vLLM；该部署配置已经
指向本轮新合并模型，而不是 Round 2 模型。

项目不再保留根目录兼容配置。以后新实验必须在
`configs/training/<实验名称>/` 中保存独立配置，并使用独立的 `outputs/`
目录。

当前数据文件为：

```text
data/processed/round-3-staging/train.json   # 1118 条
data/processed/round-3-staging/eval.json    # 124 条
```

## 其他配置

| 文件 | 用途 | 是否用于当前正式训练 |
|---|---|---:|
| `ablation.yaml` | 基于当前 Round 3 配置的消融实验计划 | 否 |
| `deployment/correction-round-2-20260904-merged.yaml` | Round 2 合并模型部署快照 | 否 |
| `deployment/correction-round-3-20260904-merged.yaml` | 当前轮次合并模型部署 | Round 3 合并后使用 |
| `judge.env.example` | LLM Judge 接口环境变量示例 | 否 |
| `archive/full-training-20260903/` | 第一次全量训练配置快照 | 否 |
| `archive/smoke/` | 历史冒烟测试配置，仅供追溯 | 否 |
| `archive/deployment/` | 历史量化部署配置，仅供追溯 | 否 |

每次新的正式实验应修改输出目录并保留独立配置或记录，避免覆盖以前的
adapter、checkpoint 和训练日志。`data/eval/test.json` 与
`data/eval/regression/regression.json` 不参与训练，只在训练完成后评测。
