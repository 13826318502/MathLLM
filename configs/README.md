# 配置文件说明

## 当前正式训练入口

当前推荐使用：

```text
configs/training/correction-round-7-20260905/train_config.yaml
```

Round 7 训练方案：

- 从 `outputs/correction-round-6-merged` 继续训练；
- 使用 150 条新增 `correction-round-7` 与 65 条原始数据回放；
- 训练集约 70% 为纠错题、30% 为原始回放；
- 使用 QLoRA，`learning_rate=5e-6`；
- 训练 1 个 epoch，避免继续叠加上一轮过拟合；
- 只运行一个模型，不做对比实验；
- 输出到 `outputs/correction-round-7/`。

正式训练命令：

```bash
python scripts/train.py --config configs/training/correction-round-7-20260905/train_config.yaml
```

Round 6 的暂存数据路径由配置固定为：

```text
data/processed/round-7-staging/train.json
data/processed/round-7-staging/eval.json
```

旧的 `correction-round-4`、`correction-round-5` 和 `correction-round-6` 纠错源在生成
Round 7 暂存数据时被排除，避免重复过采样已经学习过的纠错题。新数据生成后必须
重新运行 Round 7 专用整理程序。

训练完成后，根据最低 `eval_loss` 选择 checkpoint，再将 Round 7 LoRA 合并到
`outputs/correction-round-6-merged`，生成新的 `outputs/correction-round-7-merged`。
训练期间不需要启动 vLLM。

## 历史训练配置

| 路径 | 用途 |
|---|---|
| `training/correction-round-5-20260905/train_config.yaml` | Round 5 配置快照；仅用于复现和对照 |
| `training/correction-round-6-20260905/train_config.yaml` | Round 6 配置快照；仅用于复现和对照 |
| `training/correction-round-7-20260905/train_config.yaml` | 当前唯一正式训练入口 |
| `archive/correction-round-3-20260904/train_config.yaml` | Round 3 配置快照；曾从 Round 2 合并模型继续训练，已归档 |
| `training/correction-round-3-20260904/train_config.yaml` | Round 3 原始配置入口，保留用于复现 |
| `training/correction-round-2-20260904/train_config.yaml` | Round 2 原始配置入口，保留用于复现 |
| `archive/full-training-20260903/train_config.yaml` | 第一次全量训练快照 |
| `archive/smoke/` | 冒烟测试配置 |

历史配置不用于 Round 6，也不应复用历史 `outputs/` 目录。

## 部署配置

部署配置位于：

```text
configs/deployment/
```

只有在 Round 7 合并模型生成后，才应新增或复制对应的 Round 7 部署配置，确保
vLLM 指向新的合并模型目录，而不是历史模型。

## 评测配置

- `ablation.yaml`：消融实验计划，不参与当前正式训练；
- `judge.env.example`：LLM Judge 环境变量示例。

以下数据永远不参与训练或 checkpoint 选择：

```text
data/eval/test.json
data/eval/regression/regression.json
```

它们只在模型合并后用于最终评测和回归评测。
