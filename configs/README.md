# 配置文件说明

## 当前正式训练入口

当前推荐使用：

```text
configs/training/correction-round-10-20260908/train_config.yaml
```

Round 10 训练方案：

- 直接从原始 `models/Qwen2.5-7B-Instruct-modelscope` 开始；
- 原始基础数据扩充一倍，并新增 360 条 `correction-round-10` 纠错题；
- 清洗后训练集 2,473 条，基础数据约 85.44%，纠错数据约 14.56%；
- 使用 QLoRA，`learning_rate=1e-4`；
- 训练 1 个 epoch，避免继续叠加上一轮过拟合；
- 显式启用 `assistant_only_loss`，只让 assistant 回复参与损失计算；
- 只运行一个模型，不做对比实验；
- 输出到 `outputs/correction-round-10/`。

正式训练命令：

```bash
python scripts/train.py --config configs/training/correction-round-10-20260908/train_config.yaml
```

Round 8 的暂存数据路径由配置固定为：

```text
data/processed/round-10-staging/train.json
data/processed/round-10-staging/eval.json
```

旧的 `correction-round-4` 至 `correction-round-9` 纠错源在生成 Round 10 暂存数据时
被排除，避免重复过采样已经学习过的纠错题。新数据生成后必须重新运行 Round 8 专用
整理程序。

训练完成后，根据最低 `eval_loss` 选择 checkpoint，再将 Round 10 LoRA 合并到
原始 Qwen 基座，生成新的 `outputs/correction-round-10-merged`。
训练期间不需要启动 vLLM。

## 历史训练配置

| 路径 | 用途 |
|---|---|
| `training/correction-round-5-20260905/train_config.yaml` | Round 5 配置快照；仅用于复现和对照 |
| `training/correction-round-6-20260905/train_config.yaml` | Round 6 配置快照；仅用于复现和对照 |
| `archive/training/correction-round-7-20260905/train_config.yaml` | Round 7 配置快照；已归档 |
| `training/correction-round-8-20260907/train_config.yaml` | Round 8 配置快照；已归档 |
| `training/correction-round-9-20260907/train_config.yaml` | Round 9 配置快照；已归档 |
| `training/correction-round-10-20260908/train_config.yaml` | 当前唯一正式训练入口 |
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

只有在 Round 9 合并模型生成后，才应新增或复制对应的 Round 9 部署配置，确保
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
