# Round 5 训练配置

入口文件：

```text
train_config.yaml
```

固定关系：

```text
基座模型：outputs/correction-round-4-merged
训练集：data/processed/round-5-staging/train.json
内部验证集：data/processed/round-5-staging/eval.json
LoRA 输出：outputs/correction-round-5
最佳 checkpoint：根据 eval_loss 选择
```

本配置使用 QLoRA 4-bit + LoRA，学习率为 `2e-5`，训练 1 个 epoch，
`warmup_steps=7`，最大序列长度为 2048。训练时不启动 vLLM。

训练完成后，将选出的 checkpoint 合并到 Round 4 合并模型，输出到新的：

```text
outputs/correction-round-5-merged
```

不要覆盖 `outputs/correction-round-4-merged`，以便回滚和进行对照实验。
