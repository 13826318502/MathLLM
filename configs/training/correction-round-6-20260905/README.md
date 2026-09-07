# Round 6 训练配置

这是本轮唯一的主动训练配置入口：

```text
train_config.yaml
```

固定关系：

```text
基座模型：outputs/correction-round-5-merged
训练集：data/processed/round-6-staging/train.json
内部验证集：data/processed/round-6-staging/eval.json
LoRA 输出：outputs/correction-round-6
```

本轮使用 QLoRA 4-bit + LoRA，学习率降为 `1e-5`，训练 1 个 epoch。
降低学习率是为了在 Round 5 合并模型上继续纠错时减少对已有能力的破坏；
是否保留本轮结果，必须以原始测试集、纠错验证集和回归集的独立评测为准。

数据由 `scripts/prepare_data.py` 生成。生成时排除了 Round 4、Round 5
旧纠错源，并保护了 `data/eval/test.json` 和
`data/eval/regression/regression.json`，所以这两个集合不参与训练或内部验证。

训练完成后，先根据 `eval_loss` 选择最佳 checkpoint，再合并到新的
`outputs/correction-round-6-merged`，不要覆盖 Round 5 合并模型。
