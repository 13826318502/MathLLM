# Correction Round 7

这是单模型训练配置，不进行双模型对比。

- 基座：`outputs/correction-round-6-merged`
- 训练数据：`data/processed/round-7-staging/train.json`
- 内部验证：`data/processed/round-7-staging/eval.json`
- 训练混合：150 条新纠错题 + 65 条原始数据回放，纠错比例约 70%
- 纠错验证集：`data/eval/correction-validation/round-7.json`
- 独立测试集和回归集：仍只在合并后评测，不参与训练
- 方式：QLoRA 4-bit + LoRA，batch size 1，梯度累积 8，1 epoch
- 学习率：`5e-6`，从已合并模型继续训练时采用保守值

训练命令：

```bash
python scripts/train.py --config configs/training/correction-round-7-20260905/train_config.yaml
```

训练期间不要启动 vLLM，以免占用显存；训练结束后再合并并启动 vLLM 做
`test`、`correction-validation` 和 `regression` 三组评测。
