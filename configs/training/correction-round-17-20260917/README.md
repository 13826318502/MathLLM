# Round 17 correction-focused training

- 基座：`outputs/correction-round-16-merged`，即 Round16 合并模型。
- 训练数据：`data/processed/round-17-staging/train.json`，857 条（600 条新纠错题 + 257 条清洗原始题回放）。
- 训练过程验证：`data/processed/round-17-staging/eval.json`，120 条独立纠错验证题。
- 原始测试集和历史回归集不参与训练，继续保持冻结。
- 采用较低学习率 `2e-5` 和 1 个 epoch，降低增量纠错训练对已有能力的破坏。
- 训练前检查 `data/processed/round-17-staging/manifest.json`，确认数据和来源无误。
