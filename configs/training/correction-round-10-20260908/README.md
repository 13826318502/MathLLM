# Correction Round 10 training

- Base model: `models/Qwen2.5-7B-Instruct-modelscope` (original Qwen base; no merged adapter).
- General base data: existing 1,250 records plus 1,250 new official-source records.
- Fresh corrections: 360 records; separate correction validation: 60 records.
- Staging training set: all cleaned general data except 100 internal eval records, plus 360 corrections.
- Protected sets: `data/eval/test.json` and `data/eval/regression/regression.json`.
- QLoRA 4-bit, one epoch, batch size 1, gradient accumulation 8, learning rate `1e-4`.
- `assistant_only_loss: true`：只对 assistant 回复（解题过程和最终答案）计算 loss，
  system/user 提示词以及 assistant 标记本身被屏蔽。
- Select the best checkpoint by `best_eval_loss`, then merge only after evaluation artifacts are saved.
