# Correction Round 3 过拟合分析报告

日期：2026-09-04
实验名称：`correction-round-3-20260904`
训练方式：自定义 `scripts/train.py` + Transformers/TRL/PEFT QLoRA

## 1. 实验配置

- 训练基座：`outputs/correction-round-2-merged`
- 训练数据：`data/processed/round-3-staging/train.json`，1118 条
- 普通验证数据：`data/processed/round-3-staging/eval.json`，124 条
- 独立纠错验证集：`data/eval/correction-validation/round-3.json`，20 条
- LoRA：`r=16`、`lora_alpha=32`、`lora_dropout=0.05`
- QLoRA：4-bit NF4
- 每卡 batch size：1
- 梯度累积：8，有效 batch size 为 8
- 学习率：`1.0e-4`
- 计划训练：3 个 epoch，约 420 个优化步骤
- 最大序列长度：2048

## 2. 训练过程和证据

训练在 RTX 4090 D 上正常启动，没有出现 CUDA OOM、NaN、依赖错误或进程异常。训练进行到 330 步时，为控制 GPU 成本并保留过拟合证据而安全停止。

| 训练阶段 | Eval loss |
| --- | ---: |
| 最佳 checkpoint-110 | 0.231639 |
| checkpoint-230 附近 | 0.2486 |
| checkpoint-330 | 0.272141 |

截至 330 步的汇总记录显示：

- 最终记录的 train loss：`0.188916`
- 最终记录的 eval loss：`0.272141`
- 最佳 eval loss：`0.231639`
- 最佳 checkpoint：`outputs/correction-round-3/checkpoint-110`
- 训练日志点：67
- 验证日志点：33

loss 曲线及原始记录位于云端：

```text
outputs/correction-round-3-report/loss_curve.png
outputs/correction-round-3-report/loss_history.csv
outputs/correction-round-3-report/loss_history.json
outputs/correction-round-3-report/loss_summary.json
```

## 3. 判断

本轮存在较明确的过拟合现象：训练 loss 继续下降，但 eval loss 在 checkpoint-110 达到最低后持续回升，从 `0.231639` 上升到 `0.272141`，增幅约 17.5%。因此，`checkpoint-330` 不应作为最终模型，`checkpoint-110` 是当前应优先使用的候选。

这不是训练失败。训练过程、显存使用和 checkpoint 保存均正常；问题是继续训练已经开始损害普通验证集上的泛化能力。

## 4. 原因分析

1. 本轮是从已经微调过的 Round 2 合并模型继续训练，不是从原始基座开始，因此模型对相近数据更快达到拟合状态。
2. 1118 条样本对于 7B 模型仍然是较小的数据规模，且 3 个 epoch 会重复学习每条样本。
3. 训练集包含原始数据和多轮纠错数据，部分题型和表达方式相近，数据有效多样性低于表面数量。
4. `load_best_model_at_end=true` 只能在训练结束时恢复最佳权重，本身不会提前停止训练；本次手动停止后应直接使用已保存的 `checkpoint-110`。
5. 124 条普通验证数据规模有限，单个评估点可能有波动，但本轮从 0.231639 到 0.272141 的连续回升已经足以作为过拟合警示。

## 5. 处理结论

- 本轮不使用 `checkpoint-330` 合并；
- 保留 `checkpoint-110`，用于 LoRA 合并和后续评测；
- 暂不删除 Round 3 checkpoint，直到合并模型和评测结果确认完成；
- vLLM 应在合并后加载 `outputs/correction-round-3-merged`，不能加载训练中的 adapter 目录；
- 原始最终测试集和原错误回归集仍应独立评测，不能用于训练或 checkpoint 选择。

## 6. 下一轮建议

如果后续需要继续纠错训练，建议先使用更保守的配置：

```yaml
num_train_epochs: 1
learning_rate: 5.0e-5
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
eval_steps: 10
save_steps: 10
```

同时应保留 `eval_loss` 最低的 checkpoint，并考虑加入 early stopping。最终是否有效，仍需同时比较普通测试集、纠错验证集和原错误回归集，不能只依据训练 loss。
