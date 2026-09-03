# 89 条数据云端试跑结果

- 运行日期：2026-09-03
- GPU：NVIDIA GeForce RTX 4090 D（24GB）
- 训练方式：QLoRA 4-bit
- 训练数据：89 条
- 验证数据：10 条
- 训练配置：1 epoch、batch size 1、gradient accumulation 8
- LoRA：`r=16`、`lora_alpha=32`、`lora_dropout=0.05`
- 学习率：`2e-4`
- 最大长度：2048 tokens
- 训练耗时：约 41.785 秒
- train loss：`0.8176309463662`
- 最优 eval metric：约 `0.4618960029296306`
- checkpoint：`checkpoint-5`、`checkpoint-10`、`checkpoint-12`

最终 adapter 和 checkpoint 保留在云端，不提交到 GitHub。最终 adapter 约 154.1MB，超过 GitHub 普通单文件限制。
