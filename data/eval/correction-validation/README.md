# 纠错验证集

本目录保存纠错轮次专用的验证集，独立于最终测试集和回归集。

- `round-4.json`：Round 4 候选题中固定留出的 20 条，使用 ChatML `messages` 格式。
- `round-5.json`：Round 5 独立纠错验证集，历史保留用于对照。
- `round-6.json`：Round 6 新准备的 30 条纠错验证题，使用 ChatML `messages` 格式。
- `round-7.json`：Round 7 新准备的 30 条纠错验证题，使用 ChatML `messages` 格式；每类 6 条，与 Round 7 训练题使用不同参数和场景。
- `round-8.json`：Round 8 新准备的 40 条纠错验证题，使用 ChatML `messages` 格式；与 Round 8 训练题使用不同参数和场景。
- `round-9.json`：Round 9 新准备的 40 条纠错验证题，使用 ChatML `messages` 格式；不进入 Round 9 staging 训练集。
- `round-10.json`：Round 10 新准备的 60 条纠错验证题，使用 ChatML `messages` 格式；不进入 Round 10 staging 训练集。
- 这些题只用于观察同类型能力是否提升，不应复制回 `data/raw/`，也不应与原始错误题混合。
- `data/eval/test.json` 仍是固定最终测试集；`data/eval/regression/regression.json` 仍是原错误题回归集，二者保持不变。

正式训练前，先人工检查本轮训练候选和本验证集；若发现错误，应修正候选记录后重新用固定种子切分。

Round 6 至 Round 8 的验证题不进入对应的 staging 训练集，训练后应单独评测，
以免把纠错能力验证结果混入 checkpoint 选择。

Round 7 至 Round 9 的验证题保留用于历史对照；Round 10 的验证题是当前活动轮次。
