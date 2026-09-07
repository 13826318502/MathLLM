# Round 5 暂存数据

本目录是 Round 5 训练实际读取的数据快照，不是原始数据目录。

| 文件 | 数量 | 用途 |
|---|---:|---|
| `train.json` | 1049 | Round 5 LoRA/QLoRA 训练集 |
| `eval.json` | 116 | 训练期间的内部验证集，用于记录 `eval_loss` 和选择 checkpoint |

数据构成：

- 原始数学数据回放：929 条；
- `correction-round-5` 新纠错数据：120 条；
- Round 4 旧纠错源：已排除；
- `data/eval/test.json`：未进入训练；
- `data/eval/regression/regression.json`：未进入训练。

测试集和回归集只在模型合并后做独立评测。该目录应与
`configs/training/correction-round-5-20260905/train_config.yaml` 配套使用。
