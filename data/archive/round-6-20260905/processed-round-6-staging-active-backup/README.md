# Round 6 暂存数据

本目录是 Round 6 训练实际读取的数据快照，不是原始数据目录。

| 文件 | 数量 | 用途 |
|---|---:|---|
| `train.json` | 1155 | LoRA/QLoRA 训练集 |
| `eval.json` | 128 | 训练期间的内部验证集，用于记录 `eval_loss` 和选择 checkpoint |

构成说明：

- 原始数学数据回放：917 条；
- `correction-round-6` 新纠错数据：原始 240 条，其中 238 条通过纠错组内去重后进入训练集；
- 另外 2 条因纠错组内的重复/近似重复保护被过滤；
- Round 4、Round 5 旧纠错源：本轮排除，避免重复强化旧样本；
- `data/eval/test.json`：未进入训练或内部验证；
- `data/eval/regression/regression.json`：未进入训练或内部验证。

原始纠错训练数据位于：

```text
data/raw/corrections/correction-round-6-training.jsonl
```

独立纠错验证数据位于：

```text
data/eval/correction-validation/round-6.json
```

其中训练集为原始数据回放加 Round 6 纠错题；`eval.json` 是从允许参与内部
划分的数据中按固定种子 `42` 产生的内部验证集。纠错验证集、原始测试集和
回归集应在合并模型后单独评测，不能混入训练过程。
