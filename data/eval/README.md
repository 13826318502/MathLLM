# 独立评测数据

本目录保存不参与训练和 checkpoint 选择的数据。

- `test.json`：固定的最终测试集。用于比较基座模型、LoRA 模型和合并模型；不要把它加入训练集，也不要根据它反复调参。
- `regression/regression.json`：针对已发现错误的回归集，用于检查模型是否再次犯同类错误。

测试命令示例：

```bash
python eval/evaluate.py \
  --model_endpoint http://127.0.0.1:8000/v1 \
  --eval_data data/eval/test.json \
  --output eval/results/test-final
```

`eval/results/` 只保存评测输出，例如 `report.json`、`details.json`、`bad_cases.json` 和图表；它不是数据集目录。
