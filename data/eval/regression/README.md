# 错误回归集

本目录只用于训练完成后的回归测试，不会被 `scripts/prepare_data.py` 扫描，也不应复制到 `data/raw/`。

当前样本包括：

- 冒烟测试中正方形坐标题的原始错误题；
- 冒烟测试中未知进制题的原始错误题；
- 两道新的同类变体，用来检查模型是否真正学会方法。

运行方式：

```bash
python eval/evaluate.py \
  --model_endpoint http://localhost:8000/v1 \
  --eval_data data/eval/regression/regression.json \
  --output eval/results/regression
```

不要把这里的题目合并进训练集；否则回归测试会发生数据泄漏。
