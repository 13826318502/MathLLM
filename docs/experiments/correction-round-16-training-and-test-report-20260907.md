# Round16：原始 Qwen 基座微调与原始测试集评估报告

## 1. 实验范围

本轮实验重新从原始 `Qwen2.5-7B-Instruct` 基座开始训练，不叠加 Round7、Round10 或其他历史合并模型。

本报告只记录训练后的原始测试集结果，不把纠错验证集或历史回归集混入最终准确率。纠错验证集仍可作为训练过程中的 eval 数据，但不能替代冻结的原始测试集。

## 2. 训练配置

配置文件：`configs/training/correction-round-16-20260916/train_config.yaml`

| 项目 | 设置 |
|---|---|
| 基座模型 | `models/Qwen2.5-7B-Instruct-modelscope` |
| 方法 | QLoRA 4-bit + LoRA |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| 训练轮数 | 1 epoch |
| 训练 batch size | 1 |
| 梯度累积 | 8 |
| 学习率 | `5e-5` |
| 学习率调度 | cosine |
| warmup | 15 steps |
| 最大序列长度 | 2048 |
| assistant-only loss | 已开启 |
| 训练数据 | `data/processed/round-16-staging/train.json` |
| 训练数据规模 | 2397 条：2217 条清洗原始题 + 180 条纠错题 |
| 训练过程 eval 数据 | `data/processed/round-16-staging/eval.json`，60 条纠错验证题 |

原始测试集 `data/eval/test.json` 和历史回归集 `data/eval/regression/regression.json` 均未进入训练集。

## 3. 原始测试集结果

评测目录：`eval/results/correction-round-16-evaluation/original-test/`

### 3.1 程序直接判定

| 指标 | 结果 |
|---|---:|
| 测试题数 | 116 |
| 成功生成 | 116 |
| 正确 | 85 |
| 错误 | 31 |
| 数学准确率 | **73.28%** |
| 平均过程得分 | 0.9043 |
| 平均延迟 | 3678.72 ms |
| 首 token 延迟 | 69.50 ms |

### 3.2 大模型二次评测

大模型评测结果：`eval/results/correction-round-16-evaluation/original-test/llm-judge/llm_judged_report.json`

| 指标 | 结果 |
|---|---:|
| 总题数 | 116 |
| 判定正确 | 90 |
| 判定错误 | 26 |
| 不确定 | 0 |
| 评测失败 | 0 |
| 大模型评测准确率 | **77.59%** |

程序直接判定和大模型评测的口径不同：前者偏向答案字符串/数值规则，后者会结合过程和数学等价性判断，因此两者不应直接混为一个指标。当前更稳妥的结论是：Round16 原始测试集准确率约为 73%～78%，仍有明显提升空间。

## 4. 与 Round7 对比

| 评测口径 | Round7 | Round16 | 变化 |
|---|---:|---:|---:|
| 程序直接判定 | 66.38% | 73.28% | **+6.90 个百分点** |
| 大模型评测 | 77.19% | 77.59% | +0.40 个百分点 |

Round16 并不是比 Round7 更差。按程序直接判定，Round16 有明显提升；按大模型评测，提升较小，说明部分差异来自判定口径、答案表达形式和过程质量，而不完全是模型数学能力变化。

## 5. 评测中的 token 限制问题

`evaluate.log` 显示，116 条请求都出现了：

```text
context limit: retrying with max_tokens=1024
```

也就是说，命令虽然设置了 `--max_tokens 2048`，但由于 vLLM 的 `max_model_len=2048` 还要容纳题目和提示词，实际请求无法为回答预留完整的 2048 token，程序全部回退到了 1024 token。

这会导致长题解可能在最终答案前被截断，使部分样本表现为错误。因此，Round16 的 31 个程序判定错误不能全部解释为数学错误，其中一部分可能属于生成长度或上下文预算问题。

## 6. 本轮结论

1. Round16 的训练流程完成，且确实是基于原始 Qwen 基座重新训练。
2. 原始测试集程序准确率从 Round7 的 66.38% 提升到 73.28%，说明重新从基座训练并加入 Round16 数据是有效的。
3. 大模型评测准确率为 77.59%，相比 Round7 的 77.19% 基本持平，说明泛化提升仍然有限。
4. 本轮没有发生 API 评测失败，但所有测试请求都触发了 token 回退，评测结果受到生成长度限制影响。
5. 当前不能仅凭 loss 曲线或单一准确率断言模型已经达到目标；后续应先修复评测上下文预算，再比较模型能力。

## 7. 后续建议

1. 保持原始测试集冻结，继续作为跨轮次唯一主指标。
2. 评测时让 vLLM 的 `max_model_len` 大于提示词长度与回答上限之和，例如使用 4096；不要只把 `--max_tokens` 调大而保持上下文上限不变。
3. 单独记录“数学错误”和“输出被截断”，不要把二者合并为同一种错误。
4. 对程序判错但大模型判对的样本进行人工抽查，修正评测规则中的格式误判。
5. 新增纠错题时使用新的 round 编号，并与原始测试集、历史回归集、纠错验证集保持隔离。本报告对应的 Round16 数据不应被覆盖。

## 8. 结果文件

- 直接评测摘要：`eval/results/correction-round-16-evaluation/original-test/report.json`
- 逐题结果：`eval/results/correction-round-16-evaluation/original-test/details.json`
- 程序判错题：`eval/results/correction-round-16-evaluation/original-test/bad_cases.json`
- 大模型评测摘要：`eval/results/correction-round-16-evaluation/original-test/llm-judge/llm_judged_report.json`
- loss 与训练配置：本地若已清理 Round16 输出目录，则以 `configs/training/correction-round-16-20260916/train_config.yaml` 和本报告为实验记录；不要据此虚构缺失的 checkpoint 文件。
