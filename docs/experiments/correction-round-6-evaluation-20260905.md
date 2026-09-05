# Round 6 微调、合并与三类评测报告（2026-09-05）

## 1. 实验目的

本轮以 Round 5 合并模型为基座，加入 Round 6 纠错数据和原始数据回放，目标是
降低历史错误率，同时尽量保持原始数学题能力。训练完成后选择最佳 checkpoint，
合并 LoRA，并使用同一份合并模型评测原始测试集、Round 6 纠错验证集和历史回归集。

本报告的主要结果来自：

```text
eval/results/correction-round-6-evaluation-20260905-final-v2/
```

模型权重和完整逐题结果不提交到 GitHub；本地和云端结果目录保留完整记录。

## 2. 训练配置与数据

| 项目 | 本轮设置 |
|---|---|
| 基座模型 | `outputs/correction-round-5-merged` |
| 微调方式 | QLoRA 4-bit + LoRA |
| 训练数据 | 1155 条 |
| 原始数据回放 | 917 条 |
| Round 6 纠错数据 | 238 条实际进入训练 |
| 内部验证集 | 128 条 |
| epoch | 1 |
| batch size | 1 |
| 梯度累积 | 8 |
| learning rate | `1.0e-5` |
| 最大训练序列长度 | 2048 |

Round 6 纠错原始数据共 240 条，经过组内近似去重后有 238 条进入
`data/processed/round-6-staging/train.json`。独立纠错验证集有 30 条，未参与训练。

训练配置文件为：

```text
configs/training/correction-round-6-20260905/train_config.yaml
```

## 3. 训练结果

| 指标 | 结果 |
|---|---:|
| 优化步数 | 145 |
| 最终 train loss | 0.3569766 |
| 最终 eval loss | 0.2617419 |
| 最低 eval loss | 0.2616872 |
| 最佳 checkpoint | `outputs/correction-round-6/checkpoint-80` |
| 训练状态 | 正常完成 |

训练集 loss 在小 batch 和混合难度数据下有明显波动，但 eval loss 大体稳定并略有
下降，没有出现 NaN、梯度爆炸或 CUDA OOM。本轮曲线不能证明模型已经充分收敛，
但也没有看到明显的验证集恶化型过拟合。

最佳 checkpoint 已合并到：

```text
outputs/correction-round-6-merged
```

合并模型包含 `config.json` 和 4 个 safetensors 权重分片，可供 Transformers 和
vLLM 加载。vLLM 最终使用的服务名为 `mathllm-round6`，上下文上限为 4096。

## 4. 推理评测结果

### 4.1 程序化评测

`evaluate.py` 的基础结果如下。三类数据的请求均成功完成，没有请求失败：

| 数据集 | 数量 | 正确 | 错误 | 基础准确率 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| 原始测试集 | 116 | 76 | 40 | 65.52% | 3719.66 ms |
| Round 6 纠错验证集 | 30 | 21 | 9 | 70.00% | 3274.02 ms |
| 历史回归集 | 42 | 17 | 25 | 40.48% | 6791.32 ms |

平均首 token 延迟分别为 60.27 ms、60.01 ms 和 59.28 ms。上述基础准确率使用
程序化的文本/数值匹配，仅作为辅助指标。

### 4.2 LLM Judge 语义评测

主要结论以 `deepseek-v4-flash` 的 LLM Judge 结果为参考。评测输出只保留
`correct`、`incorrect` 和 `uncertain` 标签，同时保存了逐题模型答案和原始响应。

| 数据集 | correct | incorrect | uncertain | judge errors | 排除不确定后的准确率 |
|---|---:|---:|---:|---:|---:|
| 原始测试集 | 86 | 29 | 1 | 1 | 74.78% |
| Round 6 纠错验证集 | 26 | 4 | 0 | 0 | 86.67% |
| 历史回归集 | 17 | 23 | 2 | 0 | 42.50% |

对应报告位置：

```text
eval/results/correction-round-6-evaluation-20260905-final-v2/original-test/llm-judge/llm_judged_report.json
eval/results/correction-round-6-evaluation-20260905-final-v2/correction-validation/llm-judge/llm_judged_report.json
eval/results/correction-round-6-evaluation-20260905-final-v2/regression/llm-judge/llm_judged_report.json
```

原始测试集有 30 条需要人工复核的记录，纠错验证集有 4 条，回归集有 25 条。
因此 LLM Judge 准确率仍应视为自动语义评估结果，关键错误和不确定项需要人工抽查。

## 5. 与 Round 5 的比较

在相同的原始测试集和回归集上，LLM Judge 结果为：

| 数据集 | Round 5 | Round 6 | 变化 |
|---|---:|---:|---:|
| 原始测试集 | 74.34% | 74.78% | +0.44 个百分点 |
| 历史回归集 | 40.54% | 42.50% | +1.96 个百分点 |

原始测试集基本没有实质提升，回归集有小幅提升。Round 5 和 Round 6 的纠错验证集
题目数量和内容不同（分别为 23 条和 30 条），不能直接用两者的准确率判断提升或
下降。

## 6. 对本轮状态的判断

本轮训练和部署链路是成功的，但模型能力提升主要集中在纠错方向，尚未转化为明显
的通用测试集提升：

1. 训练过程正常，eval loss 稳定，未发现训练崩溃或明显过拟合证据。
2. 原始测试集准确率几乎不变，说明 Round 6 的学习信号没有显著改变通用能力。
3. 回归集准确率略升，说明部分历史错误得到修复，但错误率仍然较高。
4. 纠错验证集表现较好，但该集合规模较小，不能单独代表整体数学能力。
5. 之前第一次评测的 400 错误属于上下文长度配置问题，不属于模型数学错误。本轮
   将 vLLM 的 `max-model-len` 调整为 4096 后，三类评测请求全部返回 200。

## 7. 下一步建议

下一轮继续以 `outputs/correction-round-6-merged` 为基座，保留原始测试集和回归集
不变。训练数据建议调整为：

```text
60%～70%：新的、多样化的纠错训练题
30%～40%：少量原始数据回放
```

纠错题应围绕多步应用题、概率与组合、线性代数概念、代数漏项/漏系数和长步骤
推理展开，使用不同数字和不同表述，避免直接复制回归题。

学习率不建议直接恢复到 `2e-4`。可以先比较 `5e-6` 与 `1e-5`，并使用较密的
checkpoint 评估，在原始测试集、纠错验证集和回归集同时观察“修复错误”和“新增错误”。

## 8. 结论

Round 6 完成了从训练、最佳 checkpoint 选择、LoRA 合并、vLLM 部署到三类数据评测
的完整流程。当前最可靠的结论是：模型的纠错能力有小幅改善，但通用测试集没有
显著提升，下一步应优先改善纠错数据的质量、覆盖范围和训练比例，而不是简单增加
训练轮数或大幅提高学习率。
