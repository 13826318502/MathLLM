# Loss 曲线解读：如何判断训练状态

> 本文档讲解如何看懂训练过程中的 Loss 曲线，判断模型是否训练正常，以及遇到问题时如何调整。
> 阅读时间：约 20 分钟。

---

## 一、什么是 Loss？

### 1.1 Loss 的直觉理解

Loss（损失函数）衡量的是**模型预测和正确答案之间的差距**。

- Loss 越大 → 模型预测越差
- Loss 越小 → 模型预测越准
- Loss = 0 → 模型完美预测（理论上不可能达到）

训练的目标就是：**通过调整模型参数，让 Loss 尽可能小**。

### 1.2 语言模型用的是什么 Loss？

语言模型使用的是 **Cross-Entropy Loss（交叉熵损失）**。

直觉理解：模型对每个 token 预测一个概率分布，Loss 衡量的是模型给正确答案分配的概率有多低。

`
例：模型预测下一个 token

正确答案是 "解"

模型的预测概率分布：
  "解": 0.8   <- 给正确答案的概率越高，Loss 越低
  "求": 0.1
  "计": 0.05
  ...

Loss = -log(0.8) = 0.22  （很小，预测好）

如果模型预测：
  "解": 0.1   <- 给正确答案的概率很低
  "求": 0.5
  "计": 0.3
  ...

Loss = -log(0.1) = 2.30  （很大，预测差）
`

### 1.3 Loss 的典型值范围

| 场景 | Loss 值 | 含义 |
|------|---------|------|
| 随机猜测 | 约 10+ | 模型什么都不会 |
| 预训练基座模型 | 2-3 | 已有通用能力，但领域不专精 |
| 精调后的模型 | 0.5-1.5 | 领域能力显著提升 |
| 完美预测 | 0 | 理论极限，不可能达到 |

### 1.4 Loss 和准确率的关系

Loss 和准确率是**反向关系**：

- Loss 下降 → 准确率上升
- Loss 上升 → 准确率下降

但 Loss 提供的信息比准确率更丰富——它是一个连续值，能看到细微的变化趋势。

---

## 二、Train Loss 和 Eval Loss

### 2.1 两种 Loss 的区别

训练过程中会同时跟踪两种 Loss：

**Train Loss（训练集损失）**：
- 在训练数据上计算的 Loss
- 反映模型对**已见过的数据**的拟合程度
- 训练过程中实时更新

**Eval Loss（验证集损失）**：
- 在验证集（模型没见过的数据）上计算的 Loss
- 反映模型对**未见过的数据**的泛化能力
- 每隔一定步数（如每 100 步）计算一次

### 2.2 为什么要同时看两个 Loss？

只看 Train Loss 是不够的。模型可能"记住"了训练数据（Train Loss 很低），但面对新数据时表现很差（Eval Loss 很高）。这就是**过拟合**。

| 情况 | Train Loss | Eval Loss | 含义 |
|------|-----------|-----------|------|
| 正常训练 | 下降 | 下降 | 模型在学习，泛化良好 |
| 过拟合 | 持续下降 | 先降后升 | 模型记住了训练数据但无法泛化 |
| 欠拟合 | 很高 | 很高 | 模型没有充分学习 |

---

## 三、五种典型的 Loss 曲线模式

### 模式 1：正常收敛（最理想）

`
Loss
 |
 |  Train Loss:  ___________
 |              /
 |             /
 |            /
 |  Eval Loss: _____________
 |           /
 |          /
 |         /
 |________/____________________ 训练步数
`

**特征**：
- Train Loss 和 Eval Loss 都在下降
- 两者趋于平稳
- 两者差距不大

**含义**：训练成功，模型学到了知识并且能泛化。

**行动**：训练可以结束，模型可用。

---

### 模式 2：过拟合（最常见的问题）

`
Loss
 |
 |  Train Loss: ___________
 |             /
 |            /
 |           /
 |          /
 |  Eval Loss: ___
 |            /   \
 |           /     \_________  <- 开始上升！
 |          /
 |_________/__________________ 训练步数
                     ^
                   过拟合拐点
`

**特征**：
- Train Loss 持续下降
- Eval Loss 先下降后**上升**
- 两者差距越来越大

**含义**：模型开始"记住"训练数据的细节，而不是学习通用规律。

**行动**：
- 立即停止训练（Early Stopping）
- 减少训练 epoch 数
- 增加训练数据量
- 降低 LoRA rank
- 增加 lora_dropout

**判断拐点**：Eval Loss 最低的那个 checkpoint 就是最好的模型。

---

### 模式 3：欠拟合

`
Loss
 |
 |  Train Loss: ___________________
 |                 /
 |                /
 |               /
 |              /
 |  Eval Loss: ____________________
 |                /
 |               /
 |              /
 |_____________/_______________ 训练步数
`

**特征**：
- Train Loss 和 Eval Loss 都很高
- 下降非常缓慢
- 没有趋于平稳的迹象

**含义**：模型没有充分学习，可能是训练时间不够或学习率太小。

**行动**：
- 增加训练 epoch 数
- 增大学习率（从 1e-4 调到 2e-4 或 5e-4）
- 增大 LoRA rank（从 8 增到 16 或 32）
- 检查数据质量是否有问题

---

### 模式 4：学习率过大（震荡）

`
Loss
 |
 |     /\      /\
 |    /  \    /  \    /\
 |   /    \  /    \  /  \
 |  /      \/      \/    \
 | /                      \___
 |/___________________________ 训练步数
`

**特征**：
- Train Loss 上下剧烈震荡
- 没有平稳下降的趋势
- 偶尔有很大的跳变

**含义**：学习率太大，参数更新步幅过大，在最优解附近来回跳动。

**行动**：
- 降低学习率（从 5e-4 降到 2e-4，或从 2e-4 降到 1e-4）
- 增加 warmup_ratio（从 0.03 增到 0.05）
- 增大 batch_size（通过增大 gradient_accumulation_steps）

---

### 模式 5：训练崩溃

`
Loss
 |
 |                          |
 |                          |  <- Loss 突然飙升
 |                          |     或变成 NaN
 |  ________________________/
 | /
 |/___________________________ 训练步数
`

**特征**：
- Loss 突然飙升到很大的值
- 或者变成 NaN（Not a Number）
- 训练无法继续

**含义**：严重问题，通常是以下原因之一：
- 学习率太大导致梯度爆炸
- 数据格式错误（ChatML 格式不对）
- 数据中有异常值

**行动**：
- 大幅降低学习率
- 检查数据格式是否正确（特别是 ChatML 格式）
- 添加 gradient clipping（梯度裁剪）
- 检查数据中是否有异常样本

---

## 四、如何读取 Loss 曲线

### 4.1 LLaMA-Factory 的 Loss 输出

LLaMA-Factory 训练时会实时打印 Loss：

`
{'loss': 2.3456, 'learning_rate': 0.0002, 'epoch': 0.5}
{'loss': 1.8234, 'learning_rate': 0.0002, 'epoch': 1.0}
{'loss': 1.4523, 'learning_rate': 0.0002, 'epoch': 1.5}
{'loss': 1.1234, 'learning_rate': 0.0001, 'epoch': 2.0}
{'loss': 0.9876, 'learning_rate': 0.00005, 'epoch': 2.5}
`

### 4.2 用 TensorBoard 可视化

`ash
tensorboard --logdir ./outputs/math-lora --port 6006
`

然后浏览器打开 http://localhost:6006 查看 Loss 曲线图。

### 4.3 用 WandB 可视化（可选）

`ash
pip install wandb
wandb login
`

训练时 LLaMA-Factory 会自动记录到 WandB，提供更丰富的可视化。

### 4.4 关注三个关键指标

1. **最终 Train Loss 值**：越低说明模型对训练数据拟合越好
2. **Eval Loss 最低点**：这是最好的 checkpoint 对应的 Loss
3. **Train-Eval 差距**：差距越大说明过拟合越严重

---

## 五、Loss 曲线实操指南

### 5.1 训练前：小规模验证

先用 100 条数据训练 1 个 epoch，观察 Loss 是否正常下降：

`ash
llamafactory-cli train configs/train_config.yaml \
  --num_train_epochs 1 \
  --max_samples 100
`

如果 Loss 正常下降，再上全量数据。

### 5.2 训练中：持续监控

每 10 步观察一次 Loss 变化：

- Loss 是否平稳下降？
- 有没有突然跳变？
- Eval Loss 是否也开始上升？

### 5.3 训练后：对比分析

训练完成后，对比不同配置的 Loss 曲线：

- 哪个配置的 Eval Loss 最低？
- 哪个配置收敛最快？
- 有没有过拟合的迹象？

---

## 六、实际案例解读

### 案例 1：正常训练

`
Step 10:  train_loss=2.1, eval_loss=2.2
Step 50:  train_loss=1.5, eval_loss=1.6
Step 100: train_loss=1.1, eval_loss=1.2
Step 150: train_loss=0.9, eval_loss=1.0
Step 200: train_loss=0.8, eval_loss=0.95
Step 250: train_loss=0.75, eval_loss=0.93  <- 趋于平稳
`

**解读**：正常收敛，Train 和 Eval Loss 差距小，模型泛化良好。可以继续训练或结束。

### 案例 2：过拟合

`
Step 10:  train_loss=2.1, eval_loss=2.2
Step 50:  train_loss=1.5, eval_loss=1.6
Step 100: train_loss=1.1, eval_loss=1.2
Step 150: train_loss=0.8, eval_loss=1.1   <- Eval 开始上升！
Step 200: train_loss=0.6, eval_loss=1.3
Step 250: train_loss=0.4, eval_loss=1.5
`

**解读**：在 Step 100 左右出现过拟合拐点。最佳模型是 Step 100 的 checkpoint。需要减少 epoch 或增加数据。

### 案例 3：学习率过大

`
Step 10:  train_loss=2.1
Step 20:  train_loss=1.8
Step 30:  train_loss=2.5  <- 跳变！
Step 40:  train_loss=1.6
Step 50:  train_loss=2.3  <- 又跳变！
Step 60:  train_loss=1.4
`

**解读**：Loss 震荡明显，学习率过大。建议从 2e-4 降到 1e-4。

---

## 七、面试高频问题

### Q: Loss 是什么？怎么理解？

> Loss 是损失函数，衡量模型预测和正确答案的差距。语言模型用 Cross-Entropy Loss，本质是衡量模型给正确答案分配的概率有多低。Loss 越小越好，典型范围是精调后 0.5-1.5。

### Q: 怎么判断模型过拟合了？

> 看 Train Loss 和 Eval Loss 的关系。如果 Train Loss 持续下降但 Eval Loss 开始上升，就是过拟合。拐点处的 Eval Loss 最低，对应的 checkpoint 是最好的模型。

### Q: 过拟合了怎么办？

> 有几种方法：减少训练 epoch 数、增加训练数据量、降低 LoRA rank、增加 dropout、使用 Early Stopping。在我的项目里，我通过观察 Eval Loss 曲线，在拐点处停止训练，避免了过拟合。

### Q: 学习率怎么选？怎么调？

> LoRA 微调的常用学习率范围是 1e-4 到 5e-4。我项目里用的是 2e-4，这是一个安全的起点。如果 Loss 震荡说明学习率太大，如果 Loss 下降太慢说明学习率太小。可以通过观察 Loss 曲线来调整。

---

## 参考资料

- [LLaMA-Factory 训练日志](https://github.com/hiyouga/LLaMA-Factory) - Loss 输出格式
- [TensorBoard 使用教程](https://www.tensorflow.org/tensorboard) - 可视化工具
- [深度学习 Loss 曲线解读](https://) - 搜索"深度学习 loss 曲线 过拟合"
