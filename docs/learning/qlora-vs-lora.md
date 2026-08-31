# QLoRA vs LoRA：什么时候该用 QLoRA？

> 本文档讲解 QLoRA 的核心技术、与 LoRA 的区别、以及何时应该选择 QLoRA。
> 阅读时间：约 20 分钟。

---

## 一、先回顾 LoRA 的显存占用

在理解 QLoRA 之前，先搞清楚标准 LoRA 训练时显存到底花在哪里：

```
训练一个 7B 模型的 LoRA，显存组成：

┌──────────────────────────────────────────┐
│  基座模型权重（FP16）    → ~14 GB        │  ← 这部分被冻结，但仍需加载到显存
│  LoRA adapter 参数       → ~100 MB      │  ← 真正被训练的参数
│  优化器状态（AdamW）      → ~1-2 GB     │  ← 保存 A 和 B 的一阶/二阶动量
│  梯度                    → ~100 MB      │  ← A 和 B 的梯度
│  激活值（前向传播中间量） → ~1-3 GB      │  ← 和 batch_size + seq_length 成正比
└──────────────────────────────────────────┘
总计：约 16-20 GB
```

**关键观察**：14GB 的基座模型权重占了总显存的绝大部分，但它是**冻结的、不参与梯度计算**。它唯一的用处是在前向传播时做矩阵乘法。

**QLoRA 的核心思路**：既然基座模型不需要训练，能不能用更小的精度存储它？答案是：可以，用 4-bit 量化。

---

## 二、QLoRA 是什么

### 一句话定义

> **QLoRA = 4-bit 量化加载基座模型 + LoRA adapter 训练**

基座模型从 FP16（16位浮点）压缩到 NF4（4位整数），显存从 14GB 降到约 4GB。LoRA adapter 本身仍然用 BF16 训练，精度不受影响。

### 显存对比

```
标准 LoRA（FP16 基座）:
  基座模型: 14 GB (FP16)
  LoRA + 优化器 + 梯度: ~2 GB
  激活值: ~2 GB
  ─────────────────
  总计: ~18 GB  → 需要 A10 (24GB)

QLoRA（4-bit 基座）:
  基座模型: 4 GB (NF4)    ← 大幅减少
  LoRA + 优化器 + 梯度: ~2 GB
  激活值: ~2 GB
  ─────────────────
  总计: ~8 GB  → 只需 RTX 3060 (12GB)
```

---

## 三、QLoRA 的三个关键技术

### 3.1 NF4 量化（4-bit NormalFloat）

**核心思想**：预训练模型的权重通常服从正态分布。NF4 是专门为正态分布设计的量化格式。

**普通 INT4 的问题**：

INT4 把数值均匀映射到 16 个桶（2^4 = 16）。假设权重范围是 [-3, 3]，那每个桶的宽度是 6/16 = 0.375。

```
权重分布（正态分布）:
        ▓▓
      ▓▓▓▓▓▓
    ▓▓▓▓▓▓▓▓▓▓
  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
-3  -2  -1   0   1   2   3

INT4 均匀分桶:
[----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----]
-3                                                                       3

问题：0 附近（权重最密集的区域）和 ±3 附近（几乎没有权重的区域）分配了相同数量的桶。
```

**NF4 的改进**：

NF4 根据正态分布的分位数来划分桶。0 附近（权重最密集）分配更多的桶，±3 附近分配更少的桶。

```
NF4 非均匀分桶:
[--|-|-|-|-|-|-|-|--|--|-|-|-|-|-|-|--]
-3                0                  3

0 附近桶更密集 → 量化误差更小
尾部桶更稀疏 → 反正那里权重很少
```

**数学定义**：NF4 的 16 个量化级别对应标准正态分布 N(0,1) 的 16 个等概率分位数。

| 级别 | INT4 值 | NF4 值（正态分位数） |
|------|---------|-------------------|
| 0 | -3.0 | -3.0 (理论最小值) |
| 1 | -2.6 | -1.534 |
| 2 | -2.2 | -1.078 |
| ... | ... | ... |
| 7 | -0.4 | -0.131 |
| 8 | 0.0 | 0.0 |
| 9 | 0.4 | 0.131 |
| ... | ... | ... |
| 15 | 3.0 | 3.0 (理论最大值) |

**为什么 NF4 比 INT4 好**：因为正态分布的信息熵理论最优量化就是等概率分位数，NF4 接近这个最优。实验证明 NF4 比 INT4 在模型量化上的精度损失小约 20%。

### 3.2 双重量化（Double Quantization）

**问题**：量化时需要存储 scale（缩放因子）。对于 7B 模型，每个 64 个权重共享一个 scale，这些 scale 本身也占用不少显存。

**双重量化的做法**：对 scale 本身再做一次 8-bit 量化。

```
普通量化:
  权重（4-bit）: 7B × 0.5 bytes = 3.5 GB
  Scale（FP32）: 7B/64 × 4 bytes = 0.44 GB
  ─────────────────
  总计: 3.94 GB

双重量化:
  权重（4-bit）: 7B × 0.5 bytes = 3.5 GB
  Scale（8-bit）: 7B/64 × 1 byte = 0.11 GB  ← 压缩 4 倍
  Scale 的 Scale（FP32）: 7B/64/256 × 4 bytes ≈ 0.002 GB
  ─────────────────
  总计: 3.61 GB
```

**节省**：约 0.33 GB，看起来不多，但在显存紧张时很关键。

### 3.3 分页优化器（Paged Optimizers）

**问题**：训练过程中，优化器状态（AdamW 的一阶/二阶动量）会导致显存偶尔超出峰值。

**分页优化器的做法**：借鉴操作系统的虚拟内存/swap 概念，当 GPU 显存不足时，自动把优化器状态卸载到 CPU 内存，需要时再加载回来。

```
正常情况:
  GPU 显存: [模型 | LoRA | 优化器状态 | 激活值]
  CPU 内存: [空闲]

显存不足时（自动触发）:
  GPU 显存: [模型 | LoRA | ___ | 激活值]
  CPU 内存: [优化器状态]  ← 卸载到这里

需要更新参数时:
  GPU 显存: [模型 | LoRA | 优化器状态 | 激活值]  ← 加载回来
  CPU 内存: [空闲]
```

**代价**：训练速度略微变慢（CPU-GPU 之间传输有延迟），但避免了 OOM crash。

---

## 四、QLoRA vs LoRA 完整对比

| 维度 | LoRA | QLoRA |
|------|------|-------|
| **基座模型精度** | FP16 / BF16 | NF4（4-bit 量化） |
| **LoRA adapter 精度** | BF16 | BF16（不变） |
| **7B 模型显存需求** | ~16-20 GB | ~6-8 GB |
| **训练速度** | 基准 | 慢约 30-50% |
| **精度损失** | 无（和全量微调接近） | 轻微（约 1-3 个百分点） |
| **实现复杂度** | 简单 | 需要额外安装 bitsandbytes |
| **支持框架** | PEFT / LLaMA-Factory | PEFT / LLaMA-Factory（都支持） |
| **推理时** | 合并后无额外开销 | 合并后无额外开销（和 LoRA 一样） |

### 精度损失有多大？

根据 QLoRA 原始论文的实验结果：

| 任务 | LoRA (FP16) | QLoRA (NF4) | 差距 |
|------|------------|-------------|------|
| MMLU（综合） | 57.3 | 56.8 | -0.5 |
| GSM8K（数学） | 51.2 | 50.1 | -1.1 |
| HumanEval（代码） | 38.4 | 37.2 | -1.2 |
| 平均 | - | - | **-1 左右** |

结论：QLoRA 在大多数任务上比 LoRA 差 **1-3 个百分点**，对于数学推理任务差距稍大一点。

---

## 五、什么时候选 LoRA，什么时候选 QLoRA

### 决策流程图

```
你的 GPU 显存有多少？
│
├── ≥ 24GB (A10/A100/4090)
│   └── 用标准 LoRA（精度更好，速度更快）
│
├── 16GB (RTX 4080/V100)
│   ├── 能跑标准 LoRA 吗？
│   │   ├── 能（batch_size=1-2）→ 用标准 LoRA
│   │   └── OOM → 用 QLoRA
│
├── 8-12GB (RTX 3060/4060/4070)
│   └── 用 QLoRA
│
└── < 8GB
    └── 用 QLoRA + gradient_checkpointing + 更小的模型
```

### 本项目建议

| 场景 | 推荐方案 |
|------|---------|
| AutoDL 租 A10 (24GB) | **标准 LoRA**（首选） |
| AutoDL 租 T4 (16GB) | 标准 LoRA（batch_size=1） |
| 本地 RTX 3060 (12GB) | **QLoRA** |
| 本地 RTX 4060 (8GB) | **QLoRA** |
| 本地无 GPU | AutoDL 租卡，用标准 LoRA |

---

## 六、QLoRA 的代码配置差异

### 标准 LoRA 配置（PEFT）

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 标准 LoRA：FP16 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,    # FP16/BF16 加载
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

### QLoRA 配置（PEFT）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA：4-bit 量化加载基座模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                    # 使用 NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,         # 计算时用 BF16
    bnb_4bit_use_double_quant=True,                # 开启双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,                # 传入量化配置
    device_map="auto",
)

# LoRA 配置完全一样
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

**唯一的区别**：模型加载时多了 `quantization_config` 参数。LoRA 配置和训练过程完全一样。

---

## 七、面试高频问题

### Q: QLoRA 和 LoRA 有什么区别？

> QLoRA 在 LoRA 的基础上，把基座模型从 FP16 量化到 4-bit（NF4 格式）加载。LoRA adapter 本身还是用 BF16 训练，所以训练精度基本不受影响。好处是显存需求从 ~18GB 降到 ~8GB，代价是训练速度慢约 30%，精度损失约 1-3 个百分点。

### Q: NF4 量化为什么比 INT4 好？

> 因为预训练模型的权重近似服从正态分布。INT4 是均匀量化，把数值等距映射到 16 个桶。NF4 根据正态分布的分位数来分桶，在权重最密集的 0 附近分配更多的桶，在尾部（权重稀疏）分配更少的桶。这更接近信息论上的最优量化，精度损失更小。

### Q: QLoRA 会影响推理速度吗？

> 不会。推理前会把 LoRA adapter 合并回基座模型（不管是 4-bit 还是 FP16 的基座），合并后模型恢复到 FP16 精度，推理速度和标准 LoRA 完全一样。4-bit 量化只影响训练时的显存，不影响推理。

### Q: 什么情况下你会选择 QLoRA？

> 当显存不够跑标准 LoRA 时。在我的项目里，我租了 AutoDL 的 A10（24GB），显存充足所以用了标准 LoRA。但如果只有 RTX 3060（12GB），就会用 QLoRA。根据实验，QLoRA 的精度损失在数学任务上大约 1-2 个百分点，可以接受。

---

## 参考资料

- [QLoRA 原始论文](https://arxiv.org/abs/2305.14314) - QLoRA: Efficient Finetuning of Quantized Language Models
- [bitsandbytes 库](https://github.com/TimDettmers/bitsandbytes) - NF4 量化的实现
- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft) - QLoRA 的代码实现
