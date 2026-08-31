# vLLM 与 PagedAttention：高性能推理引擎

> 本文档讲解 vLLM 为什么比 HuggingFace 原生推理快，核心是 PagedAttention 和 Continuous Batching 的原理。
> 阅读时间：约 25 分钟。

---

## 一、问题：LLM 推理为什么慢？

在理解 vLLM 之前，先搞清楚 LLM 推理的瓶颈在哪里。

### 1.1 LLM 推理的两个阶段

```
用户输入: "求解方程 x² - 5x + 6 = 0"

阶段一：Prefill（预填充）
  - 一次性处理所有输入 token
  - 并行计算，速度快
  - 主要瓶颈：计算量

阶段二：Decode（逐 token 生成）
  - 每次只生成一个 token
  - 串行计算，速度慢
  - 主要瓶颈：显存带宽（memory-bound）
```

### 1.2 Decode 阶段为什么慢？

生成每个 token 时，模型需要做一次前向传播。对于 7B 模型：
- 模型权重约 14GB
- 每次生成一个 token 要把 14GB 的权重从显存读到计算单元
- 但每个 token 的实际计算量很小（只是一个向量 × 一个大矩阵）

这意味着 GPU 大部分时间在**等数据搬运**，而不是在计算。这就是所谓的 **memory-bound（显存带宽瓶颈）**。

### 1.3 KV Cache 是什么

Attention 机制需要用到之前所有 token 的 Key 和 Value 向量。如果每次生成新 token 都重新计算之前所有 token 的 K/V，会非常浪费。

**KV Cache**：把之前 token 的 K 和 V 向量缓存起来，生成新 token 时只需要计算新 token 的 K/V，然后和缓存拼接。

```
生成第 1 个 token: 计算 K₁, V₁ → 缓存
生成第 2 个 token: 计算 K₂, V₂ → 缓存 [K₁K₂, V₁V₂]
生成第 3 个 token: 计算 K₃, V₃ → 缓存 [K₁K₂K₃, V₁V₂V₃]
...
生成第 n 个 token: 计算 Kₙ, Vₙ → 缓存 [K₁...Kₙ, V₁...Vₙ]
```

**KV Cache 的显存占用**：

对于 Qwen2.5-7B（32 层，每层 32 个注意力头，头维度 128）：
- 每个 token 的 KV Cache: 32 层 × 2(K+V) × 32 头 × 128 维 × 2 bytes(FP16) = **0.5 MB/token**
- 上下文长度 2048: 0.5 × 2048 = **1 GB/请求**
- 同时服务 16 个请求: 16 GB ← 这就是为什么 KV Cache 管理如此重要

---

## 二、HuggingFace 原生推理的问题

### 2.1 静态显存分配（核心问题）

HuggingFace 原生推理（用 `model.generate()`）为每个请求**预分配最大长度的 KV Cache**：

```
假设 max_seq_len = 2048，同时有 4 个请求：

请求 A: 实际生成 100 个 token
请求 B: 实际生成 500 个 token
请求 C: 实际生成 1500 个 token
请求 D: 实际生成 800 个 token

HuggingFace 的做法:
  为每个请求预分配 2048 tokens 的 KV Cache

  请求 A: [████████░░░░░░░░░░░░] 只用了 100/2048，浪费 95%
  请求 B: [████████████░░░░░░░░] 只用了 500/2048，浪费 76%
  请求 C: [████████████████████] 用了 1500/2048
  请求 D: [████████████░░░░░░░░] 只用了 800/2048，浪费 61%

  实际使用: 2900 tokens
  预分配:   8192 tokens
  显存浪费: 64%
```

### 2.2 为什么浪费这么严重？

因为 HuggingFace 需要在训练时就分配连续显存（就像 C 语言里的数组），它不知道每个请求最终会生成多少 token，所以只能按最大值分配。

这就像停车场：每个车位预留给一辆最长的车，即使停进来的是一辆小车，车位也不会缩小。

---

## 三、PagedAttention：借鉴操作系统的虚拟内存

### 3.1 核心类比

| 操作系统虚拟内存 | PagedAttention |
|---------------|---------------|
| 虚拟地址空间 | 逻辑 KV Cache（每个请求看到的） |
| 物理内存页（4KB） | 物理 KV Cache 块（16 个 token） |
| 页表（虚拟→物理映射） | 块表（逻辑块→物理块映射） |
| 按需分页（Demand Paging） | 按需分配 KV Cache 块 |
| 内存碎片 | KV Cache 碎片（PagedAttention 解决了这个问题） |

### 3.2 PagedAttention 的工作方式

**不再预分配连续显存**，而是把 KV Cache 分成固定大小的块（block），按需分配。

```
PagedAttention 的做法:

物理显存中的 KV Cache 块池:
  Block 0  Block 1  Block 2  Block 3  Block 4  Block 5  ...
  [空]     [空]     [空]     [空]     [空]     [空]

请求 A 开始生成（需要 1 个块，装 16 个 token）:
  分配 Block 2 给请求 A 的逻辑块 0

  请求 A 块表: [逻辑块0 → 物理Block2]

请求 A 生成到第 17 个 token（需要第 2 个块）:
  分配 Block 5 给请求 A 的逻辑块 1

  请求 A 块表: [逻辑块0 → 物理Block2, 逻辑块1 → 物理Block5]

注意：Block 2 和 Block 5 在物理显存中不需要连续！
```

### 3.3 显存效率对比

```
HuggingFace 原生推理:
  请求 A (100 tokens): [████░░░░░░░░░░░░░░░░] 预分配 2048 tokens
  请求 B (500 tokens): [████████░░░░░░░░░░░░] 预分配 2048 tokens
  总预分配: 4096 tokens
  总实际使用: 600 tokens
  浪费率: 85%

PagedAttention:
  请求 A (100 tokens): [██████] 分配 7 个块 (7×16=112 tokens)
  请求 B (500 tokens): [████████████████████████████████] 分配 32 个块
  总分配: 39 块 = 624 tokens
  总实际使用: 600 tokens
  浪费率: 4%（只有最后一个块可能有浪费）
```

### 3.4 块表（Block Table）的工作方式

每个请求维护一个块表，记录逻辑块到物理块的映射：

```
请求 A 的块表:
  逻辑块 0 → 物理块 2    (token 0-15)
  逻辑块 1 → 物理块 5    (token 16-31)
  逻辑块 2 → 物理块 1    (token 32-47)
  逻辑块 3 → 物理块 8    (token 48-63)

请求 B 的块表:
  逻辑块 0 → 物理块 3    (token 0-15)
  逻辑块 1 → 物理块 7    (token 16-31)

注意：请求 A 的物理块 2, 5, 1, 8 在显存中不需要连续！
这和虚拟内存中的页表原理完全一样。
```

### 3.5 Copy-on-Write 与 KV Cache 共享

PagedAttention 还支持 **KV Cache 共享**——多个请求可以共享相同前缀的 KV Cache 块。

典型场景：同一个 system prompt 被多个请求使用。

```
请求 A: [system_prompt | user_A 的问题]
请求 B: [system_prompt | user_B 的问题]

没有共享:
  请求 A: 独立计算 system_prompt 的 KV Cache
  请求 B: 独立计算 system_prompt 的 KV Cache
  浪费了两倍计算

PagedAttention 共享:
  请求 A: [共享块0 → system_prompt KV] [独有块 → user_A KV]
  请求 B: [共享块0 → system_prompt KV] [独有块 → user_B KV]
                              ↑ 指向同一个物理块

  当请求 A 要修改共享块时，触发 Copy-on-Write:
  先复制一份，再修改自己的副本
```

---

## 四、Continuous Batching：连续批处理

### 4.1 传统 Static Batching 的问题

```
Static Batching:
  一批请求必须等所有请求都生成完毕，才能处理下一批

  Batch 1:
    请求 A: [生成中████████] 已完成 ← 要等 B 和 C
    请求 B: [生成中████████████████] 已完成
    请求 C: [生成中████████████████████████] 已完成
                                                 ↑ 三个都完成才能开始下一批

    请求 A 完成后 GPU 在空转等 B 和 C → 浪费
```

### 4.2 Continuous Batching 的做法

```
Continuous Batching:
  每个 token 生成完后，检查是否有请求完成；
  如果完成了，立即把新请求插入这个 batch

  t=1: [A, B, C] 三个请求在生成
  t=2: [A, B, C]
  t=3: [A, B, C]
  t=4: [A, B, C]
  t=5: A 完成 → 移除 A，插入 D → [D, B, C]  ← 无空转
  t=6: [D, B, C]
  t=7: [D, B, C]
  t=8: B 完成 → 移除 B，插入 E → [D, E, C]  ← 无空转
  t=9: [D, E, C]
  t=10: C 完成 → 移除 C，插入 F → [D, E, F]  ← 无空转
```

### 4.3 吞吐量对比

| 方式 | GPU 利用率 | 吞吐量 |
|------|-----------|--------|
| Static Batching | 低（等最慢的请求） | 基准 |
| **Continuous Batching** | **高（随时填满 batch）** | **5-10x** |

---

## 五、vLLM 的完整工作流程

```
vLLM 接收请求后的处理流程:

1. 调度器 (Scheduler)
   ├── 接收新请求
   ├── 决定哪些请求可以开始（基于可用 KV Cache 块数量）
   └── 维护运行队列和等待队列

2. 块管理器 (Block Manager)
   ├── 管理物理 KV Cache 块的分配和释放
   ├── 维护每个请求的块表
   └── 实现 Copy-on-Write 共享

3. 模型执行器 (Model Executor)
   ├── Prefill 阶段：并行处理所有输入 token
   ├── Decode 阶段：逐 token 生成
   └── 每生成一个 token，向块管理器请求新的 KV Cache 块

4. 输出处理
   ├── 流式输出：每生成一个 token 就发送给客户端
   └── 请求完成时：释放所有 KV Cache 块
```

---

## 六、vLLM 的部署配置详解

### 6.1 关键参数

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./model \
    --max-model-len 4096 \          # 最大上下文长度
    --gpu-memory-utilization 0.9 \  # GPU 显存使用比例
    --max-num-seqs 256 \            # 最大并发请求数
    --dtype float16 \               # 模型精度
    --quantization awq              # 量化方式（可选）
```

| 参数 | 含义 | 调整建议 |
|------|------|---------|
| `max-model-len` | 最大输入+输出长度 | 根据任务需要设置，不要设太大（浪费显存） |
| `gpu-memory-utilization` | KV Cache 可用显存比例 | 0.9 是默认值，显存紧张可以降到 0.8 |
| `max-num-seqs` | 同时处理的最大请求数 | 默认 256，受 KV Cache 总量限制 |
| `dtype` | 模型加载精度 | float16 或 auto |
| `quantization` | 量化方式 | awq / gptq / None |

### 6.2 显存分配

```
vLLM 的显存分配:

GPU 总显存 24GB (A10):
├── 模型权重: 14GB (FP16) 或 4GB (INT4)
├── KV Cache: gpu_memory_utilization × 剩余显存
│   └── 这是 PagedAttention 管理的区域
└── CUDA 运行时: ~1GB

假设 FP16 模型:
  模型: 14GB
  剩余: 24 - 14 - 1 = 9GB
  KV Cache: 9 × 0.9 = 8.1GB
  每个请求 2048 tokens: 8.1 / 1 = 8 个并发请求

假设 INT4 模型:
  模型: 4GB
  剩余: 24 - 4 - 1 = 19GB
  KV Cache: 19 × 0.9 = 17.1GB
  每个请求 2048 tokens: 17.1 / 1 = 17 个并发请求
```

---

## 七、vLLM vs 其他推理框架

| 框架 | PagedAttention | Continuous Batching | 吞吐量 | 易用性 |
|------|---------------|-------------------|--------|--------|
| **vLLM** | ✅ | ✅ | 最高 | 好（OpenAI 兼容 API） |
| HuggingFace TGI | ❌（部分支持） | ✅ | 中 | 好 |
| llama.cpp | ❌ | ❌ | 低（CPU推理） | 中 |
| TensorRT-LLM | ❌ | ✅ | 高 | 差（配置复杂） |
| SGLang | ✅ | ✅ | 高 | 中 |

**本项目选择 vLLM 的理由**：
1. 吞吐量最高（PagedAttention + Continuous Batching）
2. 提供 OpenAI 兼容 API，应用层代码可以直接用 `openai` 库调用
3. 社区活跃，文档完善
4. 支持量化模型（AWQ/GPTQ）直接加载

---

## 八、面试高频问题

### Q: vLLM 为什么比 HuggingFace 原生推理快？

> 核心是两点。第一是 PagedAttention，借鉴操作系统的虚拟内存分页机制，把 KV Cache 分成固定大小的块按需分配，避免了 HuggingFace 静态预分配导致的显存碎片化，显存利用率从 20-40% 提升到 96%。第二是 Continuous Batching，不等一个 batch 全部生成完才开始下一个，而是每个 token 生成完后动态插入新请求，GPU 利用率更高。综合下来，高并发场景下吞吐量提升 5-10 倍。

### Q: PagedAttention 和操作系统虚拟内存有什么相似之处？

> 非常相似。虚拟内存把物理内存分成固定大小的页（4KB），进程看到的是连续的虚拟地址空间，通过页表映射到物理页。PagedAttention 把 KV Cache 分成固定大小的块（16 tokens），每个请求看到的是连续的逻辑 KV Cache，通过块表映射到物理块。两者都解决了"预分配连续空间导致碎片化"的问题。PagedAttention 还支持 Copy-on-Write，多个请求可以共享相同前缀的 KV Cache 块。

### Q: Continuous Batching 和 Static Batching 有什么区别？

> Static Batching 要求一个 batch 里所有请求都生成完毕，才能开始下一个 batch。如果 batch 里有一个请求特别长，其他已完成的请求就会占用 GPU 资源空等。Continuous Batching 在每个 token 生成完后检查，如果有请求完成就立即移出并插入新请求，GPU 始终保持满载。

### Q: vLLM 的 KV Cache 能容纳多少并发请求？

> 取决于 GPU 显存减去模型权重后的剩余空间。以 A10 (24GB) 为例，FP16 模型占 14GB，剩余约 9GB 给 KV Cache。每个 token 约 0.5MB，2048 长度约 1GB/请求，所以可以容纳约 8 个并发请求。如果用 INT4 量化，模型只占 4GB，并发数可以提升到约 17 个。

### Q: vLLM 支持量化模型吗？

> 支持。可以直接加载 AWQ 或 GPTQ 量化后的模型，通过 `--quantization awq` 参数指定。量化后模型权重更小，留给 KV Cache 的显存更多，可以支持更多并发请求。

---

## 参考资料

- [vLLM 原始论文](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for Large Language Model Serving with PagedAttention
- [vLLM 官方文档](https://docs.vllm.ai/) - 部署和配置指南
- [vLLM GitHub](https://github.com/vllm-project/vllm) - 源码和 issue 讨论
