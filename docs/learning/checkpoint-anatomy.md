# LoRA/QLoRA Checkpoint 文件说明

本文档结合本项目的 Transformers + TRL + PEFT 训练流程，说明 checkpoint 的文件、作用、生成时机，以及续训、合并和推理时的区别。

## 1. Checkpoint 是什么

Checkpoint 是训练过程中的存档点。它保存某个训练步结束时的 adapter 参数、优化器状态、学习率状态和随机数状态，使训练可以从中断位置继续，也可以比较不同阶段的模型效果。

本项目中的目录通常类似：

~~~text
outputs/math-lora/
├── checkpoint-115/
├── checkpoint-280/
└── checkpoint-285/
~~~

115、280、285 表示全局 optimizer step，不是数据编号，也不是模型能力评分。

## 2. 为什么它是 LoRA checkpoint

scripts/train.py 使用 PEFT 给 Qwen2.5-7B-Instruct 注入 LoRA。基座模型被冻结，训练主要更新低秩 adapter 参数。

因此 checkpoint 通常不会复制完整的 7B 基座模型。加载时仍需要：

~~~text
同一个基座模型 + 匹配的 tokenizer + LoRA adapter
~~~

如果基座模型版本、结构或 tokenizer 不匹配，adapter 可能无法加载，或者结果不可靠。

## 3. 典型目录结构

不同 Transformers、TRL、PEFT 版本的文件名可能略有差异。常见结构如下：

~~~text
checkpoint-115/
├── adapter_model.safetensors
├── adapter_config.json
├── optimizer.pt
├── scheduler.pt
├── trainer_state.json
├── rng_state.pth
├── training_args.bin
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
~~~

可能出现的替代文件名：

~~~text
adapter_model.bin       # safetensors 的旧格式
training_args.json       # 训练参数的另一种保存形式
scaler.pt                # 某些 FP16 混合精度环境会生成
chat_template.jinja      # 某些 tokenizer 版本会单独保存
~~~

不是每个版本都会生成所有文件。判断能否推理，重点检查 adapter 权重和配置；判断能否完整续训，还要检查优化器、调度器和 Trainer 状态。

## 4. 文件作用和示例

### 4.1 adapter_model.safetensors

保存 LoRA 的低秩增量权重，是推理和合并时最重要的文件。

示意权重名称：

~~~text
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight
base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight
~~~

它不是完整模型，只表示基座权重的变化量，通常只有几十到一百多 MB。旧版本可能保存为 adapter_model.bin。

### 4.2 adapter_config.json

告诉 PEFT 如何把 adapter 挂载到基座模型。

示例：

~~~json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "bias": "none",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
~~~

其中 r 是 LoRA 秩，lora_alpha 是缩放系数，target_modules 是注入 LoRA 的线性层。没有它，程序无法正确解释 adapter 权重。

### 4.3 optimizer.pt

保存优化器状态，通常包括 AdamW 的一阶动量、二阶动量和更新步数。

~~~text
parameter → exp_avg       # 一阶动量
parameter → exp_avg_sq    # 二阶动量
step                      # 参数更新次数
~~~

它用于断点续训，使优化器沿着原来的训练轨迹继续更新。这个文件通常比 adapter 权重更大。如果只做推理或合并，可以不复制它；如果要续训，应保留它。

### 4.4 scheduler.pt

保存学习率调度器状态。项目配置中有：

~~~yaml
learning_rate: 2.0e-4
warmup_ratio: 0.03
~~~

调度器需要知道当前步数，才能恢复正确的学习率。缺少它可能导致学习率重新开始或训练轨迹改变。

### 4.5 trainer_state.json

保存 Trainer 的训练步数、日志和最佳 checkpoint 信息。

~~~json
{
  "global_step": 115,
  "best_metric": 1.284,
  "best_model_checkpoint": "outputs/math-lora/checkpoint-115",
  "log_history": [
    {"loss": 1.42, "step": 100},
    {"eval_loss": 1.284, "step": 115}
  ]
}
~~~

它可以用来查看训练进度、loss、学习率和最佳 checkpoint。它不是模型权重，但删除后会丢失训练记录和最佳模型索引。

### 4.6 rng_state.pth

保存 Python、NumPy、PyTorch 和 CUDA 的随机数状态。它有助于恢复数据打乱、dropout 等随机过程，但换硬件或软件版本后不保证完全一致。

### 4.7 training_args.bin 或 training_args.json

保存本次 Trainer 的参数，例如：

~~~text
learning_rate
per_device_train_batch_size
gradient_accumulation_steps
num_train_epochs
save_steps
eval_steps
gradient_checkpointing
bf16 / fp16
~~~

它用于追溯训练参数，但不能代替 configs/train_config.yaml。

### 4.8 tokenizer 相关文件

常见文件包括：

~~~text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
merges.txt
chat_template.jinja
~~~

这些文件保存词表、特殊 token、padding 方式和 chat template。训练和推理必须使用与 Qwen 基座匹配的 tokenizer。

## 5. Checkpoint 什么时候生成

项目配置 configs/train_config.yaml 中的关键项是：

~~~yaml
save_steps: 5
save_total_limit: 3
eval_steps: 5
logging_steps: 10
~~~

含义：

| 配置 | 作用 |
|---|---|
| save_strategy: steps | 按训练步保存 |
| save_steps: 5 | 每 5 个 optimizer step 保存一次 |
| eval_steps: 5 | 每 5 个 optimizer step 做一次验证 |
| logging_steps: 10 | 每 10 个 optimizer step 记录日志 |
| save_total_limit: 3 | 清理旧 checkpoint，避免占满磁盘 |
| load_best_model_at_end: true | 训练结束时加载最佳验证模型 |

你的 batch 配置是：

~~~yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
~~~

大致过程是读取 8 个 batch、每个 batch 2 条样本，累计梯度后更新一次参数，更新后 global_step + 1。所以 checkpoint-115 约表示完成了 115 次参数更新，不是处理了 115 条数据。

## 6. 为什么只看到几个 checkpoint

Trainer 可能曾经生成过：

~~~text
checkpoint-5
checkpoint-10
checkpoint-15
...
checkpoint-285
~~~

但 save_total_limit: 3 会自动删除旧版本。最终保留的通常是最近 checkpoint、验证集 loss 最低的最佳 checkpoint 和少量最新 checkpoint。

因此同时看到下面三个目录是正常的：

~~~text
checkpoint-115    # 可能是最佳验证点
checkpoint-280    # 最近的保存点之一
checkpoint-285    # 最近一次保存点
~~~

checkpoint 是否生成由 save_steps 决定；哪个 checkpoint 最好由 eval_loss 决定。这是两个不同过程。

## 7. 续训、合并和部署

### 7.1 断点续训

最好保留整个 checkpoint：

~~~text
adapter_model.safetensors
adapter_config.json
optimizer.pt
scheduler.pt
trainer_state.json
rng_state.pth
training_args.*
~~~

配置示例：

~~~yaml
training:
  resume_from_checkpoint: "./outputs/math-lora/checkpoint-115"
~~~

恢复训练时仍需要原来的基座模型和 tokenizer。

### 7.2 合并 LoRA

scripts/merge_lora.py 主要需要完整精度基座模型、adapter 权重和 adapter_config.json：

~~~bash
python scripts/merge_lora.py \
  --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
  --lora_path ./outputs/math-lora/checkpoint-115 \
  --output_path ./outputs/math-lora-merged \
  --dtype float16
~~~

合并后得到独立模型，体积接近完整基座模型。

### 7.3 量化和部署

推荐流程：

~~~text
最佳 LoRA checkpoint
  → merge_lora.py
  → FP16/BF16 合并模型
  → quantize.py
  → AWQ W4A16 模型
  → vLLM
~~~

不要把 optimizer.pt 当成模型权重，也不要把 QLoRA 的 NF4 加载状态直接当成 AWQ 部署模型。

## 8. 查看 checkpoint 的命令

查看目录大小：

~~~bash
cd /root/autodl-tmp/MathLLM/outputs/math-lora
du -sh checkpoint-*
~~~

查看文件列表和大小：

~~~bash
find checkpoint-115 -maxdepth 1 -type f -printf '%f\t%k KB\n' | sort
~~~

查看最佳 checkpoint：

~~~bash
python -c "import json; x=json.load(open('trainer_state.json', encoding='utf-8')); print('step:', x.get('global_step')); print('best:', x.get('best_model_checkpoint')); print('metric:', x.get('best_metric'))"
~~~

## 9. 大小和最小检查清单

按当前 LoRA 配置粗略估计：

~~~text
adapter_model.safetensors  ≈ 80–100 MB
optimizer.pt               ≈ 300–400 MB
其他状态文件               ≈ 几 MB
单个完整 checkpoint         ≈ 400–600 MB
~~~

检查 checkpoint：

- [ ] 存在 adapter_model.safetensors 或 adapter_model.bin；
- [ ] 存在 adapter_config.json；
- [ ] r、lora_alpha 和 target_modules 与训练配置一致；
- [ ] 存在匹配的基座模型和 tokenizer；
- [ ] trainer_state.json 能指出训练步数和最佳 checkpoint；
- [ ] 需要续训时，优化器、调度器和随机状态文件完整；
- [ ] 合并时使用完整精度基座；
- [ ] 部署量化前先完成合并；
- [ ] 最终模型在独立测试集上重新评测。

## 10. 关联文件

- [训练程序](../../scripts/train.py)
- [当前训练配置](../../configs/training/correction-round-2-20260904/train_config.yaml)
- [LoRA 合并程序](../../scripts/merge_lora.py)
- [LoRA 原理](lora-principle.md)
- [QLoRA 与 LoRA 对比](qlora-vs-lora.md)
- [Loss 曲线](loss-curve.md)
- [INT4 量化](int4-quantization.md)
