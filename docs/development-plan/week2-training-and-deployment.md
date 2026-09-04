# Week 2 开发计划：LoRA 训练与模型部署

## 1. 本周目标

第二周的目标是把第一周生成的数学数据用于模型训练，并完成从 LoRA adapter 到可部署推理服务的完整链路：

```text
train.json/eval.json
      ↓
100 条数据小规模试跑
      ↓
完整数据 LoRA 训练
      ↓
训练结果检查和 Loss 分析
      ↓
合并 LoRA 权重
      ↓
AWQ/INT4 量化
      ↓
vLLM 启动推理服务
      ↓
接口测试和结果记录
```

本周不做消融实验，也不追求一次得到最终最优模型。重点是建立一条可复现、能跑通、能解释的基线流程。

## 2. 本周最终交付物

```text
outputs/
├── math-lora/
│   └── checkpoint 或最终 adapter
├── math-lora-merged/
│   └── 合并后的 FP16 模型
└── math-lora-quantized/
    └── AWQ/GPTQ INT4 模型
```

同时需要保存：

- 训练配置、数据量和 GPU 型号；
- train loss、eval loss 和训练日志；
- 最佳 checkpoint 的位置；
- 合并模型和量化模型的大小；
- FP16 与 INT4 的回答对比；
- vLLM 服务测试结果。

## 3. 训练环境准备

你的本地电脑没有 NVIDIA CUDA 显卡，因此第二周的训练、合并、量化和 vLLM 部署应放到云 GPU 上完成。

### 推荐租用配置

- NVIDIA A10 24GB；
- PyTorch 2.1 或更高版本；
- CUDA 12.x；
- Python 3.10；
- CPU 4 核以上；
- 内存 16GB 起步，建议 32GB；
- 磁盘至少 50GB，建议 80GB。

项目开发指南推荐 AutoDL，也提到恒源云、潞晨云和 Colab。价格以平台当前价格为准，文档中的价格仅作参考。

### 环境检查

连接云实例后执行：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

必须确认：

- 能看到 NVIDIA GPU；
- `torch.cuda.is_available()` 为 `True`；
- PyTorch 与 CUDA 环境匹配；
- GPU 显存至少 16GB，最好 24GB。

## 4. 第一天：同步代码并安装依赖

### 任务

- 将第一周代码和数据同步到云 GPU；
- 创建 Python 3.10 环境；
- 安装项目依赖；
- 安装并检查 LLaMA-Factory；
- 确认训练集和验证集存在。

### 同步项目

```bash
git clone <你的仓库地址>
cd MathLLM
```

需要确认这些文件已同步：

```text
scripts/prepare_data.py
data/processed/train.json
data/processed/eval.json
configs/train_config.yaml
```

### 安装环境

```bash
conda create -n mathllm python=3.10 -y
conda activate mathllm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install llamafactory
```

确认数据：

```bash
ls -lh data/processed/
python -c "import json; print('train:', len(json.load(open('data/processed/train.json')))); print('eval:', len(json.load(open('data/processed/eval.json'))))"
```

如果数据文件不存在，应先回到第一周，不要直接开始训练。

## 5. 第二天：100 条数据小规模试跑

### 目标

先用约 100 条数据训练 1 个 epoch，确认模型、tokenizer、ChatML 格式、Loss 和 checkpoint 保存流程都正常。

这次试跑不用于产生最终模型，主要用于发现配置错误。

### 调整配置

当前配置在：

```text
configs/train_config.yaml
```

建议小规模试跑时临时使用：

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 1
```

正式配置中的 LoRA 参数保持：

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-4
max_seq_length: 2048
```

如果显存不足，可以把 `gradient_accumulation_steps` 调大到 16，但不要增大实际 batch size。

### 启动方式

项目现在提供两种训练入口。自定义训练脚本已经实现，基线训练优先使用：

```bash
python scripts/train.py --config configs/train_config.yaml
```

消融脚本会根据 `configs/ablation.yaml` 自动生成 LLaMA-Factory 配置，因此消融实验仍使用 LLaMA-Factory：

```bash
python eval/ablation.py --config configs/ablation.yaml --groups rank
```

需要检查：

- 模型是否成功加载；
- tokenizer 是否能处理 `messages`；
- ChatML 模板是否报错；
- 是否出现 CUDA OOM；
- 是否生成 checkpoint；
- train loss 是否有正常数值并开始下降。

自定义训练脚本会读取嵌套项目配置、应用 Qwen chat template、挂载 LoRA，并把 adapter 保存到 `training.output_dir`。正式训练前仍需在云 GPU 上完成小规模试跑，确认当前 Transformers/TRL/PEFT 版本兼容。

## 6. 第三天：完整 LoRA 基线训练

小规模试跑成功后，再恢复完整数据和正式配置。

基线配置为：

- LoRA rank：16；
- `lora_alpha`：32；
- dropout：0.05；
- target modules：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`；
- 学习率：`2e-4`；
- epoch：3；
- 最大长度：2048；
- A10 24GB 上优先使用标准 LoRA。

正式启动前记录：

```text
训练数据数量
验证数据数量
GPU 型号和显存
Git commit 或代码版本
配置文件版本
启动时间
```

### 训练期间观察 Loss

```text
train loss 下降，eval loss 也下降  → 正常
train loss 下降，eval loss 上升    → 可能过拟合
两个 loss 都不下降                 → 检查数据、学习率或配置
loss 变成 NaN                       → 检查精度、学习率和异常数据
```

如果显存不足，按以下顺序处理：

1. 将 batch size 调为 1；
2. 增大 gradient accumulation；
3. 启用 gradient checkpointing；
4. 再考虑切换 QLoRA；
5. 不要删除验证集来掩盖问题。

## 7. 第四天：检查并合并 LoRA 权重

### 任务

- 找到训练产生的最佳 checkpoint；
- 检查 adapter 文件是否完整；
- 用原始基座模型加载 adapter；
- 调用 `merge_and_unload()` 合并权重；
- 保存独立的 FP16 模型和 tokenizer。

目标目录：

```text
outputs/math-lora-merged/
```

合并模型大小接近 Qwen2.5-7B 的 FP16 模型，约 14GB。合并阶段建议继续使用 A10 24GB，并预留足够磁盘空间。

当前 `scripts/merge_lora.py` 的主要逻辑还是 `TODO`，执行前必须确认加载、合并和保存代码已经实现。

### 合并后验证

至少使用 5 道题比较：

- 基座模型回答；
- LoRA adapter 回答；
- 合并后模型回答。

检查是否出现乱码、重复、回答风格变化或明显能力退化。

## 8. 第五天：AWQ/INT4 量化

### 任务

- 准备代表数学场景的校准数据；
- 将合并后的 FP16 模型量化为 AWQ INT4；
- 保存到独立目录；
- 记录量化前后的模型大小。

输入和输出目录应分开：

```text
输入：outputs/math-lora-merged/
输出：outputs/math-lora-quantized/
```

校准数据应覆盖：

- 高等数学；
- 线性代数；
- 概率论；
- 基础代数和应用题；
- 中文和英文题目。

项目文档推荐 AWQ，GPTQ 可以作为备用方案。第一版先完成一种量化方法即可。

### 重要注意事项

当前 `scripts/quantize.py` 已使用 `llmcompressor` 实现 AWQ W4A16 量化，`requirements.txt` 已加入 `llmcompressor`。执行量化前仍要确认云服务器上的 CUDA、Transformers、llmcompressor 和 vLLM 版本兼容。

不要把 bitsandbytes 的 QLoRA 4bit 加载结果直接当成 AWQ 模型。QLoRA 的 NF4 训练加载和 AWQ 的部署 checkpoint 不是同一种格式。

### 量化后验证

使用相同的 10–20 道题分别测试 FP16 和 INT4，记录：

```text
题目
FP16 回答
INT4 回答
最终答案是否一致
解题过程是否明显退化
```

如果量化后数学答案大量错误，应检查校准数据、量化格式和推理参数；必要时改用 INT8 或暂时保留 FP16。

## 9. 第六天：启动 vLLM 服务

先启动已经确认格式正确的模型，再切换量化模型。

量化命令：

```bash
python scripts/quantize.py \\
  --model_path ./outputs/math-lora-merged \\
  --output_path ./outputs/math-lora-quantized \\
  --calibration_data ./data/processed/train.json \\
  --num_calibration_samples 256 \\
  --max_seq_length 2048 \\
  --dtype float16
```

脚本使用 AWQ 激活感知缩放和 W4A16 权重量化，并保存 `quantization_manifest.json`。量化必须在 CUDA GPU 上完成；QLoRA 的 NF4 加载结果不能直接当作 AWQ 部署模型。

典型 AWQ 启动命令：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/math-lora-quantized \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --dtype float16 \
  --quantization compressed-tensors
```

当前项目配置需要特别核对：

1. `model_path` 应指向 `math-lora-quantized`；
2. 当前脚本使用 llmcompressor 保存 compressed-tensors 格式，`quantization` 应写成 `compressed-tensors`；AWQ 是其中使用的量化算法；
3. `max_concurrent_requests` 当前没有在 `server.py` 中真正传给 vLLM；
4. 8GB 显存只适合低并发、较短上下文的测试，不要一开始设置高并发。

### 服务测试

```bash
curl http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"math-solver","messages":[{"role":"user","content":"求解 x^2-5x+6=0"}],"temperature":0.2,"max_tokens":512}'
```

检查 HTTP 状态码、回答内容、最终答案、LaTeX 和是否出现 CUDA OOM。

## 10. 第七天：端到端联调和总结

至少测试 10 道题：

- 3 道高等数学；
- 3 道线性代数；
- 2 道概率论；
- 2 道基础代数或应用题。

记录：

```text
题目
模型版本
是否答对
回答是否完整
是否有公式错误
总延迟
是否出现异常
```

最终填写：

```text
训练方法：LoRA 或 QLoRA
GPU：
训练数据量：
验证数据量：
训练 epoch：
最佳 checkpoint：
合并模型大小：
量化模型大小：
FP16 显存：
INT4 显存：
vLLM 是否启动成功：
测试题正确数量：
```

## 11. 本周不要做的事情

- 不要在没有小规模试跑的情况下直接训练完整数据；
- 不要在训练失败时立刻修改多个参数；
- 不要把训练 loss 下降直接等同于数学能力提升；
- 不要只看最终答案而忽略解题过程是否错误；
- 不要在量化模型未经测试时宣称精度没有下降；
- 不要把 QLoRA 的 NF4 checkpoint 当成 AWQ 模型部署；
- 不要同时做 rank、学习率、数据量等多项消融，消融安排在第四周；
- 不要在没有完成云 GPU 小规模试跑前直接假设完整训练流程已经可用。

## 12. 第二周验收标准

- [ ] 云 GPU 可以正常识别 CUDA；
- [ ] 第一周的 `train.json` 和 `eval.json` 已同步；
- [ ] 100 条数据、1 epoch 的小规模训练成功；
- [ ] 完整 LoRA 基线训练成功或有明确失败记录；
- [ ] 保存了 checkpoint 和训练日志；
- [ ] 能解释 train loss 和 eval loss 的变化；
- [ ] LoRA 权重成功合并为独立模型；
- [ ] 合并模型可以加载并回答问题；
- [ ] AWQ/INT4 量化在云 GPU 上成功，并完成输出加载验证；
- [ ] FP16/INT4 至少完成 10 道题对比；
- [ ] vLLM 服务能够启动；
- [ ] `/v1/models` 和聊天接口测试成功；
- [ ] 已记录模型大小、显存、延迟和回答质量。

## 13. 相关文件

- [开发指南](../development-guide.md)
- [训练配置](../../configs/train_config.yaml)
- [部署配置](../../configs/deploy_config.yaml)
- [训练脚本](../../scripts/train.py)
- [LoRA 合并脚本](../../scripts/merge_lora.py)
- [量化脚本](../../scripts/quantize.py)
- [vLLM 启动脚本](../../deploy/server.py)
