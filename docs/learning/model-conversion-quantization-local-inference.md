# 训练后模型格式转换、量化与本地推理指南

> 本文结合 MathLLM 当前的 Qwen2.5-7B-Instruct + LoRA/QLoRA + vLLM 路线，说明训练完成后模型有哪些形态、如何转换格式、不同量化方法适合什么运行环境，以及如何最终在本地电脑上运行。

## 1. 先记住一条完整链路

训练、合并、量化和推理不是同一个步骤：

```text
Qwen2.5-7B-Instruct 基座模型
        +
LoRA adapter / checkpoint
        ↓  merge_lora.py
独立的 FP16/BF16 合并模型
        ├── AWQ W4A16 → Hugging Face / compressed-tensors → vLLM/GPU
        └── GGUF F16 → GGUF Q4_K_M → Ollama/llama.cpp/CPU
```

这里有三个容易混淆的概念：

1. **格式**：模型文件如何组织，例如 Hugging Face 目录、Safetensors、GGUF。
2. **精度**：参数使用 FP32、BF16、FP16、INT8 或 INT4 等表示。
3. **推理引擎**：用什么程序加载模型，例如 Transformers、vLLM、llama.cpp 或 Ollama。

`AWQ` 和 `Q4_K_M` 都属于量化方法，但不是同一种文件格式，也不是同一种推理后端。

---

## 2. 训练完成后会得到哪些模型形态？

### 2.1 基座模型

基座模型是原始的 Qwen2.5-7B-Instruct，通常是一个 Hugging Face 模型目录：

```text
models/Qwen2.5-7B-Instruct/
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

它包含完整的模型参数、模型结构配置和 tokenizer，可以直接被 Transformers 加载。

### 2.2 LoRA adapter

LoRA checkpoint 只保存相对于基座模型的增量参数，不是一个完整大模型：

```text
outputs/math-lora/checkpoint-XXX/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json                  # 如果训练器保存了 tokenizer
├── tokenizer_config.json
├── trainer_state.json               # 训练状态
├── optimizer.pt                     # 优化器状态，可能存在
└── scheduler.pt                     # 学习率调度器状态，可能存在
```

LoRA adapter 必须和**相同的基座模型**配合使用。只拿 `adapter_model.safetensors`，不能脱离基座模型直接得到完整回答。

### 2.3 合并模型

执行 `scripts/merge_lora.py` 后，LoRA 的增量会写回基座模型副本，形成一个可以独立加载的模型：

```text
outputs/math-lora-merged/
├── config.json
├── model-00001-of-00004.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── merge_config.json
```

合并模型仍然是 FP16、BF16 或 FP32 模型，**还不是量化模型**。它是后续 AWQ、GPTQ 或 GGUF 转换最可靠的共同输入。

项目中的合并命令：

```bash
python scripts/merge_lora.py \
  --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
  --lora_path ./outputs/math-lora/checkpoint-XXX \
  --output_path ./outputs/math-lora-merged \
  --dtype float16
```

合并时不要把训练时的 NF4/4-bit 加载状态误当作最终部署格式。项目脚本会用完整精度基座模型执行合并，并输出独立模型目录。

---

## 3. 常见模型格式和适用场景

| 模型形态/格式 | 典型文件 | 主要用途 | 适合的运行环境 |
|---|---|---|---|
| Hugging Face FP16/BF16 | `config.json` + `*.safetensors` | 合并、评测、转换的中间产物 | Transformers、vLLM、转换工具 |
| LoRA adapter | `adapter_model.safetensors` | 继续训练或运行时挂载 LoRA | PEFT、Transformers、部分 vLLM 配置 |
| QLoRA/NF4 | bitsandbytes 配置 + 4-bit 权重 | 训练时节省显存 | Transformers + bitsandbytes |
| AWQ W4A16 | Safetensors + `quantization_config` | GPU 推理量化 | vLLM、部分 GPU 推理框架 |
| GPTQ INT4 | Safetensors + GPTQ 元数据 | GPU/部分专用推理 | Transformers、vLLM 等 |
| GGUF F16 | `model-f16.gguf` | GGUF 的高精度中间格式 | llama.cpp、Ollama |
| GGUF Q4_K_M | `model-Q4_K_M.gguf` | 本地低显存/CPU 推理 | Ollama、llama.cpp |

### 重要区别

- `Safetensors` 是一种安全的张量存储格式，不等于某一种量化方法。
- `GGUF` 是 llama.cpp 生态的模型封装格式，里面还可以保存 tokenizer 和聊天模板元数据。
- `AWQ`、`GPTQ`、`NF4`、`Q4_K_M` 是不同的量化方案或量化数据类型。
- 一个模型“是 4 bit”并不代表它们可以互相直接替换。具体还要看文件格式和推理引擎是否支持。

---

## 4. 训练阶段的 NF4 与部署阶段量化

### 4.1 QLoRA/NF4：为了训练

QLoRA 通常把基座模型以 4-bit NF4 形式加载，以降低训练显存；LoRA 参数仍然是可训练参数。

```text
4-bit NF4 基座模型（冻结）
        +
BF16/FP16 LoRA 参数（训练）
        ↓
LoRA checkpoint
```

NF4 主要解决的是“训练时显存不够”的问题。它不是最适合作为本地发布文件的通用格式，也不能直接当作 AWQ 或 GGUF 使用。

### 4.2 AWQ：为了 GPU 部署

AWQ 通常采用 W4A16：权重主要压缩为 4-bit，推理计算仍使用较高精度的激活值。AWQ 会用少量代表性文本进行校准，分析激活值和通道重要性，再进行权重量化。

项目中的 `scripts/quantize.py`：

- 输入：`outputs/math-lora-merged/`；
- 校准数据：`data/processed/train.json` 的训练样本；
- 输出：`outputs/math-lora-quantized/`；
- 目标格式：AWQ W4A16 / `compressed-tensors`；
- 目标推理环境：云服务器 NVIDIA GPU + vLLM。

AWQ 量化不能直接以 QLoRA/NF4 checkpoint 作为输入，必须先得到独立的合并模型。

### 4.3 Q4_K_M：为了本地 CPU/Ollama

`Q4_K_M` 是 llama.cpp 的 GGUF K-quant 方案。它主要使用 4-bit 分块量化，同时对不同张量采用适合的精度组合，因此并不是每一个张量都严格使用完全相同的 4-bit 表示。

它的输入通常是 GGUF F16/BF16，输出是一个单独的 `.gguf` 文件：

```text
mathllm-f16.gguf
        ↓
mathllm-Q4_K_M.gguf
```

llama.cpp 官方建议先生成高质量 GGUF，再使用 `llama-quantize` 生成目标量化类型；对已经量化过的输入重新量化可能导致更明显的质量损失。[llama.cpp 量化说明](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

---

## 5. 按运行目标选择格式和量化方法

### 5.1 云服务器 GPU + vLLM

推荐优先级：

```text
FP16/BF16 Hugging Face 模型 → 最高精度、显存占用较大
AWQ W4A16                 → 显存较小、适合 GPU 服务
GPTQ                      → 也可以使用，但要确认 vLLM 和 GPU 支持情况
```

项目目前选择 AWQ，原因是项目部署目标是 vLLM，并且数学任务比较关注量化后的输出质量。

### 5.2 本地没有 NVIDIA GPU

推荐：

```text
合并模型 → GGUF F16 → GGUF Q4_K_M → Ollama
```

本地使用 Ollama 推理时，通常不需要自己安装 PyTorch、Transformers 或 PEFT。Ollama 会调用 llama.cpp 生态的推理能力。

### 5.3 本地有较强 NVIDIA GPU

可以选择：

- 用 AWQ 模型配合支持 AWQ 的 GPU 推理框架；
- 用 GGUF Q4_K_M 配合 Ollama/llama.cpp；
- 显存充足时直接使用 FP16/BF16，减少量化误差。

最终选择取决于显存、吞吐量、并发量、部署接口和对精度的要求。

---

## 6. MathLLM 推荐的两条发布路线

### 路线 A：云端 vLLM 版本

```text
最佳 LoRA checkpoint
        ↓
原始基座 + LoRA → math-lora-merged
        ↓
scripts/quantize.py
        ↓
math-lora-quantized
        ↓
vLLM OpenAI 兼容 API
```

量化命令：

```bash
python scripts/quantize.py \
  --model_path ./outputs/math-lora-merged \
  --output_path ./outputs/math-lora-quantized \
  --calibration_data ./data/processed/train.json \
  --num_calibration_samples 256 \
  --max_seq_length 2048 \
  --dtype float16
```

校准数据只能来自训练数据。不要把 `eval.json` 或独立 `test.json` 用作校准数据，否则会破坏评测边界。

量化完成后，应重新启动 vLLM，并使用和量化前完全相同的测试集、生成参数和评测程序进行对比。

### 路线 B：本地 Ollama 版本

```text
最佳 LoRA checkpoint
        ↓
原始基座 + LoRA → math-lora-merged
        ↓
llama.cpp convert_hf_to_gguf.py
        ↓
mathllm-f16.gguf
        ↓
llama-quantize Q4_K_M
        ↓
mathllm-Q4_K_M.gguf
        ↓
本地 Ollama
```

如果只需要本地运行，最后通常只需要把 `mathllm-Q4_K_M.gguf` 下载到本地，不需要下载完整基座模型和全部训练 checkpoint。

---

## 7. 从合并模型转换为 GGUF

以下步骤建议在云服务器上完成，因为转换过程需要较大的磁盘空间和内存。

### 7.1 准备 llama.cpp

```bash
cd /root/autodl-tmp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

python -m pip install -r requirements.txt
cmake -B build
cmake --build build --config Release -j 8
```

### 7.2 转换 Hugging Face 合并模型

```bash
python convert_hf_to_gguf.py \
  /root/autodl-tmp/MathLLM/outputs/math-lora-merged \
  --outfile /root/autodl-tmp/MathLLM/outputs/mathllm-f16.gguf \
  --outtype f16
```

转换脚本是 llama.cpp 官方提供的 Hugging Face → GGUF 工具。[官方转换脚本](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

如果脚本提示模型架构不支持、tokenizer 不兼容或 Transformers 版本不匹配，应先更新 llama.cpp 及其依赖，不要随意修改模型配置文件。

### 7.3 量化为 Q4_K_M

```bash
./build/bin/llama-quantize \
  /root/autodl-tmp/MathLLM/outputs/mathllm-f16.gguf \
  /root/autodl-tmp/MathLLM/outputs/mathllm-Q4_K_M.gguf \
  Q4_K_M
```

如果可执行文件位于 Release 子目录，则使用：

```bash
./build/bin/Release/llama-quantize \
  /root/autodl-tmp/MathLLM/outputs/mathllm-f16.gguf \
  /root/autodl-tmp/MathLLM/outputs/mathllm-Q4_K_M.gguf \
  Q4_K_M
```

检查输出：

```bash
ls -lh /root/autodl-tmp/MathLLM/outputs/mathllm-*.gguf
```

转换期间要同时保留合并模型、GGUF F16 和最终 Q4 文件，因此云服务器的临时空间不能只按最终 Q4 文件大小准备。

---

## 8. 在本地使用 Ollama

### 8.1 准备文件

在 Windows 本地创建目录：

```text
D:\Models\MathLLM\
├── mathllm-Q4_K_M.gguf
└── Modelfile
```

`Modelfile` 示例：

```text
FROM ./mathllm-Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """你是一个专业的数学解题助手。
请先给出清晰的解题步骤，使用 LaTeX 表达公式，最后给出明确答案。"""
```

`FROM` 指向本地 GGUF 文件，路径可以相对于 `Modelfile`。Ollama 官方文档提供了通过 `FROM` 导入 GGUF 的方式。[Ollama 导入文档](https://docs.ollama.com/import)

### 8.2 创建和运行模型

在 PowerShell 中执行：

```powershell
cd D:\Models\MathLLM
ollama create mathllm -f .\Modelfile
ollama run mathllm
```

模型管理命令：

```powershell
ollama list
ollama show mathllm --modelfile
ollama ps
ollama rm mathllm
```

### 8.3 测试本地模型 API

Ollama 默认监听本机的 `11434` 端口，并提供 OpenAI 风格接口：

```powershell
$body = @{
    model = "mathllm"
    messages = @(
        @{
            role = "system"
            content = "你是数学解题助手"
        }
        @{
            role = "user"
            content = "求解 x^2 - 5x + 6 = 0"
        }
    )
    stream = $false
    temperature = 0
    max_tokens = 512
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  http://localhost:11434/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

如果项目的 FastAPI 后端原来调用 vLLM，可以把后端地址改成：

```text
http://localhost:11434/v1
```

模型名改成：

```text
mathllm
```

Ollama 的 OpenAI 兼容接口说明见官方文档：[OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)

---

## 9. tokenizer 和聊天模板为什么仍然重要？

模型权重本身并不直接理解字符串。推理时需要经过：

```text
用户文本
  ↓ tokenizer
token id
  ↓ 模型
输出 token id
  ↓ tokenizer 解码
自然语言答案
```

在 Transformers 中，输入通常是：

```python
messages = [
    {"role": "system", "content": "你是数学解题助手"},
    {"role": "user", "content": "求 2+2"},
]
```

tokenizer 的聊天模板会把这些消息转换为 Qwen 模型需要的特殊格式。训练和推理必须尽量使用同一套 tokenizer 和聊天模板，否则可能出现：

- 模型把 `<|im_start|>` 等控制符当成普通文本输出；
- 角色边界混乱；
- 模型不停止或回答格式异常；
- 训练时有效，转换后效果明显下降。

转换合并模型时要使用匹配的 Qwen tokenizer。生成 GGUF 时，转换工具通常会把 tokenizer 和聊天模板元数据写进 GGUF；导入 Ollama 后优先使用模型内置模板，不要一开始就手写复杂的 `TEMPLATE`。

---

## 10. 量化前后必须做什么验证？

量化不是“生成文件成功”就代表完成。至少要验证以下三层：

### 10.1 文件和加载验证

- 合并模型存在 `config.json` 和权重文件；
- AWQ 输出存在量化元数据；
- GGUF 文件可以被 `llama-cli` 或 Ollama 加载；
- tokenizer 和聊天模板没有报错。

### 10.2 功能验证

至少测试：

```text
简单算术题
方程题
几何题
带 LaTeX 的题目
多轮对话
空输入和超长输入
```

### 10.3 公平效果对比

使用同一份独立测试集，对以下模型分别评测：

```text
原始基座模型
LoRA 合并模型
AWQ 模型
GGUF Q4_K_M 模型
```

固定以下参数：

```text
相同测试集
相同 prompt 格式
temperature = 0
相同 max_tokens
相同停止条件
```

重点记录：

- 数学答案准确率；
- 步骤正确率；
- 量化前后新增错误数量；
- 平均延迟和首 token 延迟；
- 云端显存、本地内存和 CPU 占用。

MathLLM 的 `eval/` 评测程序应为不同模型使用不同输出目录，避免覆盖已有结果。

---

## 11. 硬件和空间估算

以 7B 模型为例，实际大小会随词表、分片和张量类型变化，以下只是规划用的粗略范围：

| 产物 | 粗略大小 | 主要用途 |
|---|---:|---|
| FP16/BF16 合并模型 | 约 14–16 GB | 合并、转换、最高精度推理 |
| AWQ INT4 | 约 4–6 GB | 云端 GPU/vLLM |
| GGUF F16 | 约 14–16 GB | GGUF 量化中间文件 |
| GGUF Q4_K_M | 约 4–5 GB | 本地 Ollama/CPU |

云服务器转换时还需要同时保存多个中间文件，并需要额外内存。建议至少预留 30GB 以上可用空间，具体以实际模型分片大小为准。

本地运行时通常只需：

- Q4_K_M 文件约 4–5GB；
- Ollama 本身的空间；
- 操作系统、上下文缓存和程序运行所需内存。

8GB 内存可能勉强运行但容易卡顿，16GB 内存更适合作为起点。CPU 可以运行，但速度取决于 CPU 核心数、内存带宽和上下文长度。

---

## 12. 常见错误和处理方式

### 错误一：把 LoRA checkpoint 当完整模型

表现：缺少模型结构或权重，无法单独加载。

处理：准备原始基座模型，先运行 `scripts/merge_lora.py`。

### 错误二：把 AWQ 目录直接交给 Ollama

表现：Ollama 无法识别目录或无法加载 Safetensors 量化元数据。

处理：从 `math-lora-merged` 重新转换为 GGUF，再量化为 `Q4_K_M`。

### 错误三：把 AWQ 再量化为 Q4_K_M

表现：虽然可能生成文件，但模型质量明显下降。

处理：使用未量化的 FP16/BF16 合并模型作为 GGUF 转换输入。只有在明确接受质量风险时，才考虑工具提供的重新量化选项。

### 错误四：量化后角色格式异常

处理顺序：

1. 确认使用了匹配的 Qwen tokenizer；
2. 检查 GGUF/Ollama 中是否有聊天模板；
3. 使用 `ollama show mathllm --modelfile` 检查导入配置；
4. 最后才考虑自定义 `TEMPLATE`。

### 错误五：本地运行很慢或内存不足

处理：

- 减小 `num_ctx`，例如从 4096 改成 2048；
- 关闭同时占用内存的程序；
- 使用 Q4_K_M，而不是 GGUF F16；
- 必要时选择更小的模型或更低量化等级；
- 不要在本地用 PyTorch 加载完整 7B FP16 模型。

---

## 13. MathLLM 的最终产物建议

### 云端部署和复现所需

```text
data/processed/train.json
data/processed/eval.json
data/eval/test.json
configs/training/*/train_config.yaml
LoRA checkpoint
outputs/math-lora-merged/
outputs/math-lora-quantized/
```

### 本地推理所需

```text
mathllm-Q4_K_M.gguf
Modelfile
Ollama
```

### Git 仓库中不建议提交

```text
基座模型权重
AWQ/GGUF 大文件
全部训练 checkpoint
optimizer.pt
云服务器缓存
临时压缩包
```

仓库中保留脚本、配置、文档、少量测试数据和评测报告即可；大模型文件应通过云盘、模型仓库或其他制品存储单独保存。

---

## 14. 最简执行清单

```text
[ ] 选择最佳 LoRA checkpoint
[ ] 用原始基座模型执行 merge_lora.py
[ ] 对合并模型进行未量化加载测试
[ ] 需要 vLLM：用 scripts/quantize.py 生成 AWQ
[ ] 需要本地 CPU：用 llama.cpp 转 GGUF F16
[ ] 从 GGUF F16 生成 Q4_K_M
[ ] 用 llama-cli 或 Ollama 验证生成
[ ] 用同一测试集比较量化前后准确率
[ ] 将最终 Q4_K_M.gguf 下载到本地
[ ] 用 Modelfile 执行 ollama create
[ ] 通过 ollama run 或 localhost:11434/v1 调用
```

## 参考资料

- [llama.cpp 模型转换说明](https://github.com/ggml-org/llama.cpp/blob/master/examples/model-conversion/README.md)
- [llama.cpp 官方 Hugging Face 转换脚本](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
- [llama.cpp 量化工具说明](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Ollama 导入 GGUF 和 Safetensors 文档](https://docs.ollama.com/import)
- [Ollama Modelfile 参考](https://docs.ollama.com/modelfile)
- [Ollama OpenAI 兼容接口](https://docs.ollama.com/api/openai-compatibility)
- [Hugging Face bitsandbytes/NF4 文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [AWQ 原始论文](https://arxiv.org/abs/2306.00978)
- [QLoRA 原始论文](https://arxiv.org/abs/2305.14314)
