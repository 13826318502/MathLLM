# MathLLM - 数学解题大模型

MathLLM 是一个面向大学数学解题场景的端到端大模型项目，计划基于 **Qwen2.5-7B-Instruct**，通过 LoRA 进行数学领域适配，并完成数据准备、训练、模型合并、量化、部署、应用和评测。

当前项目已完成数据处理管道和基座模型准备，正在进入 LoRA 训练阶段。训练、模型合并、量化、推理服务和评测代码仍需要继续实现和验证。

## 当前进展

| 模块 | 当前状态 | 说明 |
|------|----------|------|
| 原始数据 | 已准备 | 数据位于本地 `data/raw/`，原始数据文件不提交到 Git |
| 数据预处理 | 已完成 | `scripts/prepare_data.py` 已实现加载、清洗、去重、格式转换和划分 |
| 训练数据 | 已生成 | 当前 `train.json` 89 条，`eval.json` 10 条 |
| 基座模型 | 已下载 | Qwen2.5-7B-Instruct，4 个 Safetensors 权重分片，约 14.19 GiB |
| 训练配置 | 已配置 | 使用本地模型目录和 LoRA 参数 |
| LoRA 训练 | 已实现，待云 GPU 验证 | `scripts/train.py` 已实现模型加载、ChatML 处理、LoRA 挂载和 SFT 训练 |
| LoRA 合并 | 待实现 | `scripts/merge_lora.py` 中的合并逻辑仍有 TODO |
| INT4 量化 | 待实现 | `scripts/quantize.py` 中的量化逻辑仍有 TODO |
| vLLM 部署 | 待验证 | `deploy/server.py` 已提供启动器，但需要先得到可用模型 |
| Web 应用 | 待实现 | FastAPI 和 Gradio 接口中仍有 TODO |
| 自动评测 | 待实现 | 评测和消融实验脚本仍是框架代码 |

## 模型信息

- 基座模型：`Qwen/Qwen2.5-7B-Instruct`
- 模型架构：`Qwen2ForCausalLM`
- Tokenizer：`Qwen2Tokenizer`
- 微调方法：LoRA（PEFT）
- 训练最大长度：2048 tokens
- 计划部署方式：LoRA 合并后进行 INT4 量化，再使用 vLLM 部署

已下载的模型位于本地：

```text
D:\MathLLM\models\Qwen2.5-7B-Instruct-modelscope
```

模型权重较大，已通过 `.gitignore` 排除，不会提交到 GitHub。其他环境需要从 [ModelScope Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) 单独下载模型。

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 基座模型 | Qwen2.5-7B-Instruct | 通用指令模型 |
| 精调方法 | LoRA / PEFT | 数学领域适配 |
| 训练组件 | Transformers、TRL、LLaMA-Factory | 模型训练 |
| 量化组件 | AWQ / bitsandbytes | INT4/INT8 压缩 |
| 推理服务 | vLLM | 高性能模型推理 |
| 后端 | FastAPI、SSE | API 服务和流式输出 |
| 前端 | Gradio | 数学解题交互界面 |
| 数据处理 | Python、JSONL、ChatML | 数据清洗和格式转换 |
| 评测 | Python、matplotlib | 准确率、延迟和对比实验 |

## 目标架构

```text
用户
  ↓
Gradio 前端
  ↓ HTTP / SSE
FastAPI 后端
  ↓ OpenAI 兼容 API
vLLM 推理服务
  ↓
Qwen2.5-7B 数学领域适配模型
```

目标训练流程：

```text
原始数据
  → 数据清洗与 ChatML 格式转换
  → LoRA 精调
  → 合并 LoRA 权重
  → INT4 量化
  → vLLM 部署
  → FastAPI + Gradio 应用
  → 自动评测与消融实验
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.10 或更高版本：

```bash
conda create -n mathllm python=3.10
conda activate mathllm
pip install -r requirements.txt
```

如果使用 LLaMA-Factory，需要额外安装：

```bash
pip install llamafactory
```

### 2. 准备数据

将原始 JSON、JSONL 或 CSV 数据放入 `data/raw/`，运行：

```bash
python scripts/prepare_data.py
```

输出文件：

```text
data/processed/train.json
data/processed/eval.json
```

如果需要按照 Qwen tokenizer 检查 token 长度，可以运行：

```bash
python scripts/prepare_data.py \
  --tokenizer ./models/Qwen2.5-7B-Instruct-modelscope \
  --max-tokens 2048
```

### 3. 验证数据

```bash
python -c "import json; x=json.load(open('data/processed/train.json',encoding='utf-8')); print('train:',len(x)); print(x[0])"
python -c "import json; x=json.load(open('data/processed/eval.json',encoding='utf-8')); print('eval:',len(x)); print(x[0])"
```

每条记录应具有如下结构：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的数学解题助手。"},
    {"role": "user", "content": "求解 x^2 - 5x + 6 = 0。"},
    {"role": "assistant", "content": "令...因此 x=2 或 x=3。"}
  ]
}
```

### 4. 训练模型

训练配置位于 [`configs/train_config.yaml`](configs/train_config.yaml)，当前使用本地基座模型：

```yaml
model_name_or_path: "./models/Qwen2.5-7B-Instruct-modelscope"
```

LoRA 当前配置为：

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
```

当前 `scripts/train.py` 已实现基于 Transformers + PEFT + TRL 的 LoRA SFT 训练流程。训练需要在有 CUDA GPU 的云服务器上运行：

```bash
python scripts/train.py
```

脚本会读取 `messages` 数据并应用 Qwen chat template，默认使用标准 LoRA（基座模型以 BF16/FP16 加载并冻结），训练完成后把 adapter 保存到 `outputs/math-lora/`。如果显存不足，可在 `configs/train_config.yaml` 的 `training` 下设置 `use_4bit: true`，切换为 QLoRA。

建议先用 1 个 epoch、batch size 1 做小规模试跑；确认 Loss、checkpoint 和 adapter 文件正常后，再恢复正式配置。

### 5. 模型合并、量化和部署

以下目录和脚本是计划中的后续产物，目前尚未生成或尚未完整实现：

```text
outputs/math-lora/
outputs/math-lora-merged/
outputs/math-lora-quantized/
```

计划流程：

```bash
python scripts/merge_lora.py \
  --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
  --lora_path ./outputs/math-lora \
  --output_path ./outputs/math-lora-merged

python scripts/quantize.py \
  --model_path ./outputs/math-lora-merged \
  --output_path ./outputs/math-lora-quantized \
  --bits 4

python deploy/server.py
```

这些命令需要等对应脚本实现完成后再执行。

## 数据格式

原始数据位于 `data/raw/`，典型格式为：

```json
{
  "question": "求函数 f(x)=x^3-3x+1 的极值点",
  "solution": "对 f(x) 求导，得到 f'(x)=3x^2-3...",
  "answer": "x=1 为极小值点，x=-1 为极大值点"
}
```

预处理后统一转换为 ChatML 风格的 `messages` 数据，供 Qwen 的 chat template 使用。

## 项目结构

```text
MathLLM/
├── app/                              # 应用层
│   ├── api/
│   │   └── main.py                   # FastAPI 接口骨架
│   └── frontend/
│       └── gradio_app.py             # Gradio 前端骨架
├── configs/                          # 配置文件
│   ├── train_config.yaml             # 基座模型和 LoRA/训练参数
│   └── deploy_config.yaml             # vLLM 部署参数
├── data/                             # 数据目录（数据文件本地使用）
│   ├── raw/                          # 原始数学数据
│   └── processed/                    # train.json 和 eval.json
├── deploy/                           # 部署相关代码
│   ├── server.py                     # vLLM 服务启动器
│   └── Dockerfile                    # Docker 配置
├── docs/                             # 项目文档
│   ├── development-guide.md          # 开发总指南
│   ├── design.md                     # 架构和方案设计
│   ├── development-plan/
│   │   ├── week1-data-preparation.md # 第一周：数据准备
│   │   └── week2-training-and-deployment.md # 第二周：训练和部署计划
│   └── learning/                     # LoRA、QLoRA、量化、vLLM 等学习笔记
├── eval/                             # 评测和消融实验骨架
│   ├── evaluate.py                   # 多维度评测
│   └── ablation.py                   # 超参数消融实验
├── models/                           # 本地模型目录，不提交 Git
│   └── Qwen2.5-7B-Instruct-modelscope/ # Qwen 基座模型
├── scripts/                          # 数据和模型处理脚本
│   ├── prepare_data.py               # 数据加载、清洗、转换、划分
│   ├── train.py                      # LoRA 训练骨架
│   ├── merge_lora.py                 # LoRA 合并骨架
│   └── quantize.py                   # INT4/INT8 量化骨架
├── .gitignore                        # 忽略模型、数据和训练输出
├── requirements.txt                  # Python 依赖
└── README.md                         # 项目说明
```

## Git 与模型文件

模型权重不放入 Git 仓库，原因是：

- Qwen2.5-7B 权重约 14GB，远超普通 GitHub 文件限制；
- Git 仓库应主要保存代码、配置和文档；
- 模型应通过 ModelScope、Hugging Face 或对象存储单独分发；
- 项目通过 `configs/train_config.yaml` 指定本地模型路径。

因此，提交代码时不要使用会把模型目录加入暂存区的方式。当前 `.gitignore` 已忽略 `models/`、`*.safetensors`、`outputs/` 和训练数据文件。

## 后续计划

- [x] 完成原始数据读取、清洗、去重和 ChatML 格式转换
- [x] 生成训练集和验证集
- [x] 下载并验证 Qwen2.5-7B-Instruct 基座模型
- [x] 配置本地模型路径和 LoRA 超参数
- [ ] 补全 LoRA 训练脚本并完成小规模试跑
- [ ] 观察 Train Loss 和 Eval Loss
- [ ] 实现 LoRA 权重合并
- [ ] 完成 INT4 量化
- [ ] 启动并验证 vLLM 推理服务
- [ ] 完成 FastAPI 和 Gradio 联调
- [ ] 实现自动评测和消融实验
- [ ] 完善 Docker 部署和项目文档

## 文档索引

- [开发总指南](docs/development-guide.md)
- [详细设计文档](docs/design.md)
- [第一周：数据准备](docs/development-plan/week1-data-preparation.md)
- [第二周：训练与部署计划](docs/development-plan/week2-training-and-deployment.md)
- [训练配置](configs/train_config.yaml)
- [部署配置](configs/deploy_config.yaml)
- [LoRA 原理](docs/learning/lora-principle.md)
- [QLoRA 与 LoRA 对比](docs/learning/qlora-vs-lora.md)
- [INT4 量化原理](docs/learning/int4-quantization.md)
- [ChatML 数据格式](docs/learning/chatml-format.md)
- [vLLM PagedAttention](docs/learning/vllm-pagedattention.md)
- [Loss 曲线解读](docs/learning/loss-curve.md)
- [消融实验方法](docs/learning/ablation-study.md)
