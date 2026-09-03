# MathLLM：数学解题大模型

MathLLM 是一个面向大学数学解题场景的端到端项目。项目以 **Qwen2.5-7B-Instruct** 为基座模型，使用 LoRA/QLoRA 进行领域精调，目标是完成数据准备、训练、权重合并、量化、vLLM 部署、Web 应用和可复现评测。

## 当前进展

| 模块 | 状态 | 当前情况 |
|---|---|---|
| 数据下载与整理 | 已完成第一版 | GSM8K 500 条、Hendrycks MATH 500 条、中文 CMID 250 条，加上已有数据共 1360 条原始记录 |
| 数据预处理 | 已完成 | 支持 JSON/JSONL/CSV，完成字段统一、ChatML 转换、长度检查、去重和 90/10 划分 |
| 正式数据集 | 已生成 | 清洗后 1259 条：训练集 1133 条，评估集 126 条 |
| 基座模型 | 已准备 | 本地目录为 `models/Qwen2.5-7B-Instruct-modelscope/`，模型文件不提交 Git |
| LoRA/QLoRA 训练 | 已实现 | `scripts/train.py` 使用 Transformers + PEFT + TRL，已完成云端小规模试跑 |
| LoRA 合并 | 已实现 | `scripts/merge_lora.py` 可将 adapter 合并为独立模型 |
| INT4/AWQ 量化 | 待实现 | `scripts/quantize.py` 目前仍是接口骨架 |
| vLLM 服务 | 启动器已实现 | `deploy/server.py` 可启动 OpenAI 兼容服务，但正式量化模型尚未验证 |
| 自动评测 | 基础功能已实现 | `eval/evaluate.py` 支持答案、过程分、错误和延迟记录；`eval/llm_judge.py` 支持独立模型语义评测 |
| 消融实验 | 框架已实现 | `eval/ablation.py` 支持按配置运行，自动数学评测仍需配置具体命令 |
| FastAPI/Gradio 应用 | 待完成 | 当前为接口和界面骨架，仍有 TODO |

## 项目流程

```text
公开数学数据 + 人工纠错数据
          ↓
scripts/download_formal_data.py
          ↓
scripts/prepare_data.py
          ↓
ChatML train.json / eval.json
          ↓
LoRA/QLoRA SFT
          ↓
scripts/merge_lora.py
          ↓
FP16/BF16 独立模型
          ↓
AWQ/INT4 量化（待实现）
          ↓
vLLM OpenAI-compatible API
          ↓
FastAPI + Gradio 应用
```

## 环境安装

建议在云端 CUDA GPU 服务器上训练、合并和部署；本地无 NVIDIA GPU 时只适合进行数据处理、代码检查或 CPU 推理。

```bash
conda create -n mathllm python=3.10
conda activate mathllm
pip install -r requirements.txt
```

训练至少需要 PyTorch、Transformers、PEFT、TRL、Datasets 和 Accelerate。vLLM、bitsandbytes 和 AWQ 相关依赖应根据云服务器 CUDA、驱动和 vLLM 版本单独确认。

## 数据准备

### 下载正式数据

下载脚本只负责获取可复现的源数据子集，原始文件保存在 `data/raw/`：

```bash
python scripts/download_formal_data.py
```

当前使用的数据来源：

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)：英文小学数学应用题；
- [Hendrycks MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math)：代数、几何、数论、概率等竞赛数学题；
- [CMID Chinese Math Instruct](https://huggingface.co/datasets/Mxode/CMID-Chinese_Math_Instruct_Dataset)：中文数学指令数据。

数据来源、抽样数量和许可证记录在 `data/raw/SOURCES.md`。原始数据体积和许可证可能随上游数据集变化，正式训练前应保留下载日期和版本信息。

### 统一格式、清洗和划分

所有原始数据都必须经过 [scripts/prepare_data.py](scripts/prepare_data.py)。它会：

1. 读取 `data/raw/` 下的 JSON、JSONL 和 CSV；
2. 兼容 GSM8K、MATH、CMID 和项目自定义字段；
3. 统一为 `question`、`solution`、`answer` 等内部字段；
4. 转换为 Qwen ChatML 风格的 `messages`；
5. 检查角色顺序、空内容、过短回答、非法图形标记和 2048 token 长度；
6. 删除完全重复、模板重复和高相似题目；
7. 按固定随机种子划分训练集和评估集。

运行命令：

```bash
python scripts/prepare_data.py \
  --tokenizer ./models/Qwen2.5-7B-Instruct-modelscope \
  --max-tokens 2048 \
  --eval-ratio 0.1 \
  --seed 42
```

输出文件：

```text
data/processed/train.json
data/processed/eval.json
```

最终每条记录的基本结构如下，训练时不要手动添加 `<|im_start|>` 或 `<|im_end|>`：

```json
{
  "messages": [
    {"role": "system", "content": "数学解题系统提示词"},
    {"role": "user", "content": "求解 x^2 - 5x + 6 = 0。"},
    {"role": "assistant", "content": "先分析题目并给出步骤。\\n\\n**最终答案**\\n\\nx=2 或 x=3"}
  ]
}
```

本次正式数据准备结果：原始记录 1360 条，规范化 1264 条，去除重复或近似重复 5 条，最终保留 1259 条；其中 96 条因 `[asy]...[/asy]` 图形标记或无法可靠提取最终答案而被过滤。程序检查不等于数学证明，正式训练前仍应人工抽查至少 10%。

## LoRA/QLoRA 训练

训练配置位于 [configs/train_config.yaml](configs/train_config.yaml)：

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  max_seq_length: 2048
```

启动训练：

```bash
python scripts/train.py --config configs/train_config.yaml
```

默认使用标准 LoRA。24GB 显存不足时，可以将 `training.use_4bit` 改为 `true` 使用 QLoRA。训练输出默认写入 `outputs/math-lora/`，其中包括 adapter、Tokenizer、checkpoint、训练状态和 loss 记录。

LoRA 训练不会改变基座模型权重，训练输出主要是 adapter 权重和配置文件。部署前需要将 adapter 合并到基座模型，或让推理框架同时加载基座模型和 adapter。

## 模型合并

[scripts/merge_lora.py](scripts/merge_lora.py) 已实现基座模型与 LoRA adapter 的合并：

```bash
python scripts/merge_lora.py \
  --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
  --lora_path ./outputs/math-lora \
  --output_path ./outputs/math-lora-merged \
  --dtype float16
```

合并阶段必须加载完整精度的基座模型。输出是一个可以直接被 Transformers 或 vLLM 加载的独立模型，不等于量化模型，体积通常接近原始 FP16 模型。

## 量化与 vLLM 部署

当前 [scripts/quantize.py](scripts/quantize.py) 还没有完成实际量化逻辑，因此暂时不要把 `int4` 当作已经生成的模型格式。推荐的后续顺序是：

```text
LoRA adapter
  → FP16/BF16 合并模型
  → AWQ 或 GPTQ 校准量化
  → 用相同 eval.json 对比量化前后准确率和延迟
  → vLLM 部署
```

部署配置位于 [configs/deploy_config.yaml](configs/deploy_config.yaml)。量化完成后，需要把 `model.model_path` 改为实际量化模型目录；如果输出格式是 AWQ，应使用 vLLM 支持的 `awq` 量化参数，而不是笼统的 `int4`。

启动器：

```bash
python deploy/server.py
```

也可以直接使用 vLLM 的 OpenAI 兼容服务命令：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/math-lora-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096
```

服务启动后检查：

```bash
curl http://localhost:8000/v1/models
```

## 评测

### 基础评测

[eval/evaluate.py](eval/evaluate.py) 调用 OpenAI 兼容接口，记录最终答案、过程分、错误、首 token 延迟和总延迟：

```bash
python eval/evaluate.py \
  --model_endpoint http://localhost:8000/v1 \
  --eval_data data/processed/eval.json \
  --output eval/results/finetuned-model \
  --temperature 0 \
  --max_tokens 512
```

错误样本和详细结果会保存在输出目录的 `bad_cases.json`、`details.json` 和 `report.json` 中。基础评测是确定性启发式评测，不能完全证明数学推理正确性。

### 独立模型语义评测

[eval/llm_judge.py](eval/llm_judge.py) 使用另一个 OpenAI 兼容模型判断 `correct`、`incorrect` 或 `uncertain`：

```bash
python eval/llm_judge.py \
  --details eval/results/finetuned-model/details.json \
  --eval-data data/processed/eval.json \
  --output eval/results/finetuned-model/llm-judge \
  --judge-endpoint https://api.example.com/v1 \
  --judge-model your-judge-model
```

`details.json` 和 `eval.json` 必须来自同一轮评测、使用同一版本的数据。当前脚本会校验题目是否一致，避免数据集重新生成后按序号错配。`uncertain` 和 `incorrect` 样本必须人工复核。

### 回归评测和消融

真实错误题不要直接加入训练集。纠错样本放入 `data/raw/corrections/`，原错误题和同类验证题放入 `data/eval/regression/`，然后单独评测：

```bash
python eval/evaluate.py \
  --model_endpoint http://localhost:8000/v1 \
  --eval_data data/eval/regression/regression.json \
  --output eval/results/regression
```

消融实验配置位于 [configs/ablation.yaml](configs/ablation.yaml)，运行器位于 [eval/ablation.py](eval/ablation.py)。所有实验应固定数据集、随机种子和评测参数。

## 项目结构

```text
MathLLM/
├── app/
│   ├── api/
│   │   └── main.py                 # FastAPI 接口骨架
│   └── frontend/
│       └── gradio_app.py           # Gradio 界面骨架
├── configs/
│   ├── train_config.yaml           # LoRA/QLoRA 训练配置
│   ├── deploy_config.yaml          # vLLM 部署配置
│   ├── ablation.yaml               # 消融实验配置
│   └── judge.env.example            # LLM judge 环境变量示例
├── data/
│   ├── raw/                        # 原始数据，本地保存，不提交 Git
│   ├── processed/                  # train.json 和 eval.json
│   └── eval/regression/            # 回归题和回归说明
├── deploy/
│   ├── server.py                   # vLLM 启动器
│   └── Dockerfile                  # 部署镜像骨架
├── docs/
│   ├── development-plan/           # 四周开发计划
│   ├── experiments/                # 已完成实验日志
│   └── learning/                   # LoRA、QLoRA、量化、vLLM 等学习笔记
├── eval/
│   ├── evaluate.py                 # 接口、准确率和延迟评测
│   ├── llm_judge.py                # 独立大模型语义评测
│   ├── ablation.py                 # 消融实验运行器
│   └── results/                    # 实验输出和评测报告
├── scripts/
│   ├── download_formal_data.py     # 下载和抽样正式数据
│   ├── prepare_data.py             # 清洗、格式化、去重、划分
│   ├── train.py                    # LoRA/QLoRA SFT 训练
│   ├── merge_lora.py               # 合并 LoRA adapter
│   ├── quantize.py                 # 量化接口骨架
│   └── loss_curve.py               # 导出训练 loss 曲线
├── .gitignore
├── requirements.txt
└── README.md
```

## Git 与模型文件

以下内容不应提交到 GitHub：

- `models/`：Qwen2.5-7B 基座模型，体积约十几 GB；
- `outputs/`、`checkpoints/`、`lora_weights/`：训练和模型产物；
- `data/raw/`：原始数据集文件；
- 临时上传压缩包和本地评测重跑目录。

仓库主要保存代码、配置、文档、回归样例以及可控体积的处理后数据。其他机器上应先下载基座模型，再运行数据准备和训练流程。

## 后续计划

- [x] 完成正式数据下载、清洗、去重和 ChatML 转换
- [x] 生成当前版本训练集和评估集
- [x] 完成 LoRA/QLoRA 训练程序
- [x] 完成 LoRA 合并程序
- [x] 完成基础接口评测和独立模型评测框架
- [ ] 完成人工 10% 数据抽检和数学答案复核
- [ ] 实现 AWQ/GPTQ 量化并记录校准配置
- [ ] 对比合并模型与量化模型的准确率、显存和延迟
- [ ] 完成 vLLM 正式部署验证
- [ ] 完成 FastAPI 和 Gradio 联调
- [ ] 完善消融实验的自动评测命令
- [ ] 完成 Docker 部署和最终发布文档

## 文档索引

- [开发总指南](docs/development-guide.md)
- [项目设计](docs/design.md)
- [第一周：数据准备](docs/development-plan/week1-data-preparation.md)
- [第二周：训练与部署](docs/development-plan/week2-training-and-deployment.md)
- [第三周：应用开发](docs/development-plan/week3-application-development.md)
- [第四周：评测与发布](docs/development-plan/week4-evaluation-and-release.md)
- [评测说明](eval/README.md)
- [训练配置](configs/train_config.yaml)
- [部署配置](configs/deploy_config.yaml)
- [LoRA 原理](docs/learning/lora-principle.md)
- [QLoRA 与 LoRA 对比](docs/learning/qlora-vs-lora.md)
- [INT4 量化原理](docs/learning/int4-quantization.md)
- [ChatML 格式](docs/learning/chatml-format.md)
- [vLLM PagedAttention](docs/learning/vllm-pagedattention.md)
- [消融实验方法](docs/learning/ablation-study.md)
