# MathLLM - 数学解题大模型

基于 **Qwen2.5-7B LoRA 精调**的数学解题大模型系统，覆盖从数据准备、精调训练、量化部署到应用开发的完整链路。支持高等数学、线性代数、概率论等大学数学题目的自动解答，精调后的小模型在领域任务上接近大模型效果，推理成本降低 95%。

---

## 项目亮点

- **从零精调**：基于 LoRA 对 Qwen2.5-7B 进行领域适配，可训练参数仅 0.1%（~7M），单卡 16GB 显存即可完成
- **量化部署**：INT4 量化后显存占用 ≤ 6GB，基于 vLLM 实现高吞吐推理服务（PagedAttention + Continuous Batching）
- **完整应用**：FastAPI 后端 + Gradio 前端，支持流式输出、多轮对话、LaTeX 公式渲染
- **科学评测**：多维度评测体系（准确率/过程完整性/延迟/幻觉率），支持对比实验与超参数消融分析
- **领域专精**：针对数学解题场景深度优化，解题过程规范、公式格式正确

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 基座模型 | Qwen2.5-7B-Instruct | 阿里通义千问，数学能力强的开源基座 |
| 精调方法 | LoRA (PEFT) | 低秩矩阵适配，高效微调 |
| 训练框架 | Transformers + TRL / LLaMA-Factory | 两套训练方案可选 |
| 量化 | AWQ / bitsandbytes | INT4/INT8 模型压缩 |
| 推理部署 | vLLM | 高性能推理引擎，PagedAttention |
| 后端 API | FastAPI + SSE | OpenAI 兼容接口，流式输出 |
| 前端界面 | Gradio | 支持 LaTeX 渲染的对话界面 |
| 容器化 | Docker | 一键部署 |
| 评测 | 自建评测框架 + matplotlib | 多维度评估 + 可视化对比 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  用户界面 (Gradio)                                          │
│  输入数学题目 → 流式显示解题过程 → LaTeX 公式渲染             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│  应用层 (FastAPI, port 8080)                                │
│  /api/solve 解题接口  |  /api/chat 多轮对话  |  /api/health  │
└──────────────────────────┬──────────────────────────────────┘
                           │ OpenAI 兼容 API
┌──────────────────────────▼──────────────────────────────────┐
│  推理层 (vLLM, port 8000)                                   │
│  精调模型: Qwen2.5-7B-LoRA-Merged (INT4 量化)               │
│  PagedAttention | Continuous Batching | CUDA Graph          │
└─────────────────────────────────────────────────────────────┘
```

训练流程：

```
原始数据 → 数据清洗/格式转换 → LoRA 精调 → 权重合并 → INT4 量化 → vLLM 部署
 (data/raw)  (prepare_data)    (train)    (merge)    (quantize)   (server)
```

---

## 快速开始

### 环境准备

```bash
# 建议使用 Python 3.10+
conda create -n mathllm python=3.10
conda activate mathllm

# 安装依赖
pip install -r requirements.txt

# 安装 LLaMA-Factory（可选，推荐用于训练）
pip install llamafactory
```

### 第一步：准备数据

将数学 QA 数据（JSONL/CSV/JSON）放入 `data/raw/`，然后运行数据预处理：

```bash
python scripts/prepare_data.py
```

处理后会在 `data/processed/` 生成 `train.json` 和 `eval.json`，格式为 ChatML。

**数据来源建议**：
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)：8,500 条小学数学题
- [MATH](https://huggingface.co/datasets/hendrycks/competition_math)：12,500 条竞赛数学题
- 考研数学真题 / 大学高数习题：自行整理
- 用 GPT-4 生成带详细解题过程的题目：500+ 条

LoRA 数据效率高，**500-2000 条高质量数据**即可看到明显效果。

### 第二步：LoRA 精调训练

```bash
# 方式一：使用 LLaMA-Factory CLI（推荐，更稳定）
llamafactory-cli train configs/train_config.yaml

# 方式二：使用本项目训练脚本
python scripts/train.py
```

训练参数详见 `configs/train_config.yaml`，主要配置：
- LoRA rank=16, alpha=32（控制适配器参数量）
- 学习率 2e-4，3 个 epoch
- 混合精度 bf16，max_seq_length=2048

### 第三步：合并 LoRA 权重

将 LoRA adapter 合并回基座模型，得到独立可部署的模型：

```bash
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_path ./outputs/math-lora \
    --output_path ./outputs/math-lora-merged
```

### 第四步：量化压缩（可选）

INT4 量化可将显存占用从 ~14GB 降至 ~4GB：

```bash
python scripts/quantize.py \
    --model_path ./outputs/math-lora-merged \
    --output_path ./outputs/math-lora-quantized \
    --bits 4
```

### 第五步：部署推理服务

```bash
# 启动 vLLM 推理服务
python deploy/server.py

# 服务启动后，可测试接口：
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"math-solver","messages":[{"role":"user","content":"求解 x^2-5x+6=0"}]}'
```

### 第六步：启动应用

```bash
# 启动 FastAPI 后端 (port 8080)
python -m app.api.main

# 启动 Gradio 前端 (port 7860)
python app/frontend/gradio_app.py
```

打开浏览器访问 `http://localhost:7860` 即可使用。

### 第七步：评测

```bash
# 运行评测
python eval/evaluate.py \
    --model_endpoint http://localhost:8000/v1 \
    --eval_data ./data/processed/eval.json \
    --output ./eval/results

# 运行消融实验（对比不同超参数）
python eval/ablation.py --config configs/ablation.yaml
```

---

## 数据格式说明

**原始数据**（`data/raw/` 中的 JSONL）：
```jsonl
{"question": "求函数 f(x)=x³-3x+1 的极值点", "solution": "对 f(x) 求导得 f'(x)=3x²-3，令 f'(x)=0 得 x=±1...", "answer": "x=1 为极小值点, x=-1 为极大值点"}
```

**训练数据**（`data/processed/` 中的 ChatML 格式）：
```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的数学解题助手..."},
    {"role": "user", "content": "求函数 f(x)=x³-3x+1 的极值点"},
    {"role": "assistant", "content": "**分析**\n对 $f(x)$ 求导...\n**最终答案**\n$x=1$ 为极小值点，$x=-1$ 为极大值点"}
  ]
}
```

---

## 评测指标

| 维度 | 方法 | 说明 |
|------|------|------|
| 答案准确率 | 精确匹配 + 数值容差 | 最终答案是否正确 |
| 过程完整性 | LLM-as-Judge (1-5分) | 解题步骤是否完整合理 |
| 推理延迟 | 首 Token 延迟 + 总耗时 | 用户体验响应速度 |
| 幻觉率 | 错误公式/定理检测 | 是否编造不存在的内容 |
| LaTeX 规范性 | 公式格式检查 | 数学公式书写是否规范 |

---

## 项目目录结构

```
MathLLM/
├── data/
│   ├── raw/                     # 原始数据集
│   └── processed/               # 处理后的训练/验证数据
├── scripts/
│   ├── prepare_data.py          # 数据加载、清洗、格式转换、划分
│   ├── train.py                 # LoRA 精调训练
│   ├── merge_lora.py            # 合并 LoRA adapter 到基座模型
│   └── quantize.py              # INT4/INT8 量化压缩
├── deploy/
│   ├── server.py                # vLLM 推理服务启动器
│   └── Dockerfile               # Docker 部署配置
├── app/
│   ├── api/
│   │   └── main.py              # FastAPI 后端（解题/对话/健康检查）
│   └── frontend/
│       └── gradio_app.py        # Gradio Web 界面
├── eval/
│   ├── evaluate.py              # 多维度模型评测
│   └── ablation.py              # 超参数消融实验
├── configs/
│   ├── train_config.yaml        # 训练超参数配置
│   └── deploy_config.yaml       # 部署参数配置
├── docs/
│   └── design.md                # 详细设计文档
├── requirements.txt
└── README.md
```

---

## 硬件需求

| 环节 | 最低显存 | 推荐方案 | 预估成本 |
|------|---------|---------|---------|
| LoRA 精调 | 16GB | AutoDL 租 A10 (~2元/h) | ~10元 |
| INT4 部署 | 8GB | AutoDL 租 T4 (~1元/h) | ~5元 |
| 应用开发 | 无 GPU | 本地笔记本 | 0元 |
| **合计** | | | **30-50元** |

---

## 开发路线

- [ ] **Week 1**：收集数学 QA 数据集 + 数据预处理
- [ ] **Week 1**：LoRA 精调训练 + Loss 曲线观察
- [ ] **Week 2**：模型合并 + INT4 量化 + vLLM 部署
- [ ] **Week 3**：FastAPI 后端 + Gradio 前端开发
- [ ] **Week 3**：多维度评测 + 对比实验（vs 基座模型 vs GPT-4）
- [ ] **Week 4**：消融实验（rank/学习率/数据量）
- [ ] **Week 4**：Docker 打包 + 云端部署 + 文档整理

---

## 文档索引

- [详细设计文档](docs/design.md)：架构设计、LoRA 原理、vLLM 优化原理、评测体系、面试话术
- [训练配置说明](configs/train_config.yaml)
- [部署配置说明](configs/deploy_config.yaml)
