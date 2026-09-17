# MathLLM：数学解题大模型

MathLLM 是一个面向大学数学解题场景的端到端项目。项目以
**Qwen2.5-7B-Instruct** 为基座模型，使用 Transformers、TRL 和 PEFT
进行 LoRA/QLoRA 监督微调，包含数据准备、训练、LoRA 合并、评测、量化
和 vLLM 部署流程。

## 项目流程

```text
原始数据 + 人工核验的纠错样本
          ↓
data/eval/test.json（先固定，不能参与训练）
          ↓
scripts/prepare_data.py
          ↓
data/processed/train.json + eval.json
          ↓
scripts/train.py（LoRA/QLoRA）
          ↓
scripts/merge_lora.py
          ↓
FP16/BF16 独立模型
          ↓
同一 test.json 做最终评测
          ↓
scripts/quantize.py（可选 AWQ W4A16）
          ↓
vLLM OpenAI 兼容接口
```

核心原则：训练集用于学习，验证集用于选择 checkpoint 和调参，独立测试集
只在最终比较时使用，回归集用于检查已经发现的错误是否再次出现。

数据目录的分类、用途、训练/评测边界以及错误回归流程，详见
[数据分类与错误回归集说明](docs/experiments/data-classification-and-regression-guide.md)。

## 两种用法：完整版与纯功能版

本仓库同时包含**训练流程**和**应用功能**，两者可以分开使用。

| | 完整版 | 纯功能版 |
|---|---|---|
| 解题模型 | 本地微调的 `mathllm-round7`（Ollama / vLLM） | 任意 OpenAI 兼容接口（如 `deepseek-chat`） |
| 需要训练 | 是（数据准备 → LoRA → 合并 → 量化） | **否** |
| 需要 GPU | 训练与本地推理需要 | 不需要 |
| 依赖 | `requirements.txt` + 训练依赖（torch / peft / trl / vllm，需另装） | 只需 8 个运行期依赖（见下） |
| 可用功能 | 全部 | Agent 路由、知识库检索、SymPy 答案验证、运行观测、Web 前端 |

**如果只想用数学问答功能，不需要训练任何模型**——把「解题模型」也指向任意 OpenAI 兼容接口即可。

### 只保留功能模块（删掉模型微调相关内容）

纯功能版**必须保留**的文件：

```text
app/                  服务与 Agent 层（路由、工具、验证、观测）
knowledge/            知识库文档，RAG 数据源
web/                  独立前端（Agent 模式、运行观测页面）
tests/                单元测试（138 个，全部离线）
README.md             本说明
requirements.txt      运行期依赖（8 个，不含 torch）
.gitignore
configure.bat         模型配置弹窗
start_local.bat       启动入口
start_local.ps1       启动脚本
.env.local.example    配置模板
```

**只删除「模型微调」相关的目录和文件，其他一律不要删。** 可安全删除的清单：

| 路径 | 为什么可以删 |
|---|---|
| `configs/` | LoRA/QLoRA 训练与消融配置 |
| `scripts/` | 数据准备、训练、LoRA 合并、量化脚本 |
| `eval/` | 微调模型的评测脚本与评测结果 |
| `data/raw/`、`data/processed/`、`data/eval/`、`data/candidates/`、`data/behavior/`、`data/archive/` | 训练与评测数据集 |
| `docs/` | 训练/评测报告、微调与量化学习笔记、开发计划等文档 |

在仓库根目录执行（PowerShell）：

```powershell
Remove-Item -Recurse -Force configs, scripts, eval, docs
Remove-Item -Recurse -Force data/raw, data/processed, data/eval, data/candidates, data/behavior, data/archive
```

**不要删**——删了功能版会跑不起来或需要重建：

| 路径 | 原因 |
|---|---|
| `data/chroma/` | 知识库向量索引，RAG 检索依赖它；删了需重新执行 `python -m app.services.rag_service` |
| `data/traces/` | 运行轨迹落盘目录，运行时自动生成 |
| `deploy/` | vLLM 部署启动器；功能版用不到，但不属于微调，建议保留 |

> 想删掉微调内容又保持 git 工作区干净，可以用 sparse-checkout：仓库内容不变，
> 只是本地不检出这些文件，`git status` 依然干净，随时可恢复。
>
> ```powershell
> git sparse-checkout set --no-cone "/*" "!/configs/" "!/scripts/" "!/eval/" "!/data/" "!/docs/"
> git sparse-checkout disable   # 需要恢复时执行
> ```

### 安装与运行（纯功能版）

运行期依赖只有 8 个包，**不需要 torch**：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn sse-starlette pydantic httpx chromadb fastembed sympy
```

建立知识库索引，然后启动：

```powershell
python -m app.services.rag_service   # 首次会下载中文 embedding 模型（约 95MB）
start_local.bat                      # 首次运行会弹出模型配置窗口
```

在配置窗口里把**两组模型都指向同一个 OpenAI 兼容接口**即可：

```text
MATHLLM_ORCHESTRATOR_BASE_URL=https://api.deepseek.com/v1
MATHLLM_ORCHESTRATOR_MODEL=deepseek-chat
MATHLLM_ORCHESTRATOR_API_KEY=sk-xxx

MATHLLM_SOLVER_BASE_URL=https://api.deepseek.com/v1
MATHLLM_SOLVER_MODEL=deepseek-chat
MATHLLM_SOLVER_API_KEY=sk-xxx
```

浏览器打开 `http://localhost:7860`，用「Agent 模式」提问，「运行观测」页面查看调用链与指标。

### 验证纯功能版可用

```powershell
& ".venv\Scripts\python.exe" -m unittest discover -s tests
```

138 个单元测试**全部离线**（模型用脚本化假客户端注入，embedding 用确定性哈希替代），
不需要任何模型或 API Key 就能跑通。

## 项目目标与产品定位

本项目当前聚焦于“上下文感知的可靠对话”，而不是继续训练固定的数学五字段格式。
模型需要结合完整对话判断：信息足够时直接回答，能够从上下文补全省略和指代，只有
在仍缺少关键条件时才提出精确的澄清问题。

本轮行为目标可以表示为：

```text
完整对话 + 当前输入
          ↓
解析指代、承接和约束
          ↓
信息足够 ──→ 继续回答
信息不足 ──→ 询问缺少的关键条件
存在歧义 ──→ 说明歧义并请求选择
```

数学推理和普通聊天都要覆盖，避免模型看到任何简短追问都机械地要求补充信息。
该行为数据位于 `data/behavior/clarification-round-1/`，独立基线测试集位于
`data/eval/behavior/clarification-round-1-test.jsonl`。在进行 LoRA 训练前，先用原始
基座模型完成基线测试；只有确认基座模型在上下文承接、必要澄清和避免无依据猜测上
存在稳定问题，才进入微调。

## 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 正式数据整理 | 已完成当前版本 | GSM8K、Hendrycks MATH、CMID 和纠错样本已放入本地数据目录 |
| 数据预处理 | 已完成 | 支持 JSON、JSONL、CSV，包含字段统一、ChatML 转换、清洗、去重和划分 |
| 独立测试集 | 已生成 | `data/eval/test.json`，当前 116 条 |
| 当前训练数据 | 按实验轮次管理 | 当前轮次使用 `configs/training/` 中对应配置；历史数据和配置已归档 |
| 高难度对比集 | 已生成 | `data/candidates/base-error-directions-140-hard-eval.json`，140 条 |
| LoRA/QLoRA 训练 | 已实现 | `scripts/train.py` 使用 Transformers + TRL + PEFT |
| LoRA 合并 | 已实现 | `scripts/merge_lora.py` 输出独立模型 |
| loss 记录 | 已实现 | 保存 train/eval loss、CSV、JSON 和曲线图 |
| 基础模型评测 | 已实现 | 记录数学准确率、逐题答案、错题、平均延迟和首 token 延迟 |
| LLM Judge | 已实现 | `eval/llm_judge.py` 对模型答案做独立语义判断 |
| AWQ 量化 | 脚本已实现 | `scripts/quantize.py`；本地 CPU 使用已经生成的 GGUF，AWQ/GPTQ 仍建议云 GPU 验证 |
| vLLM 部署 | 已实现启动器 | `deploy/server.py` 可启动 OpenAI 兼容服务 |
| Web 应用 | 已实现 | 独立 HTML/CSS/JavaScript 前端 + FastAPI/SSE，含 Agent 模式与运行观测页面 |
| Agent 层 | 已实现 | 结构化路由 + 工具层 + 有界 ReAct 循环 + 本地 RAG + SymPy 答案验证 |
| 模型分工 | 已实现 | 编排走云端（可留空关闭）、数学求解走本地，云端不可用自动回退 |
| 可观测性 | 已实现 | 每次运行落盘 JSONL（`data/traces/`），`/api/metrics` 聚合运行指标 |

## 已记录的 Round7 训练与评测

Round7 以 Round6 合并模型为基座，使用 QLoRA 进行一轮纠错训练，之后完成 LoRA
合并、vLLM 推理和三组评测：

| 项目 | 结果 |
|---|---:|
| 训练数据 | 150 条新纠错题 + 65 条普通回放题，共 215 条 |
| 训练配置 | QLoRA 4-bit、1 epoch、batch size 1、梯度累积 8、学习率 `5e-6` |
| 最优 checkpoint | `outputs/correction-round-7/checkpoint-10` |
| 合并模型 | `outputs/correction-round-7-merged/` |
| 原始测试集 | 116 条，程序准确率 `66.38%`，LLM Judge `77.19%` |
| Round7 纠错验证集 | 30 条，程序准确率 `80.00%`，LLM Judge `96.67%` |
| 历史回归集 | 42 条，程序准确率 `38.10%`，LLM Judge `43.59%` |

本轮训练和部署链路正常完成。原始测试集和回归集的提升有限，说明 Round7 对输出
格式和部分纠错模板有效，但不能仅凭纠错验证集证明整体数学能力显著提升。后续需要
在同一批高难度 140 题上严格对比原始基座量化模型和 Round7 量化模型。

完整报告见
[Round7 训练与评测报告](docs/experiments/correction-round-7-evaluation-20260905.md)。

## 环境安装

本地 CPU 环境可以完成数据处理和代码检查。训练、LoRA 合并、量化和 vLLM
部署建议使用带 NVIDIA GPU 的云服务器，例如 24GB 显存以上的 RTX 4090D/A10。

Linux 云服务器：

```bash
conda create -n mathllm python=3.10 -y
conda activate mathllm
pip install -r requirements.txt
```

检查 GPU 和 Python：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

训练环境至少需要 PyTorch、Transformers、Datasets、Accelerate、PEFT 和 TRL。
`vllm`、`bitsandbytes` 和量化依赖应根据云服务器的 CUDA 驱动和目标版本单独安装。

## 本地一键启动和模型 API

本地前端不需要加载基座模型。当前使用独立的 HTML/CSS/JavaScript 单页前端，启动器会打开两个窗口：

```text
Web 前端：http://localhost:7860
FastAPI 后端：http://localhost:8080
```

后端通过 OpenAI 兼容的 `/chat/completions` 接口调用模型。当前默认配置已经接入
本地 Ollama 模型 `mathllm-round7`，对应的量化文件位于
`models/cpu/correction-round-7-q4_k_m.gguf`。Ollama 需要先运行，并且已经通过
`models/cpu/Modelfile` 创建该模型。

启动器会在启动 FastAPI 前检查 Ollama 和目标模型是否可用；检查通过后，项目后端
使用 `http://127.0.0.1:11434/v1` 的 OpenAI 兼容接口。也可以通过环境变量切换到
其他兼容服务商。

当前前端还提供：

- Markdown 和 LaTeX 公式渲染；
- 浏览器本地收藏题目，支持加载、移除和清空；
- “只给我提示”“讲简单一点”“检查我的答案”三个快捷学习入口；
- `Ctrl+Enter` 快捷提交题目；
- **Agent 模式**：实时显示路由结果、每步工具调用与耗时、验证结论，答案边写边流式显示，可随时暂停；
- **运行观测**页面：整体指标（失败率、路由分布、验证分布、P50/P95 延迟、token、回退率）与逐次运行的可展开调用链。

前端代码位于 `web/`：

- `web/index.html`：页面骨架和侧边栏导航；
- `web/styles.css`：响应式布局、明暗主题和组件样式；
- `web/app.js`：页面切换、Markdown 渲染、SSE 对话、Agent 轨迹卡、运行观测页面、收藏和学习记录。

### Agent 接口

除基础解题接口外，后端还提供：

| 接口 | 说明 |
|---|---|
| `POST /api/agent/run` | 非流式，返回完整执行轨迹 |
| `POST /api/agent/stream` | SSE 流式，实时推送路由、工具调用、答案分片与验证结论 |
| `GET /api/metrics?days=7` | 运行指标（失败率、路由分布、验证分布、P50/P95 延迟、token、回退率） |
| `GET /api/traces?limit=20` | 最近若干次运行的调用链明细 |

Agent 的工作方式是：先做结构化任务路由，再按需调用数学求解、知识库检索、安全计算和
答案校验工具，最后对答案做**独立验证**——方程与计算题由 SymPy 代入检验，被证伪时带
反例自动重算一次。

知识库检索得到的回答**默认跳过独立验证**：它本身就是从检索片段组织出来的，再跑一遍
来源核对只会多花 2–3 次模型调用（推理模型上很容易卡住）。需要打开时设
`MATHLLM_VERIFY_RAG=1`，恢复用检索片段做「来源核对」。

每次运行都会落盘一条 JSONL 到 `data/traces/runs.jsonl`（已 gitignore），包含路由决策、
工具调用、验证结论、耗时、token 用量与错误，可按 `run_id` 回放。

### 模型分工

编排（路由 / 规划 / 验证）走云端大模型，数学求解走本地微调模型：

```text
MATHLLM_ORCHESTRATOR_BASE_URL=https://api.deepseek.com/v1
MATHLLM_ORCHESTRATOR_MODEL=deepseek-chat
MATHLLM_ORCHESTRATOR_API_KEY=            # 留空 = 关闭云端编排，全部走本地
MATHLLM_ORCHESTRATOR_FALLBACK=local      # 云端不可用时回退本地

MATHLLM_SOLVER_BASE_URL=http://127.0.0.1:11434/v1
MATHLLM_SOLVER_MODEL=mathllm-round7
MATHLLM_SOLVER_API_KEY=ollama
```

旧的 `MATHLLM_API_BASE_URL` / `MATHLLM_MODEL_NAME` / `MATHLLM_API_KEY` 仍作为解题模型的
回退名读取。不配编排端点时，行为与只有一个本地模型时完全一致。

### 图形化配置

双击 `configure.bat` 打开配置弹窗（`start_local.bat -Configure` 也可）：填写两组模型、
点「测试连接」验证、点「保存并启动」。**API Key 留空即关闭云端编排**。弹窗只更新
`.env.local` 里模型相关的键，端口、RAG 等设置原样保留。

### 知识库

知识库检索基于本地 Chroma 向量库，首次使用前需要建立索引：

```powershell
python -m app.services.rag_service
```

索引会把 `knowledge/` 下的 Markdown 文档切分并写入 `data/chroma/`（已在 `.gitignore` 中）。
embedding 使用 `BAAI/bge-small-zh-v1.5`，首次运行会下载模型；网络受限时可设置
`HF_ENDPOINT=https://hf-mirror.com`。

首次使用时可以复制模板，也可以直接用弹窗配置：

```powershell
Copy-Item .env.local.example .env.local
```

启动器文件：

- `start_local.bat`：Windows 双击入口（首次运行会自动弹出配置窗口）；
- `start_local.ps1`：启动前后端的 PowerShell 脚本，支持 `-Configure`；
- `configure.bat`：只打开模型配置弹窗，不启动服务；
- `.env.local.example`：配置模板。

`.env.local` 已加入 `.gitignore`，不要把真实 API Key 提交到 GitHub。

## 数据准备

### 数据目录职责

```text
data/
├── raw/                         # 原始数据和纠错训练样本
│   └── corrections/             # 同类但条件不同的纠错样本
├── processed/
│   ├── train.json               # 训练数据，ChatML messages
│   └── eval.json                # checkpoint 选择用验证数据，ChatML messages
└── eval/
    ├── test.json                # 独立最终测试集，不参与训练和调参
    └── regression/              # 错误回归集，不参与训练
```

`eval/results/` 是评测输出目录，不是数据集目录。它保存 `report.json`、
`details.json`、`bad_cases.json`、延迟和准确率图表等结果。

### 原始数据

正式原始数据放在 `data/raw/`，来源、版本和抽样信息记录在
`data/raw/SOURCES.md`。当前主要来源包括：

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)：英文数学应用题；
- [Hendrycks MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math)：竞赛数学题；
- [CMID Chinese Math Instruct](https://huggingface.co/datasets/Mxode/CMID-Chinese_Math_Instruct_Dataset)：中文数学指令数据。

下载或补充原始数据：

```bash
python scripts/download_formal_data.py
```

不要直接修改原始文件中的内容；格式转换和清洗统一由
`scripts/prepare_data.py` 完成。

### 创建独立测试集

正式训练前创建一次固定测试集：

```bash
python scripts/create_test_set.py --test-ratio 0.1 --seed 42
```

脚本会从已整理的数据中按来源和学科进行固定随机种子抽样，排除纠错样本和
回归集，输出：

```text
data/eval/test.json
```

测试集使用与训练相同的 `messages` 格式，但不能复制到 `data/raw/`，也不能
用于训练、选择最佳 checkpoint 或反复调参。

### 清洗、转换和划分

运行：

```bash
python scripts/prepare_data.py \
  --raw-dir data/raw \
  --regression-dir data/eval/regression \
  --test-file data/eval/test.json \
  --output-dir data/processed \
  --eval-ratio 0.1 \
  --seed 42
```

如果本地已经下载 Qwen tokenizer，可以增加精确 token 长度检查：

```bash
python scripts/prepare_data.py \
  --tokenizer ./models/Qwen2.5-7B-Instruct-modelscope \
  --max-tokens 2048 \
  --test-file data/eval/test.json
```

程序会：

1. 读取 JSON、JSONL 和 CSV；
2. 兼容 GSM8K、MATH、CMID 和项目自定义字段；
3. 统一内部字段 `question`、`solution`、`answer`、`source`、`subject`；
4. 转换为 ChatML `messages`；
5. 修正常见全角半角、换行、LaTeX 空格和 GSM8K 计算标记；
6. 过滤缺少题目、过程或答案的样本；
7. 检查消息角色、空内容、过短回复和最大 token 长度；
8. 删除完全重复、数字/变量模板重复和高相似度题目；
9. 排除回归集和独立测试集；
10. 使用固定种子按 90%/10% 生成训练集和验证集。

每条记录的最终格式是：

```json
{
  "messages": [
    {"role": "system", "content": "数学解题系统提示词"},
    {"role": "user", "content": "求解 $x^2-5x+6=0$。"},
    {"role": "assistant", "content": "因式分解并求根。\n\n**最终答案**\n\n$x=2$ 或 $x=3$"}
  ]
}
```

不要手动写入 `[im_start]`、`[im_end]`；这些特殊标记由 tokenizer 和训练框架处理。

各轮训练数据、验证数据和纠错验证集按轮次保存于 `data/processed/`、`data/raw/` 和
`data/archive/`。`data/eval/test.json`（116 条）和
`data/eval/regression/regression.json`（42 条）保持独立。高难度 140 题位于
`data/candidates/`，只用于模型对比，不参与训练。程序检查不能替代数学证明，正式
训练前仍应人工抽查并验算参考答案。

## LoRA/QLoRA 训练

主训练程序是 `scripts/train.py`，不是 LLaMA-Factory。训练配置按轮次放在
`configs/training/`，历史配置放在 `configs/archive/training/`：

```bash
python scripts/train.py --config configs/training/<round>/train_config.yaml
```

Round7 的历史配置位于
`configs/archive/training/correction-round-7-20260905/train_config.yaml`。
具体训练轮次以所选配置文件为准，不能把历史 Round3 参数当作当前默认值。

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  use_4bit: true
  eval_steps: 10
  save_steps: 10
  max_seq_length: 2048
```

24GB 显存不足时，建议先停止 vLLM 服务，并将以下配置改为：

```yaml
training:
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_checkpointing: true
  use_4bit: true
  bf16: true
```

训练输出默认位于对应轮次的 `outputs/<round>/`，包括 LoRA adapter、checkpoint、
tokenizer、`trainer_state.json`、`train_results.json` 和 loss 记录。

LoRA/QLoRA 训练不会直接改变基座模型。adapter 可以单独挂载在基座模型上，
也可以在训练完成并选定最佳 checkpoint 后合并为独立模型。

## 选择 checkpoint 和合并 LoRA

先查看训练输出中的 `loss_summary.json`，选择验证集 loss 较低且人工评测表现
较好的 checkpoint，然后执行：

```bash
python scripts/merge_lora.py \
  --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
  --lora_path ./outputs/<round>/checkpoint-XXX \
  --output_path ./outputs/<round>-merged \
  --dtype float16
```

合并必须加载完整精度的基座模型。合并结果是可被 Transformers 或 vLLM 加载的
独立模型，不等于量化模型，体积通常接近原始 FP16 模型。

## 模型评测

### 启动 vLLM

启动服务前确认显存没有被训练进程占用：

```bash
python deploy/server.py
```

或直接运行：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/<round>-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048
```

检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

返回非空 `data` 后再开始评测。

### 验证集评测

验证集用于 checkpoint 选择和开发阶段比较：

```bash
python eval/evaluate.py \
  --model_endpoint http://127.0.0.1:8000/v1 \
  --eval_data data/processed/eval.json \
  --output eval/results/finetuned-eval \
  --temperature 0 \
  --max_tokens 512
```

先做 10 条接口冒烟测试：

```bash
python eval/evaluate.py \
  --model_endpoint http://127.0.0.1:8000/v1 \
  --eval_data data/processed/eval.json \
  --output eval/results/smoke-eval \
  --limit 10 \
  --temperature 0 \
  --max_tokens 512
```

### 独立测试集评测

模型和 checkpoint 确定后，使用同一份固定的 `test.json` 做最终比较：

```bash
python eval/evaluate.py \
  --model_endpoint http://127.0.0.1:8000/v1 \
  --eval_data data/eval/test.json \
  --output eval/results/test-final \
  --temperature 0 \
  --max_tokens 512
```

同一份测试集应分别评测基座模型、LoRA/合并模型和量化模型，并为每个模型使用
不同的输出目录。

评测输出包括：

- `report.json`：数学准确率、成功/失败数、平均总延迟、平均首 token 延迟；
- `details.json`：每道题的题目、标准答案、模型原始回答和判定信息；
- `bad_cases.json`：错误答案和请求失败样本；
- `run_config.json`：模型接口、数据文件和生成参数；
- `accuracy_comparison.png`、`latency_comparison.png`：指标图表。

基础评测是确定性的答案匹配和启发式过程检查，格式不同不一定代表数学错误。
`incorrect` 和格式边界案例应人工复核。

### 独立大模型语义评测

`eval/llm_judge.py` 用另一个能力更强的 OpenAI 兼容模型判断数学语义。必须使用
同一轮 `evaluate.py` 生成的 `details.json` 和同一份评测数据：

```bash
python eval/llm_judge.py \
  --details eval/results/test-final/details.json \
  --eval-data data/eval/test.json \
  --output eval/results/test-final/llm-judge \
  --judge-endpoint "https://api.example.com/v1" \
  --judge-model "your-judge-model"
```

API Key 不要写入代码或 Git，使用环境变量：

```bash
export JUDGE_API_KEY="your-api-key"
```

输出 `llm_judged_details.json`、`llm_judged_report.json` 和
`llm_judged_bad_cases.json`。`uncertain`、`incorrect` 以及一部分 `correct`
样本应人工抽查。

### 错误回归集

发现真实数学错误后：

- 同类但条件不同的纠错样本放入 `data/raw/corrections/`，用于后续训练；
- 原错误题和同类验证题放入 `data/eval/regression/regression.json`，只做回归测试；
- 不要把原错误题直接加入训练集。

如果已经根据 `test.json` 的错误设计了训练样本，该测试集已经参与开发迭代；正式
报告应另外保留一份未查看的新测试集作为盲测。完整的数据分流说明见
[数据分类与错误回归集说明](docs/experiments/data-classification-and-regression-guide.md)。

运行回归评测：

```bash
python eval/evaluate.py \
  --model_endpoint http://127.0.0.1:8000/v1 \
  --eval_data data/eval/regression/regression.json \
  --output eval/results/regression
```

## loss 记录

训练脚本会把训练过程保存到模型输出目录：

- `trainer_state.json`：完整训练状态和 `log_history`；
- `loss_history.json`、`loss_history.csv`：train/eval loss 记录；
- `loss_summary.json`：最终 loss、最佳 eval loss 和最佳 checkpoint；
- `loss_curve.png`：loss 曲线。

如果只有已有的 `trainer_state.json`，可以补生成曲线：

```bash
python scripts/loss_curve.py \
  --trainer_state outputs/<round>/trainer_state.json \
  --output_dir eval/results/loss-curve
```

## 量化

量化必须在 LoRA 合并后进行，不能直接把 QLoRA/NF4 checkpoint 当作部署模型：

```text
LoRA adapter
  → FP16/BF16 合并模型
  → AWQ/GPTQ 或 GGUF 量化
  → 用同一 test.json 对比量化前后效果
  → vLLM 部署
```

执行 AWQ 量化：

```bash
python scripts/quantize.py \
  --model_path ./outputs/<round>-merged \
  --output_path ./outputs/<round>-quantized \
  --calibration_data ./data/processed/train.json \
  --num_calibration_samples 256 \
  --max_seq_length 2048 \
  --dtype float16
```

校准数据只能使用训练数据，不能使用 `eval.json` 或 `test.json`。量化脚本会
保存 tokenizer、量化清单和校准摘要；正式使用前必须在目标云 GPU 上确认 vLLM
能够加载，并重新进行测试集评测。

## 消融实验

消融配置位于 `configs/ablation.yaml`，运行器位于 `eval/ablation.py`。主训练流程
使用 `scripts/train.py`；当前消融配置中的 runner 仍需根据实际训练命令检查，
不要在未确认 runner 的情况下直接启动全量消融。

消融实验必须固定：

- 同一版本的 train/eval 数据；
- 随机种子；
- system prompt；
- 评测参数；
- 一次只改变一个变量。

独立 `test.json` 不用于消融调参，只在最终模型确定后使用。

## 项目结构

```text
MathLLM/
├── app/                         # FastAPI 服务与 Agent 层
│   ├── core/                    # 共享配置和 system prompt
│   ├── api/                     # API 应用、路由和数据模型（含 metrics 观测接口）
│   ├── services/                # vLLM 调用、模型网关、流式、记忆、RAG、验证、trace、指标
│   ├── tools/                   # 独立小工具（模型配置弹窗）
│   ├── config_editor.py         # .env.local 模型配置读写
│   └── agent/                   # 结构化路由、工具层和有界 Agent 循环
│       ├── schema.py            # 结构化数据契约（含 trace 与 token 统计）
│       ├── structured.py        # 结构化补全与校验重试
│       ├── router.py            # 任务路由
│       ├── loop.py              # 有界 ReAct 循环 + 验证与重试 + trace 落盘
│       └── tools/               # 工具注册表与四个工具
├── configs/
│   ├── training/                # 按实验轮次保存训练配置
│   ├── deployment/              # 按模型状态保存部署配置
│   ├── archive/                 # 历史训练/冒烟/部署配置
│   └── ablation.yaml             # 消融实验配置
├── data/
│   ├── raw/                     # 原始数据和纠错训练样本
│   ├── processed/               # train.json、eval.json
│   └── eval/                    # test.json、regression/
├── deploy/
│   ├── server.py                # vLLM 启动器
│   └── settings.py              # vLLM 部署配置
├── web/                         # 独立 HTML/CSS/JavaScript 前端
│   ├── index.html               # 页面骨架和侧边栏
│   ├── styles.css               # 响应式布局和主题样式
│   └── app.js                   # 页面逻辑、SSE 对话、Agent 轨迹卡、运行观测页面
├── eval/
│   ├── evaluate.py              # 基础数学和延迟评测
│   ├── llm_judge.py             # 独立大模型语义评测
│   ├── ablation.py              # 消融实验运行器
│   └── results/                 # 评测输出
├── scripts/
│   ├── download_formal_data.py  # 下载正式数据
│   ├── create_test_set.py       # 创建固定独立测试集
│   ├── prepare_data.py          # 清洗、转换、去重、划分
│   ├── train.py                 # LoRA/QLoRA 训练
│   ├── merge_lora.py            # 合并 LoRA
│   ├── quantize.py              # AWQ 量化工具
│   └── loss_curve.py            # 导出 loss 曲线
├── knowledge/                   # 数学知识库文档，RAG 检索数据源
├── tests/                       # 单元测试（138 个，全部离线）
├── start_local.bat              # Windows 启动入口
├── start_local.ps1              # 启动前后端，首次运行弹出配置窗口
├── configure.bat                # 只打开模型配置弹窗
├── requirements.txt
└── README.md
```

## Git 与模型文件

以下大文件或敏感信息不要提交到 GitHub：

- `models/`：基座模型；
- `outputs/`、`checkpoints/`、`lora_weights/`：模型和训练产物；
- API Key、`.env` 和云服务器凭据；
- 临时上传压缩包。

代码、配置、文档、回归样例和固定的可控体积测试数据可以提交。云服务器更新
代码后执行：

```bash
git pull origin main
```

## 当前待完成事项

- [ ] 使用长输出上限完成高难度 140 题的原始基座量化模型评测；
- [ ] 使用相同参数完成 Round7 量化模型评测；
- [ ] 对两组结果执行 LLM Judge，并逐题统计修复、退化、截断和失败情况；
- [ ] 在目标云 GPU 上验证 AWQ/GPTQ 模型加载和量化前后效果；
- [ ] 根据双模型对照结果决定是否继续微调，而不是仅根据格式可解析率决定。

## 相关文档

- [第一周：数据准备](docs/development-plan/week1-data-preparation.md)
- [第二周：训练与部署](docs/development-plan/week2-training-and-deployment.md)
- [第三周：应用开发](docs/development-plan/week3-application-development.md)
- [数学助手功能与界面需求报告](docs/development-plan/math-assistant-product-and-interface-requirements.md)
- [第四周：评测与发布](docs/development-plan/week4-evaluation-and-release.md)
- [评测说明](eval/README.md)
- [独立评测数据说明](data/eval/README.md)
- [ChatML 格式](docs/learning/chatml-format.md)
- [LoRA 原理](docs/learning/lora-principle.md)
- [QLoRA 与 LoRA 对比](docs/learning/qlora-vs-lora.md)
- [INT4 量化原理](docs/learning/int4-quantization.md)
- [模型格式转换、量化与本地推理](docs/learning/model-conversion-quantization-local-inference.md)
- [vLLM PagedAttention](docs/learning/vllm-pagedattention.md)
- [消融实验方法](docs/learning/ablation-study.md)
- [首次全量训练与评测实验报告](docs/experiments/first-full-training-evaluation-20260903.md)
- [Round7 训练与评测报告](docs/experiments/correction-round-7-evaluation-20260905.md)
- [LoRA/QLoRA Checkpoint 文件说明](docs/learning/checkpoint-anatomy.md)
- [Agent 改造记录](docs/agent-development-log.md)：从结构化输出、工具层、有界 ReAct 循环、真实 RAG、
  前端接入、答案独立验证、模型路由到调用链日志与运行指标的完整开发日志（含 35 条踩坑记录）
