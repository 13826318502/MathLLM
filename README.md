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

## 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 正式数据整理 | 已完成当前版本 | GSM8K、Hendrycks MATH、CMID 和纠错样本已放入本地数据目录 |
| 数据预处理 | 已完成 | 支持 JSON、JSONL、CSV，包含字段统一、ChatML 转换、清洗、去重和划分 |
| 独立测试集 | 已生成 | `data/eval/test.json`，当前 116 条 |
| 当前正式训练集 | 已生成暂存版 | `data/processed/round-3-staging/train.json`，1118 条 |
| 当前正式验证集 | 已生成暂存版 | `data/processed/round-3-staging/eval.json`，124 条 |
| LoRA/QLoRA 训练 | 已实现 | `scripts/train.py` 使用 Transformers + TRL + PEFT |
| LoRA 合并 | 已实现 | `scripts/merge_lora.py` 输出独立模型 |
| loss 记录 | 已实现 | 保存 train/eval loss、CSV、JSON 和曲线图 |
| 基础模型评测 | 已实现 | 记录数学准确率、逐题答案、错题、平均延迟和首 token 延迟 |
| LLM Judge | 已实现 | `eval/llm_judge.py` 对模型答案做独立语义判断 |
| AWQ 量化 | 脚本已实现 | `scripts/quantize.py`，仍需在目标云 GPU 上实跑加载验证 |
| vLLM 部署 | 已实现启动器 | `deploy/server.py` 可启动 OpenAI 兼容服务 |
| Web 应用 | 开发中 | FastAPI/Gradio 仍需联调 |

## 最近一次完整训练与评测记录（2026-09-04）

Correction Round 2 在云端 RTX 4090D 上完成了训练、LoRA 合并、vLLM 推理和两组
评测：

| 项目 | 结果 |
|---|---:|
| 训练配置 | 3 epochs、BF16、batch size 2、gradient accumulation 8 |
| 最优 checkpoint | `outputs/correction-round-2/checkpoint-130` |
| 最优验证 loss | `0.2099656165` |
| 最终训练 loss | `0.2501130170` |
| 最终验证 loss | `0.2207271457` |
| 合并模型 | `outputs/correction-round-2-merged/` |
| 原始测试集 | 116 条，基础准确率 `59.48%`，LLM Judge `64.04%` |
| 错误回归集 | 42 条，基础准确率 `28.57%`，LLM Judge `25.64%` |

两组推理请求均全部成功，没有 HTTP 失败或 CUDA OOM。LLM Judge 的少量 JSON 解析
失败样本不代表被测模型答错，应重试或人工复核。回归集准确率较低，说明当前纠错
训练尚未充分解决原有错误，暂不应直接量化或作为最终发布版本。

本轮完整报告见
[Correction Round 2 训练与评测报告](docs/experiments/correction-round-2-evaluation-20260904.md)。

Round 3 纠错训练在第 330 步安全停止；最佳 checkpoint 为 `checkpoint-110`。
过拟合分析见
[Correction Round 3 过拟合分析报告](docs/experiments/correction-round-3-overfitting-20260904.md)。

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

当前 Round 3 暂存数据为：训练集 1118 条，普通验证集 124 条，另有 20 条
纠错验证集。`data/eval/test.json`（116 条）和
`data/eval/regression/regression.json`（42 条）保持独立。程序检查不能替代数学证明，
正式训练前仍应人工抽查并验算参考答案。

## LoRA/QLoRA 训练

主训练程序是 `scripts/train.py`，不是 LLaMA-Factory：

```bash
python scripts/train.py --config configs/training/correction-round-3-20260904/train_config.yaml
```

本轮主要配置位于
`configs/training/correction-round-3-20260904/train_config.yaml`：

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

训练输出默认位于 `outputs/math-lora/`，包括 LoRA adapter、checkpoint、
tokenizer、`trainer_state.json`、`train_results.json` 和 loss 记录。

LoRA/QLoRA 训练不会直接改变基座模型。adapter 可以单独挂载在基座模型上，
也可以在训练完成并选定最佳 checkpoint 后合并为独立模型。

## 选择 checkpoint 和合并 LoRA

先查看训练输出中的 `loss_summary.json`，选择验证集 loss 较低且人工评测表现
较好的 checkpoint，然后执行：

```bash
python scripts/merge_lora.py \
  --base_model ./models/Qwen2.5-7B-Instruct-modelscope \
  --lora_path ./outputs/math-lora/checkpoint-XXX \
  --output_path ./outputs/math-lora-merged \
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
  --model ./outputs/correction-round-3-merged \
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
  --trainer_state outputs/math-lora/trainer_state.json \
  --output_dir eval/results/loss-curve
```

## 量化

量化必须在 LoRA 合并后进行，不能直接把 QLoRA/NF4 checkpoint 当作部署模型：

```text
LoRA adapter
  → FP16/BF16 合并模型
  → AWQ W4A16 校准量化
  → 用同一 test.json 对比量化前后效果
  → vLLM 部署
```

执行 AWQ 量化：

```bash
python scripts/quantize.py \
  --model_path ./outputs/math-lora-merged \
  --output_path ./outputs/math-lora-quantized \
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
├── app/                         # FastAPI 和 Gradio 应用
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
│   └── server.py                # vLLM 启动器
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
│   ├── quantize.py              # AWQ W4A16 量化
│   └── loss_curve.py            # 导出 loss 曲线
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

## 后续计划

- [ ] 完成人工 10% 数据抽检和数学答案复核；
- [ ] 完整训练并选择最佳 checkpoint；
- [ ] 在云 GPU 上完成 AWQ 量化和 vLLM 加载验证；
- [ ] 对比合并模型与量化模型的准确率、显存和延迟；
- [ ] 完成 FastAPI/Gradio 联调；
- [ ] 完善消融实验自动评测；
- [ ] 完成 Docker 部署和最终发布文档。

## 相关文档

- [开发总指南](docs/development-guide.md)
- [项目设计](docs/design.md)
- [第一周：数据准备](docs/development-plan/week1-data-preparation.md)
- [第二周：训练与部署](docs/development-plan/week2-training-and-deployment.md)
- [第三周：应用开发](docs/development-plan/week3-application-development.md)
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
- [Correction Round 2 训练与评测报告](docs/experiments/correction-round-2-evaluation-20260904.md)
- [LoRA/QLoRA Checkpoint 文件说明](docs/learning/checkpoint-anatomy.md)
- [大模型知识文档](docs/learning/llm-knowledge-guide.md)
