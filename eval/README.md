# 模型推理与数学准确率评测

`evaluate.py` 面向 vLLM 的 OpenAI 兼容接口，完成两类检查：

1. 推理验证：确认 `/v1/models` 和 `/v1/chat/completions` 可用，模型能够处理 `messages`，并记录响应是否为空、请求错误、首 token 延迟和总延迟。
2. 数学准确率：从标准答案和模型回答中提取 `**最终答案**` 部分，先做去空白和符号规范化，再做文本匹配；数值答案允许绝对误差 `1e-6` 或相对误差 `1e-5`。

过程分是可复现的启发式完整性分数，检查回答长度、解题步骤信号、数学内容和最终答案标记。它不能证明每一个推理步骤都正确，错误样本仍需人工复核。

## 运行前提

云服务器开机并启动 vLLM 后，先在云端执行：

```bash
curl http://localhost:8000/v1/models
```

返回 JSON 且 `data` 不为空，说明服务已启动。然后运行 10 条冒烟评测：

```bash
python eval/evaluate.py \
  --model_endpoint http://localhost:8000/v1 \
  --eval_data data/processed/eval.json \
  --output eval/results/smoke-eval \
  --limit 10 \
  --temperature 0 \
  --max_tokens 512
```

冒烟测试通过后，再运行完整验证集：

```bash
python eval/evaluate.py \
  --model_endpoint http://localhost:8000/v1 \
  --eval_data data/processed/eval.json \
  --output eval/results/finetuned_model \
  --temperature 0 \
  --max_tokens 512
```

每次评测会保存：

- `report.json`：总数、成功/失败数、准确率、过程分、平均总延迟、平均首 token 延迟；
- `details.json`：每道题的题目、标准答案、模型原始回答、提取后的答案、判分原因和错误信息。
- `bad_cases.json`：所有错误答案和请求失败样本，供人工复核。
- `run_config.json`：接口、模型名、评测文件和生成参数。
- `accuracy_comparison.png`、`latency_comparison.png`：本次模型的可视化指标。

训练脚本会在训练输出目录自动保存：

- `trainer_state.json`：Transformers 的完整训练状态和 `log_history`；
- `loss_history.json`、`loss_history.csv`：筛选后的 train/eval loss 记录；
- `loss_summary.json`：最终 loss、最佳 eval loss 和最佳 checkpoint；
- `loss_curve.png`：train loss 与 eval loss 曲线。

已有云端训练结果如果只保留了 `trainer_state.json`，也可以单独补生成曲线：

```bash
python scripts/loss_curve.py \
  --trainer_state outputs/smoke-89/trainer_state.json \
  --output_dir eval/results/smoke-89
```

## 如何判断结果

- `failed > 0`：先解决服务、模型路径、tokenizer 或响应格式问题，不能把这次准确率当成最终结果。
- `accuracy`：`正确题数 / 总题数`；请求失败按不正确计入，同时单独显示 `failed`。
- `judge_reason = numeric match...`：数值在容差内相等；`normalized exact match`：规范化文本完全相等。
- 发现错误时，优先查看 `details.json`，区分最终答案错误、步骤缺失、过程与答案矛盾、LaTeX 格式错误和服务错误。

比较基座模型、LoRA 模型、合并模型或 INT4 模型时，必须使用同一份 `eval.json`、相同的 system prompt、`temperature`、`max_tokens` 和评测程序，并把每个模型输出到不同目录。

## 错误回归集

发现模型的真实数学错误后，不要直接把原错误题加入训练集。将同类但条件不同的纠错样本放入 `data/raw/corrections/`，将原错误题和新的同类验证题保存在 `data/eval/regression/regression.json`。`prepare_data.py` 会递归读取 `data/raw/`，但不会读取 `data/eval/`，因此回归集不会泄漏到训练集。

训练完成后可单独运行：

```bash
python eval/evaluate.py \
  --model_endpoint http://localhost:8000/v1 \
  --eval_data data/eval/regression/regression.json \
  --output eval/results/regression
```

## 使用独立大模型进行语义评测

`evaluate.py` 不删除也不替换。它继续负责调用被测模型、记录答案、请求错误和延迟，并提供确定性基础判定。`llm_judge.py` 是第二阶段评测器，使用另一个 OpenAI 兼容接口判断数学语义，并输出 `correct`、`incorrect` 或 `uncertain`。

先准备环境变量（也可以直接使用命令行参数）：

```bash
export JUDGE_BASE_URL="https://api.example.com/v1"
export JUDGE_MODEL="your-judge-model-name"
export JUDGE_API_KEY="your-api-key"
```

然后使用同一轮 `evaluate.py` 产生的结果：

```bash
python eval/llm_judge.py \
  --details eval/results/smoke-merged-final/details.json \
  --eval-data data/processed/eval.json \
  --output eval/results/smoke-merged-final/llm-judge
```

也可以显式指定接口和模型：

```bash
python eval/llm_judge.py \
  --details eval/results/smoke-merged-final/details.json \
  --eval-data data/processed/eval.json \
  --output eval/results/smoke-merged-final/llm-judge \
  --judge-endpoint "https://api.example.com/v1" \
  --judge-model "your-judge-model-name"
```

输出文件：

- `llm_judged_details.json`：每道题的评测标签、置信度、错误类型和简短理由；
- `llm_judged_report.json`：汇总准确率和 `uncertain` 比例；
- `llm_judged_bad_cases.json`：所有错误和不确定样本，供人工复核。

评测模型不能使用被测模型自身，推荐使用独立且能力更强的模型，并将温度固定为 0。当前 10 条冒烟数据应全部人工核查；正式数据可全量复核 `incorrect`/`uncertain`，再分层抽查一部分 `correct`。
