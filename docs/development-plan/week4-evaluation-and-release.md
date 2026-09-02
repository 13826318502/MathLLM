# Week 4 开发计划：评测、打磨与项目收尾

## 1. 本周定位

第四周是项目收尾周，目标不是继续增加功能，而是把已经完成或实现的模块变成一套可验证、可展示、可复现的项目成果。

本周最终流程：

    模型和 Web 应用
          ↓
    自动评测
          ↓
    基座模型/精调模型对比
          ↓
    消融实验
          ↓
    图表和结果分析
          ↓
    Docker 打包
          ↓
    README、截图和演示材料
          ↓
    最终验收

根据当前项目状态，数据处理管道已经完成，消融实验框架也已经存在；但训练、合并、量化、Web 应用和自动评测仍需要确认或补齐。因此本周应优先保证主链路可以工作，再做扩展实验。

## 2. 本周最终目标

本周结束时应能够回答以下问题：

- 精调模型是否比基座模型更准确？
- 模型的数学答案准确率是多少？
- 解题过程是否完整？
- 首 token 延迟和总延迟是多少？
- INT4 量化是否造成明显精度下降？
- 哪个 LoRA rank 或学习率更合适？
- 模型能否通过 Web 页面正常使用？
- 项目是否可以被别人按照 README 重新运行？

## 3. 本周最终交付物

建议最终形成以下目录：

    eval/results/
    ├── base_model/
    ├── finetuned_model/
    ├── quantized_model/
    ├── comparison/
    └── ablation/

    docs/
    ├── development-plan/
    ├── screenshots/
    └── reports/

至少保存：

- 自动评测 JSON 报告；
- 每道题的模型回答和判分结果；
- 准确率、过程分、平均延迟、首 token 延迟；
- 基座模型与精调模型对比结果；
- FP16 与 INT4 对比结果；
- 消融实验的 JSON、CSV 和图片；
- Docker 构建和启动记录；
- Web 界面截图；
- Bad Case 和错误分析；
- README 中的真实实验数据。

## 4. 开始前：确认项目收尾状态

先检查以下文件和目录：

    data/processed/train.json
    data/processed/eval.json
    outputs/math-lora/
    outputs/math-lora-merged/
    outputs/math-lora-quantized/
    app/api/main.py
    app/frontend/gradio_app.py
    eval/evaluate.py
    eval/ablation.py
    configs/ablation.yaml
    deploy/Dockerfile

当前状态中需要特别注意：

- eval/evaluate.py 仍包含多个 TODO；
- train.py、merge_lora.py、quantize.py 可能仍未完全实现；
- app/api/main.py 和 app/frontend/gradio_app.py 可能仍未完成；
- eval/ablation.py 已经可以生成实验配置、运行训练和保存结果，但依赖训练命令和评测命令正确；
- README 中写入的准确率、延迟和提升比例必须来自真实实验，不能直接使用设计文档中的目标数字。

如果模型服务和 Web 应用还不能运行，应先补齐主链路，不要直接进行大规模实验。

## 5. 第一天：确定评测方案并补齐自动评测

### 任务

修改和完善：

    eval/evaluate.py

评测必须使用同一份 eval 数据和一致的生成参数。建议先完成三个最基本指标：

1. 最终答案准确率；
2. 总生成延迟和首 token 延迟；
3. 模型回答保存。

### call_model()

需要完成：

- 调用 vLLM OpenAI 兼容接口；
- 发送 system prompt 和用户题目；
- 支持非流式或流式请求；
- 记录请求开始时间；
- 记录收到第一个 token 的时间；
- 记录完整回答结束时间；
- 返回回答文本、总延迟和首 token 延迟；
- 遇到超时或 HTTP 错误时记录失败，而不是让整批评测中断。

### judge_answer()

第一版建议采用分层判分：

- 数值答案：提取数字并使用浮点容差；
- 整数答案：允许格式差异，但数值必须一致；
- 方程或表达式：尽量使用 sympy 化简后比较；
- 无法自动判断的答案：标记为 manual_review；
- 不要把完整解题过程直接当成最终答案比较。

第一周数据的 assistant 内容包含完整解题过程和最终答案，因此需要从回答中提取最终答案，或在数据准备阶段额外保存 final_answer 字段。

### judge_process()

第一版可以先做规则评分，不必马上依赖外部 LLM：

- 是否包含分析部分；
- 是否包含计算步骤；
- 是否包含最终答案；
- 是否有明显空回答；
- 是否包含 LaTeX 公式。

如果使用 LLM-as-Judge，必须固定评分标准和 Prompt，并保存评分依据。不要只保存一个无法解释的分数。

### 输出报告

每道题至少保存：

    question
    expected_answer
    model_answer
    is_correct
    process_score
    latency_ms
    first_token_ms

最终保存：

    eval/results/report.json
    eval/results/details.json

## 6. 第二天：完成基座模型和精调模型对比

### 任务

使用相同的 eval 数据、system prompt、temperature、max_tokens 和评测代码，分别测试：

- 基座 Qwen2.5-7B-Instruct；
- LoRA adapter 模型；
- 合并后的模型；
- 如果已经完成，INT4 量化模型。

公平性要求：

- eval 题目完全相同；
- 生成参数完全相同；
- 每个模型调用次数相同；
- 不要挑选对某个模型有利的题目；
- 保存模型路径、版本和配置。

命令形式：

    python eval/evaluate.py
      --model_endpoint http://localhost:8000/v1
      --eval_data ./data/processed/eval.json
      --output ./eval/results/finetuned_model

如果基座模型和精调模型不能同时启动，可以依次更换模型服务，但必须保留各自的输出目录。

### 对比指标

至少比较：

| 指标 | 说明 |
|---|---|
| Accuracy | 最终答案正确比例 |
| Avg Process Score | 解题过程完整性 |
| Avg Latency | 平均总延迟 |
| Avg First Token | 平均首 token 延迟 |
| Bad Case 数量 | 错误题目数量 |

不要直接把设计文档中的 85%、87% 或 200ms 写进结果，除非你的实际测试确实达到这些数值。

## 7. 第三天：完成量化对比和结果分析

### 任务

对 FP16 合并模型和 INT4 量化模型使用相同题目进行测试。

至少记录：

- FP16 模型大小；
- INT4 模型大小；
- FP16 显存占用；
- INT4 显存占用；
- FP16 准确率；
- INT4 准确率；
- FP16 平均延迟；
- INT4 平均延迟。

重点回答：

    量化是否节省显存？
    量化是否提高或降低延迟？
    数学答案准确率下降了多少？
    解题过程是否出现明显退化？

建议形成一个对比表：

| 模型 | 模型大小 | 显存 | 准确率 | 平均延迟 | 首 token 延迟 |
|---|---:|---:|---:|---:|---:|
| 基座 FP16 |  |  |  |  |  |
| 精调 FP16 |  |  |  |  |  |
| 精调 INT4 |  |  |  |  |  |

如果 INT4 的错误明显增多，优先检查量化格式、校准数据和 vLLM 参数。必要时保留 INT8 或 FP16 作为最终演示版本。

## 8. 第四天：运行消融实验

当前 eval/ablation.py 和 configs/ablation.yaml 已经提供了较完整的消融实验框架。

### 实验原则

一次只改变一个变量，其他条件保持一致：

- 相同训练集；
- 相同验证集；
- 相同随机种子；
- 相同基座模型；
- 相同评测方式；
- 相同输出格式。

### 推荐顺序

先做最有价值的两组：

#### LoRA rank

    rank = 4, 8, 16, 32

#### 学习率

    learning_rate = 0.0001, 0.0002, 0.0005

如果时间和 GPU 资源允许，再做：

#### 数据量

    data_size = 100, 200, 500, 1000, 2000

#### Epoch

    epochs = 1, 2, 3, 5

### 先 dry-run

在正式训练前执行：

    python eval/ablation.py
      --config configs/ablation.yaml
      --dry-run

确认：

- 实验数量正确；
- 每个实验目录独立；
- train/eval 数据路径正确；
- 生成的训练配置正确；
- 没有意外覆盖 baseline 或其他实验。

### 正式运行

建议先只运行 rank：

    python eval/ablation.py
      --config configs/ablation.yaml
      --groups rank

再运行学习率：

    python eval/ablation.py
      --config configs/ablation.yaml
      --groups learning_rate

结果应保存到：

    eval/results/ablation/

至少检查：

    ablation_results.json
    ablation_results.csv
    rank_ablation.png
    learning_rate_ablation.png

注意：当前 configs/ablation.yaml 中 evaluation_command 默认为空，因此实验可能只能记录训练 loss 和 eval loss，不能自动记录数学准确率。只有在 evaluate.py 可用后，才应配置自动评测命令。

## 9. 第五天：生成图表和错误分析

### 图表

至少生成：

1. 基座模型、精调模型、INT4 模型准确率柱状图；
2. 各模型平均延迟对比图；
3. 首 token 延迟对比图；
4. 不同 rank 的准确率或 eval loss 曲线；
5. 不同学习率的准确率或 eval loss 曲线；
6. 训练 loss 和 eval loss 曲线。

图表应包含：

- 中文或英文标题；
- 横纵坐标名称；
- 图例；
- 单位；
- 数据来源；
- 生成时间或模型版本。

不要只生成图片，不保存对应的原始 CSV/JSON 数据。

### Bad Case 分析

将错误分为：

- 基础计算错误；
- 公式使用错误；
- 推理步骤缺失；
- 最终答案与过程不一致；
- 题目理解错误；
- LaTeX 格式错误；
- 多轮对话上下文丢失；
- API 或 SSE 处理错误；
- 量化后精度下降。

每类至少挑选 1–2 个典型案例，写出：

    题目
    模型回答
    正确答案
    错误原因
    是否属于模型问题
    是否属于应用问题
    后续改进建议

## 10. 第六天：Docker 打包和部署验证

### 任务

检查：

    deploy/Dockerfile

确认 Docker 镜像至少包含：

- vLLM 启动环境；
- 配置文件；
- 正确的模型挂载路径；
- 端口 8000；
- 正确的模型量化参数。

模型权重不建议直接复制进 Git 仓库或 Docker 镜像。可以使用挂载目录：

    docker run --gpus all
      -v /path/to/model:/app/model
      -p 8000:8000
      mathllm-vllm

构建：

    docker build -t mathllm-vllm -f deploy/Dockerfile .

启动：

    docker run --gpus all
      -v /path/to/model:/app/model
      -p 8000:8000
      mathllm-vllm

检查：

    curl http://localhost:8000/v1/models

再调用聊天接口，确认容器内的 vLLM 可以正常返回结果。

当前 Dockerfile 使用 vllm/vllm-openai:latest。最终提交前建议记录实际镜像版本，不要只依赖 latest，否则以后可能出现环境变化导致无法复现。

## 11. 第七天：项目最终验收和文档整理

### 功能验收

按顺序确认：

1. vLLM 服务启动；
2. FastAPI 健康检查正常；
3. FastAPI 非流式解题正常；
4. FastAPI SSE 流式解题正常；
5. 多轮对话正常；
6. Gradio 页面正常；
7. 数学公式正常渲染；
8. 自动评测可以运行；
9. 消融结果文件存在；
10. Docker 可以构建或有明确记录。

### README 更新

README 中应补充真实结果：

- 实际训练数据量；
- 实际使用的训练方法；
- 实际 GPU；
- 实际训练时间；
- 实际准确率；
- 实际相对提升；
- 实际 INT4 显存；
- 实际延迟；
- 实际消融结论；
- Web 界面截图；
- 评测图表。

简历中的数据只能使用真实测量结果。设计文档中的目标指标只能作为目标，不能当作实验结果。

### 最终目录建议

    docs/
    ├── development-plan/
    │   ├── week1-data-preparation.md
    │   ├── week2-training-and-deployment.md
    │   ├── week3-application-development.md
    │   └── week4-evaluation-and-release.md
    ├── screenshots/
    └── reports/

    eval/results/
    ├── base_model/
    ├── finetuned_model/
    ├── quantized_model/
    ├── comparison/
    └── ablation/

## 12. 本周不要做的事情

- 不要用设计文档中的目标数字替代真实实验结果；
- 不要在评测代码未完成时直接运行大规模消融；
- 不要用不同的 eval 集比较不同模型；
- 不要只报告准确率而不保存逐题结果；
- 不要把 train loss 下降等同于数学能力提升；
- 不要把 Docker 能构建等同于模型服务一定可用；
- 不要把量化模型未经验证的结果写进 README 或简历；
- 不要在收尾周增加与主流程无关的大功能。

## 13. 第四周验收标准

- [ ] 自动评测脚本可以运行；
- [ ] 数值答案和表达式答案有明确判分规则；
- [ ] 基座模型和精调模型使用同一 eval 集完成对比；
- [ ] 已记录准确率、过程分、平均延迟和首 token 延迟；
- [ ] FP16 与 INT4 已完成对比；
- [ ] 至少完成 rank 和学习率两组消融；
- [ ] 消融结果有 JSON、CSV 和图表；
- [ ] 已完成 Bad Case 分类和分析；
- [ ] FastAPI、Gradio 和 vLLM 联调通过；
- [ ] Docker 已构建并完成启动验证，或记录明确阻塞原因；
- [ ] README 已更新为真实实验数据；
- [ ] 已保存界面截图和评测图表；
- [ ] 项目可以向别人说明如何安装、启动和验证。

## 14. 项目结束时需要准备的面试说明

你应该能够用自己的话解释：

- 为什么选择 Qwen2.5-7B；
- 为什么选择 LoRA 或 QLoRA；
- 数据如何清洗和划分；
- 为什么使用 ChatML；
- train loss 和 eval loss 如何判断过拟合；
- 为什么选择 AWQ 或 GPTQ；
- vLLM 如何降低推理成本；
- 哪个 LoRA rank 最合适以及为什么；
- 精调模型相对基座模型提升多少；
- INT4 量化带来了什么收益和代价；
- 如何定位模型错误和应用错误。

## 15. 相关文件

- 开发指南：docs/development-guide.md
- 第三周计划：docs/development-plan/week3-application-development.md
- 评测脚本：eval/evaluate.py
- 消融实验脚本：eval/ablation.py
- 消融配置：configs/ablation.yaml
- Docker 配置：deploy/Dockerfile
- README：README.md

