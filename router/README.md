# TF-IDF 路由器

这个目录实现输入路由，不修改 Qwen 或本地 GGUF 模型的权重。路由器把输入分为：

```text
math       数学问题
non_math   普通问题
uncertain  分类置信度不足
```

分类器使用中文字符级 TF-IDF + Logistic Regression。字符 n-gram 比按空格切分更适合中文。
模型训练完成后默认保存到：

```text
outputs/router/tfidf_router.joblib
```

当前 `router/data/train.jsonl` 是可运行的冒烟数据。正式使用前，应补充更多数学、普通对话和边界样本，并单独维护 `router/data/eval.jsonl` 或 `data/eval/router-test.json` 做最终测试。

训练：

```powershell
python router/train_router.py `
  --input router/data/train.jsonl `
  --output outputs/router/tfidf_router.joblib
```

查看路由：

```powershell
python -m router.predict "解方程 2x+3=9"
python -m router.predict "帮我写一封请假邮件"
```

路由器找不到已训练的 joblib 文件时，会自动使用规则回退，因此应用仍可启动；训练出正式分类器后会优先使用 TF-IDF 分类器。

## 和本地量化模型的连接

路由器不直接加载 GGUF 文件。它先选择提示词，再由现有 FastAPI 后端通过 OpenAI 兼容接口调用同一个本地量化模型。

例如使用 Ollama 加载 `models/correction-round-7-q4_k_m.gguf`：

```powershell
ollama create mathllm-round7 -f models/Modelfile
ollama serve
```

`.env.local` 设置为：

```text
MATHLLM_API_BASE_URL=http://127.0.0.1:11434/v1
MATHLLM_MODEL_NAME=mathllm-round7
MATHLLM_API_KEY=
```

启动 MathLLM 后，`/api/chat` 会根据最新一条用户消息选择不同的 system prompt：

```text
数学问题：要求回答以 【MATH】 开头
普通问题：要求回答以 【CHAT】 开头
不确定问题：要求回答以 【UNCERTAIN】 开头
```

这些符号只是当前阶段的可观察标记，后续可以替换成 JSON 输出或在后端解析后隐藏。
