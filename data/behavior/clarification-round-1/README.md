# Clarification Round 1

本数据集训练模型结合多轮上下文判断当前输入应该直接回答、继续前文，还是主动澄清。

## 行为类型

- `answer_from_context`：上下文已经足够，直接继续回答；
- `need_clarification`：结合上下文后仍然缺少关键条件，只询问必要信息；
- `answer_directly`：当前问题本身信息完整，不需要追问；
- `ambiguous`：存在多个合理解释，指出歧义并请求用户选择。

数据只使用自然语言对话，不使用原来的五字段数学格式，也不包含路由标签。
训练和验证数据用于后续 LoRA SFT；`data/eval/behavior/clarification-round-1-test.jsonl`
是独立基线测试集，先用原始基座模型测试，不能参与训练。

多轮样本按完整对话线程划分，不能把同一线程拆到训练集和测试集。
