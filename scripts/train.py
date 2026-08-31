"""使用 LLaMA-Factory 进行 LoRA 精调

方式一（推荐）: 使用 LLaMA-Factory CLI
    llamafactory-cli train configs/train_config.yaml

方式二: 使用本脚本直接调用 transformers + peft
    python scripts/train.py

训练前确保:
    1. data/processed/train.json 已准备好
    2. 有足够的 GPU 显存（16GB+ 用于 7B LoRA）
"""

import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_name: str):
    """加载基座模型和 tokenizer，使用 4bit 加载以节省显存"""
    # TODO: 实现模型加载
    # - 使用 BitsAndBytesConfig 配置 4bit 量化加载
    # - 加载 tokenizer 并设置 padding side
    pass


def create_lora_config(config: dict) -> LoraConfig:
    """根据配置文件创建 LoRA 配置"""
    # TODO: 从 config["lora"] 读取参数创建 LoraConfig
    # - task_type = TaskType.CAUSAL_LM
    pass


def create_chat_template_fn(tokenizer):
    """创建数据处理函数：将 messages 格式转换为模型输入"""
    # TODO: 使用 tokenizer.apply_chat_template 处理 messages
    # - 返回处理后的 input_ids
    pass


def train():
    config = load_config()

    print(f"Loading model: {config['model_name_or_path']}")
    model, tokenizer = load_model_and_tokenizer(config["model_name_or_path"])

    print("Applying LoRA config...")
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    train_data = load_dataset("json", data_files=config["data"]["train_file"])
    eval_data = load_dataset("json", data_files=config["data"]["eval_file"])

    print("Starting training...")
    # TODO: 配置 SFTTrainer / Trainer 并开始训练
    # - 使用 config["training"] 中的超参数
    # - 设置 eval_strategy="steps"
    # - 保存最优 checkpoint

    print("Training complete!")


if __name__ == "__main__":
    train()
