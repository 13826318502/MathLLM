"""合并 LoRA 权重到基座模型

精调完成后，将 LoRA adapter 合并回基座模型，
得到一个完整的独立模型用于部署。

用法:
    python scripts/merge_lora.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --lora_path ./outputs/math-lora \
        --output_path ./outputs/math-lora-merged
"""

import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    print(f"Loading base model: {base_model_path}")
    # TODO: 加载基座模型（float16 精度）

    print(f"Loading LoRA adapter: {lora_path}")
    # TODO: 用 PeftModel.from_pretrained 加载 adapter

    print("Merging...")
    # TODO: model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    # TODO: 保存合并后的模型和 tokenizer

    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    merge_lora(args.base_model, args.lora_path, args.output_path)


if __name__ == "__main__":
    main()
