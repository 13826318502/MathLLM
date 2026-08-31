"""模型量化脚本：将合并后的模型量化为 INT4/INT8

量化后模型体积大幅减小，可以在更低配置的 GPU 上运行。
- FP16 7B 模型约 14GB 显存
- INT8 量化后约 7GB
- INT4 量化后约 4GB

用法:
    python scripts/quantize.py \
        --model_path ./outputs/math-lora-merged \
        --output_path ./outputs/math-lora-quantized \
        --bits 4
"""

import argparse


def quantize_model(model_path: str, output_path: str, bits: int = 4):
    """使用 bitsandbytes 或 GPTQ 进行量化

    推荐方案:
        - bits=4: AWQ 量化（速度快，精度好）
        - bits=8: bitsandbytes INT8 量化
    """
    # TODO: 实现量化逻辑
    # 方案一: 使用 AutoAWQ 进行 INT4 量化
    # 方案二: 使用 bitsandbytes 进行运行时量化
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    args = parser.parse_args()

    quantize_model(args.model_path, args.output_path, args.bits)


if __name__ == "__main__":
    main()
