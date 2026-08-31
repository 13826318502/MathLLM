import json
import random
from pathlib import Path

RAW_DIR = Path("./data/raw")
OUTPUT_DIR = Path("./data/processed")
SYSTEM_PROMPT = """你是一个专业的数学解题助手。请按照以下要求回答数学问题：
1. 先分析题目要求
2. 给出详细的解题步骤
3. 用 LaTeX 格式书写数学公式
4. 最后给出明确的答案"""


def load_raw_data() -> list[dict]:
    """从 data/raw 目录加载所有原始数据文件（支持 json/jsonl/csv）"""
    # TODO: 实现多格式数据加载
    pass


def format_example(item: dict) -> dict:
    """将单条原始数据转换为模型训练格式（ChatML）

    期望输入字段:
        item["question"]: 数学题目
        item["solution"]: 解题过程
        item["answer"]: 最终答案（可选）

    输出格式:
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution}
            ]
        }
    """
    # TODO: 实现数据格式转换
    pass


def split_dataset(data: list[dict], eval_ratio: float = 0.1) -> tuple[list, list]:
    """划分训练集和验证集"""
    # TODO: 实现数据划分
    pass


def validate_data(data: list[dict]) -> list[dict]:
    """数据清洗：去除空字段、重复项、过短样本"""
    # TODO: 实现数据校验
    pass


def main():
    print("Loading raw data...")
    raw_data = load_raw_data()
    print(f"Loaded {len(raw_data)} raw examples")

    print("Formatting examples...")
    formatted = [format_example(item) for item in raw_data]
    formatted = [item for item in formatted if item is not None]

    print("Validating data...")
    cleaned = validate_data(formatted)
    print(f"After cleaning: {len(cleaned)} examples")

    print("Splitting dataset...")
    train_data, eval_data = split_dataset(cleaned)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("Done! Data saved to data/processed/")


if __name__ == "__main__":
    main()
