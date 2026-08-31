"""模型评测脚本

评测维度:
    1. 准确率: 最终答案是否正确
    2. 过程完整性: 解题步骤是否完整合理（LLM-as-Judge）
    3. 延迟: 首 token 时间 + 总生成时间
    4. 幻觉率: 是否存在编造的错误公式/定理

评测对比:
    - 精调模型 vs 基座模型
    - 精调模型 vs GPT-4 API
    - 不同 rank 的 LoRA 效果

用法:
    python eval/evaluate.py \
        --model_endpoint http://localhost:8000/v1 \
        --eval_data ./data/processed/eval.json \
        --output ./eval/results/
"""

import json
import time
import httpx
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    process_score: float = 0.0
    latency_ms: float = 0.0
    first_token_ms: float = 0.0


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.is_correct) / len(self.results)

    @property
    def avg_process_score(self) -> float:
        if not self.results:
            return 0.0
        return np.mean([r.process_score for r in self.results])

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return np.mean([r.latency_ms for r in self.results])

    @property
    def avg_first_token_ms(self) -> float:
        if not self.results:
            return 0.0
        return np.mean([r.first_token_ms for r in self.results])

    def summary(self) -> dict:
        return {
            "total": len(self.results),
            "accuracy": round(self.accuracy, 4),
            "avg_process_score": round(self.avg_process_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_first_token_ms": round(self.avg_first_token_ms, 2),
        }


def call_model(endpoint: str, question: str) -> tuple[str, float, float]:
    """调用模型 API，返回 (回答文本, 总延迟ms, 首token延迟ms)"""
    # TODO: 实现模型调用 + 计时
    pass


def judge_answer(question: str, expected: str, actual: str) -> bool:
    """判断答案是否正确（精确匹配 + 数值容差）"""
    # TODO: 实现答案判分
    # - 对于数值答案：提取数字，考虑浮点容差
    # - 对于表达式：化简后比较
    pass


def judge_process(question: str, response: str) -> float:
    """用 LLM-as-Judge 评估解题过程完整性（0-1 分）"""
    # TODO: 调用 GPT/Qwen API 打分
    # - 评估标准：步骤是否完整、推理是否合理、是否有幻觉
    pass


def run_evaluation(endpoint: str, eval_data_path: str) -> EvalReport:
    """运行完整评测"""
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    report = EvalReport()

    for i, item in enumerate(eval_data):
        print(f"Evaluating {i+1}/{len(eval_data)}...")
        question = item["messages"][1]["content"]
        expected = item["messages"][2]["content"]

        # TODO: 调用模型、判分、记录结果
        pass

    return report


def generate_comparison_chart(reports: dict[str, EvalReport], output_dir: str):
    """生成多模型对比图表（柱状图 + 雷达图）"""
    # TODO: 用 matplotlib 生成对比图表
    # - 准确率对比柱状图
    # - 多维度雷达图（准确率/过程分/延迟）
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_endpoint", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default="./data/processed/eval.json")
    parser.add_argument("--output", type=str, default="./eval/results")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    report = run_evaluation(args.model_endpoint, args.eval_data)
    summary = report.summary()
    print(f"\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))

    with open(Path(args.output) / "report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
