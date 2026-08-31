"""消融实验脚本

对比不同超参数配置对模型效果的影响:
    - LoRA rank: 4, 8, 16, 32
    - 学习率: 1e-4, 2e-4, 5e-4
    - 训练数据量: 100, 200, 500, 1000

结果输出为表格 + 折线图

用法:
    python eval/ablation.py --config configs/ablation.yaml
"""

import json
import yaml
from pathlib import Path
from itertools import product


def load_ablation_config(config_path: str) -> dict:
    """加载消融实验配置

    配置文件格式:
        experiments:
          - lora_r: [4, 8, 16, 32]
            learning_rate: [1e-4, 2e-4, 5e-4]
            data_size: [100, 200, 500]
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_experiment(lora_r: int, learning_rate: float, data_size: int) -> dict:
    """运行单次实验：训练 + 评测，返回指标结果"""
    # TODO: 实现单次实验
    # 1. 用指定参数训练模型
    # 2. 合并 LoRA 权重
    # 3. 运行评测
    # 4. 返回结果 dict
    pass


def run_ablation_study(config: dict) -> list[dict]:
    """运行所有消融实验组合"""
    experiments = config["experiments"][0]
    param_grid = list(product(
        experiments["lora_r"],
        experiments["learning_rate"],
        experiments["data_size"],
    ))

    results = []
    for lora_r, lr, data_size in param_grid:
        print(f"Running: rank={lora_r}, lr={lr}, data_size={data_size}")
        result = run_single_experiment(lora_r, lr, data_size)
        result.update({"lora_r": lora_r, "learning_rate": lr, "data_size": data_size})
        results.append(result)

    return results


def plot_results(results: list[dict], output_dir: str):
    """绘制消融实验结果图表"""
    # TODO: 用 matplotlib 生成:
    # 1. rank vs accuracy 折线图
    # 2. learning_rate vs accuracy 折线图
    # 3. data_size vs accuracy 折线图
    # 4. 完整结果表格
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation.yaml")
    parser.add_argument("--output", type=str, default="./eval/results/ablation")
    args = parser.parse_args()

    config = load_ablation_config(args.config)
    results = run_ablation_study(config)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_results(results, args.output)
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
