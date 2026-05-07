"""评测结果分析脚本 - 汇总统计和可视化"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def analyze_evaluation_results(
    eval_result_path: str,
    output_dir: Optional[str] = None,
) -> dict:
    """
    分析评测结果，生成统计摘要。

    Args:
        eval_result_path: 评测结果文件路径（Excel 或 CSV）
        output_dir: 分析结果输出目录

    Returns:
        统计摘要字典
    """
    result_path = Path(eval_result_path)
    if result_path.suffix == ".xlsx":
        df = pd.read_excel(str(result_path))
    else:
        df = pd.read_csv(str(result_path))

    # 排除非指标列
    non_metric_cols = {"index", "source_file"}
    metric_cols = [col for col in df.columns if col not in non_metric_cols]

    # 按来源文件分组统计
    summary = {}

    if "source_file" in df.columns:
        grouped = df.groupby("source_file")[metric_cols]
        per_file_stats = grouped.agg(["mean", "std", "min", "max", "count"])
        summary["per_file_stats"] = per_file_stats.to_dict()

    # 总体统计
    overall_stats = df[metric_cols].agg(["mean", "std", "min", "max", "median"])
    summary["overall_stats"] = overall_stats.to_dict()
    summary["total_samples"] = len(df)
    summary["metrics"] = metric_cols

    # 打印摘要
    print("\n" + "=" * 60)
    print("评测结果分析摘要")
    print("=" * 60)
    print(f"总样本数: {len(df)}")
    print(f"评测指标: {metric_cols}")
    print()

    print("总体统计:")
    for metric in metric_cols:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        median_val = df[metric].median()
        print(f"  {metric}: mean={mean_val:.4f}, std={std_val:.4f}, median={median_val:.4f}")

    if "source_file" in df.columns:
        print("\n按文件统计:")
        for file_name, group in df.groupby("source_file"):
            print(f"\n  [{file_name}] ({len(group)} samples)")
            for metric in metric_cols:
                mean_val = group[metric].mean()
                print(f"    {metric}: {mean_val:.4f}")

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存统计摘要
        stats_df = overall_stats.reset_index()
        stats_df.to_excel(str(output_path / "overall_stats.xlsx"), index=False)

        if "source_file" in df.columns:
            per_file_stats.to_excel(str(output_path / "per_file_stats.xlsx"))

        print(f"\n分析结果已保存到: {output_path}")

    return summary


def compare_methods(
    eval_results_dir: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    比较不同方法的评测结果。

    Args:
        eval_results_dir: 包含多个方法评测结果的目录
        output_path: 比较结果输出路径

    Returns:
        方法比较 DataFrame
    """
    results_dir = Path(eval_results_dir)
    all_stats = []

    for eval_file in sorted(results_dir.glob("*_eval.xlsx")):
        method_name = eval_file.stem.replace("_eval", "")
        df = pd.read_excel(str(eval_file))

        metric_cols = [col for col in df.columns if col not in {"index", "source_file"}]
        stats = {"method": method_name, "sample_count": len(df)}
        for metric in metric_cols:
            stats[f"{metric}_mean"] = df[metric].mean()
            stats[f"{metric}_std"] = df[metric].std()

        all_stats.append(stats)

    if not all_stats:
        print("[WARNING] 未找到评测结果文件")
        return pd.DataFrame()

    compare_df = pd.DataFrame(all_stats)

    if output_path:
        compare_df.to_excel(output_path, index=False)
        print(f"比较结果已保存到: {output_path}")

    # 打印比较表格
    print("\n" + "=" * 60)
    print("方法比较")
    print("=" * 60)
    print(compare_df.to_string(index=False))

    return compare_df


def main():
    parser = argparse.ArgumentParser(description="评测结果分析")
    subparsers = parser.add_subparsers(dest="command")

    # 分析单个结果
    analyze_parser = subparsers.add_parser("analyze", help="分析评测结果")
    analyze_parser.add_argument("--input", type=str, required=True, help="评测结果文件路径")
    analyze_parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    # 比较多个方法
    compare_parser = subparsers.add_parser("compare", help="比较方法结果")
    compare_parser.add_argument(
        "--dir", type=str, required=True, help="评测结果目录"
    )
    compare_parser.add_argument("--output", type=str, default=None, help="输出文件路径")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_evaluation_results(args.input, args.output_dir)
    elif args.command == "compare":
        compare_methods(args.dir, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
