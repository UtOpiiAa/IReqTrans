"""批量评测脚本 - 对实验结果目录进行批量评测"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.evaluate import evaluate_from_excel


def batch_evaluate_results(
    results_dir: str,
    output_path: Optional[str] = None,
    prediction_col: str = "prediction",
    reference_col: str = "reference",
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    对结果目录中的所有 Excel 文件进行批量评测。

    Args:
        results_dir: 包含结果 Excel 文件的目录
        output_path: 评测结果输出路径（可选）
        prediction_col: 预测结果列名
        reference_col: 参考结果列名
        metrics: 使用的评测指标名称列表

    Returns:
        评测结果 DataFrame
    """
    results_path = Path(results_dir)
    all_results = []

    excel_files = list(results_path.glob("*.xlsx")) + list(results_path.glob("*.xls"))
    if not excel_files:
        print(f"[WARNING] 在 {results_dir} 中未找到 Excel 文件")
        return pd.DataFrame()

    for excel_file in sorted(excel_files):
        print(f"正在评测: {excel_file.name}")
        try:
            file_results = evaluate_from_excel(
                str(excel_file),
                prediction_col=prediction_col,
                reference_col=reference_col,
                metrics=metrics,
            )
            for result in file_results:
                result["source_file"] = excel_file.name
            all_results.extend(file_results)
        except Exception as e:
            print(f"[ERROR] 评测 {excel_file.name} 失败: {e}")

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.suffix == ".xlsx":
            df.to_excel(str(output_file), index=False)
        else:
            df.to_csv(str(output_file), index=False)
        print(f"评测结果已保存到: {output_path}")

    return df


def batch_evaluate_directories(
    base_dir: str,
    output_dir: Optional[str] = None,
    prediction_col: str = "prediction",
    reference_col: str = "reference",
    metrics: Optional[list[str]] = None,
) -> None:
    """
    对基础目录下所有子目录中的结果进行批量评测。

    Args:
        base_dir: 包含多个方法结果子目录的基础目录
        output_dir: 评测结果输出目录
        prediction_col: 预测结果列名
        reference_col: 参考结果列名
        metrics: 使用的评测指标名称列表
    """
    base_path = Path(base_dir)
    output_base = Path(output_dir) if output_dir else base_path / "evaluation_results"

    for method_dir in sorted(base_path.iterdir()):
        if not method_dir.is_dir():
            continue
        output_path = output_base / f"{method_dir.name}_eval.xlsx"
        print(f"\n{'='*60}")
        print(f"评测方法: {method_dir.name}")
        print(f"{'='*60}")

        batch_evaluate_results(
            results_dir=str(method_dir),
            output_path=str(output_path),
            prediction_col=prediction_col,
            reference_col=reference_col,
            metrics=metrics,
        )


def main():
    parser = argparse.ArgumentParser(description="批量评测实验结果")
    parser.add_argument("--results_dir", type=str, required=True, help="结果目录路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--pred_col", type=str, default="prediction", help="预测列名")
    parser.add_argument("--ref_col", type=str, default="reference", help="参考列名")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="评测指标")
    parser.add_argument(
        "--batch_dirs", action="store_true", help="是否对子目录批量评测"
    )
    args = parser.parse_args()

    if args.batch_dirs:
        batch_evaluate_directories(
            base_dir=args.results_dir,
            output_dir=args.output,
            prediction_col=args.pred_col,
            reference_col=args.ref_col,
            metrics=args.metrics,
        )
    else:
        batch_evaluate_results(
            results_dir=args.results_dir,
            output_path=args.output,
            prediction_col=args.pred_col,
            reference_col=args.ref_col,
            metrics=args.metrics,
        )


if __name__ == "__main__":
    main()
