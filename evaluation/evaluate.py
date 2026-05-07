"""评测核心逻辑 - 对单对预测/参考结果进行评测"""

from pathlib import Path
from typing import Optional

from evaluation.metrics import BaseMetric, get_metric, get_all_metrics


def evaluate_pair(
    prediction: str,
    reference: str,
    metrics: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    评测一对预测结果和参考结果。

    Args:
        prediction: 模型预测的代码
        reference: 参考代码
        metrics: 使用的评测指标名称列表，None 表示使用全部

    Returns:
        各指标的得分字典
    """
    if metrics is None:
        metric_instances = get_all_metrics()
    else:
        metric_instances = [get_metric(name) for name in metrics]

    results = {}
    for metric in metric_instances:
        try:
            result = metric.compute(prediction, reference)
            results[metric.name] = result.score
        except Exception as e:
            print(f"[WARNING] 指标 {metric.name} 计算失败: {e}")
            results[metric.name] = 0.0

    return results


def evaluate_from_files(
    prediction_file: str,
    reference_file: str,
    metrics: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    从文件读取预测和参考结果并评测。

    Args:
        prediction_file: 预测结果文件路径
        reference_file: 参考结果文件路径
        metrics: 使用的评测指标名称列表

    Returns:
        各指标的得分字典
    """
    pred_text = Path(prediction_file).read_text(encoding="utf-8").strip()
    ref_text = Path(reference_file).read_text(encoding="utf-8").strip()
    return evaluate_pair(pred_text, ref_text, metrics)


def evaluate_from_excel(
    excel_path: str,
    prediction_col: str = "prediction",
    reference_col: str = "reference",
    metrics: Optional[list[str]] = None,
) -> list[dict]:
    """
    从 Excel 文件中读取多组预测/参考结果并批量评测。

    Args:
        excel_path: Excel 文件路径
        prediction_col: 预测结果列名
        reference_col: 参考结果列名
        metrics: 使用的评测指标名称列表

    Returns:
        每行评测结果列表
    """
    import pandas as pd

    df = pd.read_excel(excel_path)
    results = []

    for idx, row in df.iterrows():
        pred = str(row.get(prediction_col, ""))
        ref = str(row.get(reference_col, ""))
        scores = evaluate_pair(pred, ref, metrics)
        scores["index"] = idx
        results.append(scores)

    return results
