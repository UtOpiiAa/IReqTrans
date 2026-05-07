"""评测模块 - 独立于 RAG 和 lspec_parser 的评测框架。

提供多种代码/文本相似度评测指标：
- ExactMatch: 精确匹配率
- EditSimilarity: 编辑相似度
- TokenF1: Token 级 F1 分数
- CodeBERTScore: 基于 CodeBERT 的语义相似度
- CodeBLEU: 代码 BLEU 分数
- SyntaxCorrectness: 语法正确性
"""

from evaluation.metrics import (
    BaseMetric,
    MetricResult,
    ExactMatch,
    EditSimilarity,
    TokenF1,
    CodeBERTScore,
    CodeBLEUScore,
    SyntaxCorrectness,
    get_metric,
    get_all_metrics,
    register_metric,
)

__all__ = [
    "BaseMetric",
    "MetricResult",
    "ExactMatch",
    "EditSimilarity",
    "TokenF1",
    "CodeBERTScore",
    "CodeBLEUScore",
    "SyntaxCorrectness",
    "get_metric",
    "get_all_metrics",
    "register_metric",
]
