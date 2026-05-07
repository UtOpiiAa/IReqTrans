"""评测指标模块"""

from evaluation.metrics.base import (
    BaseMetric,
    MetricResult,
    register_metric,
    get_metric,
    get_all_metrics,
)
from evaluation.metrics.exact_match import ExactMatch
from evaluation.metrics.edit_similarity import EditSimilarity
from evaluation.metrics.token_f1 import TokenF1
from evaluation.metrics.code_bert import CodeBERTScore
from evaluation.metrics.codebleu import CodeBLEUScore
from evaluation.metrics.syntax_correctness import SyntaxCorrectness

__all__ = [
    "BaseMetric",
    "MetricResult",
    "register_metric",
    "get_metric",
    "get_all_metrics",
    "ExactMatch",
    "EditSimilarity",
    "TokenF1",
    "CodeBERTScore",
    "CodeBLEUScore",
    "SyntaxCorrectness",
]
