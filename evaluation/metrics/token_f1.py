"""Token F1 分数指标。"""

from collections import Counter
from typing import Set

from ..metrics.base import BaseMetric, MetricResult, register_metric


def _tokenize(text: str) -> list:
    """简单分词：按空白字符拆分。"""
    return text.strip().split()


def _compute_f1(pred_tokens: list, ref_tokens: list) -> float:
    """计算 F1 分数。"""
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


@register_metric
class TokenF1(BaseMetric):
    """Token 级 F1 分数：基于 token 重叠的精确率与召回率调和平均。"""

    name = "token_f1"
    description = "Token-level F1 score based on token overlap"
    higher_is_better = True
    value_range = (0.0, 1.0)

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        ignore_case = kwargs.get("ignore_case", True)

        pred = prediction.strip()
        ref = reference.strip()

        if ignore_case:
            pred = pred.lower()
            ref = ref.lower()

        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)

        f1 = _compute_f1(pred_tokens, ref_tokens)

        # 同时计算 precision 和 recall
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0

        return MetricResult(
            name=self.name,
            score=f1,
            details={"precision": precision, "recall": recall},
        )
