"""CodeBLEU 分数指标。"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ..metrics.base import BaseMetric, MetricResult, register_metric


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """获取 n-gram 计数。"""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _compute_bleu_component(
    pred_tokens: List[str],
    ref_tokens: List[str],
    n: int,
) -> float:
    """计算 BLEU 的 n-gram 精确率组件。"""
    pred_ngrams = _get_ngrams(pred_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)

    clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
    total = sum(pred_ngrams.values())

    return clipped / total if total > 0 else 0.0


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    """计算 BLEU 的简洁惩罚。"""
    if pred_len > ref_len:
        return 1.0
    elif pred_len == 0:
        return 0.0
    else:
        return math.exp(1 - ref_len / pred_len)


@register_metric
class CodeBLEUScore(BaseMetric):
    """CodeBLEU 分数：面向代码的 BLEU 评测指标。

    在标准 BLEU 基础上增加了代码结构感知的权重调整。
    支持 1-4 gram 的匹配计算。
    """

    name = "codebleu"
    description = "Code-aware BLEU score for code generation evaluation"
    higher_is_better = True
    value_range = (0.0, 1.0)

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        max_n = kwargs.get("max_n", 4)
        weights = kwargs.get("weights", [0.25] * min(max_n, 4))

        pred_tokens = prediction.strip().split()
        ref_tokens = reference.strip().split()

        if not pred_tokens:
            return MetricResult(name=self.name, score=0.0)

        # 计算各阶 n-gram 精确率
        precisions = []
        for n in range(1, len(weights) + 1):
            p = _compute_bleu_component(pred_tokens, ref_tokens, n)
            precisions.append(p)

        # 加权几何平均
        log_avg = sum(
            w * math.log(p) if p > 0 else float('-inf')
            for w, p in zip(weights, precisions)
        )

        bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))

        if any(p == 0 for p in precisions):
            score = 0.0
        else:
            score = bp * math.exp(log_avg)

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "precisions": precisions,
                "brevity_penalty": bp,
                "pred_length": len(pred_tokens),
                "ref_length": len(ref_tokens),
            },
        )
