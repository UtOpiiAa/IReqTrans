"""精确匹配率 (Exact Match) 指标。"""

from ..metrics.base import BaseMetric, MetricResult, register_metric


@register_metric
class ExactMatch(BaseMetric):
    """精确匹配率：预测与参考完全相同的比例。"""

    name = "exact_match"
    description = "Exact match ratio between prediction and reference"
    higher_is_better = True
    value_range = (0.0, 1.0)

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        ignore_case = kwargs.get("ignore_case", False)
        strip_whitespace = kwargs.get("strip_whitespace", True)

        pred = prediction
        ref = reference

        if strip_whitespace:
            pred = pred.strip()
            ref = ref.strip()

        if ignore_case:
            pred = pred.lower()
            ref = ref.lower()

        score = 1.0 if pred == ref else 0.0
        return MetricResult(name=self.name, score=score)
