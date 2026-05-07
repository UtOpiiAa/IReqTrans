"""编辑相似度 (Edit Similarity) 指标。"""

from ..metrics.base import BaseMetric, MetricResult, register_metric


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算 Levenshtein 编辑距离。"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


@register_metric
class EditSimilarity(BaseMetric):
    """编辑相似度：基于 Levenshtein 距离的归一化相似度。

    score = 1 - edit_distance / max(len(pred), len(ref))
    """

    name = "edit_similarity"
    description = "Normalized edit similarity based on Levenshtein distance"
    higher_is_better = True
    value_range = (0.0, 1.0)

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        level = kwargs.get("level", "char")  # "char" or "token"

        pred = prediction.strip()
        ref = reference.strip()

        if level == "token":
            pred_units = pred.split()
            ref_units = ref.split()
        else:
            pred_units = pred
            ref_units = ref

        if not pred_units and not ref_units:
            return MetricResult(name=self.name, score=1.0)

        if not pred_units or not ref_units:
            return MetricResult(name=self.name, score=0.0)

        distance = _levenshtein_distance(
            pred_units if level == "char" else " ".join(pred_units),
            ref_units if level == "char" else " ".join(ref_units),
        )

        max_len = max(len(pred) if level == "char" else len(pred_units),
                      len(ref) if level == "char" else len(ref_units))
        score = max(0.0, 1.0 - distance / max_len)

        return MetricResult(
            name=self.name,
            score=score,
            details={"edit_distance": distance, "level": level},
        )
