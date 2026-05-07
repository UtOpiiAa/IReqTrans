"""评测指标基类与注册表。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type


@dataclass
class MetricResult:
    """单个评测指标的结果。"""
    name: str
    score: float
    details: Optional[dict] = None

    def __repr__(self) -> str:
        return f"MetricResult({self.name}={self.score:.4f})"


class BaseMetric(ABC):
    """评测指标基类。"""

    name: str = ""
    description: str = ""
    higher_is_better: bool = True
    value_range: tuple = (0.0, 1.0)

    @abstractmethod
    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """计算单个样本的指标值。

        Args:
            prediction: 预测文本
            reference: 参考文本
            **kwargs: 额外参数

        Returns:
            MetricResult 包含指标名称、分数和可选的详情
        """
        ...

    def compute_batch(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> List[MetricResult]:
        """批量计算指标。

        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            **kwargs: 额外参数

        Returns:
            MetricResult 列表
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                f"must have the same length"
            )
        return [
            self.compute(pred, ref, **kwargs)
            for pred, ref in zip(predictions, references)
        ]

    def compute_average(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> float:
        """计算批量指标的平均值。"""
        results = self.compute_batch(predictions, references, **kwargs)
        return sum(r.score for r in results) / len(results) if results else 0.0


# ── 指标注册表 ─────────────────────────────────────────

_METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}


def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    """注册评测指标类。"""
    if cls.name:
        _METRIC_REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> "BaseMetric":
    """获取已注册的评测指标实例。"""
    if name not in _METRIC_REGISTRY:
        available = ", ".join(_METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric: {name}. Available: {available}")
    return _METRIC_REGISTRY[name]()


def get_all_metrics() -> List[BaseMetric]:
    """获取所有已注册的评测指标实例。"""
    return [cls() for cls in _METRIC_REGISTRY.values()]
