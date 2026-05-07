"""
检索器基类
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseRetriever(ABC):
    """检索器抽象基类"""

    @abstractmethod
    def build_index(self, documents: List[str]):
        """构建索引"""
        raise NotImplementedError

    @abstractmethod
    def get_scores(self, query: str) -> np.ndarray:
        """
        返回 (len(documents),) 的分数向量，越大越好
        """
        raise NotImplementedError

    @staticmethod
    def min_max_normalize(arr: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [0, 1]"""
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.full_like(arr, 0.5)
        return (arr - min_val) / (max_val - min_val)
