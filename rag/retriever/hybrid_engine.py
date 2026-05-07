"""
混合检索引擎 - 多列加权检索
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from rag.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class ColumnConfig:
    """列检索配置"""
    name: str
    retriever: BaseRetriever
    weight: float


class HybridSearchEngine:
    """混合检索引擎 - 多列加权检索"""

    def __init__(self, excel_path: str, column_configs: List[ColumnConfig]):
        self.excel_path = excel_path
        self.column_configs = column_configs
        self.df: Optional[pd.DataFrame] = None

        # 归一化权重
        total_w = sum(cfg.weight for cfg in column_configs)
        for cfg in self.column_configs:
            cfg.weight /= total_w

        self._load_and_index()

    def _load_and_index(self):
        self.df = pd.read_excel(self.excel_path).fillna("")
        logger.info(f"加载了 {len(self.df)} 行数据")

        for cfg in self.column_configs:
            docs = self.df[cfg.name].astype(str).tolist()
            logger.info(f"为列 '{cfg.name}' 构建索引...")
            cfg.retriever.build_index(docs)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if query == "":
            return pd.DataFrame()
        assert self.df is not None, "Excel 未加载"

        final_scores = np.zeros(len(self.df), dtype=np.float32)
        for cfg in self.column_configs:
            col_scores = cfg.retriever.get_scores(query)
            if col_scores.shape[0] != len(final_scores):
                raise ValueError(f"列 {cfg.name} 的分数长度不匹配")
            final_scores += cfg.weight * col_scores

        top_idx = np.argsort(final_scores)[::-1][:top_k]
        result_df = self.df.iloc[top_idx].copy()
        result_df["__score"] = final_scores[top_idx]
        result_df = result_df.sort_values("__score", ascending=False).reset_index(drop=True)
        return result_df
