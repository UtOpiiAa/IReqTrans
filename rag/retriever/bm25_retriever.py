"""
BM25 检索器
"""

import logging
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from rag.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 关键词检索器"""

    def __init__(self, tokenizer=str.split):
        self._bm25 = None

    def build_index(self, documents: List[str]):
        self._bm25 = BM25Okapi(documents)
        logger.info(f"BM25 索引构建完成，文档数: {len(documents)}")

    def get_scores(self, query: str) -> np.ndarray:
        assert self._bm25 is not None, "BM25 index not built. Call build_index first."
        scores = np.asarray(self._bm25.get_scores(query), dtype=np.float32)
        return self.min_max_normalize(scores)
