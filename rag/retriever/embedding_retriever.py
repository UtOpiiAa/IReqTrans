"""
基于 Embedding 的向量检索器（纯 NumPy 实现，不依赖 FAISS）
"""

import os
import hashlib
import pickle
import logging
from typing import List, Dict, Optional

import numpy as np

from rag.retriever.base import BaseRetriever
from utils.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


class NumpyVectorIndex:
    """纯 NumPy 实现的向量索引，支持余弦相似度、欧氏距离、内积"""

    def __init__(self, similarity_metric: str = "cosine"):
        self.similarity_metric = similarity_metric
        self.embeddings: Optional[np.ndarray] = None
        self.num_docs = 0
        self.embedding_dim = 0

    def add(self, embeddings: np.ndarray):
        """添加向量到索引"""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings 必须是2维数组，当前维度: {embeddings.ndim}")

        self.embeddings = embeddings.astype(np.float32)
        self.num_docs, self.embedding_dim = self.embeddings.shape

        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.embeddings = self.embeddings / norms

        logger.info(f"向量索引构建完成: {self.num_docs} 个文档, {self.embedding_dim} 维")

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """搜索最相似的向量"""
        if self.embeddings is None:
            raise RuntimeError("索引未构建，请先调用 add() 方法")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        if self.similarity_metric == "cosine":
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_norm = np.where(query_norm == 0, 1, query_norm)
            query_embedding = query_embedding / query_norm
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(-similarities)[:top_k]
            top_scores = similarities[top_indices]
        elif self.similarity_metric == "l2":
            doc_norms_sq = np.sum(self.embeddings ** 2, axis=1)
            query_norm_sq = np.sum(query_embedding ** 2)
            dot_product = np.dot(self.embeddings, query_embedding.T).flatten()
            distances = np.sqrt(doc_norms_sq + query_norm_sq - 2 * dot_product)
            top_indices = np.argsort(distances)[:top_k]
            top_scores = distances[top_indices]
        elif self.similarity_metric == "ip":
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(-similarities)[:top_k]
            top_scores = similarities[top_indices]
        else:
            raise ValueError(f"不支持的相似度度量: {self.similarity_metric}")

        return top_scores, top_indices


class LocalEmbeddingRetriever(BaseRetriever):
    """使用本地模型和 NumPy 实现的密集检索器"""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "./embedding_cache",
        cache_file: str = "embeddings.pkl",
        similarity_metric: str = "cosine",
    ):
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(self.cache_dir, cache_file)
        self.similarity_metric = similarity_metric

        self.embedding_model = EmbeddingClient(
            model_name_or_path=model_name_or_path,
            device=device,
            normalize_embeddings=(similarity_metric == "cosine"),
        )
        self.index = NumpyVectorIndex(similarity_metric=similarity_metric)
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._docs: List[str] = []

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_cache()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._emb_cache = pickle.load(f)
                logger.info(f"从缓存加载 {len(self._emb_cache)} 条 embedding")
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
                self._emb_cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._emb_cache, f)
            logger.info(f"保存 {len(self._emb_cache)} 条 embedding 到缓存")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """批量获取 embeddings（带缓存）"""
        embeddings_list = []
        texts_to_encode = []
        text_to_index = []

        for idx, text in enumerate(texts):
            h = self._hash(text)
            if self.use_cache and h in self._emb_cache:
                embeddings_list.append((idx, self._emb_cache[h]))
            else:
                texts_to_encode.append(text)
                text_to_index.append((idx, h))

        if texts_to_encode:
            logger.info(f"编码 {len(texts_to_encode)} 条新文本...")
            new_embeddings = self.embedding_model.encode(texts_to_encode)
            for (original_idx, h), emb in zip(text_to_index, new_embeddings):
                if self.use_cache:
                    self._emb_cache[h] = emb
                embeddings_list.append((original_idx, emb))
            if self.use_cache:
                self._save_cache()

        embeddings_list.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in embeddings_list]).astype(np.float32)

    def build_index(self, documents: List[str]):
        logger.info(f"开始构建索引，文档数: {len(documents)}")
        self._docs = documents
        embeddings = self._get_embeddings(documents)
        self.index.add(embeddings)
        logger.info("索引构建完成")

    def get_scores(self, query: str) -> np.ndarray:
        if self.index.embeddings is None:
            raise RuntimeError("索引未构建，请先调用 build_index")

        q_emb = self._get_embeddings([query])
        scores, indices = self.index.search(q_emb, top_k=len(self._docs))

        scores_original_order = np.zeros(len(self._docs), dtype=np.float32)
        scores_original_order[indices] = scores

        if self.similarity_metric == "l2":
            scores_original_order = 1 / (1 + scores_original_order)

        return self.min_max_normalize(scores_original_order)
