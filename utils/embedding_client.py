"""
Embedding 客户端 - 本地 Embedding 模型封装
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """本地 Embedding 模型客户端，支持批量编码和缓存"""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"加载 Embedding 模型: {model_name_or_path} (设备: {self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        # 获取 embedding 维度
        with torch.no_grad():
            dummy_input = self.tokenizer(
                "test", return_tensors="pt", max_length=self.max_length, truncation=True
            ).to(self.device)
            dummy_output = self.model(**dummy_input)
            self.embedding_dim = self._mean_pooling(
                dummy_output, dummy_input["attention_mask"]
            ).shape[1]

        logger.info(f"Embedding 维度: {self.embedding_dim}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 对 token embeddings 取平均"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        批量编码文本为向量

        Returns:
            shape 为 (len(texts), embedding_dim) 的 numpy 数组
        """
        if not texts:
            return np.array([])

        # 处理空文本
        processed_texts = []
        empty_indices = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                empty_indices.append(idx)
                processed_texts.append("[EMPTY]")
            else:
                processed_texts.append(text.strip())

        all_embeddings = []
        num_batches = (len(processed_texts) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in range(0, len(processed_texts), self.batch_size):
                batch_texts = processed_texts[i : i + self.batch_size]

                if show_progress and num_batches > 1:
                    logger.info(f"处理批次 {i // self.batch_size + 1}/{num_batches}")

                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                model_output = self.model(**encoded_input)
                batch_embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )

                if self.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, p=2, dim=1
                    )

                all_embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # 空文本 embedding 设为零向量
        for idx in empty_indices:
            embeddings[idx] = np.zeros(self.embedding_dim, dtype=np.float32)

        return embeddings
