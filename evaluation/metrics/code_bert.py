"""CodeBERT Score 指标。"""

import logging
from typing import Optional

from ..metrics.base import BaseMetric, MetricResult, register_metric

logger = logging.getLogger(__name__)


@register_metric
class CodeBERTScore(BaseMetric):
    """CodeBERT 语义相似度分数。

    使用 CodeBERT 模型计算预测与参考之间的语义相似度。
    需要安装 transformers 和 torch。
    """

    name = "code_bert"
    description = "Semantic similarity score using CodeBERT model"
    higher_is_better = True
    value_range = (0.0, 1.0)

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load_model(self, model_name: str = "microsoft/codebert-base"):
        """延迟加载模型。"""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "CodeBERT score requires transformers and torch. "
                "Install with: pip install transformers torch"
            )

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        model_name = kwargs.get("model_name", "microsoft/codebert-base")

        try:
            self._load_model(model_name)
            import torch

            # 编码
            pred_inputs = self._tokenizer(
                prediction, return_tensors="pt", truncation=True, max_length=512,
            )
            ref_inputs = self._tokenizer(
                reference, return_tensors="pt", truncation=True, max_length=512,
            )

            with torch.no_grad():
                pred_emb = self._model(**pred_inputs).last_hidden_state.mean(dim=1)
                ref_emb = self._model(**ref_inputs).last_hidden_state.mean(dim=1)

            # 余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(pred_emb, ref_emb)
            score = cos_sim.item()
            # 映射到 [0, 1]
            score = (score + 1) / 2

            return MetricResult(name=self.name, score=score)

        except Exception as e:
            logger.warning(f"CodeBERT computation failed: {e}, falling back to 0.0")
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)},
            )
