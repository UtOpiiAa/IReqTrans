"""
RAG 模块 - 检索增强生成

包含:
- retriever: 检索引擎 (BM25, Embedding, Hybrid)
- models: RAG 方法实现 (NoRAG, BasicRAG, TokenRAG, FixedLengthRAG, FixedSentenceRAG, FlareRAG, DRAG)
- prompts: 提示词模板和构建器
"""

from rag.models import (
    NoRAG,
    BasicRAG,
    DynamicTokenRAG,
    FixedLengthRAG,
    FixedSentenceRAG,
    FlareRAG,
    DRAG,
)

RAG_METHODS = {
    "NoRAG": NoRAG,
    "BasicRAG": BasicRAG,
    "TokenRAG": DynamicTokenRAG,
    "FixedLengthRAG": FixedLengthRAG,
    "FixedSentenceRAG": FixedSentenceRAG,
    "FlareRAG": FlareRAG,
    "DRAG": DRAG,
}

__all__ = [
    "NoRAG", "BasicRAG", "DynamicTokenRAG", "FixedLengthRAG",
    "FixedSentenceRAG", "FlareRAG", "DRAG", "RAG_METHODS",
]
