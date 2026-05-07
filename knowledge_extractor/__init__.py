"""
知识提取模块 - 从需求文本和形式化代码中提取领域知识
"""

from knowledge_extractor.extractor import KnowledgeExtractor
from knowledge_extractor.merger import KnowledgeMerger

__all__ = ["KnowledgeExtractor", "KnowledgeMerger"]
