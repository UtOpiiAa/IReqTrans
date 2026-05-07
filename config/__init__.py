"""
RE-LLM 配置管理模块

统一管理所有配置项，支持 YAML 文件加载和环境变量覆盖。
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录（rellm_v2/）"""
    return Path(__file__).resolve().parent.parent


# ==================== 数据类定义 ====================

@dataclass
class LLMConfig:
    """LLM 模型配置"""
    model_name: str = "Qwen/Qwen3-8B"
    temperature: float = 0.0
    max_tokens: int = 8191


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    max_tokens: int = 8191


@dataclass
class RAGConfig:
    """RAG 核心配置"""
    # 知识库路径
    knowledge_path: str = "knowledge/om_knowledge.xlsx"
    implicit_knowledge_path: str = "knowledge/knowledge_implicit.xlsx"
    requirements_train_path: str = "knowledge/requirements_train.xlsx"
    requirements_test_path: str = "knowledge/requirements_test.xlsx"
    formal_code_path: str = "knowledge/reqv039.xlsx"

    # 检索参数
    retrieval_top_k: int = 3
    use_embedding: bool = True
    embedding_cache_dir: str = "model/embedding_cache"

    # OM 模式权重
    om_keyword_weight: float = 0.6
    om_function_weight: float = 0.4

    # 显式知识模式权重 (4列)
    source_keyword_weight: float = 0.6
    formal_element_weight: float = 0.2
    description_weight: float = 0.1
    example_context_weight: float = 0.1

    # 动态 RAG 参数
    hallucination_threshold: float = 0.8
    batch_size: int = 64

    # TokenRAG
    attention_threshold_percentile: float = 75.0

    # FLARE
    look_ahead_steps: int = 64
    low_confidence_threshold: float = 0.4
    filter_confidence_threshold: float = 0.8
    retrieval_frequency: int = 64
    max_query_length: int = 64

    # FixedLengthRAG
    retrieval_interval: int = 50

    # FixedSentenceRAG
    sentence_delimiter: str = ";"


@dataclass
class OutputConfig:
    """输出配置"""
    result_dir: str = "data/result"
    log_dir: str = "logs"


@dataclass
class ExtractionAPIConfig:
    """知识提取 API 配置"""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "YOUR_API_KEY"
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 120


@dataclass
class ExtractionConfig:
    """知识提取模块配置"""
    api: ExtractionAPIConfig = field(default_factory=ExtractionAPIConfig)
    requirements_file: str = "knowledge/requirements.xlsx"
    formal_code_file: str = "knowledge/reqv052.xlsx"
    output_dir: str = "extraction_output"
    sleep_interval: float = 1.0
    max_retries: int = 3
    retry_delay: int = 5
    start_index: int = 0


@dataclass
class ProjectConfig:
    """项目配置"""
    data_dir: str = "data"


@dataclass
class Config:
    """总配置"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """从 YAML 文件加载配置"""
        if not os.path.exists(yaml_path):
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        # 递归构建嵌套 dataclass
        return cls(
            project=ProjectConfig(**config_dict.get("project", {})),
            llm=LLMConfig(**config_dict.get("llm", {})),
            embedding=EmbeddingConfig(**config_dict.get("embedding", {})),
            rag=RAGConfig(**config_dict.get("rag", {})),
            output=OutputConfig(**config_dict.get("output", {})),
            extraction=ExtractionConfig(
                api=ExtractionAPIConfig(**config_dict.get("extraction", {}).get("api", {})),
                **{k: v for k, v in config_dict.get("extraction", {}).items() if k != "api"}
            ),
        )

    def resolve_path(self, relative_path: str) -> str:
        """将相对路径解析为绝对路径"""
        if not relative_path:
            return ""
        if os.path.isabs(relative_path):
            return relative_path
        full_path = get_project_root() / self.project.data_dir / relative_path
        if full_path.exists():
            return str(full_path)
        # 不存在时也返回完整路径（可能是 HuggingFace Model ID 等）
        return str(full_path)
