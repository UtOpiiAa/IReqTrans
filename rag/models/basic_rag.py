"""
BasicRAG - 基础检索增强生成

一次性检索相关文档 + 生成，支持 few-shot
"""

import os
import logging
from typing import List, Union, Optional

import pandas as pd

from config import Config
from utils.llm_client import LLMClient
from utils.file_utils import FileReader
from rag.retriever import BM25Retriever, LocalEmbeddingRetriever, HybridSearchEngine, ColumnConfig
from rag.prompts import (
    get_system_prompt_with_few_shots,
    get_mini_system_prompt_with_few_shots,
    RAG_PROMPT,
)

logger = logging.getLogger(__name__)


class BasicRAG:
    """基础 RAG：一次性检索 + 生成"""

    def __init__(self, config: Config):
        self.config = config
        rag_cfg = config.rag

        # 解析路径
        self.model_path = self._resolve_path(config.llm.model_name)
        self.knowledge_path = config.resolve_path(rag_cfg.knowledge_path)
        self.implicit_knowledge_path = config.resolve_path(rag_cfg.implicit_knowledge_path)
        self.requirement_path = config.resolve_path(rag_cfg.requirements_train_path)
        self.embedding_cache_dir = config.resolve_path(rag_cfg.embedding_cache_dir)
        self.retrieval_top_k = rag_cfg.retrieval_top_k
        self.max_tokens = config.llm.max_tokens

        # 初始化组件
        self._init_retriever()
        self.generator = LLMClient(self.model_path, max_tokens=self.max_tokens)

    def _resolve_path(self, path_str: str) -> str:
        if not path_str:
            return ""
        if os.path.isabs(path_str):
            return path_str
        return path_str

    def _init_retriever(self):
        """初始化混合检索引擎"""
        rag_cfg = self.config.rag
        emb_model_path = self._resolve_path(self.config.embedding.model_name)

        # 确保缓存目录存在
        if rag_cfg.use_embedding and not os.path.exists(self.embedding_cache_dir):
            os.makedirs(self.embedding_cache_dir, exist_ok=True)

        # OM 知识库列配置
        column_cfgs_om = [
            ColumnConfig(
                name="OM Attribute",
                retriever=BM25Retriever(),
                weight=rag_cfg.om_keyword_weight,
            ),
            ColumnConfig(
                name="OM Attribute",
                retriever=LocalEmbeddingRetriever(
                    model_name_or_path=emb_model_path,
                    cache_dir=self.embedding_cache_dir,
                    cache_file="om_attribute_cache.pkl",
                    similarity_metric="cosine",
                ),
                weight=rag_cfg.om_function_weight,
            ),
        ]

        self.retriever = HybridSearchEngine(self.knowledge_path, column_cfgs_om)

        # Few-shot 检索器（基于需求文本的 embedding 检索）
        column_cfgs_req = [
            ColumnConfig(
                name="req_text",
                retriever=LocalEmbeddingRetriever(
                    model_name_or_path=emb_model_path,
                    cache_dir=self.embedding_cache_dir,
                    cache_file="req_id_cache.pkl",
                    similarity_metric="cosine",
                ),
                weight=1.0,
            ),
        ]
        self.requirement_retriever = HybridSearchEngine(self.requirement_path, column_cfgs_req)

    def retrieve(self, query: str) -> pd.DataFrame:
        """检索相关文档"""
        return self.retriever.search(query, top_k=self.retrieval_top_k)

    def inference(self, query: str) -> str:
        """执行 RAG 推理流程"""
        docs_df = self.retrieve(query)

        docs = []
        for _, row in docs_df.iterrows():
            doc_text = f"OM Attribute: {row.get('OM Attribute', '')}\n"
            doc_text += f"Function Name: {row.get('Function in Code', '')}\n"
            docs.append(doc_text)

        few_shots = self._load_few_shots(query)
        system_prompt = get_mini_system_prompt_with_few_shots(few_shots)
        prompt = RAG_PROMPT.format(requirement=query)

        logger.info(f"检索到 {len(docs)} 条文档")

        response, _, _ = self.generator.generate(
            query_text=prompt, system_prompt=system_prompt, max_length=self.max_tokens
        )

        end_token = self.generator.end_token
        if end_token in response:
            response = response.split(end_token)[0]

        return response

    def clear_cache(self):
        self.generated_text = ""

    def _load_implicit_knowledge(self, query: str) -> str:
        """根据 Query 中的 Req ID 加载隐式知识"""
        try:
            target_req_id = query.split("\n")[0].strip()
            reader = FileReader(self.implicit_knowledge_path)
            data = reader.get_keyword_explanation_by_id("req_id", target_req_id)
            logger.info(f"隐式知识加载成功 [req_id={target_req_id}]")
            return data
        except Exception as e:
            logger.warning(f"隐式知识加载失败: {e}")
            return ""

    def _load_few_shots(self, query: str) -> str:
        """使用检索方式加载 few-shot 示例"""
        try:
            query_lines = query.split("\n")
            target_req_id = query_lines[0].strip()
            target_req_text = "\n".join(query_lines[1:]).strip() if len(query_lines) > 1 else ""

            code_path = self.config.resolve_path(self.config.rag.formal_code_path)
            code_df = pd.read_excel(code_path)

            retrieval_results = self.requirement_retriever.search(target_req_text, top_k=2)

            few_shots = []
            example_num = 1
            for _, row in retrieval_results.iterrows():
                req_id = row.get("req_id", "")
                if req_id == target_req_id:
                    continue
                req_text = row.get("req_text", "")
                code_row = code_df[code_df["名称"] == req_id]
                if len(code_row) == 0:
                    continue
                code = code_row["代码"].values[0]
                if not code:
                    continue
                few_shots.append(
                    f"【示例{example_num}】\n需求ID: {req_id}\n需求内容: {req_text}\n翻译代码:\n{code}"
                )
                example_num += 1
                if example_num > 3:
                    break

            if few_shots:
                logger.info(f"Few-shot 检索加载成功 [req_id={target_req_id}, 示例数={len(few_shots)}]")
                return "\n\n".join(few_shots)

            logger.warning("Few-shot 检索加载失败")
            return ""
        except Exception as e:
            logger.warning(f"Few-shot 检索加载异常: {e}")
            return ""

    def _perform_retrieval_for_few_shot(
        self, query_or_keywords: Union[str, List[str]], original_query: str
    ) -> str:
        """执行检索并格式化结果用于 few-shot"""
        all_docs = []
        seen_contents = set()

        if isinstance(query_or_keywords, list):
            for kw in query_or_keywords:
                docs = self._retrieve_by_keyword(kw)
                for doc in docs[:3]:
                    if doc not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc)
        else:
            all_docs = self._retrieve_by_keyword(query_or_keywords)

        if not all_docs:
            all_docs = self._retrieve_by_keyword(original_query)[:5]

        if not all_docs:
            return ""

        return "\n".join([f"{i + 1}. {doc}" for i, doc in enumerate(all_docs)])

    def _retrieve_by_keyword(self, keyword: str) -> List[str]:
        """根据关键词检索并格式化结果"""
        try:
            df = self.retrieve(keyword)
            if df.empty:
                return []
            results = []
            for _, row in df.iterrows():
                content = self._format_doc_content(row)
                if content:
                    results.append(content)
            return results
        except Exception as e:
            logger.warning(f"关键词 '{keyword}' 检索失败: {e}")
            return []

    def _format_doc_content(self, row: pd.Series) -> str:
        """格式化单条文档内容"""
        parts = []
        if row.get("OM Attribute", ""):
            parts.append(f"关键词: {str(row['OM Attribute']).strip()}")
        if row.get("Function in Code", ""):
            parts.append(f"对应的函数名称: {str(row['Function in Code']).strip()}")
        return " | ".join(parts)
