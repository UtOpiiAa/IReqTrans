"""
DynamicTokenRAG - 基于 Token 级别的动态 RAG 系统

在生成过程中监控注意力(Attention)和熵(Entropy)，动态检测并触发检索。
"""

import os
import logging
from typing import List, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

from config import Config
from rag.models.basic_rag import BasicRAG
from rag.prompts import get_system_prompt_with_few_shots, RAG_PROMPT

logger = logging.getLogger("DynamicTokenRAG")


class DynamicTokenRAG(BasicRAG):
    """基于 Token 级别的动态 RAG 系统"""

    def __init__(self, config: Config):
        super().__init__(config)

        self.END_TOKEN = self.generator.end_token
        logger.info(f"使用结束标记: {self.END_TOKEN}")

        rag_cfg = self.config.rag
        self.hallucination_threshold = rag_cfg.hallucination_threshold
        self.batch_size = rag_cfg.batch_size
        self.attention_threshold_percentile = rag_cfg.attention_threshold_percentile

        self.stop_words = ["&", "#", "->"]
        self.clear_cache()
        self.retrieval_log = []
        self.current_requirement_id = ""

    def clear_cache(self):
        self.generated_text = ""

    def inference(self, query: str, max_length: int = 8191) -> str:
        """主推理流程"""
        try:
            logger.info(f"开始推理任务 [max_length={max_length}]")
            self.clear_cache()
            self.current_requirement_id = self._extract_requirement_id(query)
            self.retrieval_log = []

            retrieve_target: Union[str, List[str]] = query
            implicit_knowledge = self._load_implicit_knowledge(query)
            few_shots = self._load_few_shots(query)

            round_num = 0
            while len(self.generated_text) < max_length:
                round_num += 1
                if self.END_TOKEN in self.generated_text:
                    logger.info("检测到结束符，停止生成")
                    break

                remaining_length = min(self.batch_size, max_length - len(self.generated_text))

                explicit_knowledge, retrieval_info = self._perform_retrieval_with_log(
                    retrieve_target, original_query=query
                )
                if not explicit_knowledge:
                    break

                self._log_retrieval(round_num, query, retrieve_target, retrieval_info)

                system_prompt = get_system_prompt_with_few_shots(
                    implicit_knowledge, explicit_knowledge, few_shots
                )
                query_prompt = RAG_PROMPT.format(requirement=query)

                new_text, tokens, attentions, logprobs, entropies, stop_focus = (
                    self.generator.generate_attn(
                        query_text=query_prompt,
                        system_prompt=system_prompt,
                        generated_text=self.generated_text,
                        max_length=remaining_length,
                        use_entropy=True,
                        use_logprob=True,
                        stop_words=self.stop_words,
                        attention_threshold_percentile=self.attention_threshold_percentile,
                    )
                )

                if not new_text:
                    logger.warning("生成结果为空，停止循环")
                    break

                retrieve_target = self._handle_stop_focus(stop_focus, retrieve_target)
                new_text, tokens, attentions, entropies = self._truncate_at_end_token(
                    new_text, tokens, attentions, entropies
                )
                self.generated_text += new_text
                logger.info(f"生成片段: {new_text.strip()[:80]}...")

            self._save_retrieval_log()
            return self._clean_end_token(self.generated_text)

        except Exception as e:
            logger.error(f"推理过程发生异常: {e}", exc_info=True)
            return f"Error during inference: {str(e)}"

    def _handle_stop_focus(self, stop_focus, current_target):
        if not stop_focus:
            return current_target
        focus_tokens = stop_focus.get("focus_tokens", [])
        following_token = stop_focus.get("following_token", "")
        if focus_tokens:
            return focus_tokens
        elif following_token:
            return [following_token]
        return current_target

    def _truncate_at_end_token(self, text, tokens, attentions, entropies):
        if self.END_TOKEN not in text:
            return text, tokens, attentions, entropies
        idx = text.index(self.END_TOKEN)
        text = text[:idx] + self.END_TOKEN
        limit_len = min(len(tokens), idx) if tokens else 0
        return text, tokens[:limit_len], attentions[:limit_len], entropies[:limit_len]

    def _clean_end_token(self, text):
        if self.END_TOKEN in text:
            return text.split(self.END_TOKEN)[0]
        return text

    def _extract_requirement_id(self, query: str) -> str:
        lines = query.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            if first_line and not first_line.startswith("If"):
                return first_line
        return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _perform_retrieval_with_log(self, query_or_keywords, original_query):
        retrieval_info = []
        all_docs = []
        seen_contents = set()

        keywords = query_or_keywords if isinstance(query_or_keywords, list) else [query_or_keywords]

        for kw in keywords:
            df = self.retrieve(kw)
            if df.empty:
                continue
            for _, row in df.iterrows():
                content = self._format_doc_content_for_log(row, kw)
                func_name = str(row.get("Function in Code", "")).strip()
                if content and content not in seen_contents:
                    all_docs.append(content)
                    seen_contents.add(content)
                    if func_name:
                        retrieval_info.append(
                            {"keyword": kw, "function_name": func_name, "content": content}
                        )

        if not all_docs:
            df = self.retrieve(original_query)
            if not df.empty:
                for _, row in df.iterrows():
                    content = self._format_doc_content_for_log(row, original_query)
                    func_name = str(row.get("Function in Code", "")).strip()
                    if content and content not in seen_contents:
                        all_docs.append(content)
                        seen_contents.add(content)
                        if func_name:
                            retrieval_info.append(
                                {
                                    "keyword": original_query,
                                    "function_name": func_name,
                                    "content": content,
                                }
                            )

        if not all_docs:
            return "", []

        formatted = "\n".join([f"{i + 1}. {doc}" for i, doc in enumerate(all_docs)])
        return formatted, retrieval_info

    def _format_doc_content_for_log(self, row, kw_prefix=""):
        parts = []
        if row.get("OM Attribute", ""):
            parts.append(f"关键词: {str(row['OM Attribute']).strip()}")
        if row.get("Function in Code", ""):
            parts.append(f"对应的函数名称: {str(row['Function in Code']).strip()}")
        content = " | ".join(parts)
        if kw_prefix and content:
            content = f"[检索词: {kw_prefix}] {content}"
        return content

    def _log_retrieval(self, round_num, original_query, retrieve_target, retrieval_info):
        keywords = retrieve_target if isinstance(retrieve_target, list) else [retrieve_target]
        for info in retrieval_info:
            self.retrieval_log.append(
                {
                    "requirement_id": self.current_requirement_id,
                    "original_query": original_query,
                    "round": round_num,
                    "search_keyword": info["keyword"],
                    "retrieved_function_name": info["function_name"],
                    "retrieved_content": info["content"],
                }
            )

    def _save_retrieval_log(self):
        if not self.retrieval_log:
            return
        df = pd.DataFrame(self.retrieval_log)
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(
            log_dir,
            f"all_retrieval_logs_p{self.attention_threshold_percentile:.0f}_k{self.retrieval_top_k}.xlsx",
        )
        existing_df = pd.DataFrame()
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_excel(filepath, engine="openpyxl")
            except Exception:
                pass
        combined_df = pd.concat([existing_df, df], ignore_index=True) if not existing_df.empty else df

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name="检索日志", index=False)

        logger.info(f"检索日志已保存至: {filepath}")
