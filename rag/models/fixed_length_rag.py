"""
FixedLengthRAG - 基于固定长度触发检索的 RAG 系统
"""

import logging
from typing import Optional

from config import Config
from rag.models.basic_rag import BasicRAG
from rag.prompts import get_mini_system_prompt_with_few_shots, RAG_PROMPT

logger = logging.getLogger("FixedLengthRAG")


class FixedLengthRAG(BasicRAG):
    """基于固定长度触发检索的 RAG"""

    def __init__(self, config: Config, retrieval_interval: Optional[int] = None):
        super().__init__(config)

        self.END_TOKEN = self.generator.end_token
        rag_cfg = self.config.rag
        self.batch_size = rag_cfg.batch_size
        self.retrieval_interval = retrieval_interval or rag_cfg.retrieval_interval
        logger.info(f"检索间隔: {self.retrieval_interval} 字符")
        self.stop_words = []
        self.clear_cache()

    def clear_cache(self):
        self.generated_text = ""

    def _build_combined_query(self, original_query: str, generated_context: str) -> str:
        """构建组合查询：原始查询 + 最新生成内容"""
        if not generated_context.strip():
            return original_query
        context_snippet = generated_context[-200:]
        return f"{original_query}\n\n最新生成内容：{context_snippet}"

    def inference(self, query: str, max_length: int = 8191) -> str:
        try:
            logger.info(
                f"开始推理 [max_length={max_length}, interval={self.retrieval_interval}]"
            )
            self.clear_cache()

            retrieve_target = query
            implicit_knowledge = self._load_implicit_knowledge(query)
            few_shots = self._load_few_shots(query)

            explicit_knowledge = self._perform_retrieval_for_few_shot(
                retrieve_target, original_query=query
            )
            if not explicit_knowledge:
                return "检索失败，无法生成内容"

            while len(self.generated_text) < max_length:
                system_prompt = get_mini_system_prompt_with_few_shots(explicit_knowledge)
                query_prompt = RAG_PROMPT.format(requirement=query)

                new_text, tokens, attentions, logprobs, entropies, stop_focus = (
                    self.generator.generate_attn(
                        query_text=query_prompt,
                        system_prompt=system_prompt,
                        generated_text=self.generated_text,
                        max_length=self.retrieval_interval,
                        use_entropy=False,
                        use_logprob=False,
                        stop_words=self.stop_words,
                    )
                )

                if not new_text:
                    break

                if self.END_TOKEN in new_text:
                    idx = new_text.index(self.END_TOKEN)
                    new_text = new_text[:idx] + self.END_TOKEN
                    self.generated_text += new_text
                    break

                self.generated_text += new_text
                logger.info(f"生成片段 [{len(new_text)} 字符]: {new_text.strip()[:80]}...")

                combined_query = self._build_combined_query(query, self.generated_text)
                new_knowledge = self._perform_retrieval_for_few_shot(
                    combined_query, original_query=query
                )
                if new_knowledge:
                    explicit_knowledge = new_knowledge

            return self._clean_end_token(self.generated_text)

        except Exception as e:
            logger.error(f"推理异常: {e}", exc_info=True)
            return f"Error during inference: {str(e)}"

    def _clean_end_token(self, text):
        if self.END_TOKEN in text:
            return text.split(self.END_TOKEN)[0]
        return text
