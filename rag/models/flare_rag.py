"""
FlareRAG - 基于低置信度检测的动态 RAG 系统

核心机制：
1. 前瞻生成(Look-ahead)
2. 低置信度检测
3. 自适应查询构建
4. 迭代生成
"""

import logging
from typing import List, Optional, Union

import numpy as np

from config import Config
from rag.models.basic_rag import BasicRAG
from rag.prompts import get_mini_system_prompt_with_few_shots, RAG_PROMPT

logger = logging.getLogger("FlareRAG")


class FlareRAG(BasicRAG):
    """基于 FLARE 机制的动态 RAG"""

    def __init__(self, config: Config):
        super().__init__(config)

        self.END_TOKEN = self.generator.end_token
        rag_cfg = self.config.rag

        self.look_ahead_steps = rag_cfg.look_ahead_steps
        self.low_confidence_threshold = rag_cfg.low_confidence_threshold
        self.filter_confidence_threshold = rag_cfg.filter_confidence_threshold
        self.retrieval_frequency = rag_cfg.retrieval_frequency
        self.batch_size = rag_cfg.batch_size
        self.max_query_length = rag_cfg.max_query_length

        self.clear_cache()

    def clear_cache(self):
        self.generated_text = ""
        self.retrieval_history = []
        self.total_tokens_generated = 0
        self.retrieval_count = 0

    def _detect_low_confidence(self, tokens: List[str], probs: List[float]):
        """检测低置信度 token"""
        low_conf_indices = []
        low_conf_tokens = []
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            if prob <= self.low_confidence_threshold:
                low_conf_indices.append(i)
                low_conf_tokens.append(token)

        has_low_confidence = len(low_conf_indices) > 0
        if has_low_confidence:
            logger.info(f"检测到 {len(low_conf_indices)} 个低置信度 token (阈值={self.low_confidence_threshold})")

        return has_low_confidence, low_conf_indices, low_conf_tokens

    def _build_query_from_low_confidence(self, tokens, probs, low_conf_indices, mask_method="simple"):
        """基于低置信度 token 构建检索查询"""
        if mask_method == "simple":
            filtered_tokens = []
            for i, (token, prob) in enumerate(zip(tokens, probs)):
                if prob > self.filter_confidence_threshold:
                    filtered_tokens.append(token)
                else:
                    filtered_tokens.append(" ")
            query = "".join(filtered_tokens).strip()
            query = " ".join(query.split())
        elif mask_method == "context":
            if not low_conf_indices:
                return "".join(tokens)
            first_low_idx = low_conf_indices[0]
            start = max(0, first_low_idx - 5)
            end = min(len(tokens), first_low_idx + 5)
            query = "".join(tokens[start:end]).strip()
        else:
            query = "".join(tokens).strip()

        if len(query) > self.max_query_length:
            query = query[: self.max_query_length]
        return query

    def _look_ahead_generate(self, query_prompt, system_prompt, max_tokens):
        """前瞻生成"""
        text, tokens, _, logprobs, _, _ = self.generator.generate_attn(
            query_text=query_prompt,
            system_prompt=system_prompt,
            generated_text=self.generated_text,
            max_length=max_tokens,
            use_logprob=True,
            use_entropy=False,
            stop_words=[],
        )
        probs = [np.exp(lp) if lp is not None else 0.0 for lp in (logprobs or [])]
        return text, tokens, probs

    def inference(self, query: str, max_length: int = 8191) -> str:
        try:
            logger.info(f"开始 FLARE 推理 [max_length={max_length}]")
            self.clear_cache()

            implicit_knowledge = self._load_implicit_knowledge(query)
            few_shots = self._load_few_shots(query)
            explicit_knowledge = self._perform_retrieval_for_few_shot(query, query)

            iteration = 0
            while self.total_tokens_generated < max_length:
                iteration += 1
                logger.info(f"=== 迭代 {iteration} (已生成 {self.total_tokens_generated} tokens) ===")

                if self.END_TOKEN in self.generated_text:
                    logger.info("检测到结束符，停止生成")
                    break

                system_prompt = get_mini_system_prompt_with_few_shots(explicit_knowledge)
                query_prompt = RAG_PROMPT.format(requirement=query)

                # FLARE 步骤1: 前瞻生成
                look_ahead_text, look_ahead_tokens, look_ahead_probs = self._look_ahead_generate(
                    query_prompt, system_prompt, self.look_ahead_steps
                )
                if not look_ahead_text:
                    break

                # 处理结束符
                if self.END_TOKEN in look_ahead_text:
                    end_idx = look_ahead_text.index(self.END_TOKEN)
                    look_ahead_text = look_ahead_text[:end_idx]
                    look_ahead_tokens = look_ahead_tokens[:end_idx]
                    look_ahead_probs = look_ahead_probs[:end_idx]

                # FLARE 步骤2: 低置信度检测
                has_low_conf, low_conf_indices, low_conf_tokens = self._detect_low_confidence(
                    look_ahead_tokens, look_ahead_probs
                )

                # FLARE 步骤3: 触发检索
                if has_low_conf:
                    logger.info("触发检索（检测到低置信度）")
                    self.retrieval_count += 1
                    retrieval_query = self._build_query_from_low_confidence(
                        look_ahead_tokens, look_ahead_probs, low_conf_indices, mask_method="simple"
                    )
                    if len(retrieval_query.strip()) < 3:
                        retrieval_query = query
                    new_knowledge = self._perform_retrieval_for_few_shot(retrieval_query, query)
                    if new_knowledge:
                        explicit_knowledge = new_knowledge

                # FLARE 步骤4: 正式生成
                system_prompt = get_mini_system_prompt_with_few_shots(explicit_knowledge)
                remaining = max_length - self.total_tokens_generated
                gen_length = min(self.retrieval_frequency, remaining)

                new_text, tokens, _, _, _, _ = self.generator.generate_attn(
                    query_text=query_prompt,
                    system_prompt=system_prompt,
                    generated_text=self.generated_text,
                    max_length=gen_length,
                    use_logprob=False,
                    use_entropy=False,
                    stop_words=[],
                )

                if not new_text:
                    break

                if self.END_TOKEN in new_text:
                    end_idx = new_text.index(self.END_TOKEN)
                    new_text = new_text[:end_idx] + self.END_TOKEN

                self.generated_text += new_text
                self.total_tokens_generated += len(tokens) if tokens else len(new_text)
                logger.info(f"生成片段: {new_text.strip()[:80]}...")

            final_text = self._clean_end_token(self.generated_text)
            logger.info(
                f"推理完成，共 {iteration} 次迭代，"
                f"生成 {self.total_tokens_generated} tokens，"
                f"触发 {self.retrieval_count} 次检索"
            )
            return final_text

        except Exception as e:
            logger.error(f"推理异常: {e}", exc_info=True)
            return f"Error during inference: {str(e)}"

    def _clean_end_token(self, text):
        if self.END_TOKEN in text:
            return text.split(self.END_TOKEN)[0]
        return text
