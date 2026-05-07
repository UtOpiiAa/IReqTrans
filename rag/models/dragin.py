"""
DRAG - 基于注意力+幻觉检测的动态 RAG 系统

在生成过程中检测幻觉，回溯并重新检索。
"""

import logging
from typing import List, Tuple, Optional, Union

import torch
import numpy as np

from config import Config
from rag.models.basic_rag import BasicRAG
from rag.prompts import get_mini_system_prompt_with_few_shots, RAG_PROMPT

logger = logging.getLogger("DRAG")


class DRAG(BasicRAG):
    """基于幻觉检测的动态 RAG"""

    def __init__(self, config: Config):
        super().__init__(config)

        self.END_TOKEN = self.generator.end_token
        rag_cfg = self.config.rag
        self.hallucination_threshold = rag_cfg.hallucination_threshold
        self.batch_size = rag_cfg.batch_size
        self.stop_words = []
        self.clear_cache()

    def clear_cache(self):
        self.generated_text = ""
        self.retrieval_count = 0

    def inference(self, query: str, max_length: int = 8191) -> str:
        try:
            logger.info(f"开始推理任务 [max_length={max_length}]")
            self.clear_cache()

            retrieve_target: Union[str, List[str]] = query
            implicit_knowledge = self._load_implicit_knowledge(query)
            few_shots = self._load_few_shots(query)

            while len(self.generated_text) < max_length:
                if self.END_TOKEN in self.generated_text:
                    logger.info("检测到结束符，停止生成")
                    break

                remaining_length = min(self.batch_size, max_length - len(self.generated_text))

                explicit_knowledge = self._perform_retrieval_for_few_shot(
                    retrieve_target, original_query=query
                )
                if not explicit_knowledge:
                    break

                system_prompt = get_mini_system_prompt_with_few_shots(explicit_knowledge)
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
                    )
                )

                if not new_text:
                    logger.warning("生成结果为空，停止循环")
                    break

                retrieve_target = self._handle_stop_focus(stop_focus, retrieve_target)

                new_text, tokens, attentions, entropies = self._truncate_at_end_token(
                    new_text, tokens, attentions, entropies
                )

                # 幻觉检测
                is_hallucination, tokens, hallucinated_idx, norm_attentions, metric_values = (
                    self.modifier(new_text, tokens, attentions, weight=entropies)
                )

                if not is_hallucination:
                    self.generated_text += new_text
                    logger.info(f"生成片段(无幻觉): {new_text.strip()[:80]}...")
                else:
                    self.retrieval_count += 1
                    retrieve_target = self._handle_hallucination(
                        hallucinated_idx, tokens, new_text, query_prompt, query
                    )

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

    def _handle_hallucination(self, hallucinated_idx, tokens, new_text, query_prompt, original_query):
        """处理幻觉检测后的回溯逻辑"""
        cut_idx = hallucinated_idx[0]
        if cut_idx > 0:
            self.generated_text += new_text[:cut_idx]

        try:
            current_full_prompt = query_prompt + self.generated_text
            attention_matrix = self._get_full_attention_matrix(
                prompt_text=current_full_prompt, generated_tokens=tokens
            )

            new_keywords = self._extract_retrieval_query_from_attention(
                hallucinated_idx=hallucinated_idx,
                generated_tokens=tokens,
                attention_matrix=attention_matrix,
                prompt_text=current_full_prompt,
                top_k=3,
            )

            if new_keywords:
                logger.info(f"幻觉回溯提取关键词: {new_keywords}")
                return new_keywords

            return original_query

        except Exception as e:
            logger.error(f"注意力回溯分析失败: {e}")
            return original_query

    def _get_full_attention_matrix(self, prompt_text, generated_tokens):
        """获取生成 Token 对 Prompt 的 Cross-Attention 矩阵"""
        if not generated_tokens or not prompt_text:
            return np.array([]).reshape(0, 0)

        tokenizer = self.generator.tokenizer
        model = self.generator.model

        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = prompt_ids.shape[1]

        gen_text = "".join(generated_tokens)
        gen_ids = tokenizer.encode(gen_text, add_special_tokens=False, return_tensors="pt").to(
            model.device
        )

        if gen_ids.shape[1] == 0:
            return np.array([]).reshape(0, 0)

        attention_rows = []
        current_input = prompt_ids

        with torch.no_grad():
            for i in range(gen_ids.shape[1]):
                outputs = model(current_input, output_attentions=True)
                attn_map = outputs.attentions[-1][0]
                mean_attn = torch.mean(attn_map, dim=0)
                last_token_attn = mean_attn[-1, :prompt_len]
                attention_rows.append(last_token_attn.cpu().numpy())
                next_token = gen_ids[:, i : i + 1]
                current_input = torch.cat([current_input, next_token], dim=1)

        return np.array(attention_rows)

    def _extract_retrieval_query_from_attention(
        self, hallucinated_idx, generated_tokens, attention_matrix, prompt_text, top_k=3
    ):
        """从注意力矩阵中提取检索关键词"""
        if attention_matrix.size == 0:
            return []

        tokenizer = self.generator.tokenizer
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = [tokenizer.decode([pid]) for pid in prompt_ids]
        prompt_len = len(prompt_ids)

        valid_idxs = [i for i in hallucinated_idx if i < attention_matrix.shape[0]]
        if not valid_idxs:
            return []

        hallucination_attention = np.max(attention_matrix[valid_idxs], axis=0)

        # 差分注意力
        hallucination_attention = self._apply_differential_attention(
            hallucination_attention, attention_matrix, valid_idxs
        )

        # 位置权重衰减
        position_weights = np.linspace(0.3, 1.0, prompt_len)
        hallucination_attention = hallucination_attention * position_weights

        # 局部归一化
        hallucination_attention = self._apply_local_normalization(
            hallucination_attention, prompt_len
        )

        # 提取语义短语
        phrases = self._extract_semantic_phrases_from_attention(
            attention_scores=hallucination_attention,
            prompt_token_ids=prompt_ids,
            prompt_tokens=prompt_tokens,
            attention_threshold=0.01,
            max_phrase_length=5,
        )

        # 相关性排序
        current_generated = "".join(generated_tokens).lower()
        for p in phrases:
            p["adjusted_score"] = self._compute_relevance_score(
                p["text"], p["score"], current_generated
            )

        phrases.sort(key=lambda x: x["adjusted_score"], reverse=True)

        seen = set()
        result_keywords = []
        for p in phrases:
            text = p["text"].strip()
            if text and text not in seen:
                result_keywords.append(text)
                seen.add(text)
                if len(result_keywords) >= top_k:
                    break

        return result_keywords

    def _apply_differential_attention(self, hallucination_attention, attention_matrix, valid_idxs):
        if valid_idxs[0] > 0:
            prev_idx = max(0, valid_idxs[0] - 1)
            prev_attention = attention_matrix[prev_idx]
            diff_attention = np.maximum(hallucination_attention - prev_attention, 0)
            return 0.7 * diff_attention + 0.3 * hallucination_attention
        return hallucination_attention

    def _apply_local_normalization(self, attention, prompt_len):
        window_size = min(64, prompt_len // 4)
        if window_size <= 0:
            return attention
        local_normalized = np.zeros_like(attention)
        for i in range(0, prompt_len, window_size):
            end = min(i + window_size, prompt_len)
            window = attention[i:end]
            if window.max() > 0:
                local_normalized[i:end] = window / (window.max() + 1e-9)
        global_normalized = attention / (attention.max() + 1e-9)
        return 0.6 * local_normalized + 0.4 * global_normalized

    def _compute_relevance_score(self, phrase_text, base_score, current_generated):
        phrase_lower = phrase_text.lower()
        if phrase_lower in current_generated:
            return base_score * 0.3
        return base_score

    def _extract_semantic_phrases_from_attention(
        self, attention_scores, prompt_token_ids, prompt_tokens,
        attention_threshold=0.01, max_phrase_length=5
    ):
        phrases = []
        prompt_length = len(attention_scores)
        i = 0

        while i < prompt_length:
            if attention_scores[i] >= attention_threshold:
                start_idx = i
                end_idx = i
                while end_idx < prompt_length and attention_scores[end_idx] >= attention_threshold:
                    end_idx += 1
                end_idx = min(start_idx + max_phrase_length, end_idx)

                if end_idx > start_idx:
                    phrase_ids = prompt_token_ids[start_idx:end_idx]
                    phrase_score = np.mean(attention_scores[start_idx:end_idx])

                    try:
                        tokenizer = self.generator.tokenizer
                        text = tokenizer.decode(phrase_ids, skip_special_tokens=True).strip()
                        if text:
                            phrases.append({
                                "text": text,
                                "score": float(phrase_score),
                                "start_idx": start_idx,
                                "end_idx": end_idx - 1,
                            })
                    except Exception:
                        pass

                i = end_idx
            else:
                i += 1

        return phrases

    def modifier(self, text, tokens, attentions, weight):
        """幻觉检测器: (归一化Attention * Entropy) > 阈值"""
        hallucinated_indices = []
        metric_values = []

        attentions_np = np.array(attentions)
        norm_attentions = attentions_np / (np.sum(attentions_np) + 1e-9)

        for i in range(len(tokens)):
            val = norm_attentions[i] * weight[i]
            metric_values.append(val)
            if val > self.hallucination_threshold:
                hallucinated_indices.append(i)

        is_hallucination = len(hallucinated_indices) > 0
        return is_hallucination, tokens, hallucinated_indices, norm_attentions.tolist(), metric_values
