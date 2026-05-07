"""
LLM 客户端 - 支持本地模型 (HuggingFace Transformers) 和 OpenAI API 两种模式
"""

import os
import logging
from typing import Optional, List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from config import LLMConfig
from rag.prompts.template_adapter import get_prompt_template, BasePromptTemplate

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端，封装本地模型加载与文本生成"""

    def __init__(self, model_name_or_path: str, max_tokens: int = 8191):
        """
        Args:
            model_name_or_path: 本地模型路径或 HuggingFace Model ID
            max_tokens: 最大生成 token 数
        """
        logger.info(f"加载模型: {model_name_or_path}")
        self.model_name_or_path = model_name_or_path
        self.max_tokens = max_tokens
        self.device_map = "auto"

        trust_remote_code = "falcon" in model_name_or_path.lower() or "glm" in model_name_or_path.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=self.device_map, trust_remote_code=trust_remote_code
        )
        self.model.set_attn_implementation("eager")

        # 初始化提示词模板适配器
        self.prompt_template: BasePromptTemplate = get_prompt_template(
            model_name_or_path, self.tokenizer
        )
        logger.info(f"使用提示词模板: {type(self.prompt_template).__name__}")

        # 确定空格 Token
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            try:
                self.space_token = self.tokenizer.tokenize(" ")[0]
            except IndexError:
                self.space_token = " "

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @property
    def end_token(self) -> str:
        return self.prompt_template.end_token

    def _prepare_inputs(
        self, query_text: str, system_prompt: str, generated_text: str = ""
    ):
        """构建 Prompt 并编码"""
        text = self.prompt_template.format_chat(
            query=query_text,
            system_prompt=system_prompt,
            add_generation_prompt=True,
        )
        if generated_text:
            text += generated_text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        return input_ids, input_ids.shape[1]

    def _apply_stop_tokens(self, text: str, stop_tokens: Optional[List[str]]) -> str:
        """应用停止词截断文本"""
        if stop_tokens:
            for stop_token in stop_tokens:
                if stop_token in text:
                    text = text.split(stop_token)[0] + stop_token
        return text

    def generate(
        self,
        query_text: str,
        system_prompt: str,
        max_length: int = 5120,
        generated_text: str = "",
        return_logprobs: bool = False,
        stop_tokens: Optional[List[str]] = None,
    ):
        """
        基础文本生成

        Returns:
            (text, tokens, logprobs_list)
        """
        input_ids, input_length = self._prepare_inputs(query_text, system_prompt, generated_text)
        attention_mask = torch.ones_like(input_ids)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_length,
        }

        if return_logprobs:
            generate_kwargs["return_dict_in_generate"] = True
            generate_kwargs["output_scores"] = True
            outputs = self.model.generate(**generate_kwargs)
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            generated_sequence = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_sequence[0])
            tokens = [self.tokenizer.decode(t) for t in generated_sequence[0]]
            logprobs_list = [float(p) for p in transition_scores[0].cpu().numpy()]
            text = self._apply_stop_tokens(text, stop_tokens)
            return text, tokens, logprobs_list
        else:
            outputs = self.model.generate(**generate_kwargs)
            generated_sequence = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_sequence[0])
            text = self._apply_stop_tokens(text, stop_tokens)
            return text, None, None

    def generate_attn(
        self,
        query_text: str,
        system_prompt: str,
        max_length: int,
        generated_text: str = "",
        solver: str = "max",
        use_entropy: bool = False,
        use_logprob: bool = False,
        stop_words: Optional[List[str]] = None,
        focus_window: int = 512,
        attention_threshold_percentile: float = 75.0,
    ):
        """
        带注意力分析的文本生成

        Returns:
            (text, seq_list, attn_list, seq_logprobs, seq_entropies, following_focus)
        """
        import numpy as np

        focus_window = max(32, focus_window)
        input_ids, input_length = self._prepare_inputs(query_text, system_prompt, generated_text)
        attention_mask = torch.ones_like(input_ids)

        # 模型生成
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # 初始处理
        full_generated_sequence = outputs.sequences[:, input_length:]
        full_generated_ids = full_generated_sequence[0]
        full_text = self.tokenizer.decode(full_generated_ids)

        # 停止词截断
        cut_char_idx = len(full_text)
        matched_stop_word = None
        for stop_word in (stop_words or []):
            if not stop_word:
                continue
            pos = full_text.find(stop_word)
            if pos != -1:
                end_pos = pos + len(stop_word)
                if end_pos < cut_char_idx:
                    cut_char_idx = end_pos
                    matched_stop_word = stop_word

        # 处理截断与 Following Focus
        final_token_count = full_generated_ids.shape[0]
        text_output = full_text
        following_focus = None

        if cut_char_idx < len(full_text):
            trimmed_text = full_text[:cut_char_idx]
            trimmed_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
            final_token_count = len(trimmed_ids)
            text_output = trimmed_text

            next_token_pos = final_token_count
            if next_token_pos < full_generated_sequence.shape[1]:
                query_ids = self.tokenizer.encode(
                    query_text, return_tensors="pt", add_special_tokens=False
                ).to(self.model.device)
                query_token_ids = query_ids[0].tolist()

                following_focus = self._extract_focus_from_following_token(
                    prompt_ids=query_ids,
                    prompt_token_ids=query_token_ids,
                    generated_ids=full_generated_sequence,
                    follow_token_idx=next_token_pos,
                    stop_word=matched_stop_word,
                    focus_window=focus_window,
                    attention_threshold_percentile=attention_threshold_percentile,
                )

        # 有效 tokens
        valid_generated_ids = full_generated_ids[:final_token_count]
        valid_tokens = self.tokenizer.convert_ids_to_tokens(valid_generated_ids)

        # Attention Grouping
        token_intervals = self._group_tokens_by_semantics(valid_tokens, valid_generated_ids)

        # 计算 Attention Score
        with torch.no_grad():
            forward_input = valid_generated_ids.unsqueeze(0).to(self.model.device)
            output = self.model(forward_input, output_attentions=True)

        raw_atten = output.attentions[-1][0]
        mean_atten = self._compute_attention_metrics(raw_atten, valid_tokens, solver)

        # 聚合 Attention
        seq_list = []
        attn_list = []
        for start, end in token_intervals:
            token_seq = "".join(valid_tokens[start : end + 1]).replace(self.space_token, "")
            attn_val = sum(mean_atten[start : end + 1]).item()
            seq_list.append(token_seq)
            attn_list.append(attn_val)

        # LogProbs
        seq_logprobs = None
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0][:final_token_count].cpu().numpy()
            seq_logprobs = []
            for start, end in token_intervals:
                avg_lp = sum(logprobs[start : end + 1]) / (end - start + 1)
                seq_logprobs.append(avg_lp)

        # Entropy
        seq_entropies = None
        if use_entropy:
            scores_list = [s.cpu() for s in outputs.scores[:final_token_count]]
            if scores_list:
                scores_tensor = torch.cat(scores_list, dim=0)
                probs = torch.softmax(scores_tensor, dim=-1).numpy()
                entropies = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
                seq_entropies = []
                for start, end in token_intervals:
                    avg_ent = sum(entropies[start : end + 1]) / (end - start + 1)
                    seq_entropies.append(avg_ent)
            else:
                seq_entropies = []

        return text_output, seq_list, attn_list, seq_logprobs, seq_entropies, following_focus

    def _compute_attention_metrics(self, atten: torch.Tensor, tokens: List[str], solver: str) -> torch.Tensor:
        """计算聚合注意力分数"""
        if solver == "max":
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            seq_len = mean_atten.shape[0]
            for i in range(seq_len):
                mean_atten[i] /= seq_len - i
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError(f"Solver {solver} not implemented")

        if mean_atten.shape[0] > 1 and tokens[0] == "</s>":
            sum_val = sum(mean_atten[1:]).item()
            if sum_val != 0:
                mean_atten = mean_atten / sum_val

        return mean_atten

    def _group_tokens_by_semantics(self, tokens: List[str], generated_ids: torch.Tensor) -> List[List[int]]:
        """根据空格/换行将 Token 分组"""
        intervals = []
        for i, t in enumerate(tokens):
            is_start = (
                i == 0
                or t.startswith(self.space_token)
                or generated_ids[i] == 13
                or (i > 0 and tokens[i - 1] == "</s>")
            )
            if is_start:
                intervals.append([i, i])
            else:
                if intervals:
                    intervals[-1][-1] += 1
                else:
                    intervals.append([i, i])
        return intervals

    def _extract_focus_from_following_token(
        self,
        prompt_ids: torch.Tensor,
        prompt_token_ids: list,
        generated_ids: torch.Tensor,
        follow_token_idx: int,
        stop_word: Optional[str] = None,
        top_k: int = 5,
        focus_window: int = 512,
        attention_threshold_percentile: int = 75,
        max_span_length: int = 32,
    ) -> Optional[Dict]:
        """基于被截断后的下一个 token 的注意力，提取其关注的前文关键词"""
        import numpy as np
        import re

        if follow_token_idx is None or follow_token_idx >= generated_ids.shape[1]:
            return None

        STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
            "of", "to", "in", "on", "at", "by", "for", "with", "about", "against",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "it", "its", "it's", "they", "them", "their", "this", "that", "these", "those",
            "no", "so", "very", "can", "will",
            "just", "don", "should", "now", "return", "true", "false", "null",
            "via", "from", "as", "such", "do", "does", "did", "which", "who", "whom", "hold", "\n",
        }

        focus_window = max(32, focus_window)
        prompt_ids = prompt_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[1]
        slice_start = max(0, prompt_length - focus_window)
        prompt_slice = prompt_ids[:, slice_start:]

        generated_prefix = generated_ids[:, :follow_token_idx].to(self.model.device)
        combined_input = torch.cat([prompt_slice, generated_prefix], dim=1)

        with torch.no_grad():
            outputs = self.model(combined_input, output_attentions=True)
            if outputs.attentions is None or len(outputs.attentions) == 0:
                return None
            attn = outputs.attentions[-1][0]
            mean_attn = torch.mean(attn[:, -1], dim=0)

        prompt_slice_length = prompt_slice.shape[1]
        prompt_attention = mean_attn[:prompt_slice_length].detach().cpu().to(torch.float32).numpy()

        if prompt_attention.size == 0:
            return None

        threshold = np.percentile(prompt_attention, attention_threshold_percentile)

        # 查找连续 span
        spans = []
        i = 0
        len_atten = len(prompt_attention)
        while i < len_atten:
            if prompt_attention[i] >= threshold:
                start_idx = i
                end_idx = i + 1
                while (
                    end_idx < len_atten
                    and prompt_attention[end_idx] >= threshold
                    and (end_idx - start_idx) < max_span_length
                ):
                    end_idx += 1
                span_score = np.sum(prompt_attention[start_idx:end_idx])
                spans.append((span_score, start_idx, end_idx))
                i = end_idx
            else:
                i += 1

        spans.sort(key=lambda x: x[0], reverse=True)

        focus_tokens = []
        focus_scores = []
        seen = set()
        total_prompt_len = len(prompt_token_ids)

        boundary_chars = {".", ",", "!", "?", ";", ":", "\n", "\r", "\t"}

        for score, start, end in spans:
            if len(focus_tokens) >= top_k:
                break
            original_start = slice_start + start
            original_end = slice_start + end
            if original_start >= total_prompt_len:
                continue

            # Token 补全（向左）
            curr_l = original_start
            search_count = 0
            while curr_l > 0 and search_count < 10:
                curr_token_str = self.tokenizer.decode([prompt_token_ids[curr_l]])
                if curr_token_str.startswith(" ") or curr_token_str.startswith("Ġ"):
                    break
                if any(c in curr_token_str for c in boundary_chars):
                    break
                prev_token_str = self.tokenizer.decode([prompt_token_ids[curr_l - 1]])
                if any(c in prev_token_str for c in boundary_chars):
                    break
                curr_l -= 1
                search_count += 1

            # Token 补全（向右）
            curr_r = original_end
            search_count = 0
            while curr_r < total_prompt_len and search_count < 10:
                token_str = self.tokenizer.decode([prompt_token_ids[curr_r]])
                if token_str.startswith(" ") or token_str.startswith("Ġ"):
                    break
                if any(c in token_str for c in boundary_chars):
                    break
                curr_r += 1
                search_count += 1

            expanded_ids = prompt_token_ids[curr_l:curr_r]
            full_text = self.tokenizer.decode(expanded_ids, skip_special_tokens=True)

            segments = re.split(r"[\n\r]+", full_text)
            segments = [s.strip() for s in segments if s.strip()]
            if segments:
                full_text = max(segments, key=len)
            cleaned_text = full_text.strip().strip('.,;:!?()[]{}""''`\n\t').lower()

            if not cleaned_text:
                continue

            sub_words = re.split(r"[^\w]+", cleaned_text)
            sub_words = [w for w in sub_words if w]

            if not sub_words:
                continue

            is_pure_stopword = all(w in STOP_WORDS or len(w) <= 1 for w in sub_words)
            if is_pure_stopword:
                continue

            if cleaned_text in seen:
                continue

            if "GSS-SyRS" in full_text or "hold" in full_text:
                continue

            focus_tokens.append(full_text.strip())
            focus_scores.append(float(score))
            seen.add(cleaned_text)

        following_token_text = self.tokenizer.decode(
            [generated_ids[0, follow_token_idx].item()], skip_special_tokens=True
        ).strip()

        return {
            "stop_word": stop_word,
            "following_token": following_token_text,
            "focus_tokens": focus_tokens,
            "focus_scores": focus_scores,
        }
