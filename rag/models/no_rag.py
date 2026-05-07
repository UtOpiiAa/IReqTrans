"""
NoRAG - 无检索，纯 LLM 推理 (zero-shot)
"""

import os
import logging
from typing import Optional

from config import Config
from utils.llm_client import LLMClient
from rag.prompts import get_system_prompt_zero_shot, RAG_PROMPT

logger = logging.getLogger(__name__)


class NoRAG:
    """无检索模型，直接使用 LLM 进行推理生成"""

    def __init__(self, config: Config):
        self.config = config
        self.model_path = self._resolve_path(config.llm.model_name)
        self.max_tokens = config.llm.max_tokens
        self.generator = LLMClient(self.model_path, max_tokens=self.max_tokens)

    def _resolve_path(self, path_str: str) -> str:
        if not path_str:
            return ""
        if os.path.isabs(path_str):
            return path_str
        return path_str

    def inference(self, query: str) -> str:
        """执行纯 LLM 推理"""
        system_prompt = get_system_prompt_zero_shot()
        prompt = RAG_PROMPT.format(requirement=query)

        logger.info("NoRAG 模式 - 无检索直接推理")
        logger.info(f"System Prompt 长度: {len(system_prompt)}")

        response, _, _ = self.generator.generate(
            query_text=prompt, system_prompt=system_prompt, max_length=self.max_tokens
        )

        end_token = self.generator.end_token
        if end_token in response:
            response = response.split(end_token)[0]

        return response

    def clear_cache(self):
        self.generated_text = ""
