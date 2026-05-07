"""
Prompt 模块 - 提示词模板、构建器和适配器
"""

from rag.prompts.system_prompt import (
    assemble_system_prompt,
    get_system_prompt,
    get_system_prompt_with_few_shots,
    get_system_prompt_zero_shot,
    get_system_prompt_few_shot,
    get_mini_system_prompt_with_few_shots,
)
from rag.prompts.rag_prompt import RAG_PROMPT
from rag.prompts.template_adapter import get_prompt_template, BasePromptTemplate

__all__ = [
    "assemble_system_prompt",
    "get_system_prompt",
    "get_system_prompt_with_few_shots",
    "get_system_prompt_zero_shot",
    "get_system_prompt_few_shot",
    "get_mini_system_prompt_with_few_shots",
    "RAG_PROMPT",
    "get_prompt_template",
    "BasePromptTemplate",
]
