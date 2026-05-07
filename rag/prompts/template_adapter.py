"""
Prompt 模板适配器 - 根据模型类型自动适配提示词格式

支持: Qwen3, Deepseek, GLM, Llama, Mistral, 以及通用兜底模板
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class BasePromptTemplate(ABC):
    """提示词模板基类"""

    MODEL_IDENTIFIERS: List[str] = []

    def __init__(self, tokenizer: AutoTokenizer = None):
        self.tokenizer = tokenizer

    @abstractmethod
    def format_messages(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        raise NotImplementedError

    def format_chat(
        self,
        query: str,
        system_prompt: str = "",
        history: List[Dict[str, str]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})
        return self.format_messages(messages, add_generation_prompt)

    @property
    @abstractmethod
    def end_token(self) -> str:
        pass

    @property
    def stop_tokens(self) -> List[str]:
        return [self.end_token]

    @classmethod
    def match(cls, model_name_or_path: str) -> bool:
        model_name_lower = model_name_or_path.lower()
        return any(identifier.lower() in model_name_lower for identifier in cls.MODEL_IDENTIFIERS)


class Qwen3Template(BasePromptTemplate):
    """Qwen3 ChatML 格式"""

    MODEL_IDENTIFIERS = ["qwen3", "qwen-3", "qwen/qwen3"]
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def format_messages(self, messages, add_generation_prompt=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
                )
            except Exception as e:
                logger.warning(f"Qwen3 apply_chat_template 失败: {e}")

        parts = []
        for msg in messages:
            parts.append(f"{self.IM_START}{msg['role']}\n{msg['content']}{self.IM_END}")
        if add_generation_prompt:
            parts.append(f"{self.IM_START}assistant\n")
        return "\n".join(parts)

    @property
    def end_token(self):
        return self.IM_END

    @property
    def stop_tokens(self):
        return [self.IM_END, ""]


class DeepseekTemplate(BasePromptTemplate):
    """Deepseek Instruct 格式"""

    MODEL_IDENTIFIERS = ["deepseek", "deep-seek", "deepseek-coder"]
    DEFAULT_SYSTEM = (
        "You are an AI programming assistant, utilizing the Deepseek Coder model, "
        "developed by Deepseek Company."
    )

    def format_messages(self, messages, add_generation_prompt=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"Deepseek apply_chat_template 失败: {e}")

        system_content = self.DEFAULT_SYSTEM
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        prompt = f"{system_content}\n### Instruction:\n{user_content}\n### Response:"
        if assistant_content:
            prompt += f"\n{assistant_content}"
        return prompt

    @property
    def end_token(self):
        return "<|EOT|>"

    @property
    def stop_tokens(self):
        return ["<|EOT|>", "</s>", "### Instruction:"]


class GLMTemplate(BasePromptTemplate):
    """GLM-4 Chat 格式"""

    MODEL_IDENTIFIERS = ["glm", "chatglm", "glm-4", "glm4", "zhipu"]

    def format_messages(self, messages, add_generation_prompt=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"GLM apply_chat_template 失败: {e}")

        parts = [""]
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        if add_generation_prompt:
            parts.append("")
        return "".join(parts)

    @property
    def end_token(self):
        return ""

    @property
    def stop_tokens(self):
        return ["", ""]


class LlamaTemplate(BasePromptTemplate):
    """Llama Chat 格式"""

    MODEL_IDENTIFIERS = ["llama", "llama2", "llama-2", "meta-llama"]

    def format_messages(self, messages, add_generation_prompt=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"Llama apply_chat_template 失败: {e}")

        system_content = ""
        conversation_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content:
                    conversation_parts.append(
                        f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{msg['content']} [/INST]"
                    )
                    system_content = ""
                else:
                    conversation_parts.append(f"<s>[INST] {msg['content']} [/INST]")
            elif msg["role"] == "assistant":
                conversation_parts.append(f" {msg['content']} </s>")
        return "".join(conversation_parts)

    @property
    def end_token(self):
        return "<|eot_id|>"

    @property
    def stop_tokens(self):
        return ["<|eot_id|>", "[INST]"]


class MistralTemplate(BasePromptTemplate):
    """Mistral Instruct 格式"""

    MODEL_IDENTIFIERS = ["mistral", "mixtral"]

    def format_messages(self, messages, add_generation_prompt=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"Mistral apply_chat_template 失败: {e}")

        parts = ["<s>"]
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_msg = f"{system_content}\n\n{msg['content']}" if system_content else msg["content"]
                parts.append(f"[INST] {user_msg} [/INST]")
                system_content = ""
            elif msg["role"] == "assistant":
                parts.append(f" {msg['content']}</s>")
        return "".join(parts)

    @property
    def end_token(self):
        return "</s>"

    @property
    def stop_tokens(self):
        return ["</s>", "[INST]"]


class GenericTemplate(BasePromptTemplate):
    """通用模板（兜底）"""

    MODEL_IDENTIFIERS = []

    def format_messages(self, messages, add_generation_prompt=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"通用模板 apply_chat_template 失败: {e}")

        parts = []
        for msg in messages:
            parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts)

    @property
    def end_token(self):
        return "</s>"


class PromptTemplateFactory:
    """提示词模板工厂"""

    TEMPLATE_CLASSES = [
        Qwen3Template,
        DeepseekTemplate,
        GLMTemplate,
        LlamaTemplate,
        MistralTemplate,
    ]

    @classmethod
    def create(cls, model_name_or_path: str, tokenizer: AutoTokenizer = None) -> BasePromptTemplate:
        for template_class in cls.TEMPLATE_CLASSES:
            if template_class.match(model_name_or_path):
                logger.info(f"模型 '{model_name_or_path}' 匹配模板: {template_class.__name__}")
                return template_class(tokenizer)
        logger.warning(f"模型 '{model_name_or_path}' 未匹配到特定模板，使用通用模板")
        return GenericTemplate(tokenizer)

    @classmethod
    def register(cls, template_class: type):
        if not issubclass(template_class, BasePromptTemplate):
            raise TypeError("模板类必须继承自 BasePromptTemplate")
        cls.TEMPLATE_CLASSES.insert(0, template_class)

    @classmethod
    def list_supported_models(cls) -> Dict[str, List[str]]:
        return {
            tc.__name__: tc.MODEL_IDENTIFIERS
            for tc in cls.TEMPLATE_CLASSES
            if tc.MODEL_IDENTIFIERS
        }


def get_prompt_template(model_name_or_path: str, tokenizer: AutoTokenizer = None) -> BasePromptTemplate:
    """便捷函数：获取提示词模板"""
    return PromptTemplateFactory.create(model_name_or_path, tokenizer)
