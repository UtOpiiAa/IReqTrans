"""
知识提取器 - 从需求文本和形式化代码中提取领域知识

调用外部 LLM API，提取变量映射、函数映射、术语解释、翻译技巧等原子知识。
"""

import os
import json
import re
import time
import logging
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests

from config import Config, ExtractionConfig, ExtractionAPIConfig

logger = logging.getLogger(__name__)


# ==================== 提示词模板 ====================

SYSTEM_PROMPT = """你是一个专业的铁路信号联锁系统领域专家，精通形式化规范语言和需求工程。
你的任务是分析需求文本及其对应的形式化代码翻译，提取出翻译过程中涉及的领域知识。
生成的代码语法规则如下：
```Lspec Grammar (BNF) ```
This grammar enforces operator precedence: ~, PRE > & > # > -> (lowest).

Specification = FormulaID , ":=" , Formula , ";" ;
FormulaID = "\"" , ID , "\"" ;
ID = ("GSS" | "RS" | "REQ") , "-" , { Letter | Digit | "-" } ;
Formula =
      "true" | "false"
    | PredicateFunction | StateFunction | UserDefinedFunction
    | "~" , Formula | Formula , "&" , Formula | Formula , "#" , Formula
    | Formula , "->" , Formula | "ALL" , Object , Formula | "SOME" , Object , Formula
    | Object , "=" , Object | "PRE" , Formula | "X" , Formula
    | Formula , "S" , Formula | Formula , "U" , Formula ;
Object = Identifier ;
Identifier = LowercaseLetter , { LowercaseLetter | Digit } ;
PredicateFunction = FunctionName , "(" , ParameterList , ")" ;
StateFunction = FunctionName , "(" , ParameterList , ")" ;
UserDefinedFunction = FunctionName , "(" , ParameterList , ")" ;
FunctionName = Identifier ;
ParameterList = Object , { "," , Object } ;
这一部分是已知内容无需重复。
请严格按照用户要求的JSON格式输出结果，不要添加任何额外的说明文字。"""

EXTRACTION_PROMPT = """请分析以下需求文本及其对应的形式化代码翻译，提取出翻译过程中涉及的领域知识。

## 需求ID：{req_id}

## 原始需求文本：
{req_text}

## 形式化代码翻译：
```
{code}
```

## 任务要求：
请提取以下类型的原子知识（每条知识尽量细粒度，一条知识只描述一个变量/函数/术语/规则）：

1. **variable_mapping（变量映射）**：需求文本中的概念/名词与形式化代码中变量名的对应关系
2. **function_mapping（函数映射）**：需求文本中的操作/动作/状态与形式化代码中函数/谓词的对应关系
3. **term_explanation（术语解释）**：领域专业术语的含义解释
4. **translation_technique（翻译技巧）**：从自然语言到形式化语言的翻译规则和技巧

## 输出格式要求：
请以JSON对象格式输出，内容为英文：
```json
{{
  "req_id": "{req_id}",
  "knowledge": [
    {{
      "source_keyword": "原文中的关键词或短语（英文，用于检索）",
      "formal_element": "对应的形式化元素（变量名/函数名/谓词等）",
      "description": "知识的详细描述",
      "example_context": "该知识在本需求中的具体应用示例"
    }}
  ]
}}
```

## 重要提示：
- 每条知识必须足够原子化
- source_keyword必须是原文中实际出现的词汇或短语
- 描述要清晰、准确、专业
- 每条知识用英文描述
- 请直接输出JSON对象，不要包含```json标记或其他说明文字
"""


class KnowledgeExtractor:
    """知识提取器"""

    def __init__(self, config: Config):
        self.config = config
        self.ext_cfg = config.extraction
        self.api_cfg = config.extraction.api

    def _call_llm_api(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
        """调用大模型 API"""
        headers = {
            "Authorization": f"Bearer {self.api_cfg.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.api_cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.api_cfg.max_tokens,
            "temperature": self.api_cfg.temperature,
        }

        for attempt in range(self.api_cfg.max_retries):
            try:
                response = requests.post(
                    f"{self.api_cfg.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.api_cfg.timeout,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时，第{attempt + 1}次重试...")
            except requests.exceptions.RequestException as e:
                logger.warning(f"API请求失败: {e}")
                if attempt < self.api_cfg.max_retries - 1:
                    time.sleep(self.api_cfg.retry_delay)
            except (KeyError, IndexError) as e:
                logger.error(f"API响应格式错误: {e}")
                return None
        return None

    @staticmethod
    def extract_json_from_response(response: str) -> Dict:
        """从 API 响应中提取 JSON"""
        if not response:
            return {}

        response = response.strip()

        def try_parse(text):
            try:
                result = json.loads(text)
                if isinstance(result, dict) and "knowledge" in result:
                    return result
                if isinstance(result, list):
                    return {"knowledge": result}
            except json.JSONDecodeError:
                pass
            return None

        # 直接解析
        parsed = try_parse(response)
        if parsed:
            return parsed

        # 提取代码块中的 JSON
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        for match in re.findall(json_pattern, response):
            parsed = try_parse(match.strip())
            if parsed:
                return parsed

        # 查找对象边界
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = try_parse(response[start:end])
            if parsed:
                return parsed

        logger.warning("无法解析 JSON 响应")
        return {}

    def extract_single(self, req_id: str, req_text: str, code: str) -> Tuple[Dict, Optional[str]]:
        """处理单条需求，提取知识"""
        prompt = EXTRACTION_PROMPT.format(req_id=req_id, req_text=req_text, code=code)
        response = self._call_llm_api(prompt)

        if not response:
            return {"req_id": req_id, "knowledge": []}, None

        parsed = self.extract_json_from_response(response)
        knowledge_list = parsed.get("knowledge", [])

        # 清洗：只保留四个核心字段
        valid_knowledge = []
        for item in knowledge_list:
            if not item.get("source_keyword"):
                continue
            valid_knowledge.append({
                "source_keyword": item.get("source_keyword", ""),
                "formal_element": item.get("formal_element", ""),
                "description": item.get("description", ""),
                "example_context": item.get("example_context", ""),
            })

        return {"req_id": req_id, "knowledge": valid_knowledge}, response

    def run(self):
        """运行知识提取主流程"""
        logger.info("知识提取程序启动")

        if self.api_cfg.api_key == "YOUR_API_KEY":
            logger.error("请先配置 API 密钥！在 config/default.yaml 的 extraction.api.api_key 中设置")
            return

        # 加载数据
        req_file = self.config.resolve_path(self.ext_cfg.requirements_file)
        code_file = self.config.resolve_path(self.ext_cfg.formal_code_file)
        output_dir = os.path.join(
            self.config.resolve_path(self.ext_cfg.output_dir), "knowledge_files"
        )
        os.makedirs(output_dir, exist_ok=True)

        try:
            req_df = pd.read_excel(req_file)
            code_df = pd.read_excel(code_file)
        except FileNotFoundError as e:
            logger.error(f"文件不存在: {e}")
            return

        merged_df = pd.merge(req_df, code_df, on="req_id", how="inner")
        logger.info(f"匹配记录: {len(merged_df)} 条")

        if merged_df.empty:
            logger.error("没有匹配的记录！请检查两个文件的 req_id 列")
            return

        start_idx = self.ext_cfg.start_index
        end_idx = len(merged_df)

        all_knowledge = []
        for idx in range(start_idx, end_idx):
            row = merged_df.iloc[idx]
            req_id = str(row["req_id"])
            req_text = str(row["req_text"])
            code = str(row["code"])

            # 跳过已处理的
            safe_id = re.sub(r'[<>:"/\\|?*]', "_", req_id)
            success_file = os.path.join(output_dir, f"{safe_id}_knowledge.txt")
            if os.path.exists(success_file):
                logger.info(f"[{idx + 1}] {req_id} 已存在，跳过")
                continue

            logger.info(f"[{idx + 1}/{end_idx}] 处理需求: {req_id}")
            result_obj, raw_response = self.extract_single(req_id, req_text, code)
            knowledge_list = result_obj.get("knowledge", [])

            # 保存到文件
            file_path = os.path.join(
                output_dir,
                f"{safe_id}_knowledge{'_FAILED' if not knowledge_list else ''}.txt",
            )
            with open(file_path, "w", encoding="utf-8") as f:
                if knowledge_list:
                    json.dump(result_obj, f, ensure_ascii=False, indent=2)
                elif raw_response:
                    f.write("=== 原始响应（JSON解析失败）===\n\n")
                    f.write(raw_response)
                else:
                    f.write("=== API调用失败 ===\n")

            if knowledge_list:
                all_knowledge.extend(knowledge_list)
                logger.info(f"  提取到 {len(knowledge_list)} 条知识")
            else:
                logger.warning(f"  未提取到有效知识")

            time.sleep(self.ext_cfg.sleep_interval)

        # 汇总 Excel
        if all_knowledge:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(
                self.config.resolve_path(self.ext_cfg.output_dir),
                f"all_knowledge_{timestamp}.xlsx",
            )
            df = pd.DataFrame(all_knowledge)
            df.to_excel(excel_path, index=False, engine="openpyxl")
            logger.info(f"汇总 Excel 已保存: {excel_path} ({len(all_knowledge)} 条知识)")

        logger.info("知识提取完成")
