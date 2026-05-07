"""
知识合并器 - 读取提取的知识文件，按 (source_keyword, formal_element) 去重合并
"""

import os
import json
import re
import logging
from typing import List, Dict, Tuple

import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class KnowledgeMerger:
    """知识合并工具"""

    @staticmethod
    def load_knowledge_files(knowledge_dir: str) -> List[Dict]:
        """加载所有成功的知识文件"""
        all_knowledge = []

        if not os.path.exists(knowledge_dir):
            logger.warning(f"目录不存在: {knowledge_dir}")
            return all_knowledge

        for filename in os.listdir(knowledge_dir):
            if not filename.endswith("_knowledge.txt") or "_FAILED" in filename:
                continue

            filepath = os.path.join(knowledge_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())

                req_id = data.get("req_id", "")
                if not req_id:
                    match = re.match(r"(.+)_knowledge\.txt$", filename)
                    req_id = match.group(1) if match else ""

                for item in data.get("knowledge", []):
                    item["req_id"] = req_id
                    all_knowledge.append(item)

                logger.info(f"加载: {filename} ({len(data.get('knowledge', []))} 条)")
            except Exception as e:
                logger.warning(f"加载失败: {filename} - {e}")

        return all_knowledge

    @staticmethod
    def merge_keep_all_sources(knowledge_list: List[Dict]) -> Tuple[List[Dict], Dict]:
        """按 (source_keyword, formal_element) 去重，合并所有来源 req_id"""
        merged = {}
        duplicate_count = 0

        for item in knowledge_list:
            source_keyword = item.get("source_keyword", "").strip().lower()
            formal_element = item.get("formal_element", "").strip()
            if not source_keyword or not formal_element:
                continue

            key = (source_keyword, formal_element)
            if key not in merged:
                merged[key] = item.copy()
                merged[key]["req_ids"] = [item.get("req_id", "")]
            else:
                duplicate_count += 1
                existing_req_id = item.get("req_id", "")
                if existing_req_id and existing_req_id not in merged[key]["req_ids"]:
                    merged[key]["req_ids"].append(existing_req_id)

        result = []
        for item in merged.values():
            req_ids = item.pop("req_ids", [])
            item["req_id"] = ", ".join(sorted(set(req_ids)))
            result.append(item)

        stats = {"total": len(knowledge_list), "merged": len(merged), "duplicates": duplicate_count}
        return result, stats

    @staticmethod
    def save_to_excel(knowledge_list: List[Dict], output_path: str):
        """保存合并后的知识到 Excel"""
        if not knowledge_list:
            logger.warning("没有知识可以保存")
            return

        columns = ["req_id", "source_keyword", "formal_element", "description", "example_context"]
        rows = [{col: item.get(col, "") for col in columns} for item in knowledge_list]
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values(by=["source_keyword", "formal_element"])
        df.to_excel(output_path, index=False, engine="openpyxl")
        logger.info(f"已保存合并 Excel: {output_path}")

    def run(self, knowledge_dir: str, output_dir: str):
        """运行合并主流程"""
        logger.info("知识合并程序启动")

        all_knowledge = self.load_knowledge_files(knowledge_dir)
        logger.info(f"共加载 {len(all_knowledge)} 条知识")

        if not all_knowledge:
            logger.warning("没有找到知识文件")
            return

        merged, stats = self.merge_keep_all_sources(all_knowledge)
        logger.info(f"合并统计: 原始={stats['total']}, 去重后={stats['merged']}, 重复={stats['duplicates']}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"merged_knowledge_{timestamp}.xlsx")
        self.save_to_excel(merged, output_path)

        logger.info("合并完成")
