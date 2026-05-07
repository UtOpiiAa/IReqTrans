"""
文件读写工具 - Excel/文本/JSON 文件的读写操作
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import chardet

logger = logging.getLogger(__name__)


class FileReader:
    """文件读取工具，支持 Excel、文本、JSON 格式"""

    ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1", "cp1252"]

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    def read_excel(self, sheet_name=0) -> pd.DataFrame:
        """读取 Excel 文件"""
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        logger.info(f"读取 Excel: {len(df)} 行, {len(df.columns)} 列")
        return df

    def get_keyword_explanation_by_id(self, id_column: str, target_id: str) -> str:
        """从 Excel 中根据 ID 获取 keyword + explanation 两列，拼接返回"""
        df = self.read_excel()
        filtered = df[df[id_column] == target_id]
        lines = []
        for _, row in filtered.iterrows():
            kw = str(row.get("keyword", ""))
            ex = str(row.get("explanation", ""))
            lines.append(f"{kw}: {ex}")
        return "\n".join(lines)

    def read_txt(self, encoding: Optional[str] = None) -> str:
        """读取文本文件，自动检测编码"""
        if encoding:
            try:
                with open(self.file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                logger.warning(f"指定编码 {encoding} 失败，尝试自动检测")

        for enc in self.ENCODINGS:
            try:
                with open(self.file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # 使用 chardet 检测
        try:
            with open(self.file_path, "rb") as f:
                raw_data = f.read()
            detected = chardet.detect(raw_data)
            if detected and detected["encoding"]:
                return raw_data.decode(detected["encoding"], errors="ignore")
        except Exception as e:
            logger.warning(f"chardet 检测失败: {e}")

        with open(self.file_path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

    def read_json(self, encoding: Optional[str] = None) -> Dict:
        """读取 JSON 文件"""
        content = self.read_txt(encoding=encoding)

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)


class ExcelWriter:
    """Excel 文件写入工具"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write(
        self,
        data_list: List[Dict],
        output_filename: str,
        sheet_name: str = "Sheet1",
        include_index: bool = False,
    ) -> str:
        """将字典列表写入 Excel 文件"""
        if not data_list:
            logger.warning("数据列表为空，无法写入")
            return ""

        df = pd.DataFrame(data_list)
        output_path = os.path.join(self.output_dir, output_filename)
        df.to_excel(output_path, sheet_name=sheet_name, index=include_index)
        logger.info(f"写入 {len(data_list)} 行数据到 {output_path}")
        return output_path

    def append(
        self,
        data_list: List[Dict],
        output_filename: str,
        sheet_name: str = "Sheet1",
    ) -> str:
        """追加数据到现有 Excel 文件"""
        output_path = os.path.join(self.output_dir, output_filename)
        new_df = pd.DataFrame(data_list)

        if os.path.exists(output_path):
            existing_df = pd.read_excel(output_path, sheet_name=sheet_name)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_excel(output_path, sheet_name=sheet_name, index=False)
        logger.info(f"追加 {len(data_list)} 行数据到 {output_path}")
        return output_path


class TextWriter:
    """文本文件写入工具"""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
        self.content = ""

    def append_content(self, content: str):
        self.content += content

    def write(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding=self.encoding) as f:
            f.write(self.content)
        logger.info(f"写入文本到 {self.file_path}")

    def clear(self):
        self.content = ""


class ExcelReader:
    """Excel 批量读取工具"""

    def __init__(self, input_dir: str = "./input"):
        self.input_dir = input_dir
        os.makedirs(input_dir, exist_ok=True)

    def read(self, filename: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
        excel_path = os.path.join(self.input_dir, filename)
        df = pd.read_excel(
            excel_path, sheet_name=sheet_name, engine="openpyxl", dtype=str
        )
        return df
