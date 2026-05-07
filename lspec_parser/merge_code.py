"""LSpec 代码合并工具。

将解析后的 AST 结果与已有代码进行合并/对齐。
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .parser import parse_lspec, ParserError
from .lexer import LexerError
from .ast_nodes import LSpecDocument


def merge_lspec_with_code(
    lspec_path: str,
    code_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """将 LSpec 解析结果与代码文件合并。

    Args:
        lspec_path: LSpec 文件路径
        code_path: 代码文件路径
        output_path: 输出文件路径

    Returns:
        合并结果字典
    """
    # 解析 LSpec
    with open(lspec_path, "r", encoding="utf-8") as f:
        lspec_source = f.read()

    try:
        doc = parse_lspec(lspec_source)
    except (LexerError, ParserError) as e:
        return {"status": "error", "error": f"Failed to parse LSpec: {e}"}

    # 读取代码
    with open(code_path, "r", encoding="utf-8") as f:
        code_lines = f.readlines()

    # 构建 ID -> 描述 映射
    id_to_desc: Dict[str, str] = {}
    for section in doc.sections:
        for item in section.items:
            if item.identifier:
                id_to_desc[item.identifier] = item.description

    # 在代码中查找引用并标注
    annotations: List[dict] = []
    for i, line in enumerate(code_lines):
        for item_id, desc in id_to_desc.items():
            if item_id in line:
                annotations.append({
                    "line": i + 1,
                    "code": line.rstrip(),
                    "requirement_id": item_id,
                    "requirement_desc": desc,
                })

    result = {
        "status": "success",
        "lspec_file": lspec_path,
        "code_file": code_path,
        "total_requirements": len(id_to_desc),
        "matched_annotations": len(annotations),
        "annotations": annotations,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="Merge LSpec with code")
    parser.add_argument("lspec_file", help="LSpec file path")
    parser.add_argument("code_file", help="Code file path")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    result = merge_lspec_with_code(args.lspec_file, args.code_file, args.output)
    if result["status"] == "success":
        print(f"Requirements: {result['total_requirements']}, Matches: {result['matched_annotations']}")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
