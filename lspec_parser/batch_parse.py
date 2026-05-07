"""LSpec 批量解析工具。

批量解析目录下的 .lspec / .txt 文件，输出解析结果。
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .parser import parse_lspec, ParserError
from .lexer import LexerError


def parse_single_file(filepath: str) -> dict:
    """解析单个 LSpec 文件，返回结果字典。"""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        doc = parse_lspec(source)
        result = {
            "file": filepath,
            "status": "success",
            "document": asdict(doc),
        }
    except (LexerError, ParserError) as e:
        result = {
            "file": filepath,
            "status": "error",
            "error": str(e),
        }
    return result


def batch_parse(
    input_dir: str,
    extensions: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> List[dict]:
    """批量解析目录下的 LSpec 文件。

    Args:
        input_dir: 输入目录路径
        extensions: 要处理的文件扩展名列表，默认 [".lspec", ".txt"]
        output_path: 输出 JSON 文件路径，为 None 则不输出

    Returns:
        解析结果列表
    """
    if extensions is None:
        extensions = [".lspec", ".txt"]

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    results = []
    for ext in extensions:
        for filepath in sorted(input_path.rglob(f"*{ext}")):
            print(f"Parsing: {filepath}")
            result = parse_single_file(str(filepath))
            results.append(result)

    # 统计
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nTotal: {len(results)}, Success: {success}, Errors: {errors}")

    # 输出到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")

    return results


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Batch parse LSpec files")
    parser.add_argument("input_dir", help="Directory containing LSpec files")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    parser.add_argument(
        "-e", "--extensions",
        nargs="+",
        default=[".lspec", ".txt"],
        help="File extensions to process (default: .lspec .txt)",
    )
    args = parser.parse_args()
    batch_parse(args.input_dir, args.extensions, args.output)


if __name__ == "__main__":
    main()
