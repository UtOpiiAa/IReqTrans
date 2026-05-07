"""
合并多任务分片产生的 Excel 结果文件

用法:
    python -m scripts.merge_results --model qwen3 --method DRAG --total-tasks 4
    python -m scripts.merge_results --input-dir ./data/result --output-dir ./merged
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_project_root


def merge_excel_files(file_paths, output_path):
    """合并多个 Excel 文件"""
    all_dfs = []
    for f in file_paths:
        try:
            df = pd.read_excel(f)
            all_dfs.append(df)
            print(f"  - {Path(f).name}: {len(df)} 行")
        except Exception as e:
            print(f"  读取失败: {Path(f).name} - {e}")

    if not all_dfs:
        print("没有成功读取任何文件")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    if "req_id" in merged_df.columns:
        merged_df = merged_df.sort_values("req_id").reset_index(drop=True)
    merged_df.to_excel(output_path, index=False)
    print(f"\n合并完成！总行数: {len(merged_df)}，输出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并多任务分片 Excel 结果")
    parser.add_argument("--input-dir", type=str, default=None, help="输入目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--output", type=str, default="merged_results.xlsx", help="输出文件名")
    parser.add_argument("--pattern", type=str, default="*.xlsx", help="文件匹配模式")
    args = parser.parse_args()

    root = get_project_root()
    input_dir = Path(args.input_dir) if args.input_dir else root / "data" / "result"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"在 {input_dir} 中未找到匹配 {args.pattern} 的文件")
        return

    print(f"找到 {len(files)} 个文件")
    output_path = output_dir / args.output
    merge_excel_files(files, str(output_path))


if __name__ == "__main__":
    main()
