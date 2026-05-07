"""
运行所有 RAG 方法的脚本

用法:
    python -m scripts.run_all_methods --model-path /path/to/model
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(SCRIPT_DIR, "run.py")

RAG_METHODS = [
    "NoRAG",
    "BasicRAG",
    "FixedLengthRAG",
    "FixedSentenceRAG",
    "TokenRAG",
    "DRAG",
    "FlareRAG",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="运行所有 RAG 方法")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    args = parser.parse_args()

    for method in RAG_METHODS:
        print(f"\n{'#' * 60}")
        print(f"# Starting: {method}")
        print(f"{'#' * 60}")

        cmd = [sys.executable, RUN_PY, "--method", method, "--model-path", args.model_path]
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"ERROR: {method} failed with code {result.returncode}")
            sys.exit(1)
        else:
            print(f"SUCCESS: {method} completed")

    print(f"\n{'=' * 60}")
    print("All methods completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
