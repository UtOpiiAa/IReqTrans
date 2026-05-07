"""
知识合并入口脚本

用法:
    python -m scripts.merge_knowledge
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, get_project_root
from knowledge_extractor import KnowledgeMerger


def main():
    root = get_project_root()
    config_path = str(root / "config" / "default.yaml")
    config = Config.from_yaml(config_path) if os.path.exists(config_path) else Config()

    knowledge_dir = os.path.join(config.resolve_path(config.extraction.output_dir), "knowledge_files")
    output_dir = config.resolve_path(config.extraction.output_dir)

    merger = KnowledgeMerger()
    merger.run(knowledge_dir, output_dir)


if __name__ == "__main__":
    main()
