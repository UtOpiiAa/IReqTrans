"""
知识提取入口脚本

用法:
    python -m scripts.extract_knowledge
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, get_project_root
from knowledge_extractor import KnowledgeExtractor


def main():
    root = get_project_root()
    config_path = str(root / "config" / "default.yaml")
    config = Config.from_yaml(config_path) if os.path.exists(config_path) else Config()

    extractor = KnowledgeExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
