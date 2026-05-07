"""语法正确性 (Syntax Correctness) 指标。"""

import logging
import subprocess
import tempfile
import os
from pathlib import Path

from ..metrics.base import BaseMetric, MetricResult, register_metric

logger = logging.getLogger(__name__)

# 语言到编译/检查命令的映射
SYNTAX_CHECK_COMMANDS = {
    "python": ["python", "-m", "py_compile", "{file}"],
    "java": ["javac", "-d", "{temp_dir}", "{file}"],
    "c": ["gcc", "-fsyntax-only", "{file}"],
    "cpp": ["g++", "-fsyntax-only", "{file}"],
    "javascript": ["node", "--check", "{file}"],
}


@register_metric
class SyntaxCorrectness(BaseMetric):
    """语法正确性：检查生成代码是否能通过语法检查。"""

    name = "syntax_correctness"
    description = "Whether the generated code is syntactically correct"
    higher_is_better = True
    value_range = (0.0, 1.0)

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        language = kwargs.get("language", "python")

        if language not in SYNTAX_CHECK_COMMANDS:
            logger.warning(f"Unsupported language: {language}, defaulting to Python")
            language = "python"

        # 尝试语法检查
        is_correct = self._check_syntax(prediction, language)

        return MetricResult(
            name=self.name,
            score=1.0 if is_correct else 0.0,
            details={"language": language},
        )

    def _check_syntax(self, code: str, language: str) -> bool:
        """执行语法检查。"""
        ext_map = {
            "python": ".py",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp",
            "javascript": ".js",
        }
        ext = ext_map.get(language, ".txt")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=ext, delete=False, encoding="utf-8",
            ) as f:
                f.write(code)
                temp_path = f.name

            cmd_template = SYNTAX_CHECK_COMMANDS[language]
            cmd = [
                arg.replace("{file}", temp_path).replace("{temp_dir}", os.path.dirname(temp_path))
                for arg in cmd_template
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
                text=True,
            )

            return result.returncode == 0

        except FileNotFoundError:
            # 编译器未安装，回退到简单检查
            logger.warning(f"Compiler/checker not found for {language}, using fallback")
            return self._fallback_check(code, language)

        except subprocess.TimeoutExpired:
            logger.warning("Syntax check timed out")
            return False

        except Exception as e:
            logger.warning(f"Syntax check failed: {e}")
            return False

        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    @staticmethod
    def _fallback_check(code: str, language: str) -> bool:
        """简单回退语法检查。"""
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False
        # 其他语言无法回退检查，默认通过
        return True
