"""LSpec 抽象语法树节点定义。"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASTNode:
    """AST 基类。"""
    pass


@dataclass
class LSpecDocument(ASTNode):
    """LSpec 文档根节点。"""
    name: str = ""
    sections: List['Section'] = field(default_factory=list)

    def __str__(self) -> str:
        sections_str = "\n".join(str(s) for s in self.sections)
        return f"Document({self.name}):\n{sections_str}"


@dataclass
class Section(ASTNode):
    """章节节点。"""
    title: str = ""
    content: str = ""
    items: List['Item'] = field(default_factory=list)

    def __str__(self) -> str:
        items_str = "\n".join(f"  {it}" for it in self.items)
        return f"[{self.title}] {self.content}\n{items_str}"


@dataclass
class Item(ASTNode):
    """条目节点。"""
    identifier: str = ""
    item_type: str = ""  # e.g., "requirement", "constraint", "scenario"
    description: str = ""
    attributes: dict = field(default_factory=dict)
    sub_items: List['SubItem'] = field(default_factory=list)

    def __str__(self) -> str:
        attrs_str = f" {self.attributes}" if self.attributes else ""
        subs_str = "\n".join(f"    {s}" for s in self.sub_items)
        return f"<{self.item_type}> {self.identifier}: {self.description}{attrs_str}\n{subs_str}"


@dataclass
class SubItem(ASTNode):
    """子条目节点。"""
    identifier: str = ""
    sub_type: str = ""
    description: str = ""
    attributes: dict = field(default_factory=dict)

    def __str__(self) -> str:
        attrs_str = f" {self.attributes}" if self.attributes else ""
        return f"<{self.sub_type}> {self.identifier}: {self.description}{attrs_str}"


@dataclass
class Attribute(ASTNode):
    """属性节点。"""
    key: str = ""
    value: str = ""

    def __str__(self) -> str:
        return f"{self.key}={self.value}"
