"""LSpec 语法分析器。"""

from typing import List, Optional

from .ast_nodes import (
    ASTNode, LSpecDocument, Section, Item, SubItem, Attribute,
)
from .lexer import Lexer, Token, TokenType, LexerError


class ParserError(Exception):
    """语法分析错误。"""

    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        loc = f" at L{token.line}:C{token.column}" if token else ""
        super().__init__(f"Parser error{loc}: {message}")


class Parser:
    """LSpec 语法分析器。

    支持解析 LSpec 格式的需求规约文本，生成 AST。

    LSpec 文本格式示例::

        [系统名称]

        ## 功能需求

        REQ-001: 系统应支持用户登录
          Actor: 用户
          Action: 输入用户名和密码
          When: 用户访问登录页面
          Then: 系统验证身份并跳转主页

        ## 非功能需求

        CON-001: 响应时间应小于2秒
          Type: 性能
          Value: <2s
    """

    def __init__(self, source: str):
        self.lexer = Lexer(source)
        self.tokens: List[Token] = []
        self.pos = 0

    def _current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, '', 0, 0)

    def _peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token(TokenType.EOF, '', 0, 0)

    def _advance(self) -> Token:
        token = self._current()
        self.pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        token = self._current()
        if token.type != token_type:
            raise ParserError(
                f"Expected {token_type.name}, got {token.type.name} ({token.value!r})",
                token,
            )
        return self._advance()

    def _match(self, *token_types: TokenType) -> Optional[Token]:
        if self._current().type in token_types:
            return self._advance()
        return None

    def _skip_newlines(self) -> None:
        while self._current().type == TokenType.NEWLINE:
            self._advance()

    # ── 主解析入口 ──────────────────────────────────────

    def parse(self) -> LSpecDocument:
        """解析 LSpec 文本，返回 AST。"""
        self.tokens = self.lexer.tokenize()
        self.pos = 0
        self._skip_newlines()

        document = LSpecDocument()
        document.name = self._parse_document_header()
        self._skip_newlines()

        while self._current().type != TokenType.EOF:
            section = self._try_parse_section()
            if section:
                document.sections.append(section)
            else:
                # 跳过无法识别的行
                self._advance()
            self._skip_newlines()

        return document

    # ── 文档头 ──────────────────────────────────────────

    def _parse_document_header(self) -> str:
        """解析文档标题，如 [系统名称]。"""
        if self._match(TokenType.LBRACKET):
            name_parts = []
            while self._current().type not in (TokenType.RBRACKET, TokenType.EOF):
                name_parts.append(self._advance().value)
            self._match(TokenType.RBRACKET)
            return " ".join(name_parts).strip()
        return ""

    # ── 章节解析 ────────────────────────────────────────

    def _try_parse_section(self) -> Optional[Section]:
        """尝试解析一个章节（以 ## 或 SECTION 关键词开头）。"""
        # 形式1: ## 标题
        if self._current().value == '##' or (
            self._current().type == TokenType.HASH
            and self._peek().type == TokenType.HASH
        ):
            return self._parse_section_heading()

        # 形式2: section 标题
        if self._current().type == TokenType.SECTION:
            return self._parse_section_keyword()

        return None

    def _parse_section_heading(self) -> Section:
        """解析 ## 标题 形式的章节。"""
        self._advance()  # skip first #
        self._match(TokenType.HASH)  # skip second #
        title_parts = []
        while self._current().type not in (TokenType.NEWLINE, TokenType.EOF):
            title_parts.append(self._advance().value)
        title = " ".join(title_parts).strip()

        self._skip_newlines()
        items = self._parse_items()

        section = Section(title=title, items=items)
        return section

    def _parse_section_keyword(self) -> Section:
        """解析 section 标题 形式的章节。"""
        self._advance()  # skip 'section'
        title_parts = []
        while self._current().type not in (TokenType.NEWLINE, TokenType.LBRACKET, TokenType.EOF):
            title_parts.append(self._advance().value)
        title = " ".join(title_parts).strip()

        self._skip_newlines()
        items = self._parse_items()
        return Section(title=title, items=items)

    # ── 条目解析 ────────────────────────────────────────

    def _parse_items(self) -> List[Item]:
        """解析章节下的条目列表。"""
        items = []
        while self._current().type != TokenType.EOF:
            # 检查是否是新一章节
            if self._current().value == '##' or self._current().type == TokenType.SECTION:
                break
            if self._current().type == TokenType.HASH and self._peek().type == TokenType.HASH:
                break

            item = self._try_parse_item()
            if item:
                items.append(item)
            else:
                # 跳过空行
                if self._current().type == TokenType.NEWLINE:
                    self._advance()
                else:
                    break
        return items

    def _try_parse_item(self) -> Optional[Item]:
        """尝试解析一个条目。

        支持格式:
        - REQ-001: 描述
        - <requirement> REQ-001: 描述
        - requirement REQ-001: 描述
        """
        self._skip_newlines()
        if self._current().type == TokenType.EOF:
            return None

        item_type = ""
        identifier = ""
        description = ""

        # 解析类型标签 <requirement>
        if self._current().type == TokenType.LANGLE:
            item_type = self._parse_type_tag()

        # 解析类型关键词 requirement
        if not item_type and self._current().type in (
            TokenType.REQUIREMENT, TokenType.CONSTRAINT, TokenType.SCENARIO,
        ):
            item_type = self._advance().value.lower()

        # 解析标识符（如 REQ-001）
        if self._current().type == TokenType.IDENTIFIER:
            identifier = self._advance().value
            # 追加后续标识符部分（如 -001）
            while self._current().type == TokenType.DASH:
                identifier += self._advance().value
                if self._current().type in (TokenType.IDENTIFIER, TokenType.NUMBER):
                    identifier += self._advance().value

        # 解析冒号后的描述
        if self._match(TokenType.COLON):
            desc_parts = []
            while self._current().type not in (TokenType.NEWLINE, TokenType.EOF):
                desc_parts.append(self._advance().value)
            description = " ".join(desc_parts).strip()

        if not identifier and not description:
            return None

        self._skip_newlines()

        # 解析子条目
        sub_items = self._parse_sub_items()

        return Item(
            identifier=identifier,
            item_type=item_type,
            description=description,
            sub_items=sub_items,
        )

    def _parse_type_tag(self) -> str:
        """解析 <type> 标签。"""
        self._advance()  # skip <
        type_parts = []
        while self._current().type not in (TokenType.RANGLE, TokenType.EOF):
            type_parts.append(self._advance().value)
        self._match(TokenType.RANGLE)
        return "".join(type_parts).strip().lower()

    # ── 子条目解析 ──────────────────────────────────────

    def _parse_sub_items(self) -> List[SubItem]:
        """解析缩进的子条目。"""
        sub_items = []
        while self._current().type != TokenType.EOF:
            # 新章节开始则停止
            if self._current().value == '##' or self._current().type == TokenType.SECTION:
                break
            if self._current().type == TokenType.HASH and self._peek().type == TokenType.HASH:
                break

            sub = self._try_parse_sub_item()
            if sub:
                sub_items.append(sub)
            else:
                break
        return sub_items

    def _try_parse_sub_item(self) -> Optional[SubItem]:
        """尝试解析一个子条目。

        支持格式:
        - Actor: 用户
        - Action: 输入凭据
        - When: 访问登录页
        - Then: 验证并跳转
        """
        if self._current().type in (
            TokenType.ACTOR, TokenType.ACTION, TokenType.WHEN, TokenType.THEN,
        ):
            sub_type = self._advance().value
            self._match(TokenType.COLON)
            desc_parts = []
            while self._current().type not in (TokenType.NEWLINE, TokenType.EOF):
                desc_parts.append(self._advance().value)
            description = " ".join(desc_parts).strip()
            self._skip_newlines()
            return SubItem(sub_type=sub_type.lower(), description=description)

        # 通用 Key: Value 格式
        if self._current().type == TokenType.IDENTIFIER and self._peek().type == TokenType.COLON:
            key = self._advance().value
            self._advance()  # skip :
            value_parts = []
            while self._current().type not in (TokenType.NEWLINE, TokenType.EOF):
                value_parts.append(self._advance().value)
            value = " ".join(value_parts).strip()
            self._skip_newlines()
            return SubItem(identifier=key, description=value)

        return None


def parse_lspec(source: str) -> LSpecDocument:
    """便捷函数：解析 LSpec 文本。"""
    return Parser(source).parse()
