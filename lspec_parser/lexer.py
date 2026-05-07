"""LSpec 词法分析器。"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    """Token 类型枚举。"""
    # 结构符号
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    LANGLE = auto()         # <
    RANGLE = auto()         # >
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    COLON = auto()          # :
    SEMICOLON = auto()      # ;
    COMMA = auto()          # ,
    DOT = auto()            # .
    EQUALS = auto()         # =
    HASH = auto()           # #
    DASH = auto()           # -
    ARROW = auto()          # ->

    # 关键词
    SECTION = auto()
    REQUIREMENT = auto()
    CONSTRAINT = auto()
    SCENARIO = auto()
    ACTOR = auto()
    ACTION = auto()
    WHEN = auto()
    THEN = auto()
    IF = auto()
    ELSE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    MUST = auto()
    SHOULD = auto()
    MAY = auto()

    # 字面量
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()

    # 特殊
    NEWLINE = auto()
    INDENT = auto()
    EOF = auto()


@dataclass
class Token:
    """Token 结构。"""
    type: TokenType
    value: str
    line: int = 0
    column: int = 0

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line})"


# 关键词映射
KEYWORDS = {
    'section': TokenType.SECTION,
    'requirement': TokenType.REQUIREMENT,
    'constraint': TokenType.CONSTRAINT,
    'scenario': TokenType.SCENARIO,
    'actor': TokenType.ACTOR,
    'action': TokenType.ACTION,
    'when': TokenType.WHEN,
    'then': TokenType.THEN,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'must': TokenType.MUST,
    'should': TokenType.SHOULD,
    'may': TokenType.MAY,
}

# 符号映射
SYMBOLS = {
    '[': TokenType.LBRACKET,
    ']': TokenType.RBRACKET,
    '<': TokenType.LANGLE,
    '>': TokenType.RANGLE,
    '{': TokenType.LBRACE,
    '}': TokenType.RBRACE,
    '(': TokenType.LPAREN,
    ')': TokenType.RPAREN,
    ':': TokenType.COLON,
    ';': TokenType.SEMICOLON,
    ',': TokenType.COMMA,
    '.': TokenType.DOT,
    '=': TokenType.EQUALS,
    '#': TokenType.HASH,
    '-': TokenType.DASH,
}


class LexerError(Exception):
    """词法分析错误。"""

    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.line = line
        self.column = column
        super().__init__(f"Lexer error at L{line}:C{column}: {message}")


class Lexer:
    """LSpec 词法分析器。"""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else ''

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.source) and self.source[self.pos] in ' \t':
            self._advance()

    def _skip_comment(self) -> None:
        if self._peek() == '#' and self._peek(1) != '[':
            while self.pos < len(self.source) and self.source[self.pos] != '\n':
                self._advance()

    def _read_string(self, quote: str) -> Token:
        line, col = self.line, self.column
        self._advance()  # skip opening quote
        value = []
        while self.pos < len(self.source):
            ch = self._peek()
            if ch == '\\':
                self._advance()
                esc = self._advance()
                escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                value.append(escape_map.get(esc, esc))
            elif ch == quote:
                self._advance()
                return Token(TokenType.STRING, ''.join(value), line, col)
            else:
                value.append(self._advance())
        raise LexerError("Unterminated string", line, col)

    def _read_number(self) -> Token:
        line, col = self.line, self.column
        value = []
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == '.'):
            value.append(self._advance())
        return Token(TokenType.NUMBER, ''.join(value), line, col)

    def _read_identifier(self) -> Token:
        line, col = self.line, self.column
        value = []
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] in '_-'):
            value.append(self._advance())
        text = ''.join(value)
        token_type = KEYWORDS.get(text.lower(), TokenType.IDENTIFIER)
        return Token(token_type, text, line, col)

    def _read_arrow(self) -> Token:
        line, col = self.line, self.column
        self._advance()  # skip '-'
        self._advance()  # skip '>'
        return Token(TokenType.ARROW, '->', line, col)

    def tokenize(self) -> List[Token]:
        """执行词法分析，返回 token 列表。"""
        while self.pos < len(self.source):
            self._skip_whitespace()
            if self.pos >= len(self.source):
                break

            ch = self._peek()

            # 换行
            if ch == '\n':
                self._advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\\n', self.line, self.column))
                continue

            # 注释
            if ch == '#':
                self._skip_comment()
                continue

            # 字符串
            if ch in '"\'':
                self.tokens.append(self._read_string(ch))
                continue

            # 箭头 ->
            if ch == '-' and self._peek(1) == '>':
                self.tokens.append(self._read_arrow())
                continue

            # 数字
            if ch.isdigit():
                self.tokens.append(self._read_number())
                continue

            # 标识符 / 关键词
            if ch.isalpha() or ch == '_':
                self.tokens.append(self._read_identifier())
                continue

            # 符号
            if ch in SYMBOLS:
                line, col = self.line, self.column
                self._advance()
                self.tokens.append(Token(SYMBOLS[ch], ch, line, col))
                continue

            raise LexerError(f"Unexpected character: {ch!r}", self.line, self.column)

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
