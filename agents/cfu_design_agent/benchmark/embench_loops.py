"""Conservative, source-position-based discovery of C loop statements.

This lexer is intentionally not a full C parser. It skips preprocessor directives,
comments and literals and returns only loops whose complete statement is
unambiguous; unsupported/macro-generated loops are left uninstrumented.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopSpan:
    function: str
    ordinal: int
    line: int
    start: int
    end: int
    kind: str


def _tokens(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    i = 0
    line_start = True
    while i < len(text):
        c = text[i]
        if c == "\n":
            line_start = True
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if c == "#" and line_start:
            # A directive may span several lines via backslash-newline.
            while i < len(text):
                end = text.find("\n", i)
                if end < 0:
                    i = len(text)
                    break
                continuation = text[i:end].rstrip().endswith("\\")
                i = end + 1
                if not continuation:
                    break
            line_start = True
            continue
        line_start = False
        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        if c in "\"'":
            quote = c
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            continue
        start = i
        if c.isalpha() or c == "_":
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                i += 1
        else:
            i += 1
        tokens.append((text[start:i], start, i))
    return tokens


def _pairs(tokens: list[tuple[str, int, int]], left: str, right: str) -> dict[int, int]:
    stack: list[int] = []
    matched: dict[int, int] = {}
    for index, (token, _, _) in enumerate(tokens):
        if token == left:
            stack.append(index)
        elif token == right and stack:
            matched[stack.pop()] = index
    return matched


def discover_loops(text: str, functions: list[str]) -> list[LoopSpan]:
    """Return loops in named functions, ordered by source position.

    `while` closing a do/while is not an independent loop. An unbraced loop
    body is parsed as a full statement (including nested control flow); an
    ambiguous statement is skipped rather than instrumented at a guessed end.
    """
    from .embench_instrument import find_definition_brace

    tokens = _tokens(text)
    by_start = {start: i for i, (_, start, _) in enumerate(tokens)}
    braces = _pairs(tokens, "{", "}")
    parens = _pairs(tokens, "(", ")")
    intervals: list[tuple[str, int, int]] = []
    for name in functions:
        brace = find_definition_brace(text, name)
        pos = by_start.get(brace)
        if pos is not None and pos in braces:
            intervals.append((name, pos, braces[pos]))
    def statement(i: int, depth: int = 0) -> int | None:
        """Index of a complete statement's last token (bounded recursion)."""
        if depth > 48 or i >= len(tokens):
            return None
        word = tokens[i][0]
        if word == "{":
            return braces.get(i)
        if word in {"for", "while", "switch", "if"}:
            if i + 1 >= len(tokens) or tokens[i + 1][0] != "(":
                return None
            condition = parens.get(i + 1)
            if condition is None:
                return None
            end = statement(condition + 1, depth + 1)
            if end is None:
                return None
            if word == "if" and end + 1 < len(tokens) and tokens[end + 1][0] == "else":
                return statement(end + 2, depth + 1)
            return end
        if word == "do":
            end = statement(i + 1, depth + 1)
            if end is None or end + 2 >= len(tokens) or tokens[end + 1][0] != "while" or tokens[end + 2][0] != "(":
                return None
            condition = parens.get(end + 2)
            if condition is None or condition + 1 >= len(tokens) or tokens[condition + 1][0] != ";":
                return None
            return condition + 1
        if word == ";":
            return i
        nesting = 0
        for end in range(i, len(tokens)):
            token = tokens[end][0]
            if token == "(":
                nesting += 1
            elif token == ")":
                nesting -= 1
            elif token in {"{", "}"} and nesting == 0:
                return None
            elif token == ";" and nesting == 0:
                return end
        return None

    loops: list[LoopSpan] = []
    counts: dict[str, int] = {}
    do_tails: set[int] = set()
    for i, (word, start, _) in enumerate(tokens):
        owner = next((name for name, first, last in intervals if first < i < last), "")
        if not owner or word not in {"for", "while", "do"} or i in do_tails:
            continue
        # A loop token inside a function-like macro invocation is not a C
        # statement at this position; leave it to the compiler, not our lexer.
        if i > 0 and tokens[i - 1][0] in {"#", "\\"}:
            continue
        if word == "do":
            body_end = statement(i + 1)
            if body_end is None or body_end + 1 >= len(tokens):
                continue
            do_tails.add(body_end + 1)
        last = statement(i)
        if last is None:
            continue
        counts[owner] = counts.get(owner, 0) + 1
        loops.append(LoopSpan(owner, counts[owner], text.count("\n", 0, start) + 1,
                              start, tokens[last][2], word))
    return loops


def loop_id(file: str, function: str, ordinal: int) -> str:
    """Unique, stable C identifier across source files in the same workload."""
    stem = re.sub(r"\W", "_", function)[:32]
    digest = hashlib.sha256(file.encode()).hexdigest()[:8]
    return f"obs_loop_{digest}_{stem}_{ordinal}"
