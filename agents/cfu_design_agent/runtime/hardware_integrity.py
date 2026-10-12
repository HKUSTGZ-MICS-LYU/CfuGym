"""Bounded source lint for fixed-answer CFUs; never a semantic-equivalence proof.

Computational value replies must not be provably constant. A dummy operand/LSU
read, or waiting on ``lsu.done``, does not make a literal result computational.
This checks literal replies and simple aliases/helpers/registers in the agent
Scala surface, including legacy ``ctx.reply``. Unknown Scala expressions are
not certified: differential conformance and workload verification still own
correctness. No model calls, compiler, subprocess, or generated-file writes.
"""
from __future__ import annotations

import ast
import re

from .. import isa
from . import instruction_encoding

VERSION = 2
# Keep strings (instruction names and Spinal literals), blank comments, retain lines.
_LEX = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
_IDENT = re.compile(r'[A-Za-z_]\w*\Z')
_UNKNOWN = object()


def _code(text: str) -> str:
    return instruction_encoding.code(text)


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Content of an opening parenthesis, ignoring quoted Scala text."""
    depth, quote, escape = 0, '', False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = ''
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return '', start


def _arguments(text: str) -> list[str]:
    depth, quote, escape, start = 0, '', False, 0
    result = []
    for i, ch in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = ''
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch in '({[':
            depth += 1
        elif ch in ')}]':
            depth -= 1
        elif ch == ',' and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    result.append(text[start:].strip())
    return result


def _searchable(text: str) -> str:
    """Hide quoted text from syntax searches without shifting source offsets."""
    return instruction_encoding.code(text, strings=True)


def _expression_at(text: str, start: int) -> str:
    depth, quote, escape = 0, '', False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = ''
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch in '({[':
            depth += 1
        elif ch in ')}]':
            if not depth:
                return text[start:i].strip()
            depth -= 1
        elif depth == 0 and ch in '\n;,':
            return text[start:i].strip()
    return text[start:].strip()


def _definitions(text: str) -> dict[str, list[str]]:
    """Only straight-line definitions; conditional assignments remain unknown.

    Indexed/lane writes mark their whole result unknown, so an initial zero-filled
    vector populated with computed lanes is not mistaken for a constant reply.
    """
    values: dict[str, list[str]] = {}
    searchable = _searchable(text)
    pattern = re.compile(r'\b(?:val|var|def)\s+([A-Za-z_]\w*)'
                         r'(?:\s*\(\s*\))?\s*(?::[^=\n]+)?\s*=\s*')
    for match in pattern.finditer(searchable):
        rhs = _expression_at(text, match.end())
        # Defs with parameters, complex blocks and nested scope shadowing are
        # deliberately unknown rather than inferred as pure helper functions.
        values.setdefault(match.group(1), []).append(rhs)
    for match in re.finditer(r'\b([A-Za-z_]\w*)\s*(\([^\n]*?\))?\s*:=\s*', searchable):
        values.setdefault(match.group(1), []).append(
            '<indexed write>' if match.group(2) else _expression_at(text, match.end()))
    # A definition may get updated through foreach/assignFromBits, etc.; avoid
    # claiming its initializer is the final value.
    for match in re.finditer(r'\b([A-Za-z_]\w*)\.(?:foreach|assignFromBits|assignFrom)\s*\(', text):
        values.setdefault(match.group(1), []).append('<mutation>')
    return values


def _constant(expr: str, values: dict[str, list[str]], seen: frozenset[str] = frozenset()):
    expr = expr.strip()
    if len(seen) > 32 or not expr or len(expr) > 16_384:
        return _UNKNOWN
    if expr.startswith('{') and expr.endswith('}'):
        inner = expr[1:-1].strip()
        if inner.startswith('return '):
            inner = inner[7:].strip()
        if not re.search(r'\b(?:val|var|def|when)\b|;', inner):
            return _constant(inner, values, seen)
    # Conversions preserve the fact that the data is constant (not its width).
    expr = re.sub(r'\.(?:asBits|asUInt|asSInt)\b', '', expr)
    expr = re.sub(r'\.(?:resize|resized)\s*\([^()]*\)\s*$', '', expr).strip()
    if expr.startswith('('):
        content, end = _balanced(expr, 0)
        if end == len(expr):
            return _constant(content, values, seen)
    literal = re.fullmatch(r'(?:B|U|S)\s*"([^"\n]+)"', expr)
    if literal:
        token = literal.group(1).replace('_', '')
        sized = re.fullmatch(r"(\d+)'([hbd])([0-9a-fA-F]+)", token)
        try:
            if sized:
                return int(sized[3], {'h': 16, 'b': 2, 'd': 10}[sized[2]]) % (1 << min(int(sized[1]), 4096))
            if re.fullmatch('[01]+', token): return int(token, 2)
        except ValueError:
            pass
        return ('spinal_literal', literal.group(1))
    wrapped = re.match(r'(?:B|U|S|BigInt)\s*\(', expr)
    if wrapped:
        content, end = _balanced(expr, expr.index('('))
        if end == len(expr):
            return _constant(_arguments(content)[0], values, seen)
    # An init-only register carrying a fixed answer is also not computation.
    reg = re.match(r'Reg\s*\(', expr)
    if reg:
        _body, end = _balanced(expr, expr.index('('))
        init = re.match(r'\s*init\s*\(', expr[end:])
        if init:
            start = end + init.end() - 1
            content, finish = _balanced(expr, start)
            if finish == len(expr):
                return _constant(content, values, seen)
    name = expr[:-2].strip() if expr.endswith('()') else expr
    if _IDENT.fullmatch(name) and name in values and name not in seen:
        candidates = [_constant(rhs, values, seen | {name}) for rhs in values[name]]
        if candidates and all(value is not _UNKNOWN and value == candidates[0] for value in candidates):
            return candidates[0]
        return _UNKNOWN
    # Scala indexed/sliced XOR-self is fixed too (Python cannot parse downto).
    halves = expr.split('^')
    if len(halves) == 2 and halves[0].strip() == halves[1].strip():
        return 0
    # Fold a small safe subset, including x * 0 and x - x disguises. No eval.
    normalized = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)[lL]\b', r'\1', expr)
    try:
        node = ast.parse(normalized, mode='eval').body
    except (SyntaxError, ValueError, RecursionError):
        return _UNKNOWN
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return node.value if node.value.bit_length() <= 4096 else _UNKNOWN
    if isinstance(node, ast.BinOp):
        left = _constant(ast.unparse(node.left), values, seen)
        right = _constant(ast.unparse(node.right), values, seen)
        if isinstance(node.op, (ast.Mult, ast.BitAnd)) and (left == 0 or right == 0):
            return 0
        if isinstance(node.op, (ast.Sub, ast.BitXor)) and ast.dump(node.left) == ast.dump(node.right):
            return 0
        if isinstance(left, int) and isinstance(right, int):
            if left.bit_length() + right.bit_length() > 4096:
                return _UNKNOWN
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.BitOr): return left | right
            if isinstance(node.op, ast.BitAnd): return left & right
            if isinstance(node.op, ast.BitXor): return left ^ right
    if isinstance(node, ast.UnaryOp):
        value = _constant(ast.unparse(node.operand), values, seen)
        if isinstance(value, int):
            if isinstance(node.op, ast.USub): return -value
            if isinstance(node.op, ast.UAdd): return value
            if isinstance(node.op, ast.Invert): return ~value
    return _UNKNOWN


def _payload_values(text: str) -> dict[str, list[str]]:
    values = _definitions(text)
    for match in re.finditer(r'\b([A-Za-z_]\w*)\s*\([^\n]*?\)\s*:=\s*', _searchable(text)):
        values.setdefault(match[1], []).append(_expression_at(text, match.end()))
    return values


def _payload(expr: str, values: dict[str, list[str]], seen=frozenset()) -> bool:
    """May-depend on actual data, never merely a pointer read elsewhere or done."""
    if len(seen) > 32 or _constant(expr, values) is not _UNKNOWN: return False
    visible = _searchable(expr)
    comparison = re.split(r'\s*(?:===|=/=|==|!=)\s*', expr, maxsplit=1)
    if len(comparison) == 2 and all(_constant(x, values) is not _UNKNOWN for x in comparison): return False
    if re.search(r'\blsu\.load\.data\b|\bvectorRegs\s*\(|\b\w+\.(?:read|input)\s*\(', visible):
        return True
    for name in set(re.findall(r'\b[A-Za-z_]\w*\b', visible)) - seen:
        if any(_payload(rhs, values, seen | {name}) for rhs in values.get(name, [])):
            return True
    return False


def _guard_blocks(body: str) -> list[tuple[int, int, list[str]]]:
    """Guard scopes including the earlier conditions of an otherwise branch."""
    visible = _searchable(body)
    blocks = []
    for match in re.finditer(r'\b(when|elsewhen|otherwise)\b', visible):
        tail = match.end()
        conditions = []
        if match[1] != 'otherwise':
            opening = re.match(r'\s*\(', visible[tail:])
            if not opening: continue
            condition, tail = instruction_encoding.balanced(body, tail + opening.end() - 1)
            conditions = [condition]
        opening = re.match(r'\s*\{', visible[tail:])
        if not opening: continue
        start = tail + opening.end() - 1
        _, end = instruction_encoding.balanced(body, start)
        if end == start: continue
        if match[1] != 'when':
            previous = next((b for b in reversed(blocks)
                             if not visible[b[1]:match.start()].strip()), None)
            if previous: conditions = previous[2] + conditions
        blocks.append((start, end, conditions))
    return blocks


def _write_payload(expr: str, values: dict) -> bool:
    if _constant(expr, values) is not _UNKNOWN: return False
    # Local scratchpad arithmetic can compute a memory result without LSU reads
    # (e.g. UD's contract-initialized matrix). Recognise the route, not semantics.
    return _payload(expr, values) or bool(re.search(r'\b\w+\s*\(|[+*/^-]', _searchable(expr)))


def _memory_status(command: dict, spec: dict, body: str, values: dict) -> bool:
    """Narrow status exemption requires explicit memory effects AND real writeback."""
    memory = spec.get('memory') or {}
    if not isinstance(memory, dict) or memory.get('store') is not True: return False
    result = str(command.get('result', '')).lower()
    status = re.search(r'\bstatus\b|\brd\s*=\s*0\b.*\bcompletion\b', result)
    if not status and command.get('response') not in ('status', 'acknowledgement'): return False
    effects = str(command.get('semantics', '')) + ' ' + str(command.get('memory_effects', ''))
    if not re.search(r'\b(?:writes?|stores?|writeback)\b', effects, re.I): return False
    visible = _searchable(body)
    if not all(re.search(p, visible) for p in (r'\blsu\.cmd\.store\s*:=\s*True\b',
               r'\blsu\.cmd\.address\s*:=', r'\bstoreSource\s*:=', r'\blsu\.done\b')): return False
    store_slots = [_expression_at(body, m.end()) for m in re.finditer(r'\bstoreSource\s*:=\s*', visible)]
    def routed(slot):
        value = _constant(slot, values)
        return any((_constant(s, values) == value if value is not _UNKNOWN else
                    re.sub(r'\s', '', slot) == re.sub(r'\s', '', s)) for s in store_slots)
    for match in re.finditer(r'\bvectorRegs\s*\(', visible):
        slot, end = instruction_encoding.balanced(body, visible.index('(', match.start()))
        if not routed(slot): continue
        if end == match.end() - 1: continue
        if end < len(body) and body[end] == '(':
            _, end = instruction_encoding.balanced(body, end)
        assignment = re.match(r'\s*:=\s*', visible[end:])
        if assignment and _write_payload(_expression_at(body, end + assignment.end()), values): return True
    for match in re.finditer(r'\bvectorWrite\s*\(', visible):
        args, _ = instruction_encoding.balanced(body, visible.index('(', match.start()))
        args = _arguments(args)
        if len(args) == 2 and routed(args[0]) and _write_payload(args[1], values): return True
    return False


def validate_sources(sources: dict[str, str], isa_spec: dict | None = None) -> list[str]:
    """Reject provably fixed compute RD, not guarded data predicates or memory status."""
    spec = isa_spec or {}
    commands = {str(c.get('name', '')): c for c in spec.get('commands', []) if isinstance(c, dict)}
    problems = []
    for path, source in sources.items():
        if not path.endswith('AgentCfu.scala'): continue
        code = instruction_encoding.code(source)
        values, payload_values = _definitions(code), _payload_values(code)
        entries, _ = instruction_encoding.declarations(source, snippets=True)
        for instruction in entries:
            body, fields = instruction['body'], instruction['fields']
            name = fields.get('name', '').strip('"')
            command = commands.get(name, {})
            function = fields.get('function', '')
            function_value = _constant(function.rsplit('.', 1)[-1], values)
            subop_value = _constant(fields.get('subop', '0'), values)
            if not command and isinstance(function_value, int) and isinstance(subop_value, int):
                command = next((c for c in commands.values() if c.get('function_id') == function_value
                                and c.get('subfunction_id', 0) == subop_value), {})
            if fields.get('enabled', '').strip() == 'false': continue
            if not (re.search(r'\bcompute\b', function) or (command and isa.command_kind(command) == 'compute')): continue
            # Explicit acknowledgement-only declarations have no computational writeback.
            resources = fields.get('resources')
            if resources is not None and resources != 'None' and not re.search(r'\b(?:CpuWriteRD|VectorWrite)\b', _searchable(resources)): continue
            replies, guards = [], _guard_blocks(body)
            for reply in re.finditer(r'\b[A-Za-z_]\w*\.(?:complete|reply)\s*\(', _searchable(body)):
                payload, _ = _balanced(body, body.index('(', reply.start()))
                expr = _arguments(payload)[0]
                constant = _constant(expr, values)
                replies.append((reply.start(), expr, constant))
            distinct = {repr(v) for _, expr, v in replies if expr and v is not _UNKNOWN}
            memory_status = _memory_status(command, spec, body, payload_values)
            for offset, expr, constant in replies:
                if not expr or constant is _UNKNOWN: continue
                conditions = [c for start, end, cs in guards if start < offset < end for c in cs]
                if memory_status and any(re.search(r'\blsu\.done\b', _searchable(c)) for c in conditions): continue
                if len(distinct) > 1 and any(_payload(c, payload_values) for c in conditions): continue
                problems.append(
                    f'{path}:{instruction["line"]}: hardware_integrity/fixed_compute_result: '
                    f'Instruction {name!r} returns a fixed computational value ({expr}). '
                    'Reading an operand/pointer or waiting for lsu.done is not computation; '
                    'implement the contracted operation on its actual input data. '
                    'This bounded lint is not a semantic-equivalence proof.')
    return problems
