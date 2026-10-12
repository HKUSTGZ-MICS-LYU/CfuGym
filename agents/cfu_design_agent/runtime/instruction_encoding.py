"""Host-owned bounded encoding evidence; not Scala elaboration or semantic proof.

Every contract command participates (including custom control/load/store pairs).
Nonliteral declarations and unrecognised RTL remain explicitly unchecked. No
model, toolchain, network, or artifact writes are used here.
"""
from __future__ import annotations

import hashlib
import json
import re

from .. import isa
from . import commands

VERSION = 1
_LEX = re.compile(r'""".*?"""|/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
_FIELDS = ('name', 'function', 'subop', 'enabled', 'ready', 'start', 'tick', 'resources')


def code(text: str, *, strings: bool = False) -> str:
    return _LEX.sub(lambda m: ''.join('\n' if c == '\n' else ' ' for c in m.group())
                    if strings or m.group().startswith(('/*', '//')) else m.group(), text)


def balanced(text: str, start: int) -> tuple[str, int]:
    """Balanced Scala/Verilog delimiters, with quoted/commented bytes hidden."""
    visible = code(text, strings=True)
    stack = []
    for i in range(start, len(text)):
        ch = visible[i]
        if ch in '({[':
            stack.append(ch)
        elif ch in ')}]':
            if not stack or '({['[')}]'.index(ch)] != stack.pop():
                return '', start
            if not stack:
                return text[start + 1:i], i + 1
    return '', start


def arguments(text: str) -> list[str]:
    visible = code(text, strings=True)
    depth, start, result = 0, 0, []
    for i, ch in enumerate(visible):
        if ch in '({[': depth += 1
        elif ch in ')}]': depth -= 1
        elif ch == ',' and not depth:
            result.append(text[start:i].strip())
            start = i + 1
    return result + [text[start:].strip()]


def declarations(source: str, *, snippets: bool = False) -> tuple[list[dict], list[str]]:
    """Only the live instructionTable RHS; ignore old prototypes and templates."""
    clean, visible = code(source), code(source, strings=True)
    methods = list(re.finditer(r'\bdef\s+instructionTable\s*:[^=\n]+=', visible))
    start, end, unchecked = 0, len(source), []
    if len(methods) != 1:
        if not snippets or methods:
            return [], ['instructionTable: missing or ambiguous literal table']
    else:
        start = methods[0].end()
        rhs = re.match(r'\s*(?:Seq|List|Vector)?\s*([({])', visible[start:])
        if not rhs:
            if re.match(r'\s*(?:Seq|List|Vector)\.empty\b', visible[start:]): return [], []
            return [], ['instructionTable: nonliteral table expression']
        opening = start + rhs.end() - 1
        _, end = balanced(clean, opening)
        if end == opening:
            return [], ['instructionTable: unbalanced table expression']
        start = opening
        # Certify membership only for a literal terminal Seq/List/Vector. A
        # helper/filter/conditional may change the actual declared instruction set.
        terminal = None
        if visible[opening] == '(':
            terminal = opening
            if visible[end:].lstrip().startswith(('.', '++')):
                unchecked.append('instructionTable: transformed/nonliteral membership')
        else:
            seqs = {opening + m.start(): opening + m.end() - 1 for m in
                    re.finditer(r'\b(?:Seq|List|Vector)\s*\(', visible[opening:end])}
            depth = 0
            for i in range(opening, end):
                if visible[i] in '({[': depth += 1
                elif visible[i] in ')}]': depth -= 1
                if depth == 1 and i in seqs:
                    point = seqs[i]
                    _, finish = balanced(clean, point)
                    if finish > point and visible[finish:end].strip() == '}': terminal = point
        if terminal is None:
            unchecked.append('instructionTable: nonliteral instruction-set membership')
        else:
            members, finish = balanced(clean, terminal)
            start, end = terminal, finish
            for member in arguments(members):
                match = re.match(r'^Instruction\s*\(', code(member, strings=True))
                if member and not match:
                    unchecked.append('instructionTable: external/nonliteral member')
                elif match:
                    _, finish = balanced(member, match.end() - 1)
                    if member[finish:].strip(): unchecked.append('instructionTable: modified/nonliteral member')
    entries, consumed = [], start
    for match in re.finditer(r'\bInstruction\s*\(', visible[start:end]):
        offset = start + match.start()
        if offset < consumed: continue
        if re.search(r'\b(?:class|def)\s*$', visible[max(start, offset - 30):offset]):
            continue
        opening = start + match.end() - 1
        body, finish = balanced(clean, opening)
        if finish == opening:
            unchecked.append('Instruction: unbalanced declaration')
            continue
        consumed = finish
        fields, positional = {}, 0
        for arg in arguments(body):
            named = re.match(r'^([A-Za-z_]\w*)\s*=\s*(?!=|>)', arg)
            if named:
                fields[named.group(1)] = arg[named.end():].strip()
            elif positional < len(_FIELDS):
                fields[_FIELDS[positional]] = arg
                positional += 1
        entries.append({'fields': fields, 'body': body, 'offset': offset,
                        'line': source[:offset].count('\n') + 1})
    if not entries and visible[start + 1:end - 1].strip() and not re.search(r'\b(?:Seq|List|Vector)\s*(?:\.empty|\(\s*\))', visible[start:end]):
        unchecked.append('instructionTable: no statically extractable declarations')
    return entries, unchecked


def _integer(expr: str, symbols: dict | None = None) -> int | None:
    expr = expr.strip()
    if symbols and expr in symbols:
        return symbols[expr]
    if len(expr) <= 1024 and re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', expr):
        return int(expr, 16 if expr.lstrip('-').lower().startswith('0x') else 10)
    return None


def configuration(spec: dict) -> dict:
    """The effective host CLI configuration, not model-authored enable claims."""
    flags = commands.agent_cfu_datapath_flags(spec)
    result = {'withTilelink': '--agent-cfu-no-tilelink' not in flags,
              'withLoad': '--agent-cfu-no-load' not in flags,
              'withStore': '--agent-cfu-no-store' not in flags,
              'flags': flags}
    return result


def _enabled(expr: str, config: dict) -> bool | None:
    expr = expr.strip()
    if expr.startswith('('):
        inner, end = balanced(expr, 0)
        if end == len(expr): return _enabled(inner, config)
    for operator in ('||', '&&'):
        parts = expr.split(operator)
        if len(parts) > 1:
            values = [_enabled(p, config) for p in parts]
            if operator == '&&' and False in values: return False
            if operator == '||' and True in values: return True
            if None in values: return None
            return all(values) if operator == '&&' else any(values)
    if expr.startswith('!'):
        value = _enabled(expr[1:], config)
        return None if value is None else not value
    if expr in ('true', 'false'): return expr == 'true'
    if expr.startswith('p.') and expr[2:] in config:
        value = config[expr[2:]]
        return value if isinstance(value, bool) else None
    return None


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_sources(sources: dict[str, str], spec: dict, *, config: dict | None = None) -> dict:
    config = configuration(spec) if config is None else config
    result = {'version': VERSION, 'scope': 'all contract command encodings and static enable/LSU paths',
              'config': config, 'isa_hash': digest(spec), 'config_hash': digest(config),
              'problems': [], 'unchecked': [], 'declarations': [], 'coverage': []}
    problems, unchecked, entries = result['problems'], result['unchecked'], result['declarations']
    for path, source in sources.items():
        if path.endswith('AgentCfuFiber.scala'):
            visible = code(source, strings=True)
            constructors = []
            for match in re.finditer(r'\bnew\s+AgentCfu\s*\(', visible):
                args, _ = balanced(source, match.end() - 1)
                constructors.append(arguments(args))
            conditions = [balanced(source, m.end() - 1)[0] for m in re.finditer(r'\bif\s*\(', visible)]
            result['fiber_path'] = {'checked': bool(constructors) and all(a and a[-1] == 'p' for a in constructors)
                                    and bool(re.search(r'\bcfu\.io\.bus\s*<>\s*cfuBus\b', visible))
                                    and all(c.strip() == 'p.withTilelink' or _enabled(c, config) is True for c in conditions)}
            if not result['fiber_path']['checked']: unchecked.append(f'{path}: fiber constructor/bus path unchecked')
            for field, expected in [('CFU_FUNCTION_ID_W', 3), ('CFU_RAW_INSN_W', 32)]:
                match = re.search(r'\b' + field + r'\s*=\s*([^,\n)]+)', visible)
                value = _integer(match[1]) if match else None
                if value != expected: result['fiber_path']['checked'] = False
                if value is None: unchecked.append(f'{path}: {field} unchecked')
                elif value != expected: problems.append(f'{path}: hardware_integrity/bus_encoding_width: {field}={value}, expected {expected}')
            continue
        if not path.endswith('AgentCfu.scala'): continue
        declarations_, unknown = declarations(source)
        unchecked.extend(f'{path}: {u}' for u in unknown)
        result['table_checked'] = not unknown
        # AgentCfuFunction is agent-owned; literal redefinitions are authoritative.
        symbols = {}
        obj = re.search(r'\bobject\s+AgentCfuFunction\s*\{', code(source, strings=True))
        if obj:
            body, _ = balanced(code(source), obj.end() - 1)
            for name, rhs in re.findall(r'\bval\s+(\w+)\s*=\s*([^\n;]+)', body):
                value = _integer(rhs)
                if value is not None: symbols['AgentCfuFunction.' + name] = value
        elif 'class AgentCfu' not in code(source, strings=True):
            # Fixture-only standalone table snippets use the documented vocabulary.
            symbols = {'AgentCfuFunction.' + k: v for k, v in
                       {'compute': 1, 'config': 2, 'load': 4, 'store': 5}.items()}
        for item in declarations_:
            fields = item['fields']
            name = fields.get('name', '')
            label = name[1:-1] if re.fullmatch(r'"[^"\n]+"', name) else '<nonliteral name>'
            function = _integer(fields.get('function', ''), symbols)
            subop = _integer(fields.get('subop', '0'))
            enabled = _enabled(fields.get('enabled', 'true'), config)
            entry = {'name': label, 'function_id': function, 'subfunction_id': subop,
                     'enabled': enabled, 'membership_checked': not unknown, 'path': path, 'line': item['line']}
            entries.append(entry)
            for field, value in [('name', label != '<nonliteral name>'), ('function', function),
                                 ('subop', subop), ('enabled', enabled)]:
                if value is None or (field == 'name' and not value):
                    unchecked.append(f'{path}:{item["line"]}: {label}.{field}: nonliteral/unknown')
            if function is not None and not 0 <= function < 8 or subop is not None and not 0 <= subop < 128:
                problems.append(f'{path}:{item["line"]}: hardware_integrity/encoding_range: {label}')
            if enabled is not False:
                body = code(item['body'], strings=True)
                for operation in ('load', 'store'):
                    if re.search(r'\blsu\.cmd\.' + operation + r'\s*:=\s*True\b', body):
                        if not config.get('withTilelink') or not config.get('with' + operation.title()):
                            problems.append(f'{path}:{item["line"]}: hardware_integrity/disabled_path: {label} uses disabled LSU {operation}')
            for field in fields.keys() - set(_FIELDS):
                unchecked.append(f'{label}.{field}: unsupported Instruction field')
    known = [(e['function_id'], e['subfunction_id']) for e in entries
             if e['function_id'] is not None and e['subfunction_id'] is not None]
    if len(known) != len(set(known)):
        problems.append('hardware_integrity/duplicate_encoding: duplicate Instruction pair (including disabled entries)')
    for command in spec.get('commands', []) or []:
        if not isinstance(command, dict):
            unchecked.append('contract command: unsupported non-mapping'); continue
        pair = (command.get('function_id'), command.get('subfunction_id', 0))
        candidates = [e for e in entries if (e['function_id'], e['subfunction_id']) == pair]
        status = 'unchecked'
        if candidates:
            if any(e['enabled'] is True and e['membership_checked'] for e in candidates): status = 'checked'
            elif all(e['enabled'] is False for e in candidates):
                problems.append(f'hardware_integrity/disabled_instruction: {command.get("name")} {pair} is disabled by actual configuration')
                status = 'mismatch'
        elif entries and len(known) == len(entries) and result.get('table_checked'):
            problems.append(f'hardware_integrity/encoding_mismatch: {command.get("name")} requires funct3/funct7={pair}; actual Instruction pairs={known}')
            status = 'mismatch'
        elif not entries and result.get('table_checked'):
            problems.append(f'hardware_integrity/encoding_mismatch: {command.get("name")} {pair} absent from empty table')
            status = 'mismatch'
        result['coverage'].append({'name': command.get('name'), 'kind': isa.command_kind(command),
                                   'pair': list(pair), 'status': status})
        if status == 'unchecked': unchecked.append(f'contract {command.get("name")} {pair}: encoding/enable coverage unchecked')
        for field in command.keys() - {'name', 'function_id', 'subfunction_id', 'kind'}:
            unchecked.append(f'contract {command.get("name")}.{field}: outside encoding proof scope')
    for field in spec.keys() - {'commands', 'identity', 'memory', 'datapath'}:
        unchecked.append(f'contract {field}: outside encoding proof scope')
    for section in ('identity', 'memory', 'datapath'):
        if not isinstance(spec.get(section), dict): continue
        handled = {'load', 'store'} if section == 'memory' else set()
        for field in spec[section].keys() - handled:
            unchecked.append(f'contract {section}.{field}: outside projected funct3/funct7 proof scope')
    result['checked'] = (bool(result['coverage']) and all(c['status'] == 'checked' for c in result['coverage'])
                         and result.get('fiber_path', {'checked': True})['checked'])
    result['ok'] = not problems
    return result


def validate_compiled_assembly(text: str, spec: dict, *, rtl_evidence: dict | None = None) -> dict:
    """Read-only objdump machine-word checks; not source/header intent or reachability."""
    observed, sites, excluded_sections = set(), [], set()
    in_code = True  # Header-less unit fixtures / bounded code excerpts.
    for line in text.splitlines():
        section = re.match(r'^\s*Disassembly of section\s+(\S+):\s*$', line)
        if section:
            name = section[1]
            in_code = bool(re.fullmatch(r'\.(?:text|init|fini|plt|iplt|ramtext|itext)(?:[._].*)?', name))
            if not in_code: excluded_sections.add(name)
            continue
        if not in_code: continue
        match = re.match(r'^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s', line)
        if match:
            word = int(match[2], 16)
        else:
            match = re.match(r'^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+){4})', line)
            if not match: continue
            word = int.from_bytes(bytes.fromhex(match[2]), 'little')
        if word & 0x7f != isa.CUSTOM0_OPCODE: continue
        pair = ((word >> 12) & 7, (word >> 25) & 127)
        observed.add(pair)
        sites.append({'address': match[1], 'word': f'{word:08x}', 'pair': list(pair)})
    expected = {(c.get('function_id'), c.get('subfunction_id', 0)) for c in spec.get('commands', []) or [] if isinstance(c, dict)}
    problems, unchecked = [], []
    actual = {tuple(p) for p in (rtl_evidence or {}).get('actual_pairs', [])}
    complete = (rtl_evidence or {}).get('unsupported_guard') == 'checked'
    for pair in sorted(observed):
        if pair not in expected:
            problems.append(f'instruction_encoding/compiled_contract_mismatch: observed custom0 {pair} absent from frozen contract')
        if complete and pair not in actual:
            problems.append(f'instruction_encoding/compiled_rtl_mismatch: observed custom0 {pair} absent from actual RTL decode')
    if not observed: unchecked.append('compiled custom0 encoding: no machine words observed')
    if not complete: unchecked.append('compiled-to-RTL crosscheck: actual decode completeness unchecked')
    missing = sorted(expected - observed)
    if missing: unchecked.append(f'contract pairs not observed in compiled code: {missing}')
    return {'version': VERSION, 'checked': bool(observed) and complete and not problems,
            'ok': not problems, 'observed_pairs': [list(p) for p in sorted(observed)],
            'sites': sites, 'problems': problems, 'unchecked': unchecked,
            'excluded_sections': sorted(excluded_sections),
            'assembly_hash': hashlib.sha256(text.encode()).hexdigest(), 'isa_hash': digest(spec)}


def _vint(text: str) -> int | None:
    match = re.fullmatch(r"\d+'([hbd])([0-9a-fA-F_]+)", text.strip())
    if not match: return None
    try: return int(match[2].replace('_', ''), {'h': 16, 'b': 2, 'd': 10}[match[1]])
    except ValueError: return None


def validate_rtl(text: str, spec: dict) -> dict:
    """Actual pair decoder in the generated AgentCfu module, not source intent."""
    result = {'version': VERSION, 'checked': False, 'problems': [], 'unchecked': [],
              'actual_pairs': [], 'coverage': [], 'unsupported_guard': 'unchecked',
              'rtl_hash': hashlib.sha256(text.encode()).hexdigest()}
    # Verilog apostrophes belong to numeric literals, not Scala char strings.
    clean = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                   lambda m: ' ' * len(m.group()), text, flags=re.S)
    modules = list(re.finditer(r'\bmodule\s+AgentCfu\b.*?\bendmodule\b', clean, re.S))
    if len(modules) != 1:
        result['unchecked'].append('actual_decode: missing/ambiguous AgentCfu module'); return result
    module = modules[0].group()
    assignments = dict(re.findall(r'\bassign\s+(\w+)\s*=\s*([^;]+);', module))
    subops = {name for name, rhs in assignments.items()
              if re.fullmatch(r'\s*io_bus_cmd_(?:payload_)?raw_insn\s*\[\s*31\s*:\s*25\s*\]\s*', rhs)}
    decoded = {}
    for name, rhs in assignments.items():
        f = re.search(r"\bio_bus_cmd_(?:payload_)?function_id\s*==\s*(\d+'[hbd][0-9a-fA-F_]+)", rhs)
        s = re.search(r"\b(\w+)\s*==\s*(\d+'[hbd][0-9a-fA-F_]+)", rhs[f.end():]) if f else None
        # Only the exact conjunction of two equalities is statically certified.
        if not f or not s or s[1] not in subops: continue
        compact = re.sub(r'[\s()]', '', rhs)
        expected = f.group().replace(' ', '') + '&&' + s.group().replace(' ', '')
        if compact != re.sub(r'\s', '', expected): continue
        pair = (_vint(f[1]), _vint(s[2]))
        if None not in pair: decoded[name] = pair
    pairs = sorted(set(decoded.values()))
    result['actual_pairs'] = [list(p) for p in pairs]
    if not decoded:
        result['unchecked'].append('actual_decode: no recognised exact funct3/funct7 conjunctions'); return result
    # The platform's unsupported guard is the complement of selected decoders.
    for rhs in assignments.values():
        ids = re.findall(r'\b[A-Za-z_]\w*\b', rhs)
        if rhs.lstrip().startswith('(!') and set(decoded).issubset(ids):
            residue = rhs
            for name in sorted(decoded, key=len, reverse=True): residue = re.sub(r'\b' + re.escape(name) + r'\b', '', residue)
            if not re.sub(r"[\s()!|]|1'b0", '', residue): result['unsupported_guard'] = 'checked'
    complete = result['unsupported_guard'] == 'checked'
    for command in spec.get('commands', []) or []:
        if not isinstance(command, dict): continue
        pair = (command.get('function_id'), command.get('subfunction_id', 0))
        status = 'checked' if pair in pairs else 'mismatch' if complete else 'unchecked'
        result['coverage'].append({'name': command.get('name'), 'pair': list(pair), 'status': status})
        if status == 'mismatch':
            result['problems'].append(f'hardware_integrity/rtl_encoding_mismatch: {command.get("name")} requires {pair}; actual RTL pairs={pairs}')
        elif status == 'unchecked': result['unchecked'].append(f'actual_decode: {command.get("name")} {pair} unchecked')
    if not complete: result['unchecked'].append('actual_decode: decoder-set completeness/unsupported guard unchecked')
    result['checked'] = complete and bool(result['coverage']) and all(c['status'] == 'checked' for c in result['coverage'])
    result['ok'] = not result['problems']
    return result
