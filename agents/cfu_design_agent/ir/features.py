"""Static LLVM IR feature extraction: data movement, typed ops, dependencies.

Combines LLVM 14 ``opt`` analysis prints (LoopInfo, ScalarEvolution,
DependenceAnalysis) with a bounded deterministic text parse of the same module.
Everything produced is static evidence: no measured cycles and no runtime byte
counts are derived; unprovable values stay the string ``"unknown"`` (never 0).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
import xml.etree.ElementTree as ET

SCHEMA_VERSION = "1"
MAX_OUTPUT_BYTES = 200_000
MAX_EDGES_PER_LOOP = 120
MAX_OPS_PER_LOOP = 60
MAX_ACCESS_GROUPS = 40
MAX_MEM_EDGES = 60
TIMEOUT_SECONDS = 15
MAX_CAPTURE_BYTES = 2_000_000
_TRUSTED_OPT_ROOTS = ("/usr/bin", "/usr/local/bin", "/usr/lib/llvm-14/bin")
_TOOL_NOTE = "static_ir_text_parse+opt_analysis; not LLVM LoopInfo bindings"
_LIMITATIONS = (
    "Static IR evidence only: no measured cycles, no runtime byte counts. "
    "Trip counts are known only when ScalarEvolution reports a constant "
    "backedge-taken count. Loop bodies are inclusive of nested loops. "
    "Memory aliasing is known only for same base object or DependenceAnalysis "
    "results. Text parsing (not LLVM C++ APIs) counts opcodes and types."
)

_DEFINE_RE = re.compile(r'^define\b.*?@(?:"([^"]+)"|([\w.$-]+))\s*\(')
_BLOCK_RE = re.compile(r'^(?:"([^"]+)"|([\w.$-]+)):\s*(?:;.*)?$')
_ASSIGN_RE = re.compile(r'^(%[\w.$-]+|%"[^"]*")\s*=\s*(.*)$')
_META_STRIP = re.compile(r',?\s*![\w.$-]*\s+!\d+')
_ALIGN_STRIP = re.compile(r',?\s*align\s+\d+')
_ATTRGRP_STRIP = re.compile(r'\s+#[\w.]+')
_DBG_RE = re.compile(r'!dbg\s+!(\d+)')
_LOOPMETA_RE = re.compile(r'!llvm\.loop\s+!\d+')
_DIDEF_RE = re.compile(r'^\s*!(\d+)\s*=\s*(?:distinct\s+)?!DILocation\(line:\s*(\d+)')
_VALUE_RE = re.compile(r'(%[\w.$-]+|%"[^"]*")')
_GLOBAL_RE = re.compile(r'(@[\w.$-]+|@"[^"]*")')
_CONST_RE = re.compile(r'^(?:-?\d+$|-?\d+\.\d+(?:[eE][-+]?\d+)?$|true$|false$|null$|undef$|poison$|zeroinitializer$)')
_SCEV_SECTION_RE = re.compile(r"Printing analysis 'Scalar Evolution Analysis' for function '([^']+)':")
_SCEV_LOOP_RE = re.compile(r'^Loop %([\w.$-]+): (backedge-taken count|max backedge-taken count|Trip multiple|Predicated backedge-taken count) is (.*)$')
_SCEV_VALUE_RE = re.compile(r'^\s+(%[\w.$-]+|%"[^"]*") = ')
_SCEV_ARROW_RE = re.compile(r'^\s+-->\s+(.*?)\s+U:')
_ADDREC_RE = re.compile(r'\{[^{}]*,\+,(-?\d+)\}')
_LOOPINFO_RE = re.compile(r'^(\s*)Loop at depth (\d+) containing: (.*)$')
_LOOP_BLOCK_RE = re.compile(r'%([\w.$-]+)((?:<[^>]+>)*)')
_DA_SECTION_RE = re.compile(r"^'Dependence Analysis' for function '([^']+)':")
_DA_PAIR_RE = re.compile(r'^Src:\s+(.*?)\s+--> Dst:\s+(.*?)\s*$')
_DA_RESULT_RE = re.compile(r'^\s*da analyze - (.*?)!\s*$')

_BINOPS = {
    "add": "integer_add", "sub": "integer_sub", "mul": "integer_mul",
    "udiv": "integer_div", "sdiv": "integer_div", "urem": "integer_rem",
    "srem": "integer_rem", "and": "bitwise_and", "or": "bitwise_or",
    "xor": "bitwise_xor", "shl": "shift_left", "lshr": "shift_right_logic",
    "ashr": "shift_right_arith", "fadd": "fadd", "fsub": "fsub",
    "fmul": "fmul", "fdiv": "fdiv", "frem": "frem",
}
_CASTS = {
    "trunc": "cast_trunc", "zext": "cast_zext", "sext": "cast_sext",
    "fptrunc": "cast_fp_trunc", "fpext": "cast_fp_ext",
    "fptoui": "cast_fp_to_int", "fptosi": "cast_fp_to_int",
    "uitofp": "cast_int_to_fp", "sitofp": "cast_int_to_fp",
    "ptrtoint": "cast_ptr_to_int", "inttoptr": "cast_int_to_ptr",
    "bitcast": "cast_bitcast", "addrspacecast": "cast_bitcast",
}
_OTHER_OPS = {
    "alloca": "other", "atomicrmw": "atomic_rmw", "cmpxchg": "atomic_rmw",
    "fence": "other", "va_arg": "other", "landingpad": "other",
    "catchpad": "other", "cleanuppad": "other", "freeze": "other",
    "ret": "other", "br": "other", "switch": "other", "unreachable": "other",
}
_BIN_FLAGS = {"nuw", "nsw", "exact", "fast", "disjoint"}
_UNSUPPORTED_CONSTRUCTS = ("invoke", "indirectbr", "callbr", "catchswitch",
                           "catchret", "cleanupret", "resume")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _discover_opt(opt_binary: str | None = None) -> str | None:
    """Accept only trusted system opt; never a workspace or PATH hijack."""
    if opt_binary == "":
        return None
    candidates = [opt_binary] if opt_binary else ["opt", "opt-14"]
    for name in candidates:
        if not name:
            continue
        found = name if Path(name).is_absolute() else shutil.which(name)
        if not found:
            continue
        real = Path(found).resolve()
        if (real.is_file() and os.access(real, os.X_OK)
                and any(real.is_relative_to(root) for root in _TRUSTED_OPT_ROOTS)):
            return str(real)
    return None


def _execute(argv: list[str], *, limit: int = MAX_CAPTURE_BYTES) -> dict[str, Any]:
    """Run a bounded command through OS temp files; capture, never stream."""
    result: dict[str, Any] = {"argv": argv, "returncode": None, "stderr": ""}
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                done = subprocess.run(
                    argv, cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, timeout=TIMEOUT_SECONDS, check=False,
                )
                result["returncode"] = done.returncode
            except subprocess.TimeoutExpired:
                result["stderr"] = f"timed out after {TIMEOUT_SECONDS}s"
            stdout.seek(0)
            data = stdout.read(limit + 1)
            stderr.seek(0)
            err = stderr.read(65537)
            if err:
                result["stderr"] += err[:65536].decode("utf-8", "replace")
            if len(data) > limit:
                result["overflow"] = True
            else:
                result["stdout"] = data
    except OSError as exc:
        result["stderr"] = f"{type(exc).__name__}: {exc}"
    return result


def _run_opt_passes(ll_text: str, passes: str, opt: str,
                    runner: Callable[..., dict[str, Any]] | None = None) -> tuple[str, str]:
    """Run one opt pass print on a temp module; returns (combined_output, note)."""
    run = runner or _execute
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False, encoding="utf-8") as handle:
        handle.write(ll_text)
        path = handle.name
    try:
        argv = [opt, f"-passes={passes}", "-disable-output", path]
        result = run(argv, limit=MAX_CAPTURE_BYTES)
        if result.get("returncode") != 0 or "stdout" not in result:
            return "", f"{passes} failed: {result.get('stderr', '')[:200]}"
        out = (result["stdout"].decode("utf-8", "replace")
               + "\n" + result.get("stderr", ""))
        return out, ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------- text parsing

def _strip_metadata(text: str) -> str:
    text = _META_STRIP.sub("", text)
    text = _ALIGN_STRIP.sub("", text)
    text = _ATTRGRP_STRIP.sub("", text)
    return text.strip()


def _take_type(text: str) -> tuple[str, str]:
    """Split the leading type token (handles <N x T>, [N x T], { ... }) off."""
    text = text.lstrip()
    if text.startswith("<"):
        depth = 0
        for index, char in enumerate(text):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    return text[:index + 1].strip(), text[index + 1:].lstrip()
        return text, ""
    if text.startswith("["):
        depth = 0
        for index, char in enumerate(text):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[:index + 1].strip(), text[index + 1:].lstrip()
        return text, ""
    if text.startswith("{"):
        depth = 0
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[:index + 1].strip(), text[index + 1:].lstrip()
        return text, ""
    match = re.match(r'[^,\s]+', text)
    if not match:
        return "", ""
    return match.group(0), text[match.end():].lstrip()


def _type_info(type_str: str) -> dict[str, Any]:
    """Return {'kind','bits','lanes'}; bits/lanes are int or 'unknown'."""
    t = type_str.strip()
    if not t:
        return {"kind": "unknown", "bits": "unknown", "lanes": "unknown"}
    vec = re.fullmatch(r'<(\d+)\s+x\s+(.+)>', t)
    if vec:
        inner = _type_info(vec.group(2))
        return {"kind": "vector", "bits": inner["bits"], "lanes": int(vec.group(1)),
                "elem": inner}
    scalar = re.fullmatch(r'i(\d+)', t)
    if scalar:
        return {"kind": "int", "bits": int(scalar.group(1)), "lanes": 1}
    floats = {"half": 16, "bfloat": 16, "float": 32, "double": 64, "fp128": 128, "x86_fp80": 80}
    if t in floats:
        return {"kind": "float", "bits": floats[t], "lanes": 1}
    if t == "void":
        return {"kind": "void", "bits": "unknown", "lanes": 1}
    if t.endswith("*") or "addrspace(" in t:
        return {"kind": "pointer", "bits": 32, "lanes": 1}
    return {"kind": "aggregate", "bits": "unknown", "lanes": 1}


def _elem_bytes(type_str: str) -> int | None:
    info = _type_info(type_str)
    bits, lanes = info["bits"], info["lanes"]
    if type(bits) is int and type(lanes) is int:
        return bits * lanes // 8
    return None


def _parse_instruction(raw: str) -> dict[str, Any]:
    text = raw.strip()
    rec: dict[str, Any] = {
        "raw": raw, "name": "", "opcode": "other", "class": "other",
        "type": "", "bits": "unknown", "lanes": "unknown",
        "operands": [], "constants": 0, "dbg": None,
        "has_loop_meta": bool(_LOOPMETA_RE.search(text)),
    }
    dbg = _DBG_RE.search(text)
    rec["dbg"] = dbg.group(1) if dbg else None
    cleaned = _strip_metadata(text)
    name = ""
    rest = cleaned
    match = _ASSIGN_RE.match(cleaned)
    if match:
        name, rest = match.group(1), match.group(2).strip()
    rec["name"] = name
    words = rest.split()
    if not words:
        return rec
    opcode = words[0]
    cursor = 1
    if opcode in ("tail", "musttail", "notail") and len(words) > 1 and words[1] == "call":
        opcode, cursor = "call", 2
    elif opcode == "getelementptr" and len(words) > 1 and words[1] == "inbounds":
        cursor = 2
    elif opcode in ("load", "store") and len(words) > 1 and words[1] in ("volatile", "atomic", "weak"):
        cursor = 2
    rec["opcode"] = opcode
    operand_text = rest
    if opcode in _BINOPS:
        rec["class"] = _BINOPS[opcode]
        tail = " ".join(words[cursor:])
        while tail.split() and tail.split()[0] in _BIN_FLAGS:
            tail = tail.split(" ", 1)[1]
        type_str, operand_text = _take_type(tail)
    elif opcode in ("icmp", "fcmp"):
        rec["class"] = "compare_icmp" if opcode == "icmp" else "compare_fcmp"
        predicate = words[cursor] if cursor < len(words) else ""
        rec["predicate"] = predicate
        type_str, operand_text = _take_type(" ".join(words[cursor + 1:]))
    elif opcode == "phi":
        rec["class"] = "phi"
        type_str, operand_text = _take_type(" ".join(words[cursor:]))
    elif opcode == "select":
        rec["class"] = "select"
        # select i1 %c, <ty> %a, <ty> %b -> value type is the second type.
        type_str, after = _take_type(" ".join(words[cursor:]))
        type_str, operand_text = _take_type(after.split(",", 1)[-1] if "," in after else after)
    elif opcode in _CASTS:
        rec["class"] = _CASTS[opcode]
        type_str, operand_text = _take_type(" ".join(words[cursor:]))
    elif opcode == "load":
        rec["class"] = "load"
        type_str, operand_text = _take_type(" ".join(words[cursor:]))
    elif opcode == "store":
        rec["class"] = "store"
        type_str, operand_text = _take_type(" ".join(words[cursor:]))
    elif opcode == "getelementptr":
        rec["class"] = "gep"
        type_str, operand_text = _take_type(" ".join(words[cursor:]))
        rec["gep_source_type"] = type_str
    elif opcode == "call":
        rec["class"] = "call"
        tail = " ".join(words[cursor:])
        # strip call-site attributes before the return type.
        while tail.split() and tail.split()[0] in {
            "ccc", "fastcc", "coldcc", "cc", "signext", "zeroext", "noundef",
            "nofree", "nounwind", "readonly", "writeonly", "speculatable",
            "willreturn", "memory", "immarg",
        }:
            tail = tail.split(" ", 1)[1]
        type_str, operand_text = _take_type(tail)
    else:
        rec["class"] = _OTHER_OPS.get(opcode, "other")
        type_str, operand_text = ("", rest)
        if opcode in ("ret", "br", "switch", "unreachable") or opcode in _UNSUPPORTED_CONSTRUCTS:
            rec["unsupported"] = opcode if opcode in _UNSUPPORTED_CONSTRUCTS else None
    rec["type"] = type_str
    info = _type_info(type_str) if type_str else {"bits": "unknown", "lanes": "unknown"}
    rec["bits"] = info["bits"]
    rec["lanes"] = info["lanes"]
    rec["operands"] = re.findall(r'%(?:[\w.$-]+|"[^"]*")|@(?:[\w.$-]+|@"[^"]*")', operand_text)
    rec["constants"] = sum(1 for tok in re.split(r'[,\s]+', operand_text) if _CONST_RE.match(tok))
    if opcode == "getelementptr":
        rec["gep_indices"] = [tok.strip() for tok in operand_text.split(",")[1:]]
    return rec


def _parse_module(ll_text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse functions and the metadata line map; raise ValueError if unusable."""
    if "define" not in ll_text:
        raise ValueError("IR text contains no function definitions")
    lines = ll_text.splitlines()
    di_lines: dict[str, int] = {}
    for line in lines:
        match = _DIDEF_RE.match(line)
        if match:
            di_lines[match.group(1)] = int(match.group(2))
    functions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    block: dict[str, Any] | None = None
    for raw in lines:
        if raw.startswith("define"):
            match = _DEFINE_RE.match(raw)
            if not match:
                continue
            name = match.group(1) or match.group(2)
            args = _VALUE_RE.findall(raw.split("(", 1)[1].rsplit(")", 1)[0])
            current = {"name": name, "args": ["%" + a if not a.startswith("%") else a
                                              for a in args], "blocks": [],
                       "unsupported": []}
            functions.append(current)
            block = None
            continue
        if current is None:
            continue
        if raw.startswith("}"):
            current, block = None, None
            continue
        if raw and not raw[0].isspace():
            label = _BLOCK_RE.match(raw)
            if label:
                block = {"label": label.group(1) or label.group(2), "instructions": []}
                current["blocks"].append(block)
                continue
        if raw.strip():
            if block is None:
                block = {"label": "entry", "instructions": []}
                current["blocks"].append(block)
            inst = _parse_instruction(raw)
            if inst.get("unsupported"):
                current["unsupported"].append(inst["unsupported"])
            block["instructions"].append(inst)
    if not functions:
        raise ValueError("IR text contains no parseable functions")
    return functions, di_lines


def _build_cfg(function: dict[str, Any]) -> dict[str, list[str]]:
    succ: dict[str, list[str]] = {}
    for block in function["blocks"]:
        label = block["label"]
        tail = block["instructions"][-1] if block["instructions"] else None
        targets: list[str] = []
        if tail:
            raw = tail["raw"]
            if re.search(r'\bbr\b', raw) and "label" in raw:
                targets = re.findall(r'label\s+%([\w.$-]+|"[^"]*")', raw)
            elif re.search(r'\bswitch\b', raw):
                targets = re.findall(r'label\s+%([\w.$-]+|"[^"]*")', raw)
        succ[label] = targets
    return succ


def _natural_loops_fallback(function: dict[str, Any]) -> list[dict[str, Any]]:
    """Dominator-based natural loops when opt is unavailable."""
    succ = _build_cfg(function)
    labels = [block["label"] for block in function["blocks"]]
    if not labels:
        return []
    entry = labels[0]
    preds: dict[str, list[str]] = {label: [] for label in labels}
    for src, dsts in succ.items():
        for dst in dsts:
            if dst in preds:
                preds[dst].append(src)
    dom: dict[str, set[str]] = {label: set(labels) for label in labels}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for label in labels:
            if label == entry:
                continue
            incoming = [dom[p] for p in preds[label] if p in dom]
            new = set.intersection(*incoming) | {label} if incoming else {label}
            if new != dom[label]:
                dom[label] = new
                changed = True
    loops: list[dict[str, Any]] = []
    for src, dsts in succ.items():
        for dst in dsts:
            if dst not in dom.get(src, ()):
                continue
            body = {dst}
            stack = [src]
            while stack:
                node = stack.pop()
                if node in body:
                    continue
                body.add(node)
                stack.extend(p for p in preds.get(node, ()) if p not in body)
            loops.append({"header": dst, "blocks": sorted(body), "depth": 0,
                          "llvm_loop_meta": False, "source": "text_fallback"})
    # depth via strict body containment
    for loop in loops:
        loop["depth"] = 1 + sum(
            1 for other in loops
            if other is not loop and set(loop["blocks"]) < set(other["blocks"])
        )
    return loops


# ------------------------------------------------------------------ opt parsing

def _parse_loopinfo(text: str) -> list[dict[str, Any]]:
    loops: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _LOOPINFO_RE.match(line)
        if not match:
            continue
        blocks: list[str] = []
        header = ""
        for block_match in _LOOP_BLOCK_RE.finditer(match.group(3)):
            name = block_match.group(1)
            tags = block_match.group(2)
            blocks.append(name)
            if "<header>" in tags and not header:
                header = name
        if blocks:
            loops.append({"header": header or blocks[0], "blocks": blocks,
                          "depth": int(match.group(2)), "llvm_loop_meta": False,
                          "source": "opt_loopinfo"})
    return loops


def _parse_scev(text: str) -> dict[str, dict[str, Any]]:
    per_function: dict[str, dict[str, Any]] = {}
    current_fn = ""
    pending_value = ""
    for line in text.splitlines():
        section = _SCEV_SECTION_RE.match(line)
        if section:
            current_fn = section.group(1)
            per_function.setdefault(current_fn, {"loops": {}, "values": {}})
            continue
        if not current_fn:
            continue
        loop_match = _SCEV_LOOP_RE.match(line)
        if loop_match:
            entry = per_function[current_fn]["loops"].setdefault(loop_match.group(1), {})
            key = {"backedge-taken count": "backedge", "max backedge-taken count": "max_backedge",
                   "Predicated backedge-taken count": "predicated_backedge",
                   "Trip multiple": "trip_multiple"}[loop_match.group(2)]
            entry[key] = loop_match.group(3).strip()
            continue
        value = _SCEV_VALUE_RE.match(line)
        if value:
            pending_value = value.group(1)
            continue
        arrow = _SCEV_ARROW_RE.match(line)
        if arrow and pending_value:
            per_function[current_fn]["values"][pending_value] = arrow.group(1)
            pending_value = ""
    return per_function


def _parse_da(text: str) -> dict[str, list[tuple[str, str, str]]]:
    per_function: dict[str, list[tuple[str, str, str]]] = {}
    current_fn = ""
    pending: tuple[str, str] | None = None
    for line in text.splitlines():
        section = _DA_SECTION_RE.match(line)
        if section:
            current_fn = section.group(1)
            per_function.setdefault(current_fn, [])
            continue
        pair = _DA_PAIR_RE.match(line)
        if pair:
            pending = (pair.group(1).strip(), pair.group(2).strip())
            continue
        result = _DA_RESULT_RE.match(line)
        if result and pending and current_fn:
            per_function[current_fn].append((pending[0], pending[1], result.group(1).strip()))
            pending = None
    return per_function


def _norm_inst(text: str) -> str:
    return re.sub(r'\s+', ' ', _strip_metadata(text)).strip()


# ------------------------------------------------------------------- features

def _base_object(ptr: str, defs: dict[str, dict[str, Any]],
                 args: set[str], depth: int = 0) -> tuple[str, str]:
    if ptr.startswith("@"):
        return "global", ptr
    if ptr in args:
        return "argument", ptr
    if depth > 8 or ptr not in defs:
        return "unknown", ptr or "unknown"
    inst = defs[ptr]
    if inst["opcode"] == "alloca":
        return "alloca", ptr
    if inst["opcode"] == "getelementptr":
        operands = [o for o in inst["operands"] if o.startswith(("%", "@"))]
        return _base_object(operands[0], defs, args, depth + 1) if operands else ("unknown", ptr)
    if inst["opcode"] == "bitcast" or inst["class"].startswith("cast"):
        operands = [o for o in inst["operands"] if o.startswith(("%", "@"))]
        return _base_object(operands[0], defs, args, depth + 1) if operands else ("unknown", ptr)
    return "unknown", ptr


_SCOPED_ADDREC_RE = re.compile(r'\{[^{}]*,\+,(-?\d+)\}(?:<[a-z]+>)*<%([\w.$-]+)>')


def _scev_stride(index_value: str, function_scev: dict[str, Any],
                 elem_bytes: int | None, header: str = "") -> int | None:
    """Constant byte stride: pointer-expression addrec for this loop, else
    index addrec scaled by element size."""
    expr = function_scev.get("values", {}).get(index_value, "")
    if header:
        for step, scope in _SCOPED_ADDREC_RE.findall(expr):
            if scope == header:
                return abs(int(step))
        scoped = re.findall(r'\{[^{}]*,\+,' + r'(-?\d+)\}<%([\w.$-]+)>', expr)
        for step, scope in scoped:
            if scope == header:
                return abs(int(step))
    steps = [int(step) for step in _ADDREC_RE.findall(expr)]
    if steps and elem_bytes:
        return abs(steps[-1]) * elem_bytes
    return None


def _induction_stride(index_value: str, defs: dict[str, dict[str, Any]],
                      elem_bytes: int | None) -> int | None:
    """Phi with a constant-add update gives a constant step in elements."""
    inst = defs.get(index_value)
    if not inst or inst["opcode"] != "phi" or not elem_bytes:
        return None
    for incoming in _VALUE_RE.findall(inst["raw"]):
        updater = defs.get(incoming)
        if updater and updater["opcode"] == "add":
            constants = [int(t) for t in re.split(r'[,\s]+', _strip_metadata(updater["raw"]))
                         if re.fullmatch(r'-?\d+', t)]
            if len(constants) == 1:
                return constants[0] * elem_bytes
    return None


def _classify_recurrence(phi: dict[str, Any], defs: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Classify the update feeding a loop-carried phi."""
    incoming = [o for o in _VALUE_RE.findall(phi["raw"]) if o in defs]
    for value in incoming:
        updater = defs[value]
        if updater["opcode"] not in _BINOPS:
            continue
        operands = updater["operands"]
        if phi["name"] not in operands:
            continue
        others = [o for o in operands if o != phi["name"]]
        constants = [t for t in re.split(r'[,\s]+', _strip_metadata(updater["raw"]))
                     if _CONST_RE.match(t)]
        op = updater["opcode"]
        if op == "add":
            return ("induction" if constants else "add_reduction"), op
        if op == "xor":
            return "xor_reduction", op
        if op == "or":
            return "or_reduction", op
        if op == "and":
            return "and_reduction", op
        if op == "mul":
            return "mul_reduction", op
        if others:
            return "unknown_chain", op
    return "unknown_chain", "none"


def _operation_groups(body_insts: list[dict[str, Any]]) -> dict[str, Any]:
    """Typed operation counts and call targets for one region body."""
    op_groups: dict[tuple, dict[str, Any]] = {}
    calls: dict[str, dict[str, Any]] = {}
    for inst in body_insts:
        if inst["class"] == "call":
            callee_match = re.search(r'@("[^"]*"|[\w.$-]+)', inst["raw"])
            callee = callee_match.group(1).strip('"') if callee_match else "unknown"
            intrinsic = callee.startswith("llvm.")
            entry = calls.setdefault(callee, {"callee": callee, "count": 0,
                                              "intrinsic": intrinsic})
            entry["count"] += 1
            continue
        key = (inst["class"], inst.get("opcode", ""), str(inst["type"]),
               str(inst["bits"]), str(inst["lanes"]))
        group = op_groups.setdefault(key, {
            "class": inst["class"], "opcode": inst.get("opcode", ""),
            "type": str(inst["type"]) or "unknown", "bits": inst["bits"],
            "lanes": inst["lanes"], "count": 0, "constant_operands": 0,
        })
        group["count"] += 1
        group["constant_operands"] += inst["constants"]
        if inst["class"] in ("compare_icmp", "compare_fcmp"):
            group["predicate"] = inst.get("predicate", "")
    return {
        "ops": sorted(op_groups.values(),
                      key=lambda g: (g["class"], str(g["type"]), g.get("opcode", ""))),
        "calls": sorted(calls.values(), key=lambda c: c["callee"]),
    }


def _loop_source_line(body_instructions: list[dict[str, Any]],
                      di_lines: dict[str, int]) -> tuple[int | None, str]:
    lines = [di_lines[str(inst["dbg"])] for inst in body_instructions
             if inst.get("dbg") and str(inst["dbg"]) in di_lines and di_lines[str(inst["dbg"])] > 0]
    if lines:
        return min(lines), "debug_location"
    return None, "unmapped"


def extract_ir_features(
    ll_text: str, *, source_file: str = "", function: str = "",
    hotspot_line: int | None = None, opt_binary: str | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (features, audit) for one textual IR module; pure besides opt.

    ``opt_binary=None`` discovers a trusted opt; ``opt_binary=""`` forces the
    text-only fallback. ``runner`` overrides subprocess execution for tests.
    """
    functions, di_lines = _parse_module(ll_text)
    audit: dict[str, Any] = {
        "input_sha256": _digest(ll_text.encode()),
        "commands": [], "warnings": [], "analysis_source": "text_only",
    }
    opt = _discover_opt(opt_binary)
    loopinfo: list[dict[str, Any]] = []
    scev: dict[str, dict[str, Any]] = {}
    da: dict[str, list[tuple[str, str, str]]] = {}
    if opt:
        run = runner or _execute
        version = run([opt, "--version"], limit=4096)
        audit["commands"].append({k: v for k, v in version.items() if k != "stdout"})
        audit["opt_version"] = (version.get("stdout", b"").decode("utf-8", "replace").splitlines()
                                or ["unknown"])[0]
        for passes in ("print<loops>,print<scalar-evolution>", "print<da>"):
            output, note = _run_opt_passes(ll_text, passes, opt, runner)
            audit["warnings"].extend([note] if note else [])
            if passes.startswith("print<loops>"):
                loopinfo = _parse_loopinfo(output)
                scev = _parse_scev(output)
            else:
                da = _parse_da(output)
        audit["analysis_source"] = "llvm14_opt+text" if loopinfo else "text_only"
    features: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "source_file": source_file,
        "analysis_source": audit["analysis_source"], "tool_note": _TOOL_NOTE,
        "limitations": _LIMITATIONS, "functions": [],
    }
    blocks_to_function: dict[str, str] = {}
    for func in functions:
        for block in func["blocks"]:
            blocks_to_function.setdefault(f"{func['name']}:{block['label']}", func["name"])
    for func in functions:
        name = func["name"]
        if function and name != function:
            continue
        blocks = func["blocks"]
        body_index = {block["label"]: index for index, block in enumerate(blocks)}
        defs = {inst["name"]: inst for block in blocks for inst in block["instructions"]
                if inst["name"]}
        all_insts = [inst for block in blocks for inst in block["instructions"]]
        loops = [dict(loop) for loop in loopinfo]
        scev_loops = scev.get(name, {}).get("loops", {})
        # Attribute opt loops to this function: SCEV section first (header id
        # association), then block membership (all-but-one blocks in function).
        owned: list[dict[str, Any]] = []
        for loop in loops:
            known = [b for b in loop["blocks"] if b in body_index]
            header_known = loop["header"] in body_index
            if loop["header"] in scev_loops and header_known:
                loop["blocks"] = sorted(known, key=lambda b: body_index[b])
                owned.append(loop)
            elif not scev_loops and header_known and known and len(known) >= max(1, len(loop["blocks"]) - 1):
                loop["blocks"] = sorted(known, key=lambda b: body_index[b])
                owned.append(loop)
        if not owned and not loopinfo:
            owned = _natural_loops_fallback(func)
        for loop in owned:
            header = loop["header"]
            raw_body = [b for b in loop["blocks"] if b in body_index]
            body_insts = [inst for b in raw_body for inst in
                          next(blk for blk in blocks if blk["label"] == b)["instructions"]]
            loop_defs = {inst["name"]: inst for inst in body_insts if inst["name"]}
            # Backedge metadata presence from the header's predecessors in body.
            loop["llvm_loop_meta"] = any(inst["has_loop_meta"] for inst in body_insts)
            source_line, source_link = _loop_source_line(body_insts, di_lines)
            scev_loop = scev.get(name, {}).get("loops", {}).get(header, {})
            backedge = scev_loop.get("backedge", "")
            trip = int(backedge) + 1 if re.fullmatch(r'\d+', backedge) else "unknown"
            multiple = (int(scev_loop["trip_multiple"])
                        if re.fullmatch(r'\d+', scev_loop.get("trip_multiple", "")) else "unknown")
            # ---- data movement
            groups: dict[tuple, dict[str, Any]] = {}
            loads = stores = 0
            bytes_static: int | None = 0
            for inst in body_insts:
                if inst["class"] not in ("load", "store"):
                    continue
                pointer = [o for o in inst["operands"] if o.startswith(("%", "@"))]
                pointer = pointer[-1] if pointer else ""
                kind, base = _base_object(pointer, defs, set(func["args"]))
                elem = inst["type"]
                elem_bits = inst["bits"] if type(inst["bits"]) is int else "unknown"
                lanes = inst["lanes"] if type(inst["lanes"]) is int else 1
                stride: int | None = None
                confidence = "unknown"
                if inst["opcode"] == "getelementptr":
                    pass
                # stride via the pointer's own GEP (when pointer is a GEP result)
                gep_inst = defs.get(pointer)
                cast_seen = 0
                while (gep_inst and gep_inst["class"].startswith("cast")
                       and cast_seen < 4):
                    cast_operands = [o for o in gep_inst["operands"] if o.startswith(("%", "@"))]
                    if not cast_operands:
                        break
                    gep_inst = defs.get(cast_operands[0])
                    cast_seen += 1
                if gep_inst and gep_inst["opcode"] == "getelementptr":
                    indices = [re.sub(r'^\S+\s+', '', token).strip()
                               for token in gep_inst.get("gep_indices", [])]
                    index_value = None
                    for token in reversed(indices):
                        values = _VALUE_RE.findall(token)
                        if values:
                            index_value = values[0] if values[0].startswith("%") else "%" + values[0]
                            break
                    if index_value:
                        src_elem = gep_inst.get("gep_source_type", "")
                        # element stepped over: after leading constant indices
                        stepped = src_elem
                        for token in indices[:-1]:
                            if re.fullmatch(r'-?\d+', token.strip()):
                                inner = re.fullmatch(r'\[(\d+)\s+x\s+(.+)\]', stepped)
                                vec = re.fullmatch(r'<(\d+)\s+x\s+(.+)>', stepped)
                                if inner:
                                    stepped = inner.group(2)
                                elif vec:
                                    stepped = vec.group(2)
                        ebytes = _elem_bytes(stepped)
                        function_scev = scev.get(name, {})
                        stride = _scev_stride(pointer, function_scev, None, header)
                        if stride is not None:
                            confidence = "scev_pointer_step"
                        else:
                            stride = _scev_stride(index_value, function_scev, ebytes, header)
                            if stride is not None:
                                confidence = "scev_constant_step"
                            else:
                                stride = _induction_stride(index_value, defs, ebytes)
                                if stride is not None:
                                    confidence = "induction_phi"
                key = (base, kind, inst["class"], str(elem), str(elem_bits), str(lanes), str(stride))
                group = groups.setdefault(key, {
                    "base": base, "base_kind": kind, "access": inst["class"],
                    "element_type": elem, "element_bits": elem_bits, "lanes": lanes,
                    "instruction_count": 0, "stride_bytes": stride if stride is not None else "unknown",
                    "stride_confidence": confidence,
                })
                group["instruction_count"] += 1
                if inst["class"] == "load":
                    loads += 1
                else:
                    stores += 1
                width = _elem_bytes(elem)
                if width is None:
                    bytes_static = None
                elif bytes_static is not None:
                    bytes_static += width
            upper = "unknown"
            if (type(trip) is int and bytes_static is not None):
                upper = bytes_static * trip
            movement = {
                "accesses": sorted(groups.values(), key=lambda g: (str(g["base"]), g["access"],
                                                                   str(g["stride_bytes"]))),
                "totals": {
                    "load_instructions": loads, "store_instructions": stores,
                    "bytes_touched_static": bytes_static if bytes_static is not None else "unknown",
                    "bytes_total": "unknown", "trip_count": trip, "trip_multiple": multiple,
                    "bytes_upper_bound": upper,
                    "estimate_kind": "static_upper_bound" if upper != "unknown" else "not_computable",
                    "limitation": ("static instruction counts; execution multiplicity and "
                                   "true runtime traffic are unknown"),
                },
            }
            # ---- operations
            operations = _operation_groups(body_insts)
            # ---- dependencies
            ssa_edges: list[dict[str, str]] = []
            loop_edges: list[dict[str, str]] = []
            recurrences: list[dict[str, str]] = []
            body_blocks = [b for b in blocks if b["label"] in raw_body]
            for block in body_blocks:
                for inst in block["instructions"]:
                    is_header_phi = (block["label"] == header and inst["class"] == "phi")
                    for operand in inst["operands"]:
                        if operand not in loop_defs or not inst["name"] or operand == inst["name"]:
                            continue
                        if is_header_phi:
                            loop_edges.append({"from": operand, "to": inst["name"],
                                               "kind": "loop_carried", "via": "phi"})
                        else:
                            ssa_edges.append({"from": operand, "to": inst["name"],
                                              "kind": "ssa"})
                    if is_header_phi and inst["name"] in loop_defs:
                        update_class, update_op = _classify_recurrence(inst, defs)
                        recurrences.append({
                            "phi": inst["name"], "update_class": update_class,
                            "update_op": update_op, "bit_width": str(inst["bits"]),
                        })
            mem_edges: list[dict[str, str]] = []
            inst_by_norm: dict[str, dict[str, Any]] = {}
            for inst in body_insts:
                inst_by_norm.setdefault(_norm_inst(inst["raw"]), inst)
            for src_text, dst_text, result in da.get(name, []):
                src_inst = inst_by_norm.get(_norm_inst(src_text))
                dst_inst = inst_by_norm.get(_norm_inst(dst_text))
                if not src_inst or not dst_inst:
                    continue
                kind = "consistent" if result.startswith("consistent") else result.split()[0]
                mem_edges.append({"src": src_inst["name"] or src_inst["raw"].strip()[:40],
                                  "dst": dst_inst["name"] or dst_inst["raw"].strip()[:40],
                                  "result": kind if kind in ("flow", "anti", "output", "none",
                                                             "confused", "consistent") else "unknown"})
            # static depth: longest forward SSA chain in the body (phis excluded)
            forward: dict[str, list[str]] = {}
            for inst in body_insts:
                if not inst["name"] or inst["class"] == "phi":
                    continue
                for operand in inst["operands"]:
                    if operand in loop_defs and operand != inst["name"]:
                        forward.setdefault(operand, []).append(inst["name"])
            depth_memo: dict[str, int] = {}

            def chain_depth(node: str, seen: frozenset[str] = frozenset()) -> int:
                if node in depth_memo:
                    return depth_memo[node]
                if node in seen:
                    return 0
                best = 1
                for nxt in forward.get(node, ()):
                    best = max(best, 1 + chain_depth(nxt, seen | {node}))
                depth_memo[node] = best
                return best

            static_depth = max((chain_depth(node) for node in forward), default=0)
            conflicts: dict[str, str] = {}
            for base, kind, access, *_ in groups:
                prior = conflicts.get(base)
                if prior and prior != access:
                    conflicts[base] = "read_write"
                else:
                    conflicts[base] = access
            dependencies = {
                "edges": (sorted(ssa_edges, key=lambda e: (e["from"], e["to"]))[:MAX_EDGES_PER_LOOP]
                          + sorted(loop_edges, key=lambda e: (e["from"], e["to"]))[:MAX_EDGES_PER_LOOP]),
                "recurrences": sorted(recurrences, key=lambda r: r["phi"]),
                "mem_edges": sorted(mem_edges, key=lambda e: (e["src"], e["dst"]))[:MAX_MEM_EDGES],
                "conflicts": [{"base": base, "kind": kind}
                              for base, kind in sorted(conflicts.items())],
                "summary": {
                    "ssa_edges": len(ssa_edges), "loop_carried_edges": len(loop_edges),
                    "mem_edges": len(mem_edges), "static_depth": static_depth,
                    "static_depth_not_cycles": True,
                    "unknown_base_objects": any(g["base_kind"] == "unknown"
                                                for g in movement["accesses"]),
                },
            }
            hotspot = "false"
            if hotspot_line is not None:
                hotspot = ("true" if source_line == hotspot_line else "false")
            features["functions"].append({
                "name": name, "kind": "loop", "id": f"loop:{name}:{header}",
                "header_block": header, "depth": loop.get("depth", 1),
                "blocks": raw_body, "llvm_loop_metadata": loop["llvm_loop_meta"],
                "source_line": source_line if source_line is not None else "unknown",
                "source_link": source_link,
                **({"hotspot_match": hotspot} if hotspot_line is not None else {}),
                "trip_count": trip, "trip_count_expr": backedge or "unknown",
                "trip_multiple": multiple,
                "max_backedge_taken": (int(scev_loop["max_backedge"])
                                       if re.fullmatch(r'\d+', scev_loop.get("max_backedge", ""))
                                       else "unknown"),
                "data_movement": movement, "operations": operations,
                "dependencies": dependencies,
            })
        if not owned:
            # Function without loops: one function-scope region.
            source_line, source_link = _loop_source_line(all_insts, di_lines)
            features["functions"].append({
                "name": name, "kind": "function_scope", "id": f"function:{name}",
                "header_block": blocks[0]["label"] if blocks else "entry", "depth": 0,
                "blocks": [b["label"] for b in blocks], "llvm_loop_metadata": False,
                "source_line": source_line if source_line is not None else "unknown",
                "source_link": source_link, "trip_count": "unknown",
                "trip_count_expr": "unknown", "trip_multiple": "unknown",
                "max_backedge_taken": "unknown",
                "data_movement": {"accesses": [], "totals": {
                    "load_instructions": sum(1 for i in all_insts if i["class"] == "load"),
                    "store_instructions": sum(1 for i in all_insts if i["class"] == "store"),
                    "bytes_touched_static": "unknown", "bytes_total": "unknown",
                    "trip_count": "unknown", "trip_multiple": "unknown",
                    "bytes_upper_bound": "unknown", "estimate_kind": "not_computable",
                    "limitation": "function scope; no loop analysis"}},
                "operations": _operation_groups(all_insts),
                "dependencies": {"edges": [], "recurrences": [], "mem_edges": [],
                                 "conflicts": [],
                                 "summary": {"ssa_edges": 0, "loop_carried_edges": 0,
                                             "mem_edges": 0, "static_depth": 0,
                                             "static_depth_not_cycles": True,
                                             "unknown_base_objects": False}},
            })
        if function and name == function:
            break
    return features, audit


# ------------------------------------------------------------------ rendering

def _set_attrs(element: ET.Element, attrs: list[tuple[str, Any]]) -> None:
    for key, value in attrs:
        element.set(key, str(value))


def render_features_xml(features: dict[str, Any]) -> str:
    """Deterministic XML rendering with explicit provenance and hard caps."""
    root = ET.Element("ir_features", {
        "schema_version": str(features.get("schema_version", SCHEMA_VERSION)),
        "provenance": "static_ir",
        "analysis_source": str(features.get("analysis_source", "text_only")),
        "source_file": str(features.get("source_file", "")),
    })
    tool = ET.SubElement(root, "toolchain")
    tool.set("note", str(features.get("tool_note", _TOOL_NOTE)))
    limits = ET.SubElement(root, "limitations")
    limits.text = str(features.get("limitations", _LIMITATIONS))
    truncated = False
    for func in features.get("functions", []):
        fn = ET.SubElement(root, "function")
        _set_attrs(fn, [("name", func.get("name", "")), ("kind", func.get("kind", "")),
                        ("id", func.get("id", "")), ("header_block", func.get("header_block", "")),
                        ("depth", func.get("depth", 0))])
        if "hotspot_match" in func:
            fn.set("hotspot_match", str(func["hotspot_match"]))
        if func.get("source_line") not in (None, "unknown"):
            fn.set("source_line", str(func["source_line"]))
        fn.set("source_link", str(func.get("source_link", "unmapped")))
        fn.set("trip_count", str(func.get("trip_count", "unknown")))
        fn.set("trip_count_expr", str(func.get("trip_count_expr", "unknown"))[:200])
        fn.set("trip_multiple", str(func.get("trip_multiple", "unknown")))
        fn.set("max_backedge_taken", str(func.get("max_backedge_taken", "unknown")))
        movement = func.get("data_movement", {})
        dm = ET.SubElement(fn, "data_movement")
        accesses = movement.get("accesses", [])[:MAX_ACCESS_GROUPS]
        truncated |= len(movement.get("accesses", [])) > MAX_ACCESS_GROUPS
        for access in accesses:
            node = ET.SubElement(dm, "access")
            _set_attrs(node, [(key, access.get(key, "unknown"))
                              for key in ("base", "base_kind", "access", "element_type",
                                          "element_bits", "lanes", "instruction_count",
                                          "stride_bytes", "stride_confidence")])
        totals = ET.SubElement(dm, "totals")
        _set_attrs(totals, [(key, movement.get("totals", {}).get(key, "unknown"))
                            for key in ("load_instructions", "store_instructions",
                                        "bytes_touched_static", "bytes_total", "trip_count",
                                        "trip_multiple", "bytes_upper_bound", "estimate_kind")])
        totals.set("limitation", str(movement.get("totals", {}).get("limitation", "")))
        ops_node = ET.SubElement(fn, "operations")
        ops = func.get("operations", {}).get("ops", [])[:MAX_OPS_PER_LOOP]
        truncated |= len(func.get("operations", {}).get("ops", [])) > MAX_OPS_PER_LOOP
        for op in ops:
            node = ET.SubElement(ops_node, "op")
            _set_attrs(node, [(key, op.get(key, "unknown"))
                              for key in ("class", "opcode", "type", "bits", "lanes",
                                          "count", "constant_operands")])
            if "predicate" in op:
                node.set("predicate", str(op["predicate"]))
        calls_node = ET.SubElement(ops_node, "calls")
        for call in func.get("operations", {}).get("calls", []):
            node = ET.SubElement(calls_node, "call")
            _set_attrs(node, [("callee", call.get("callee", "unknown")),
                              ("count", call.get("count", 0)),
                              ("intrinsic", str(call.get("intrinsic", False)).lower())])
        deps = ET.SubElement(fn, "dependencies")
        all_edges = func.get("dependencies", {}).get("edges", [])
        truncated |= len(all_edges) > MAX_EDGES_PER_LOOP * 2
        for edge in all_edges[:MAX_EDGES_PER_LOOP * 2]:
            node = ET.SubElement(deps, "edge")
            _set_attrs(node, [(key, edge.get(key, ""))
                              for key in ("from", "to", "kind")
                              if edge.get(key)])
            if edge.get("via"):
                node.set("via", str(edge["via"]))
        for rec in func.get("dependencies", {}).get("recurrences", []):
            node = ET.SubElement(deps, "recurrence")
            _set_attrs(node, [(key, rec.get(key, "unknown"))
                              for key in ("phi", "update_class", "update_op", "bit_width")])
        all_mem = func.get("dependencies", {}).get("mem_edges", [])
        truncated |= len(all_mem) > MAX_MEM_EDGES
        for mem in all_mem[:MAX_MEM_EDGES]:
            node = ET.SubElement(deps, "mem_edge")
            _set_attrs(node, [(key, mem.get(key, "")) for key in ("src", "dst", "result")])
        for conflict in func.get("dependencies", {}).get("conflicts", []):
            node = ET.SubElement(deps, "conflict")
            _set_attrs(node, [("base", conflict.get("base", "unknown")),
                              ("kind", conflict.get("kind", "unknown"))])
        summary = ET.SubElement(deps, "summary")
        _set_attrs(summary, [(key, str(value).lower() if isinstance(value, bool) else value)
                             for key, value in
                             sorted(func.get("dependencies", {}).get("summary", {}).items())])
        summary.set("provenance", "static_ir")
    if truncated:
        root.set("truncated", "true")
    text = ET.tostring(root, encoding="unicode")
    # Whole-element drops keep the document well-formed; never cut mid-tag.
    function_nodes = list(root.findall("function"))
    while len(text.encode()) > MAX_OUTPUT_BYTES and function_nodes:
        root.remove(function_nodes.pop())
        root.set("truncated", "true")
        text = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + text + "\n"


def summarize_features(features: dict[str, Any]) -> str:
    """Bounded plain-text digest for prompts; static evidence caveats included."""
    lines = [f"IR features ({features.get('analysis_source', 'text_only')}, static_ir): "
             "not measured cycles or runtime traffic."]
    for func in features.get("functions", [])[:6]:
        totals = func.get("data_movement", {}).get("totals", {})
        summary = func.get("dependencies", {}).get("summary", {})
        lines.append(
            f"- {func.get('id', '?')} kind={func.get('kind')} "
            f"source_line={func.get('source_line', 'unknown')} "
            f"trip={func.get('trip_count', 'unknown')} "
            f"loads={totals.get('load_instructions', '?')} "
            f"stores={totals.get('store_instructions', '?')} "
            f"bytes_static={totals.get('bytes_touched_static', 'unknown')} "
            f"bytes_total=unknown depth={summary.get('static_depth', '?')}"
        )
        ops = func.get("operations", {}).get("ops", [])
        if ops:
            top = ", ".join(f"{op['class']}({op['type']})x{op['count']}" for op in ops[:6])
            lines.append(f"  ops: {top}")
        recurrences = func.get("dependencies", {}).get("recurrences", [])
        if recurrences:
            lines.append("  recurrences: " + ", ".join(
                f"{r['phi']}:{r['update_class']}" for r in recurrences[:6]))
        calls = [c for c in func.get("operations", {}).get("calls", [])
                 if not str(c.get("callee", "")).startswith("llvm.dbg")]
        if calls:
            lines.append("  calls: " + ", ".join(
                f"{c['callee']}x{c['count']}" for c in calls[:6]))
    lines.append("All figures are static upper/counting evidence; runtime byte counts "
                 "and cycles remain unknown.")
    return "\n".join(lines[:40])
