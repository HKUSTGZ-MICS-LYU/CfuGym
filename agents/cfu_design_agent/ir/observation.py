"""Read-only, bounded evidence lookup for applied Embench observation points.

Dynamic cycle measurements refer to the instrumented profile configuration;
ASM and LLVM IR are *candidates* from the uninstrumented reference. Neither
static artifact identifies the measured loop's exact machine-code address.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..runtime.policy import PolicyError

if TYPE_CHECKING:
    from ..runtime.tools import RepoToolbox

_RUN = re.compile(r"agents/generated(?:/[A-Za-z0-9_-]+)?/runs/[A-Za-z0-9_-]+\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.c\Z")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SYMBOL = re.compile(r"^\s*(?:[0-9a-fA-F]+\s+)?<([A-Za-z_$][\w.$-]*)>:\s*$")
_DEFINE = re.compile(r'^\s*define\b.*?@(?:"((?:[^"\\]|\\.)+)"|([A-Za-z_$][\w.$-]*))\s*\(')
_METADATA = re.compile(r"^\s*!(\d+)\s*=\s*(?:distinct\s+)?!(\w+)\((.*)\)\s*$")
_DBG = re.compile(r"!dbg\s+!(\d+)\b")
_FIELD_SCOPE = re.compile(r"\bscope:\s*!(\d+)\b")
_FIELD_FILE = re.compile(r"\bfile:\s*!(\d+)\b")
_FIELD_LINE = re.compile(r"\bline:\s*(\d+)\b")
_FILENAME = re.compile(r'\bfilename:\s*"([^"\\]*)"')
MAX_READ_CHARS = 600_000
MAX_OUTPUT_CHARS = 12_000


def _safe(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def _number(value: object, *, minimum: int = 0) -> int | None:
    return value if type(value) is int and value >= minimum else None


def _paths(root: str, benchmark: str, filename: str) -> tuple[str, str, str]:
    if not (_safe(root, _RUN) and _safe(benchmark, _NAME) and _safe(filename, _FILE)):
        raise ValueError("unsafe run root, benchmark or source file")
    stem = filename[:-2]
    return (f"{root}/workspace/embench_{benchmark}/reference/embench_main.asm",
            f"{root}/07_analysis/ir/{stem}.ll",
            f"{root}/07_analysis/ir/{stem}.features.xml")


def build_observation_index(*, run_root: str, benchmark: str, observation: dict,
                            profiles: dict, ir_result: dict) -> dict:
    """Pure index of valid applied points; missing artifacts are not asserted present.

    Malformed, conflicting or unlocated points are omitted. Only an exact
    recorded run-scoped IR file with the source file's stem is accepted.
    """
    _paths(run_root, benchmark, "check.c")
    points = observation.get("points", []) if isinstance(observation, dict) else []
    applied = observation.get("applied", []) if isinstance(observation, dict) else []
    measured = profiles.get("profile", {}) if isinstance(profiles, dict) else {}
    recorded = ir_result.get("ir_files", []) if isinstance(ir_result, dict) else []
    applied_ids = {item for item in applied if _safe(item, _IDENT)} if isinstance(applied, list) else set()
    records = {item for item in recorded if isinstance(item, str)} if isinstance(recorded, list) else set()
    results: dict[str, dict] = {}
    duplicate: set[str] = set()
    for point in points if isinstance(points, list) else []:
        if not isinstance(point, dict):
            continue
        ident, filename, function = (point.get(k) for k in ("id", "file", "function"))
        if (not _safe(ident, _IDENT) or ident not in applied_ids
                or not _safe(filename, _FILE) or not _safe(function, _IDENT)):
            continue
        if ident in results:
            duplicate.add(ident)
            continue
        kind = point.get("kind")
        if kind not in ("function", "loop"):
            continue
        line = _number(point.get("line"), minimum=1) if "line" in point else None
        ordinal = _number(point.get("ordinal")) if "ordinal" in point else None
        if ("line" in point and line is None) or ("ordinal" in point and ordinal is None):
            continue
        asm, expected_ir, expected_features = _paths(run_root, benchmark, filename)
        row = measured.get(ident, {}) if isinstance(measured, dict) else {}
        row = row if isinstance(row, dict) else {}
        target_match = (ir_result.get("toolchain", {}).get("target_match")
                        if isinstance(ir_result.get("toolchain"), dict) else None) if isinstance(ir_result, dict) else None
        results[ident] = {
            "id": ident, "kind": kind, "file": filename, "function": function,
            "line": line, "ordinal": ordinal,
            "cycles": _number(row.get("cycles")), "calls": _number(row.get("calls")),
            "asm_path": asm, "ir_path": expected_ir if expected_ir in records else None,
            "features_path": (expected_features if isinstance(ir_result, dict)
                              and expected_features in set(
                                  item for item in ir_result.get("feature_files", [])
                                  if isinstance(item, str)) else None),
            "ir_target_match": target_match if type(target_match) is bool else None,
        }
    for ident in duplicate:
        results.pop(ident, None)
    return {"schema_version": "1", "run_root": run_root, "benchmark": benchmark,
            "observations": results}


def _read(toolbox: RepoToolbox, path: str) -> tuple[list[str], bool, str]:
    try:
        text = toolbox.read_text(path, limit=MAX_READ_CHARS + 1)
    except PolicyError:
        raise
    except (OSError, ValueError, PermissionError) as exc:
        return [], False, f"unavailable ({type(exc).__name__})"
    if not isinstance(text, str) or not text:
        return [], False, "missing or empty"
    return text[:MAX_READ_CHARS].splitlines(), len(text) > MAX_READ_CHARS, "readable"


def _asm_absence(state: str) -> str:
    """Say what a missing reference disassembly means and where to look instead.

    It is a *build output*: a run that reused the cached measurement has none, so
    "missing or empty" alone reads as "this evidence does not exist at all".
    """
    if state == "readable":
        return "function symbol not found"
    if state == "missing or empty":
        return ("no reference disassembly in this run's workspace (it is a reference "
                "build output; a reused profile measurement has none) — use view='ir'")
    return state


def _asm_candidates(lines: list[str], function: str, excerpt_limit: int
                   ) -> list[tuple[str, int, list[tuple[int, str]]]]:
    sections: list[tuple[str, int, list[tuple[int, str]]]] = []
    current: tuple[str, int, list[tuple[int, str]]] | None = None
    for lineno, text in enumerate(lines, 1):
        match = _SYMBOL.match(text)
        if match:
            symbol = match.group(1)
            # A dotted prefix alone is insufficient: avoid unrelated foo.bar.
            clone = re.fullmatch(re.escape(function) +
                                 r"\.(?:isra|constprop|part|cold|clone|llvm|lto_priv)(?:\.\d+)*", symbol)
            current = (symbol, lineno, []) if symbol == function or clone else None
            if current is not None:
                sections.append(current)
        elif current is not None and len(current[2]) < excerpt_limit:
            current[2].append((lineno, text))
    return sections


def _ir_candidates(lines: list[str], function: str, filename: str, source_line: int | None,
                   expected_source: str, absolute_source: str) -> tuple[list[tuple[int, str]], str]:
    if source_line is None:
        return [], "unmapped (no source line in observation)"
    # Metadata definitions are usually at the end of the translation unit. The
    # lookup is bounded by MAX_READ_CHARS, never an unbounded full-file scan.
    meta: dict[str, tuple[str, str]] = {}
    for text in lines:
        match = _METADATA.match(text)
        if match:
            meta[match.group(1)] = (match.group(2), match.group(3))
    def source_matches(metadata_filename: str) -> bool:
        # Clang may record either the TU's bare filename or its full run-scoped
        # reference path; do not equate arbitrary same-basename paths.
        return metadata_filename in (filename, expected_source, absolute_source)

    locations: set[str] = set()
    for ident, (kind, body) in meta.items():
        if kind != "DILocation":
            continue
        line_match = _FIELD_LINE.search(body)
        if not line_match or int(line_match.group(1)) != source_line:
            continue
        # Follow lexical scopes to exclude known locations from other files.
        # Missing scope metadata remains a tentative candidate, not proof.
        scope = _FIELD_SCOPE.search(body)
        visited: set[str] = set()
        scope_id = scope.group(1) if scope else ""
        wrong_file = False
        while scope_id and scope_id not in visited and len(visited) < 32:
            visited.add(scope_id)
            entry = meta.get(scope_id)
            if entry is None:
                break
            kind, scope_body = entry
            if kind == "DIFile":
                named = _FILENAME.search(scope_body)
                if named and not source_matches(named.group(1)):
                    wrong_file = True
                break
            file_ref = _FIELD_FILE.search(scope_body)
            if file_ref:
                file_entry = meta.get(file_ref.group(1))
                if file_entry and file_entry[0] == "DIFile":
                    named = _FILENAME.search(file_entry[1])
                    if named and not source_matches(named.group(1)):
                        wrong_file = True
                    break
            parent = _FIELD_SCOPE.search(scope_body)
            scope_id = parent.group(1) if parent else ""
        if not wrong_file:
            locations.add(ident)
    inside = False
    found_define = False
    matches: list[tuple[int, str]] = []
    for lineno, text in enumerate(lines, 1):
        start = _DEFINE.match(text)
        if start:
            inside = (start.group(1) or start.group(2)) == function
            found_define |= inside
            continue
        if inside and text.lstrip().startswith("}"):
            inside = False
        elif inside and any(ref in locations for ref in _DBG.findall(text)):
            if len(matches) < 100:
                matches.append((lineno, text))
    if not found_define:
        return [], "unmapped (matching define function absent)"
    if not matches:
        return [], f"unmapped (no !dbg instruction at source line {source_line} in matching function)"
    return matches, (f"{len(matches)} candidate instruction line(s) at source line {source_line}; "
                     "header-line candidate only, not loop-body or loop-address mapping")


def _features_excerpt(text: str, function: str, source_line: int | None,
                      budget: int) -> tuple[list[str], str]:
    """Pick the smallest feature-XML subtree for this observation, schema-tolerant."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return [], f"unparseable feature XML ({type(exc).__name__})"
    # Match by priority: explicit hotspot flag, function+line, line alone
    # (loops may omit the function attribute), then function scope alone.
    ranked: list[tuple[int, object]] = []
    for element in root.iter():
        if element is root:
            continue
        name = element.get("function", "") or element.get("name", "")
        line = element.get("source_line", "")
        hotspot = element.get("hotspot_match", "")
        rank = 0
        if hotspot == "true":
            rank = 1
        elif name == function and source_line is not None and line == str(source_line):
            rank = 2
        elif not name and source_line is not None and line == str(source_line):
            rank = 3
        elif name == function and source_line is None:
            rank = 4
        if rank:
            ranked.append((rank, element))
    ranked.sort(key=lambda item: item[0])
    chosen = ranked[0][1] if ranked else root
    try:
        rendered = ET.tostring(chosen, encoding="unicode")
    except (TypeError, ValueError) as exc:
        return [], f"cannot render feature XML ({type(exc).__name__})"
    lines = rendered.splitlines()
    if len(lines) > budget:
        lines = lines[:budget - 1] + ["... feature excerpt line limit reached"]
    if chosen is root:
        return lines, "whole document (no loop matched this observation)"
    return lines, "loop/function-focused excerpt (static_ir)"


def inspect_observation(index: dict, toolbox: RepoToolbox, observation_id: str,
                        view: str = "summary", max_lines: int = 60) -> str:
    """Read only approved artifact paths through ``toolbox.read_text``.

    Results are capped to 8..100 output lines and 12,000 characters. ASM
    symbols and LLVM debug locations are candidate evidence, not loop identity.
    """
    if view not in ("summary", "asm", "ir", "features"):
        raise ValueError("view must be summary, asm, ir or features")
    if not isinstance(index, dict) or not _safe(observation_id, _IDENT):
        raise ValueError("unknown observation ID")
    records = index.get("observations")
    if not isinstance(records, dict) or observation_id not in records:
        raise ValueError("unknown observation ID")
    item = records[observation_id]
    if not isinstance(item, dict) or item.get("id") != observation_id:
        raise ValueError("invalid observation index")
    root, benchmark, filename = index.get("run_root"), index.get("benchmark"), item.get("file")
    asm, expected_ir, expected_features = _paths(root, benchmark, filename)
    function = item.get("function")
    if not _safe(function, _IDENT) or item.get("kind") not in ("function", "loop"):
        raise ValueError("invalid observation index")
    if (item.get("asm_path") != asm or item.get("ir_path") not in (None, expected_ir)
            or item.get("features_path") not in (None, expected_features)):
        raise ValueError("invalid observation artifact path")
    if getattr(toolbox, "run_root", None) != root:
        raise ValueError("observation belongs to another run")
    if getattr(toolbox, "stage", None) != "analysis":
        raise ValueError("observation inspection requires analysis toolbox")
    line, ordinal = item.get("line"), item.get("ordinal")
    if (line is not None and _number(line, minimum=1) is None) or (ordinal is not None and _number(ordinal) is None):
        raise ValueError("invalid observation index")
    if type(max_lines) is not int:
        raise ValueError("max_lines must be an integer")
    budget = min(100, max(8, max_lines))
    asm_lines, asm_cut, asm_state = (_read(toolbox, asm) if view == "asm"
                                     else ([], False, "not inspected in this view"))
    sections = _asm_candidates(asm_lines, function, budget) if asm_lines else []
    ir_path = item["ir_path"]
    ir_lines, ir_cut, ir_state = (_read(toolbox, ir_path) if view == "ir" and ir_path
                                  else ([], False, "not recorded" if not ir_path else "not inspected in this view"))
    source_path = f"{root}/workspace/embench_{benchmark}/reference/src/{filename}"
    matches, ir_mapping = _ir_candidates(
        ir_lines, function, filename, line, source_path, str(toolbox.workdir / source_path),
    ) if ir_lines else ([], "unmapped")
    target_match = item.get("ir_target_match")
    if target_match is not None and type(target_match) is not bool:
        raise ValueError("invalid observation index")
    target_note = ("approximate target; not faithful GCC machine code" if target_match is False
                   else "target match reported" if target_match is True else "target match unknown")
    features_path = item.get("features_path")
    feature_lines: list[str] = []
    feature_cut = False
    feature_state = "not recorded"
    if view == "features" and features_path:
        feature_lines, feature_cut, feature_state = _read(toolbox, features_path)
    elif features_path:
        feature_state = "not inspected in this view"
    measured = (f"cycles={item.get('cycles') if _number(item.get('cycles')) is not None else 'unavailable'} "
                f"calls={item.get('calls') if _number(item.get('calls')) is not None else 'unavailable'}")
    output = [f"Observation {observation_id}: {item['kind']} {filename}:{line or '?'} "
              f"function={function} ordinal={ordinal if ordinal is not None else '?'}",
              f"Measured profile (instrumented): {measured}",
              f"ASM reference candidate: {asm} — " +
              (f"{len(sections)} function symbol(s); {'ambiguous clones' if len(sections) > 1 else 'function-level only'}"
               if sections else _asm_absence(asm_state)) +
              (" [bounded scan truncated; further symbols may exist]" if asm_cut else ""),
              f"LLVM IR candidate: {ir_path or 'not recorded'} — " +
              (ir_mapping if ir_lines else ir_state) + f" [{target_note}]" +
              (" [bounded scan truncated; mapping incomplete]" if ir_cut else ""),
              f"Feature XML: {features_path or 'not recorded'} — " +
              ("available (view='features')" if features_path and view != "features" else feature_state),
              "Caution: static reference candidates are not measured loop addresses or cycle estimates."]
    if view == "asm" and sections:
        output.append("ASM symbols (function-level only; no source-line link):")
        for symbol, physical, excerpt in sections[:4]:
            output.append(f"{asm}:{physical} <{symbol}>:")
            output.extend(f"{asm}:{lineno} {text}" for lineno, text in excerpt)
        if len(sections) > 4:
            output.append(f"... {len(sections) - 4} additional symbols omitted")
    if view == "ir" and matches:
        output.append("LLVM IR candidate !dbg instruction lines (may be discontiguous; not an exact loop match):")
        output.extend(f"{ir_path}:{lineno} {text}" for lineno, text in matches)
    if view == "features":
        if feature_lines:
            excerpt, focus = _features_excerpt("\n".join(feature_lines), function, line, budget)
            output.append(f"Feature XML excerpt: {focus} [{target_note}]")
            output.extend(excerpt)
            if feature_cut:
                output.append(f"Feature XML scan truncated to {MAX_READ_CHARS} characters.")
        else:
            output.append(f"Feature XML unavailable: {feature_state}")
    if (view == "asm" and asm_cut) or (view == "ir" and ir_cut):
        output.append(f"Artifact scan truncated to {MAX_READ_CHARS} characters; mapping may be incomplete.")
    if len(output) > budget:
        output = output[:budget - 1] + ["... output line limit reached"]
    text = "\n".join(output)
    suffix = "\n... character limit reached"
    return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS - len(suffix)] + suffix
