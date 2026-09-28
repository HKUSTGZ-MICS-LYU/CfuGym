"""Observation-point instrumentation for Embench workloads.

The default instrumenter observes each discoverable loop as a whole statement,
plus its containing function for context. Scope exit is handled by GCC's cleanup
attribute in sw/embench/embench_obs.h, including early return/break/continue.
The reference and CFU builds are never instrumented.

Two rules matter for correctness and were validated against Embench's crc32:

* Only *definitions* may be instrumented. Matching a prototype would put the
  marker in whatever function follows it.
* The instrumented file must include "embench_obs.h"; otherwise the macro is an
  implicit function call and the link fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .generic import static_hotspot_candidates
from .embench_loops import discover_loops, loop_id


OBS_INCLUDE = '#include "embench_obs.h"'
OBS_MARKER_PREFIX = "EMBENCH_OBS_ENTER("
DEFAULT_MAX_POINTS = 24


def source_key(rel_path: str) -> str:
    """Stable benchmark-relative source identity, independent of run/config."""
    normalized = rel_path.replace("\\", "/")
    return normalized.split("/src/", 1)[-1] if "/src/" in normalized else normalized


@dataclass
class ObservationPlan:
    """Requested observation points plus what was actually applied."""

    points: list[dict] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"points": self.points, "applied": self.applied, "skipped": self.skipped}


def find_definition_brace(text: str, name: str) -> int:
    """Return the index of the opening brace of `name`'s definition, or -1.

    Skips prototypes and call sites: the name must be followed by a parameter
    list (balanced parens), optional __attribute__((...)) blocks, and then a
    brace.
    """
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    for match in pattern.finditer(text):
        i = match.end()
        while i < len(text) and text[i] in " \t\n\r":
            i += 1
        if i >= len(text) or text[i] != "(":
            continue
        depth = 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        i += 1
        while i < len(text) and text[i] in " \t\n\r":
            i += 1
        while text.startswith("__attribute__", i):
            j = text.find("(", i)
            if j < 0:
                break
            depth = 0
            while j < len(text):
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1
            while i < len(text) and text[i] in " \t\n\r":
                i += 1
        if i < len(text) and text[i] == "{":
            return i
    return -1


def ensure_include(text: str) -> str:
    """Add the observation header after the first unconditional include."""
    if OBS_INCLUDE in text:
        return text
    # Place it immediately after the first unconditional include. Putting it
    # before all includes changes feature macros (e.g. wikisort's bool typedef);
    # putting it after the last include can hide it under #ifdef DEBUG.
    depth = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("#if ", "#if\t", "#ifdef", "#ifndef")):
            depth += 1
        elif stripped.startswith("#endif"):
            depth = max(0, depth - 1)
        elif depth == 0 and re.match(r"#\s*include\b", stripped):
            end = offset + len(line)
            return text[:end] + OBS_INCLUDE + "\n" + text[end:]
        offset += len(line)
    return OBS_INCLUDE + "\n" + text


def normalize_plan(plan: dict | list | None) -> list[dict]:
    """Accept a list of function/loop points or {"points": [...]} without guessing locations."""
    if plan is None:
        return []
    raw = plan.get("points", []) if isinstance(plan, dict) else plan
    points: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("function", "") or "").strip()
        if not name:
            continue
        kind = str(item.get("kind", "function") or "function").strip()
        points.append(
            {
                "file": str(item.get("file", "") or "").strip(),
                "function": name,
                "kind": kind,
                "id": str(item.get("id", "") or name).strip(),
                **({"ordinal": item["ordinal"]} if "ordinal" in item else {}),
                **({"line": item["line"]} if "line" in item else {}),
                **({"loop_kind": item["loop_kind"]} if "loop_kind" in item else {}),
            }
        )
    return points


def default_plan(sources: dict[str, str], *, limit: int = DEFAULT_MAX_POINTS) -> list[dict]:
    """Measure whole loops by default; retain containing functions as context."""
    points: list[dict] = []
    for rel_path, text in sources.items():
        key = source_key(rel_path)
        functions = [name for name in static_hotspot_candidates(text) if find_definition_brace(text, name) >= 0]
        # Embench sometimes places __attribute__((noinline)) between the return
        # type and benchmark_body; the generic candidate regex misses it.
        if find_definition_brace(text, "benchmark_body") >= 0 and "benchmark_body" not in functions:
            functions.insert(0, "benchmark_body")
        loops = discover_loops(text, functions)
        selected = []
        owners: set[str] = set()
        for loop in loops:
            additional = 1 + (loop.function not in owners)
            if len(points) + len(selected) + len(owners) + additional > limit:
                break
            selected.append(loop)
            owners.add(loop.function)
        for name in functions:
            if name in owners:
                points.append({"file": key, "function": name, "kind": "function",
                               "id": loop_id(key, name, 0).replace("obs_loop_", "obs_fn_", 1)})
        for loop in selected:
            points.append({"file": key, "function": loop.function, "kind": "loop",
                           "id": loop_id(key, loop.function, loop.ordinal),
                           "ordinal": loop.ordinal, "line": loop.line, "loop_kind": loop.kind})
        if len(points) >= limit:
            break
    return points


def instrument_text(
    text: str,
    points: list[dict],
    *,
    rel_path: str = "",
) -> tuple[str, list[str], list[dict]]:
    """Instrument complete loops and functions, with edits on original offsets.

    All edits are applied from right to left: nested loops and function markers
    cannot invalidate each other's source positions. A loop with a changed or
    ambiguous source location is skipped, not placed at a guessed location.
    """
    applied: list[str] = []
    skipped: list[dict] = []
    edits: list[tuple[int, str]] = []
    matching = [point for point in points if not point.get("file") or not rel_path
                or point["file"] in (rel_path, source_key(rel_path), Path(rel_path).name)]
    funcs = {str(point["function"]) for point in matching}
    spans = discover_loops(text, sorted(funcs))
    for point in matching:
        ident = str(point.get("id", ""))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", ident):
            skipped.append({**point, "reason": "id must be a C identifier"})
            continue
        marker = f"{OBS_MARKER_PREFIX}{ident});"
        if marker in text:
            # Refreshing a profile in the same workspace must not double-wrap
            # existing points, but they remain active measurement points.
            applied.append(ident)
            continue
        if ident in applied:
            skipped.append({**point, "reason": "duplicate observation id"})
            continue
        kind = point.get("kind", "function")
        if kind == "function":
            brace = find_definition_brace(text, str(point["function"]))
            if brace < 0:
                skipped.append({**point, "reason": "definition not found"})
                continue
            edits.append((brace + 1, f"\n  {marker}"))
        elif kind == "loop":
            matches = [loop for loop in spans if loop.function == point["function"]
                       and (point.get("ordinal") is None or loop.ordinal == point["ordinal"])
                       and (point.get("line") is None or loop.line == point["line"])]
            if len(matches) != 1:
                skipped.append({**point, "reason": "loop location not found or ambiguous; specify ordinal/line"})
                continue
            loop = matches[0]
            # A new enclosing scope times the full loop statement once per entry.
            # It does not add rdcycle to each iteration or alter break/continue.
            edits.append((loop.start, f"{{ {marker}\n"))
            edits.append((loop.end, "\n}"))
        else:
            skipped.append({**point, "reason": f"unsupported observation kind {kind}"})
            continue
        applied.append(ident)
    instrumented = text
    for offset, insert in sorted(edits, key=lambda item: item[0], reverse=True):
        instrumented = instrumented[:offset] + insert + instrumented[offset:]
    if applied:
        instrumented = ensure_include(instrumented)
    return instrumented, applied, skipped


def instrument_sources(
    sources: dict[str, str],
    plan: dict | list | None = None,
    *,
    auto: bool = True,
) -> tuple[dict[str, str], ObservationPlan]:
    """Instrument a {rel_path: text} mapping and report what happened."""
    points = normalize_plan(plan)
    if not points and auto:
        points = default_plan(sources)
    result: dict[str, str] = {}
    report = ObservationPlan(points=points)
    used_ids: set[str] = set()
    for rel_path, text in sources.items():
        file_points = [point for point in points if not point.get("file") or
                       point["file"] in (rel_path, source_key(rel_path), Path(rel_path).name)]
        conflicting = [point for point in file_points if point["id"] in used_ids]
        report.skipped.extend({**point, "reason": "duplicate observation id across files"} for point in conflicting)
        file_points = [point for point in file_points if point["id"] not in used_ids]
        new_text, applied, skipped = instrument_text(text, file_points, rel_path=rel_path)
        used_ids.update(applied)
        result[rel_path] = new_text
        report.applied.extend(applied)
        report.skipped.extend(skipped)
    return result, report
