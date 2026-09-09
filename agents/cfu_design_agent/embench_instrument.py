"""Observation-point instrumentation for Embench workloads.

The instrumenter inserts a single `EMBENCH_OBS_ENTER(id)` at the start of a
chosen function definition. Scope exit is handled by GCC's cleanup attribute in
sw/embench/embench_obs.h, so early `return`/`break`/`continue` are covered
without rewriting control flow.

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

from .benchmarks import static_hotspot_candidates


OBS_INCLUDE = '#include "embench_obs.h"'
OBS_MARKER_PREFIX = "EMBENCH_OBS_ENTER("
DEFAULT_MAX_POINTS = 24


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
    """Add the observation header include once, after the last existing include."""
    if OBS_INCLUDE in text:
        return text
    last = text.rfind("#include")
    if last < 0:
        return OBS_INCLUDE + "\n" + text
    eol = text.find("\n", last)
    if eol < 0:
        return text + "\n" + OBS_INCLUDE + "\n"
    return text[: eol + 1] + OBS_INCLUDE + "\n" + text[eol + 1 :]


def normalize_plan(plan: dict | list | None) -> list[dict]:
    """Accept a list of points or {"points": [...]}; keep only function points."""
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
            }
        )
    return points


def default_plan(sources: dict[str, str], *, limit: int = DEFAULT_MAX_POINTS) -> list[dict]:
    """Function-level candidates: functions that contain a loop."""
    points: list[dict] = []
    for rel_path, text in sources.items():
        for name in static_hotspot_candidates(text):
            points.append({"file": rel_path, "function": name, "kind": "function", "id": name})
            if len(points) >= limit:
                return points
    return points


def instrument_text(
    text: str,
    points: list[dict],
    *,
    rel_path: str = "",
) -> tuple[str, list[str], list[dict]]:
    """Apply function-level observation points to one file's text.

    Returns (new_text, applied_ids, skipped_entries). Loop-level points are
    reported as skipped with a reason rather than silently ignored.
    """
    applied: list[str] = []
    skipped: list[dict] = []
    instrumented = text
    for point in points:
        if point.get("file") and rel_path and point["file"] not in (rel_path, Path(rel_path).name):
            continue
        name = point["function"]
        if point.get("kind", "function") != "function":
            skipped.append({**point, "reason": "loop-level instrumentation is not supported yet"})
            continue
        marker = f"{OBS_MARKER_PREFIX}{point['id']});"
        if marker in instrumented:
            skipped.append({**point, "reason": "already instrumented"})
            continue
        brace = find_definition_brace(instrumented, name)
        if brace < 0:
            skipped.append({**point, "reason": "definition not found"})
            continue
        instrumented = (
            instrumented[: brace + 1]
            + f"\n  {OBS_MARKER_PREFIX}{point['id']});"
            + instrumented[brace + 1 :]
        )
        applied.append(point["id"])
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
    for rel_path, text in sources.items():
        new_text, applied, skipped = instrument_text(text, points, rel_path=rel_path)
        result[rel_path] = new_text
        report.applied.extend(applied)
        report.skipped.extend(skipped)
    return result, report
