"""Pure formatting for durable, already-validated candidate progress lines.

Review owns admission and selection. This module neither evaluates a candidate
nor reads mutable artifacts, and must never promote a prediction to a result.
"""
from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if isinstance(value, int) or math.isfinite(value) else None


def _fmt(value: Any) -> str:
    value = _number(value)
    return "N/A" if value is None else str(value) if isinstance(value, int) else f"{value:g}"


def _text(value: Any) -> str:
    return " ".join(str(value).split())[:320] if value is not None else "N/A"


def _speedup(entry: dict) -> str:
    value = _number(entry.get("speedup_x100"))
    return "N/A" if value is None else f"{value / 100:.2f}x"


def _basis(cost: dict) -> tuple:
    backend = cost.get("backend") or ("yosys" if "cells" in cost else None)
    metric = cost.get("metric") or ("yosys_generic_mapped_cells" if backend == "yosys" else None)
    return backend, metric, cost.get("part"), cost.get("target_frequency_mhz")


def _cost(cost: dict) -> str:
    backend, metric, part, frequency = _basis(cost)
    if backend == "vivado":
        return (f"backend=vivado metric={_text(metric)} CFU_LUT={_fmt(cost.get('lut'))} "
                f"FF={_fmt(cost.get('ff'))} DSP={_fmt(cost.get('dsp'))} "
                f"BRAM36={_fmt(cost.get('bram_36k'))} part={_text(part)} "
                f"target_MHz={_fmt(frequency)}")
    return (f"backend={_text(backend)} metric={_text(metric)} cells={_fmt(cost.get('cells'))} "
            f"wire_bits={_fmt(cost.get('wire_bits'))} memories={_fmt(cost.get('memories'))} "
            f"memory_bits={_fmt(cost.get('memory_bits'))}")


def _delta(entry: dict, parent: dict | None) -> str:
    if not parent or not parent.get("measured") or not parent.get("validated"):
        return "vs_parent=N/A"
    pieces = [f"vs_round={parent.get('round')}"]
    old, new = _number(parent.get("cfu_cycles")), _number(entry.get("cfu_cycles"))
    if old is not None and old > 0 and new is not None and entry.get("reference_cycles") == parent.get("reference_cycles"):
        pieces.append(f"cycles={new - old:+g} ({(new / old - 1) * 100:+.1f}%)")
    else:
        pieces.append("cycles=N/A (measurement basis differs)")
    old_cost, new_cost = parent.get("cost") or {}, entry.get("cost") or {}
    if _basis(old_cost) != _basis(new_cost):
        pieces.append("cost=N/A (EDA basis differs)")
    else:
        key = "lut" if _basis(new_cost)[0] == "vivado" else "cells"
        before, after = _number(old_cost.get(key)), _number(new_cost.get(key))
        pieces.append(f"{key}={after - before:+g}" if before is not None and after is not None else f"{key}=N/A")
    return " ".join(pieces)


def validated_candidate_lines(state: dict, entry: dict, rounds: list[dict],
                              selected: dict | None, candidate: dict) -> list[str]:
    """Format a new complete result, not historical/partial/failed measurements."""
    if entry.get("measured") is not True or entry.get("validated") is not True or entry.get("snapshot_error"):
        return []
    previous = [item for item in rounds if item.get("round") != entry.get("round")
                and item.get("measured") and item.get("validated")]
    if entry.get("parent_round") is not None:
        parent = next((item for item in previous if item.get("round") == entry["parent_round"]), None)
    else:
        parent = previous[-1] if previous else None
    selected_valid = selected and selected.get("measured") and selected.get("validated")
    selection = "yes" if selected_valid and selected["round"] == entry["round"] else "no"
    lines = [
        f"    [candidate] VALIDATED round={entry.get('round')} target={_text(entry.get('target'))} "
        f"isa_revision={entry.get('isa_revision', 'N/A')} evaluation={_text(entry.get('evaluation_id') or None)} "
        f"verify={entry.get('verify_submission', 'N/A')}/{state.get('max_iters', 'N/A')}",
        f"    [performance] reference_cycles={_fmt(entry.get('reference_cycles'))} "
        f"cfu_cycles={_fmt(entry.get('cfu_cycles'))} speedup={_speedup(entry)}",
        "    [cost] " + _cost(entry.get("cost") or {}),
        "    [delta] " + _delta(entry, parent),
    ]
    if selected_valid:
        best_cost = selected.get("cost") or {}
        area = (f"CFU_LUT={_fmt(best_cost.get('lut'))}" if _basis(best_cost)[0] == "vivado"
                else f"cells={_fmt(best_cost.get('cells'))}")
        lines.append(f"    [best] objective={_text(state.get('loop_objective') or 'pareto')} "
                     f"round={selected['round']} cfu_cycles={_fmt(selected.get('cfu_cycles'))} "
                     f"speedup={_speedup(selected)} {area} current_selected={selection} "
                     f"frontier={_text(state.get('optimization_frontier', []))}")
        if candidate.get("round") == selected["round"] and candidate.get("manifest"):
            lines.append("    [source] " + _text(candidate["manifest"]))
    return lines
