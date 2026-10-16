"""Current-vs-historical FPGA evidence and compact, scope-labelled reporting."""
from __future__ import annotations

from typing import Any
from . import dependencies


def view(state: dict[str, Any]) -> dict[str, Any]:
    gate = (state.get("gates") or {}).get("cost") or {}
    data = gate.get("ppa") or {}
    if not data:
        for result in reversed(state.get("cost_results") or []):
            data = (result.get("parsed") or {}).get("ppa") or {}
            if data:
                break
    if not data:
        return {"data": {}, "current": False, "valid": False, "reason": "not measured"}
    evidence = data.get("evidence") or {}
    current = bool(evidence.get("input_sha256") and
                   evidence["input_sha256"] == dependencies.input_fingerprints(state)["cost"] and
                   (not state.get("evaluation_id") or evidence.get("evaluation_id") == state["evaluation_id"]))
    valid = bool(current and gate.get("ok") and not gate.get("stale") and not gate.get("cancelled")
                 and data.get("status") == "passed")
    return {"data": data, "current": current, "valid": valid,
            "reason": "current validated FPGA PPA" if valid else "current partial/failed FPGA evidence" if current
            else "historical FPGA evidence; does not certify the current candidate"}


def number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def power(value: dict | None) -> str:
    value = value or {}
    if value.get("value") is not None:
        return number(value["value"])
    if value.get("upper_bound") is not None:
        return "<" + number(value["upper_bound"])
    return "n/a"


def lines(state: dict[str, Any]) -> list[str]:
    result = view(state)
    data = result["data"]
    if not data:
        return ["FPGA PPA: not measured."] if state.get("cost_backend") == "vivado" else []
    rows = [f"- {result['reason']}; implementation status: {data.get('status', 'unknown')}",
            f"- Vivado {(data.get('vivado') or {}).get('version')}; part {data.get('part')}; target {data.get('target_frequency_mhz')} MHz",
            "", "| Scope (post-route) | LUT | FF | BRAM36 tiles | DSP | Dynamic W | Setup slack ns | Hold slack ns |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope, label in (("soc", "Whole SoC"), ("cfu", "CFU hierarchy")):
        info = data.get(scope) or {}
        area = info.get("area") or {}
        timing = info.get("timing") or {}
        if scope == "cfu":
            # No synthetic standalone Fmax or misleading min-of-category scalar.
            setup = hold = "see paths below"
        else:
            setup = number((timing.get("setup") or {}).get("slack_ns"))
            hold = number((timing.get("hold") or {}).get("slack_ns"))
        rows.append(f"| {label} | " + " | ".join(number(area.get(key)) for key in ("lut", "ff", "bram_36k", "dsp"))
                    + f" | {power((info.get('power') or {}).get('dynamic'))} | {setup} | {hold} |")
    soc_power = (data.get("soc") or {}).get("power") or {}
    rows.extend(["", f"- SoC device static: {power(soc_power.get('device_static'))} W (not apportioned to CFU).",
                 f"- Power: vectorless Vivado estimate; confidence {soc_power.get('confidence') or 'unknown'}; not board/workload measurement.",
                 f"- CFU instance: {(data.get('cfu') or {}).get('instance') or 'not found'}"])
    for kind, modes in ((data.get("cfu") or {}).get("timing") or {}).items():
        values = []
        for mode in ("setup", "hold"):
            path = modes.get(mode) or {}
            values.append(f"{mode}=" + ("not applicable (no path)" if path.get("exists") is False else number(path.get("slack_ns")) + " ns"))
        rows.append(f"- CFU {kind}: " + ", ".join(values))
    rows.append("- CFU timing is measured inside the implemented SoC, not standalone CFU Fmax; path groups can overlap.")
    board = (data.get("soc") or {}).get("board_readiness") or {}
    rows.append(f"- Board readiness: {board.get('status', 'not validated')}; part-only PPA is not board/bitstream certification.")
    drc = (data.get("soc") or {}).get("drc") or {}
    if drc:
        rows.append(f"- DRC: errors={number(drc.get('errors'))}, raw critical={number(drc.get('critical_warnings'))}, implementation critical={number(drc.get('implementation_critical_warnings'))} (only exact UCIO-1 may be board-only).")
    if (data.get("evidence") or {}).get("summary_path"):
        rows.append(f"- Complete PPA evidence: {data['evidence']['summary_path']}")
    return rows
