"""Whole-MiCoSoc Vivado PPA: stdlib-only argv, validation and evidence parsing.

Public integration seam: ``build_command(state, rtl_dir, out_dir)`` returns safe
argv (no execution/filesystem writes); ``load_summary(workdir, out_dir)`` reads
and revalidates the emitted summary. No stage_services or agent imports.
All measured timing is at the requested clock, never an inferred Fmax.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_PART = "xcau25p-ffvb676-2-e"
DEFAULT_FREQUENCY_MHZ = 100.0
SCHEMA_VERSION = 1
RTL_SUFFIXES = {".v", ".sv", ".vh", ".svh"}
DATA_SUFFIXES = {".bin", ".mem", ".hex", ".dat"}
AREA_LABELS = {
    "lut": ("Slice LUTs", "CLB LUTs"),
    "ff": ("Slice Registers", "CLB Registers"),
    "bram_36k": ("Block RAM Tile",),
    "dsp": ("DSPs", "DSP48E1 only", "DSP48E2 only"),
    "lutram": ("LUT as Distributed RAM",),
    "ramb18": ("RAMB18E1 only", "RAMB18E2 only", "RAMB18"),
    "ramb36": ("RAMB36E1 only", "RAMB36E2 only", "RAMB36/FIFO"),
}
CORE_AREA_METRICS = frozenset({"lut", "ff", "bram_36k", "dsp"})
PATH_GROUPS = ("internal", "entering", "leaving", "through")


def _resolve(base: Path, path: str | Path) -> Path:
    item = Path(path)
    return (item if item.is_absolute() else base / item).resolve()


def _within(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _clean_path(path: Path) -> None:
    if any(c in str(path) for c in "\x00\t\r\n"):
        raise ValueError("Paths must not contain NUL, tabs or newlines")


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def validate_inputs(run_dir: Path, rtl_root: Path, files: Sequence[Path], out_dir: Path,
                    *, part: str = DEFAULT_PART, target_frequency_mhz: float = 100.0,
                    threads: int = 4) -> dict[str, Any]:
    """Validate an explicit complete hardware fileset, without creating output.

    Caller must supply generator provenance; names/content are additionally
    checked for simulator wrappers. This is a local EDA boundary, not a sandbox
    for running hostile HDL or arbitrary vendor plugins.
    """
    run, root, out = (Path(p).resolve() for p in (run_dir, rtl_root, out_dir))
    if not re.fullmatch(r"xc[a-z0-9]+-[a-z0-9-]+", part):
        raise ValueError("Invalid FPGA part token")
    frequency = _positive(target_frequency_mhz, "target frequency")
    if isinstance(threads, bool) or int(threads) != threads or not 1 <= int(threads) <= 64:
        raise ValueError("threads must be an integer in [1, 64]")
    if not run.is_dir() or not root.is_dir():
        raise ValueError("run_dir and rtl_root must exist")
    if not _within(root, run) or not _within(out, run) or out == run:
        raise ValueError("RTL input and output must be confined to the run")
    if _within(out, root) or _within(root, out):
        raise ValueError("Output must not overlap source RTL")
    resolved = [Path(p).resolve() for p in files]
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError("Explicit, nonempty, duplicate-free complete fileset required")
    for path in [run, root, out, *resolved]:
        _clean_path(path)
    texts: dict[Path, str] = {}
    for path in resolved:
        if not path.is_file() or not _within(path, root):
            raise ValueError(f"Input is missing or outside rtl_root: {path}")
        if path.suffix.lower() not in RTL_SUFFIXES | DATA_SUFFIXES:
            raise ValueError(f"Unsupported hardware fileset member: {path}")
        lower = str(path.relative_to(run)).lower()
        if any(token in lower for token in ("simworkspace", "simrtl", "sim_rtl", "micosocsim")):
            raise ValueError("Simulator RTL is forbidden")
        if path.suffix.lower() in RTL_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            # Remove comments before looking at declarations and file references.
            text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
            if re.search(r"\bmodule\s+(?:MiCoSocSim\w*|\w*Testbench)\b", text, re.I):
                raise ValueError("Simulator/testbench module is forbidden")
            texts[path] = text
    if sum(len(re.findall(r"\bmodule\s+MiCoSoc\s*(?:#\s*\(|\()", t))
           for t in texts.values()) != 1:
        raise ValueError("Fileset must contain exactly one hardware MiCoSoc top")
    declared = set(resolved)
    for path, text in texts.items():
        references = re.findall(r'`include\s+"([^"\n]+)"', text)
        references += re.findall(r'\$readmem[hb]\s*\(\s*"([^"\n]+)"', text)
        if len(re.findall(r"\$readmem[hb]\s*\(", text)) != len(
                re.findall(r'\$readmem[hb]\s*\(\s*"([^"\n]+)"', text)):
            raise ValueError("Nonliteral memory initialization file is not supported")
        if len(re.findall(r"`include\b", text)) != len(re.findall(r'`include\s+"([^"\n]+)"', text)):
            raise ValueError("Nonliteral include is not supported")
        for ref in references:
            # Absolute paths would escape the private staged fileset on replay.
            if Path(ref).is_absolute():
                raise ValueError("Absolute HDL file references are not supported")
            candidates = {(path.parent / ref).resolve(), (root / ref).resolve()}
            if not any(p in declared and _within(p, root) for p in candidates):
                raise ValueError(f"Undeclared or escaping HDL file reference: {ref}")
    return {"run_dir": run, "rtl_root": root, "files": resolved, "out_dir": out,
            "part": part, "target_frequency_mhz": frequency, "threads": int(threads)}


def build_command(state: Mapping[str, Any], rtl_dir: str | Path,
                  out_dir: str | Path) -> list[str]:
    """Build standalone driver's argv from an immutable ``rtl_snapshot``.

    Snapshot files are ordered dictionaries {path, sha256}, where path may be
    repo-relative or relative to rtl_dir. Hashes are checked before execution
    by the driver as well. Does not glob a possibly simulator-contaminated tree.
    """
    workdir = Path(str(state.get("workdir") or ".")).resolve()
    run_name = state.get("run_root")
    if not run_name:
        raise ValueError("run_root is required")
    run = _resolve(workdir, str(run_name))
    root, out = _resolve(workdir, rtl_dir), _resolve(workdir, out_dir)
    snapshot = state.get("rtl_snapshot") or {}
    entries = snapshot.get("files") or []
    if not entries:
        raise ValueError("Frozen hardware rtl_snapshot.files is required")
    files, hashes = [], []
    for entry in entries:
        if not isinstance(entry, Mapping) or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError("Every RTL snapshot member requires a sha256")
        name = Path(str(entry["path"]))
        path = _resolve(workdir, name)
        if not path.is_relative_to(root):
            path = _resolve(root, name)
        files.append(path)
        hashes.append(str(entry["sha256"]))
    cfg = validate_inputs(run, root, files, out,
                          part=str(state.get("fpga_part") or DEFAULT_PART),
                          target_frequency_mhz=state.get("target_frequency_mhz", 100.0),
                          threads=state.get("vivado_threads", 4))
    timeout = _positive(state.get("eda_timeout") or 3600, "timeout")
    script = Path(__file__).resolve().parents[3] / "skills/cfu-designer/scripts/vivado_ppa_report.py"
    def relative(path: Path) -> str:
        if not path.is_relative_to(workdir):
            raise ValueError("Command paths must be inside workdir")
        return path.relative_to(workdir).as_posix()

    argv = ["python3", relative(script), "--run-dir", relative(run), "--rtl-root", relative(root),
            "--out-dir", relative(out), "--part", cfg["part"], "--target-frequency-mhz",
            str(cfg["target_frequency_mhz"]), "--threads", str(cfg["threads"]),
            "--timeout", str(timeout)]
    for path, digest in zip(files, hashes):
        argv.extend(["--rtl", relative(path), "--sha256", digest])
    for key in ("sha256", "hardware_input_sha256"):
        value = snapshot.get(key)
        if value:
            if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
                raise ValueError(f"Invalid snapshot {key}")
            argv.extend(["--snapshot-" + key.replace("_", "-"), str(value)])
    return argv


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any) -> float | None:
    """Parse finite scalar; unknown, NaN and comparison bounds are not zero."""
    try:
        result = float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def power_value(value: Any) -> dict[str, Any]:
    text = str(value).strip()
    if text.startswith("<"):
        bound = number(text[1:].strip())
        return {"value": None, "upper_bound": bound if bound is not None and bound > 0 else None,
                "bound_exclusive": True, "unit": "W", "reported": text}
    val = number(text)
    return {"value": val if val is not None and val >= 0 else None,
            "upper_bound": None, "bound_exclusive": False, "unit": "W", "reported": text}


def parse_metrics_tsv(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 2 or not re.fullmatch(r"[a-zA-Z0-9_.]+", columns[0]):
            raise ValueError("Malformed machine metrics TSV")
        key, value = columns
        if key in result and key != "phase":
            raise ValueError(f"Duplicate machine metric: {key}")
        result[key] = value
    return result


def parse_utilization(text: str) -> dict[str, float | None]:
    """Parse only named summary rows, preserving fractional BRAM tile counts."""
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) >= 2:
                rows.setdefault(cols[0].rstrip("* "), []).append(cols[1])
    result: dict[str, float | None] = {}
    for key, labels in AREA_LABELS.items():
        candidates = next(([number(value) for value in rows[label]] for label in labels if label in rows), [])
        # Vivado repeats some summary rows in Slice Distribution. Equal values
        # are harmless; contradictory values or absent rows remain unknown.
        result[key] = candidates[0] if candidates and len(set(candidates)) == 1 else None
        if result[key] is not None and result[key] < 0:
            result[key] = None
    # Vivado omits zero-valued subcategories from its synthesized Slice Logic
    # table. A nonnegative subset of an explicitly zero parent is exactly zero;
    # this is evidence-based inference, not a default for a missing report row.
    parent_memory = [number(value) for value in rows.get("LUT as Memory", [])]
    if "LUT as Distributed RAM" not in rows and parent_memory and set(parent_memory) == {0.0}:
        result["lutram"] = 0.0
    return result


def parse_power(text: str, cfu_instance: str | None) -> dict[str, Any]:
    """Parse exact summary labels and an indentation-reconstructed hierarchy path.

    Vivado's hierarchy Power(W) is dynamic only. Never distribute device static
    power. The full report and vectorless assumptions are retained separately.
    """
    result: dict[str, Any] = {"soc_dynamic": power_value(None),
                              "soc_static": power_value(None),
                              "cfu_dynamic": power_value(None), "confidence": None}
    in_hierarchy = False
    stack: list[str] = []
    matches = []
    seen: dict[str, int] = {}
    for line in text.splitlines():
        if re.match(r"\s*\d+(?:\.\d+)*\s+By Hierarchy\s*$", line):
            in_hierarchy = True
            continue
        if not line.lstrip().startswith("|"):
            continue
        raw = line.strip().strip("|").split("|")
        cols = [c.strip() for c in raw]
        if len(cols) < 2:
            continue
        label, value = cols[:2]
        for expected, key in (("Dynamic (W)", "soc_dynamic"), ("Device Static (W)", "soc_static")):
            if label == expected:
                seen[key] = seen.get(key, 0) + 1
                result[key] = power_value(value)
        if label == "Confidence Level":
            result["confidence"] = value
        if in_hierarchy and len(cols) == 2 and (number(value) is not None or value.startswith("<")):
            indent = len(raw[0]) - len(raw[0].lstrip())
            depth = max(0, (indent - 1) // 2)
            if depth > len(stack):
                # Malformed hierarchy cannot identify the unique instance.
                continue
            stack = stack[:depth] + [label]
            hierarchy = "/".join(stack[1:])
            if cfu_instance and hierarchy == cfu_instance:
                matches.append(power_value(value))
    for key, count in seen.items():
        if count != 1:
            result[key] = power_value(None)
    if len(matches) == 1:
        result["cfu_dynamic"] = matches[0]
    return result


def _path_metrics(metrics: Mapping[str, str], prefix: str) -> dict[str, Any]:
    return {"exists": {"0": False, "1": True}.get(metrics.get(prefix + ".exists", "")),
            "slack_ns": number(metrics.get(prefix + ".slack_ns")),
            "datapath_delay_ns": number(metrics.get(prefix + ".datapath_delay_ns")),
            "total_negative_slack_ns": number(metrics.get(prefix + ".total_negative_slack_ns")),
            "total_negative_slack_method": "sum_of_reported_worst_endpoint_slacks",
            "failing_endpoints": number(metrics.get(prefix + ".failing_endpoints")),
            "startpoint": metrics.get(prefix + ".startpoint") or None,
            "endpoint": metrics.get(prefix + ".endpoint") or None}


def make_summary(metrics: Mapping[str, str], soc_area: str = "", cfu_area: str = "",
                 power: str = "", *, provenance: Mapping[str, Any] | None = None,
                 returncode: int | None = None, execution_issue: str | None = None,
                 soc_synthesis_area: str = "", cfu_synthesis_area: str = "") -> dict[str, Any]:
    """Produce a conservative schema; absence of evidence can never pass."""
    frequency = number(metrics.get("target_frequency_mhz"))
    instance = metrics.get("cfu.instance") or None
    power_data = parse_power(power, instance)
    cfu_dynamic = power_data["cfu_dynamic"]
    cfu_power_method = "whole_soc_hierarchical_vectorless_estimate"
    if cfu_dynamic["value"] is None and cfu_dynamic["upper_bound"] is None:
        # Some Vivado versions omit leaf-module rows from the hierarchy table.
        # The total SoC dynamic power is a conservative upper bound for any
        # nonnegative component; never present it as a CFU point estimate.
        soc_dynamic = power_data["soc_dynamic"]
        soc_bound = (soc_dynamic["value"] if soc_dynamic["value"] is not None
                     else soc_dynamic["upper_bound"])
        if soc_bound is not None:
            cfu_dynamic = {
                "value": None, "upper_bound": soc_bound,
                "bound_exclusive": bool(soc_dynamic["bound_exclusive"]),
                "unit": "W", "reported": f"<= SoC dynamic total ({soc_bound:g} W)",
            }
            cfu_power_method = "whole_soc_dynamic_power_upper_bound"
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "backend": "vivado",
        "vivado": {"version": metrics.get("vivado.version") or None},
        "part": metrics.get("part") or None,
        "target_frequency": {"value": frequency, "unit": "MHz"},
        "target_frequency_mhz": frequency, "phase": metrics.get("phase", "not_started"),
        "soc": {"top": "MiCoSoc", "area": parse_utilization(soc_area),
                "synthesis_area": parse_utilization(soc_synthesis_area),
                "power": {"dynamic": power_data["soc_dynamic"],
                          "device_static": power_data["soc_static"],
                          "method": "vectorless_estimate", "confidence": power_data["confidence"]},
                "timing": {mode: _path_metrics(metrics, f"soc.{mode}") for mode in ("setup", "hold")},
                "unconstrained_internal_endpoints": number(metrics.get("soc.unconstrained_internal_endpoints")),
                "no_clock": number(metrics.get("soc.no_clock")),
                "route": {key: {"0": False, "1": True}.get(metrics.get("route." + key, ""))
                          for key in ("fully_routed", "errors_in_routes", "fully_placed")},
                "drc": {"errors": number(metrics.get("drc.errors")),
                        "critical_warnings": number(metrics.get("drc.critical_warnings")),
                        "warnings": number(metrics.get("drc.warnings"))}},
        "cfu": {"instance": instance, "instance_count": number(metrics.get("cfu.instance_count")),
                "area": parse_utilization(cfu_area),
                "synthesis_area": parse_utilization(cfu_synthesis_area),
                "power": {"dynamic": cfu_dynamic, "device_static": None,
                          "method": cfu_power_method},
                "timing": {group: {mode: _path_metrics(metrics, f"cfu.{group}.{mode}")
                                   for mode in ("setup", "hold")} for group in PATH_GROUPS}},
        "provenance": dict(provenance or {}), "status": "failed", "issues": [],
    }
    drc = summary["soc"]["drc"]
    board_count = number(metrics.get("drc.board_unbound.critical_warnings"))
    board_classified = (metrics.get("drc.board_unbound.rule_code") == "UCIO-1"
                        and metrics.get("drc.board_unbound.raw_severity") == "Critical Warning"
                        and metrics.get("drc.board_unbound.scope") == "part_only_no_bitstream"
                        and board_count is not None and board_count >= 0 and board_count.is_integer()
                        and drc["critical_warnings"] is not None and board_count <= drc["critical_warnings"])
    # Legacy metrics have no classification and therefore subtract NOTHING.
    drc["implementation_critical_warnings"] = (
        drc["critical_warnings"] - board_count if board_classified else drc["critical_warnings"])
    summary["soc"]["board_readiness"] = {
        "status": "unbound" if board_classified and board_count > 0 else "not_validated",
        "scope": "part_only_no_bitstream", "bitstream_ready": False,
        "classification_evidence": board_classified,
        "unbound_critical_warnings": board_count if board_classified else None,
        "exceptions": [{"code": "UCIO-1", "raw_severity": "Critical Warning", "count": board_count,
                        "classification": "board_pin_locations_unbound"}]
                      if board_classified and board_count > 0 else [],
    }
    issues: list[dict[str, str]] = summary["issues"]

    def issue(code: str, detail: str, severity: str = "error") -> None:
        issues.append({"code": code, "severity": severity, "detail": detail})

    if returncode != 0:
        issue("tool_exit", f"Vivado return code: {returncode!r}")
    if execution_issue:
        issue("execution", execution_issue)
    if metrics.get("flow.error"):
        issue("tcl_error", metrics["flow.error"])
    if summary["phase"] != "post_route":
        issue("incomplete_phase", "Full synthesis/opt/place/physopt/route evidence required")
    if not summary["vivado"]["version"]:
        issue("missing_version", "Vivado version unavailable")
    if not re.fullmatch(r"xc[a-z0-9]+-[a-z0-9-]+", str(summary["part"])) or frequency is None or frequency <= 0:
        issue("invalid_target", "Approved part and a finite positive target frequency required")
    requested = summary["provenance"]
    if requested.get("requested_part") and requested["requested_part"] != summary["part"]:
        issue("target_mismatch", "Vivado part differs from requested part")
    if requested.get("requested_frequency_mhz") and requested["requested_frequency_mhz"] != frequency:
        issue("target_mismatch", "Vivado target frequency differs from requested frequency")
    if summary["cfu"]["instance_count"] != 1 or not instance:
        issue("cfu_hierarchy", "Exactly one preserved AgentCfu instance is required")
    for scope in ("soc", "cfu"):
        for phase_key in ("area", "synthesis_area"):
            for name, value in summary[scope][phase_key].items():
                if value is None:
                    issue("missing_area" if name in CORE_AREA_METRICS else "missing_auxiliary_area",
                          f"{scope}.{phase_key}.{name} is unknown",
                          "error" if name in CORE_AREA_METRICS else "warning")
        p = summary[scope]["power"]["dynamic"]
        if p["value"] is None and p["upper_bound"] is None:
            issue("missing_dynamic_power", f"{scope} hierarchical dynamic power is unknown")
    required_timing = ("slack_ns", "datapath_delay_ns", "startpoint", "endpoint",
                       "total_negative_slack_ns", "failing_endpoints")
    for mode in ("setup", "hold"):
        timing = summary["soc"]["timing"][mode]
        if any(timing[key] is None for key in required_timing):
            issue("missing_timing", f"Whole SoC {mode} critical path fields are unknown")
        if timing["exists"] is not True or timing["slack_ns"] is None:
            issue("missing_timing", f"Whole SoC {mode} path/slack is unknown")
        elif timing["slack_ns"] < 0:
            issue("timing_not_closed", f"Whole SoC {mode} slack {timing['slack_ns']} ns")
        if timing["total_negative_slack_ns"] != 0 or timing["failing_endpoints"] != 0:
            issue("timing_not_closed", f"Whole SoC {mode} total negative slack/endpoints are nonzero or unknown")
    for group, data in summary["cfu"]["timing"].items():
        for mode, timing in data.items():
            if timing["exists"] is None or (timing["exists"] and any(timing[key] is None for key in required_timing)):
                issue("missing_cfu_timing", f"CFU {group}/{mode} evidence missing")
            elif timing["exists"] and (timing["slack_ns"] < 0 or timing["total_negative_slack_ns"] != 0
                                       or timing["failing_endpoints"] != 0):
                issue("timing_not_closed", f"CFU {group}/{mode} negative slack or failing endpoints")
    if not any(t["setup"]["exists"] for t in summary["cfu"]["timing"].values()):
        issue("no_cfu_paths", "No timed path touching the AgentCfu hierarchy")
    for key in ("unconstrained_internal_endpoints", "no_clock"):
        if summary["soc"][key] != 0:
            issue("unconstrained_timing", f"{key}: {summary['soc'][key]!r}")
    route = summary["soc"]["route"]
    if route != {"fully_routed": True, "errors_in_routes": False, "fully_placed": True}:
        issue("route_not_closed", f"Route status: {route}")
    for key in ("errors", "implementation_critical_warnings"):
        if summary["soc"]["drc"][key] != 0:
            issue("drc_not_closed", f"DRC {key}: {summary['soc']['drc'][key]!r}")
    if board_count is not None and not board_classified:
        issue("invalid_board_classification", "UCIO-1 classification evidence is invalid/inconsistent")
    if board_classified and board_count > 0:
        issue("board_unbound", f"Exact UCIO-1: {board_count:g} raw Critical Warning; no board LOC map, no bitstream readiness", "warning")
    issue("board_not_validated", "Part-only PPA does not establish board compatibility or bitstream readiness", "warning")
    if summary["soc"]["drc"]["warnings"] is None:
        issue("missing_drc", "DRC warning count unavailable")
    issue("power_assumptions", "Vectorless vendor estimate, not measured workload power; device static is SoC-only", "warning")
    if not any(item["severity"] == "error" for item in issues):
        summary["status"] = "passed"
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Whole MiCoSoc Vivado PPA", "", f"- Status: **{summary['status']}**",
             f"- Phase: {summary['phase']}; Vivado: {summary['vivado']['version']}",
             f"- Part: {summary['part']}; target: {summary['target_frequency_mhz']} MHz",
             f"- AgentCfu: `{summary['cfu']['instance']}`", "",
             "| Scope/phase | LUT | FF | LUTRAM | BRAM36 tiles | RAMB18 | RAMB36 | DSP | Dynamic W |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in ("soc", "cfu"):
        p = summary[scope]["power"]["dynamic"]
        dynamic = p["value"] if p["value"] is not None else (
            ("<" if p["bound_exclusive"] else "≤") + str(p["upper_bound"])
            if p["upper_bound"] is not None else "unknown")
        for phase, key in (("synthesis", "synthesis_area"), ("post_route", "area")):
            area = summary[scope][key]
            power_text = dynamic if phase == "post_route" else "not reported"
            lines.append(f"| {scope}/{phase} | " + " | ".join(str(area[k]) if area[k] is not None else "unknown"
                         for k in ("lut", "ff", "lutram", "bram_36k", "ramb18", "ramb36", "dsp")) + f" | {power_text} |")
    lines += ["", "## Timing at target clock (ns)", "",
              "| Scope | Setup slack | Hold slack | TNS | THS | Setup failing endpoints | Hold failing endpoints |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    timing_scopes = {"soc": summary["soc"]["timing"], **{
        f"cfu/{key}": val for key, val in summary["cfu"]["timing"].items()}}
    for scope, data in timing_scopes.items():
        values = [str(data[mode]["slack_ns"]) if data[mode]["slack_ns"] is not None else
                  ("no paths" if data[mode]["exists"] is False else "unknown") for mode in ("setup", "hold")]
        for key in ("total_negative_slack_ns", "failing_endpoints"):
            values.extend(str(data[mode][key]) if data[mode][key] is not None else
                          ("no paths" if data[mode]["exists"] is False else "unknown")
                          for mode in ("setup", "hold"))
        lines.append(f"| {scope} | " + " | ".join(values) + " |")
    lines += ["", "Internal = endpoints in CFU; entering/leaving = paths crossing its input/output pins;",
              "through = a path crossing both. Categories may overlap. No Fmax is inferred."]
    if summary["cfu"]["power"]["method"] == "whole_soc_dynamic_power_upper_bound":
        lines.append("CFU dynamic power is only bounded by total SoC dynamic power because Vivado omitted the CFU leaf from its hierarchy report; this is not a CFU point estimate.")
    lines += [
              "TNS/THS method: sum_of_reported_worst_endpoint_slacks. Vivado's per-path SLACK is already rounded;",
              "these sums are approximate, not the exact native global totals. The exact vendor SoC totals remain in the",
              "[native timing summary](<timing_summary.rpt>); failing endpoint counts are integer counts, not slack sums.",
              "Reset/UART are explicitly asynchronous exceptions; other unexpected I/O fails constraints.",
              "No DONT_TOUCH. Hierarchy is preserved by flatten_hierarchy none.",
              "Vectorless power is an estimate, not workload measurement. Device static is not apportioned.",
              "", f"- Internal unconstrained: {summary['soc']['unconstrained_internal_endpoints']}",
              f"- No-clock registers: {summary['soc']['no_clock']}",
              f"- Route: `{summary['soc']['route']}`", f"- DRC (raw counts retained): `{summary['soc']['drc']}`",
              f"- Board readiness: `{summary['soc']['board_readiness']}`",
              "- Exact UCIO-1 critical warnings alone may be classified board-unbound; all other critical/errors still fail.",
              "- Part-only PPA is NOT board validation or bitstream readiness.", "", "## Issues", ""]
    lines.extend(f"- {item['severity']}: {item['code']}: {item['detail']}" for item in summary["issues"])
    lines += ["", "Full reports, checkpoints, machine metrics, launch argv and SHA-256 provenance are retained in this directory."]
    return "\n".join(lines) + "\n"


def load_summary(workdir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Read structured output; never let a hand-edited/stale 'passed' bypass evidence.

    Driver provenance records the return code and hashes of machine metrics and
    reports. Re-read those inputs and reconstruct the verdict, verifying hashes.
    Missing/invalid output raises ValueError/OSError for the caller's cost gate.
    """
    out = _resolve(Path(workdir).resolve(), out_dir)
    if not out.is_relative_to(Path(workdir).resolve()) or not (out / "summary.json").resolve().is_relative_to(out):
        raise ValueError("Vivado summary is outside workdir/output")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary.get("backend") != "vivado" or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Vivado summary schema")
    provenance = summary.get("provenance") or {}
    run = Path(str(provenance.get("run_dir") or "")).resolve()
    if not run.is_relative_to(Path(workdir).resolve()) or out == run or not out.is_relative_to(run):
        raise ValueError("Vivado summary output is outside its recorded run")
    artifacts = provenance.get("artifacts") or {}
    expected = ("metrics.tsv", "utilization_soc.rpt", "utilization_cfu.rpt", "power.rpt",
                "utilization_post_synth.rpt", "utilization_cfu_post_synth.rpt")
    texts: dict[str, str] = {}
    for name in expected:
        path = out / name
        if name not in artifacts:
            texts[name] = ""
            continue
        if not path.resolve().is_relative_to(out) or sha256_file(path) != artifacts[name]:
            raise ValueError(f"Changed/out-of-scope Vivado artifact: {name}")
        texts[name] = path.read_text(encoding="utf-8", errors="replace")
    try:
        metrics = parse_metrics_tsv(texts["metrics.tsv"])
    except ValueError:
        metrics = {}
    metrics.setdefault("part", str(provenance.get("requested_part") or ""))
    metrics.setdefault("target_frequency_mhz", str(provenance.get("requested_frequency_mhz") or ""))
    reconstructed = make_summary(metrics, texts["utilization_soc.rpt"],
                                 texts["utilization_cfu.rpt"], texts["power.rpt"], provenance=provenance,
                                 returncode=provenance.get("returncode"),
                                 execution_issue=provenance.get("execution_issue"),
                                 soc_synthesis_area=texts["utilization_post_synth.rpt"],
                                 cfu_synthesis_area=texts["utilization_cfu_post_synth.rpt"])
    if reconstructed != summary:
        raise ValueError("Vivado summary does not match retained machine evidence")
    return reconstructed
