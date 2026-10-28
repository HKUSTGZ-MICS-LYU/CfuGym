"""Bounded input fingerprints for CFU validation gates.

Fingerprints attest to the *current* inputs, not to the model's assertions. A
missing source or a changed upstream contract makes a previously passed gate
stale and demands the normal validation ladder again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .tools import design_workspace_paths
from . import cost_budget, vivado


DOWNSTREAM = {
    "analysis": ("isa", "implementation_hw", "hw_build", "hw_conformance", "implementation_sw", "sw_build", "sim", "cost"),
    "isa": ("implementation_hw", "hw_build", "hw_conformance", "implementation_sw", "sw_build", "sim", "cost"),
    "implementation_hw": ("hw_build", "hw_conformance", "sw_build", "sim", "cost"),
    "implementation_sw": ("sw_build", "sim", "cost"),
    "hw_build": ("hw_conformance", "sw_build", "sim", "cost"),
    "hw_conformance": ("sw_build", "sim", "cost"),
    "sw_build": ("sim", "cost"),
    "sim": ("cost", "validation_pair"),
    "cost": ("validation_pair",),
    "validation_pair": (),
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _source_hash(root: Path, rel: str) -> str:
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"source fingerprint escapes workdir: {rel}")
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_fingerprints(state: dict[str, Any]) -> dict[str, str]:
    """Recompute ordered stage inputs from authoritative contract and source bytes."""
    root = Path(state.get("workdir", ".")).resolve()
    run_root = str(state.get("run_root", "") or "")
    hw = {path: _source_hash(root, path) for path in design_workspace_paths(run_root).values()} if run_root else {}
    project = state.get("benchmark_project") or {}
    sw_paths = set(state.get("changed_sw_files", []) or [])
    sw_paths.update(path for path in (project.get("source_path"), project.get("header_path")) if path)
    if run_root:
        workspace = (root / run_root / "workspace").resolve()
        if not workspace.is_relative_to(root):
            raise ValueError("run workspace fingerprint escapes workdir")
        if workspace.is_dir():
            # Capture additional agent-editable C/header sources rather than
            # trusting the last model-reported changed-file list.
            sw_paths.update(
                path.relative_to(root).as_posix()
                for suffix in ("*.c", "*.h", "*.cpp", "*.hpp")
                for path in workspace.rglob(suffix)
                if path.is_file() and not any(part in {"build", "target", "checks", "soc"} for part in path.relative_to(workspace).parts)
            )
    if project.get("kind") == "embench_benchmark" and run_root:
        # Include the tracked reference inputs too: changes to the original
        # workload must invalidate measurements made from an older copy.
        sw_paths.update(project.get("source_files") or [])
    sw = {path: _source_hash(root, path) for path in sorted(sw_paths)}
    analysis = _hash({"profile": state.get("profile_result"), "workload": state.get("workload_spec"),
                      "analysis": state.get("workload_analysis"), "ir": state.get("ir_result")})
    isa = _hash({"analysis": analysis, "isa": state.get("isa_spec")})
    from . import hardware_integrity
    hw_input = _hash({"isa": isa, "hardware": hw,
                      "hardware_integrity_version": hardware_integrity.VERSION})
    sw_input = _hash({"isa": isa, "software": sw})
    # Old checkpoints retain their exact fingerprint vocabulary. New hardware
    # includes the same declared clock frequency as simulation and Vivado.
    if "target_frequency_mhz" in state:
        hw_input = _hash({"hardware": hw_input, "frequency_mhz": state["target_frequency_mhz"]})
    built = _hash({"hardware": hw_input})
    conformance_inputs = {name: _source_hash(root, f"{run_root}/09_implementation/conformance/{name}")
                          for name in ("oracle.yaml", "cfu_conformance.c")} if run_root else {}
    checked = _hash({"hardware": built, "software": sw_input, "conformance": conformance_inputs,
                     "delivery": state.get("conformance_delivery")})
    sw_built = _hash({"sw": sw_input, "hardware": built})
    measured = _hash({"build": sw_built, "profile": analysis,
                      "build_configuration": {"project": project.get("build_cmd"),
                                              "run": project.get("run_cmd"),
                                              "test": project.get("test_cmd"),
                                              "isa": state.get("isa_spec"),
                                              "benchmark": project.get("benchmark"),
                                               **({"measurement": project.get("measurement"),
                                                   "benchmark_integrity": project.get("integrity"),
                                                    "benchmark_patches": project.get("patches")}
                                                  if project.get("kind") == "embench_benchmark" else {}),
                                              "flags": state.get("soc_flags")}})
    cost_input = measured
    if state.get("cost_backend") == "vivado":
        cost_input = _hash({"measurement": measured, "backend": "vivado",
                            "part": state.get("fpga_part", vivado.DEFAULT_PART),
                            "frequency_mhz": state.get("target_frequency_mhz", 100.0),
                            "rtl_sha256": (state.get("rtl_snapshot") or {}).get("sha256"),
                            "vivado_threads": int(state.get("vivado_threads", 4)),
                            "vivado_parser": _source_hash(root, "agents/cfu_design_agent/runtime/vivado.py"),
                            "flow_schema": 2, "top": "MiCoSoc", "cfu": "AgentCfu",
                            "yosys_precheck": "standalone-agent-cfu-or-selected-frozen-cfu-v1",
                            "cost_stage": _source_hash(root, "agents/cfu_design_agent/stages/verify/cost.py"),
                            "cost_commands": _source_hash(root, "agents/cfu_design_agent/runtime/commands.py"),
                            "yosys_runner": _source_hash(root, "skills/cfu-designer/scripts/yosys_cost_report.py"),
                            "vivado_runner": _source_hash(root, "skills/cfu-designer/scripts/vivado_ppa_report.py"),
                            "tcl": _source_hash(root, "skills/cfu-designer/scripts/vivado_ppa.tcl")})
    # Budget policy affects cost acceptance only, never HW/simulation evidence.
    # Wrap both backends, including legacy/missing state defaults, so a pre-wall
    # cost pass cannot satisfy the current Cost Wall without revalidation.
    cost_input = _hash({"cost": cost_input,
                        "max_yosys_cells": cost_budget.max_cells(state),
                        "cost_wall_version": cost_budget.VERSION})
    return {
        "analysis": analysis, "isa": isa, "implementation_hw": hw_input,
        "hw_build": hw_input, "hw_conformance": checked,
        "implementation_sw": sw_input, "sw_build": sw_built,
        "sim": measured, "cost": cost_input,
        **({"validation_pair": _hash({"sim": measured, "cost": cost_input})}
           if int(state.get("flow_version", 1)) >= 2 else {}),
    }


def completed_gate_inputs(state: dict[str, Any], *, resume: bool = False) -> None:
    """Refuse stale gates; resume additionally validates prepared workload protocol."""
    project = state.get("benchmark_project") or {}
    if resume and project.get("kind") == "embench_benchmark":
        from ..benchmark import embench, embench_integrity, embench_patchers
        expected = embench.measurement_parameters()
        if (project.get("measurement") != expected
                or project.get("patches") != embench_patchers.patch_manifest(str(project.get("benchmark", "")))
                or (project.get("integrity") or {}).get("version") != embench_integrity.INTEGRITY_VERSION):
            raise ValueError("legacy Embench checkpoint measurement; start a new run")
        problems = embench_integrity.validate_candidate(
            Path(state.get("workdir", ".")), str(state.get("run_root", "")),
            str(project.get("benchmark", "")))
        if problems:
            raise ValueError("checkpoint benchmark integrity: " + "; ".join(problems))
    current = input_fingerprints(state)
    saved = state.get("input_fingerprints") or {}
    gates = state.get("gates") or {}
    if saved and any(current.get(name) != digest for name, digest in saved.items() if name in current):
        raise ValueError("checkpoint source or contract changed since the saved graph state")
    for name, gate in gates.items():
        if name in current and isinstance(gate, dict) and gate.get("ok"):
            if saved.get(name) != current[name]:
                raise ValueError(f"checkpoint gate {name} has stale or missing input fingerprint")
            if name in {"implementation_hw", "hw_build", "hw_conformance", "sim", "cost"}:
                run_root = str(state.get("run_root", "") or "")
                hw_paths = design_workspace_paths(run_root).values() if run_root else ()
                if not hw_paths or any(_source_hash(Path(state["workdir"]).resolve(), path) == "missing"
                                       for path in hw_paths):
                    raise ValueError(f"checkpoint gate {name} lacks hardware source files")
            if name in {"implementation_sw", "sw_build", "sim", "cost"}:
                project = state.get("benchmark_project") or {}
                source = project.get("source_path")
                if source and _source_hash(Path(state["workdir"]).resolve(), source) == "missing":
                    raise ValueError(f"checkpoint gate {name} lacks software source file")


def invalidate_changed_gates(gates: dict[str, dict[str, Any]], before: dict[str, str],
                             after: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Mark changed gate and its dependents stale, never reuse an old pass."""
    updated = {name: dict(gate) for name, gate in gates.items()}
    changed = {name for name in DOWNSTREAM if before.get(name) != after.get(name)}
    for name in changed:
        for target in (name, *DOWNSTREAM[name]):
            if target in updated:
                updated[target].update(ok=False, stale=True, reason=f"upstream input changed: {name}")
    return updated
