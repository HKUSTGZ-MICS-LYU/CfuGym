"""Policy-confined source inputs and transactional exploration-parent restore."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from ...runtime import conformance_delivery, stage_services

RESTORE_KEYS = (
    "isa_spec", "isa_spec_text", "design_contract", "isa_revision", "isa_extension_base", "isa_added_symbols",
    "workload_analysis", "benchmark_project", "workload_spec", "design_files", "design_workspace",
    "hardware_plan", "software_plan", "implementation_hw_summary", "implementation_sw_summary",
    "conformance_delivery", "conformance_tests", "implementation_design_type",
    "implementation_unimplemented", "implementation_unverified",
)
_SUFFIXES = {".c", ".h", ".cpp", ".hpp", ".scala", ".S", ".mk"}
ALIASES = {"07_analysis/workload_analysis.yaml": "analysis", "08_isa/cfu_isa.yaml": "isa",
           "08_isa/cfu_isa.md": "isa", "08_isa/validation.yaml": "isa",
           "09_implementation/hw_plan.md": "implementation", "09_implementation/sw_plan.md": "implementation"}


def snapshot_aliases(state, toolbox, artifacts, prefix):
    records = []
    for relative, stage in ALIASES.items():
        source = f"{state['run_root']}/{relative}"
        path = conformance_delivery._safe_path(toolbox, source)
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        ref = artifacts.write_text("review", f"{prefix}/context/{Path(relative).name}", content,
                                   inputs=[source], kind="validated_context")
        records.append({"relative": relative, "stage": stage, "snapshot": ref["immutable_path"],
                        "sha256": ref["sha256"]})
    return records


def input_paths(state: dict[str, Any], toolbox: Any) -> list[str]:
    paths = set(state.get("design_files") or [])
    paths.update(path for path, _ in stage_services._software_targets(state, toolbox))
    workspace = Path(toolbox.workdir) / state.get("run_root", "") / "workspace"
    if workspace.is_dir():
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [name for name in dirs if name not in {"build", "target", ".git", "soc", "checks"}]
            for name in files:
                path = Path(root) / name
                rel = path.relative_to(toolbox.workdir).as_posix()
                if (path.suffix in _SUFFIXES or name == "Makefile") and toolbox.is_read_allowed(rel) and toolbox.is_write_allowed(rel):
                    paths.add(rel)
    header = (state.get("gates", {}).get("implementation_sw") or {}).get("intrinsics_header")
    if header:
        paths.add(header)
    if state.get("run_root"):
        paths.update(path for path in conformance_delivery.delivery_paths(state).values()
                     if (Path(toolbox.workdir) / path).is_file())
    for path in paths:
        conformance_delivery._safe_path(toolbox, path, write=True)
    return sorted(paths)


def restore_parent(state: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Restore only verified text inputs, never reuse a parent's passing build gates."""
    if state.get("dry_run") or not state.get("apply_patch"):
        raise ValueError("restoring an exploration parent requires patch application")
    toolbox = stage_services._toolbox_for_stage(state, "implementation")
    manifest_path = str(candidate.get("manifest") or "")
    full = conformance_delivery._safe_path(toolbox, manifest_path)
    raw = full.read_bytes()
    if not candidate.get("sha256") or hashlib.sha256(raw).hexdigest() != candidate["sha256"]:
        raise ValueError("candidate manifest hash is missing or mismatched")
    manifest = yaml.safe_load(raw)
    if not isinstance(manifest, dict) or manifest.get("round") != candidate.get("round"):
        raise ValueError("candidate manifest round is mismatched")
    saved = manifest.get("restore_state")
    if not isinstance(saved, dict):
        raise ValueError("legacy candidate has no restorable state; choose current or start a new run")
    if candidate.get("files") != manifest.get("files"):
        raise ValueError("candidate source inventory is mismatched")
    allowed_root = f"{state.get('run_root', '')}/workspace/"
    formal = set(conformance_delivery.delivery_paths(state).values())
    desired: dict[str, bytes | None] = {}
    for record in manifest.get("files") or []:
        source = str(record.get("source") or "")
        if not source.startswith(allowed_root) and source not in formal:
            raise ValueError(f"candidate source is not a run-workspace input: {source}")
        conformance_delivery._safe_path(toolbox, source, write=True)
        snapshot = conformance_delivery._safe_path(toolbox, str(record.get("snapshot") or ""))
        data = snapshot.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise ValueError(f"candidate source hash mismatched: {source}")
        desired[source] = data
    # Remove new text dependencies introduced by a later experiment. Fixed
    # oracle paths also need removal when the parent did not have them.
    current_paths = set(input_paths(state, toolbox)) | formal
    for path in current_paths - desired.keys():
        desired[path] = None
    aliases = []
    for record in manifest.get("aliases") or []:
        relative, stage = record.get("relative"), record.get("stage")
        if ALIASES.get(relative) != stage:
            raise ValueError("candidate context alias is outside its fixed stage paths")
        snapshot = conformance_delivery._safe_path(toolbox, str(record.get("snapshot") or ""))
        data = snapshot.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise ValueError(f"candidate context hash mismatched: {relative}")
        alias_box = stage_services._toolbox_for_stage(state, stage)
        source = f"{state['run_root']}/{relative}"
        previous = conformance_delivery._snapshot(alias_box, {source: source}, write=True)
        aliases.append((relative, stage, data, alias_box, previous))
    before = conformance_delivery._snapshot(toolbox, {path: path for path in desired}, write=True)
    try:
        conformance_delivery._restore(toolbox, desired)
        for relative, stage, data, _box, _previous in aliases:
            stage_services._artifacts_for_stage(state, stage).write_text(
                stage, relative, data.decode("utf-8"), kind="restored_candidate_context")
    except Exception:
        conformance_delivery._restore(toolbox, before)
        for _relative, _stage, _data, alias_box, previous in aliases:
            conformance_delivery._restore(alias_box, previous)
        raise
    update = {key: deepcopy(saved[key]) for key in RESTORE_KEYS if key in saved}
    preserved = {"load_context", "workload_spec", "prepare_benchmark", "profile", "hotspot", "ir", "analysis", "isa"}
    update.update(
        gates={key: deepcopy(value) for key, value in (manifest.get("gates") or {}).items() if key in preserved},
        input_fingerprints={}, isa_draft={}, evaluation_id="", evaluation={}, rtl_snapshot={}, flow_memory={}, stopped_at="",
        repair_scope="", repair_route={}, repair_entry={}, changed_design_files=[], changed_sw_files=[],
        repair_evidence_epoch={key: len(state.get(key) or []) for key in (
            "validation_results", "cost_results", "hw_attempts", "sw_attempts", "conformance_attempts")},
    )
    return update
