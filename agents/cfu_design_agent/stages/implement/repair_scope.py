"""Delivery scope for a Stage3 repair, frozen from authoritative entry evidence."""
from typing import Any

from ...runtime import stage_services
from . import intrinsics


def required_domains(state, toolbox, failure_info) -> tuple[bool, bool]:
    """First delivery changes both halves; repair may reuse hash-verified healthy bytes."""
    entry = [item for item in failure_info["failures"].values()
             if item.get("status", "current") == "current"]
    rungs = {str(item.get("rung") or "") for item in entry}
    gates = state.get("gates") or {}
    isa_hash = intrinsics._isa_digest(state)
    hw_hash = intrinsics._hardware_candidate_hash(state, toolbox)
    sw_hash = intrinsics._software_candidate_hash(
        toolbox, stage_services._software_targets(state, toolbox))

    def reusable(gate: dict[str, Any], candidate_hash: str, *, software=False) -> bool:
        return bool(gate.get("ok") and gate.get("applied") and not gate.get("stale")
                    and not gate.get("skipped") and not gate.get("errors")
                    and gate.get("candidate_hash") == candidate_hash
                    and (not software or gate.get("isa_hash") == isa_hash))

    oracle_only = state.get("repair_scope") == "conformance"
    return (not reusable(gates.get("implementation_hw") or {}, hw_hash)
            or (not oracle_only and any(rung.startswith("hw") for rung in rungs)),
            not reusable(gates.get("implementation_sw") or {}, sw_hash, software=True)
            or (not oracle_only and any(rung.startswith("sw") for rung in rungs)))


def changed_paths(toolbox, session) -> set[str]:
    """No-op and edit-then-revert do not count as an applied design change."""
    return {path for path in session.touched_paths
            if (toolbox.workdir / path).is_file()
            and (toolbox.workdir / path).read_bytes() != session.snapshots[path]}
