"""The implementation stage's final audit of a submitted delivery.

The model chooses to submit; this function decides whether the submitted bytes
may enter the validation ladder. It lives beside the stage rather than inside it
so the stage module stays under its size budget and the audit itself is testable
without a model, a tool loop or a graph run.

Nothing here writes a Gate verdict: it answers "is this delivery admissible",
and the ladder's own rungs decide whether it is correct.
"""

from __future__ import annotations

from typing import Any

from ...runtime import stage_services
from ...runtime.policy import PolicyError
from ...state import CfuDesignState
from . import intrinsics, repair_scope


def audit_delivery(
    state: CfuDesignState,
    toolbox: Any,
    delivery: str,
    *,
    toolsets: list[Any],
    oracle_session: Any,
    hardware_targets: frozenset[str],
    software_targets: frozenset[str],
    need_hw: bool,
    need_sw: bool,
) -> tuple[bool, str]:
    """``(accepted, reason)``; the reason is what a rejected submission carries."""
    if not delivery.strip() or not toolsets:
        return False, "the delivery was empty"
    tools = toolsets[-1]
    if tools.repair_route and tools.repair_route["target"] in ("analysis", "contract"):
        return True, ""
    problems = stage_services.benchmark_integrity_problems(toolbox)
    if problems:
        return False, "benchmark integrity: " + "; ".join(problems)
    # Oracle staging may precede diagnosis; submitting a repair may not.
    if tools.diagnosis_required and not tools.hardware_session.diagnosis:
        return False, (
            "a repair round must file submit_repair_diagnosis before it can be submitted: "
            "state the observed failure, the root cause, the repair mapping and the "
            "regression risks, then submit again"
        )
    oracle_changed = bool(oracle_session is not None and oracle_session.changed)
    need_change = True
    if state.get("repair_scope") == "conformance":
        # A conformance-scope round may legitimately change only the test/oracle.
        need_change = bool(need_hw or need_sw or oracle_changed)
    hw_changed = repair_scope.changed_paths(
        toolbox, tools.hardware_session).intersection(hardware_targets)
    sw_changed = repair_scope.changed_paths(
        toolbox, tools.software_session).intersection(software_targets)
    if need_change and not hw_changed and not sw_changed and not oracle_changed:
        return False, (
            "no hardware target was edited and no software target was edited, so "
            "nothing is applied: call edit_hardware/edit_software on the approved "
            "targets with the SEARCH/REPLACE edits, then submit a summary"
        )
    if need_hw and not hw_changed:
        return False, (
            "the current failures are hardware-side but no hardware target was edited: "
            f"call edit_hardware on {', '.join(sorted(hardware_targets))} with the "
            "SEARCH/REPLACE edits, then submit a summary"
        )
    if need_sw and not sw_changed:
        return False, (
            "the current failures are software-side but no software target was edited: "
            f"call edit_software on {', '.join(sorted(software_targets))}, "
            "then submit a summary"
        )
    if oracle_session is not None:
        _delivery, delivery_errors = oracle_session.current()
        if delivery_errors:
            return False, (
                "conformance oracle is missing/stale/invalid: call write_conformance; "
                + "; ".join(delivery_errors)
            )
    if any(not (toolbox.workdir / path).is_file() for path in (*hw_changed, *sw_changed)):
        return False, "an edited target is missing from the run workspace"
    problems = stage_services.hardware_integrity_problems(state, toolbox)
    if problems:
        return False, "hardware integrity: " + "; ".join(problems)
    try:
        exercises, why = intrinsics._software_exercises_cfu(
            state, toolbox, stage_services._software_targets(state, toolbox)
        )
    except (ValueError, PolicyError, OSError) as exc:
        return False, f"the CFU-usage check could not run: {exc}"
    if not exercises:
        return False, f"the accelerated path does not exercise the CFU: {why}"
    return True, ""
