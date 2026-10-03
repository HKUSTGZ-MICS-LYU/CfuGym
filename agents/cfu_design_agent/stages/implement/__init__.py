"""The Implement stage: one entry point, two ways to work.

The stage is where the agent is free: it decides the order of editing hardware,
editing software, writing the conformance oracle, and optional compile/syntax
checks. Simulation runs only in Verify, never in the implementation tool loop. The
ladder afterwards reruns every check independently and owns the verdict — the
model cannot certify its own design.

Two modes, chosen by capability rather than by a flag:

* **tool mode** (a tool-capable model): the bounded transactional tool loop in
  :mod:`.unified`;
* **delivery mode** (``--no-agent-tools``, or a model that cannot call tools):
  the EDIT/FILE/PATCH delivery protocol, hardware then software, merged into one
  update.

Both write the same two gates, so the ladder and the repair routing are
unaffected by which mode ran.
"""

from __future__ import annotations

from ...runtime import llm_step, observations, stage_services
from ...state import CfuDesignState
from . import conformance as oracle_author, delivery, hw as hardware, intrinsics, repair_scope, sw as software, unified

from ...flow import outcome


def implement(state: CfuDesignState) -> CfuDesignState:
    """Deliver both sides of the accepted ISA contract.

    The mode is a property of the environment, not a user-facing switch: a run
    without tools still has to be able to implement something, and a run with
    tools gets the free loop.
    """
    client = llm_step._llm(state)
    if (llm_step._tools_enabled(state, client) and not state.get("dry_run")
            and state.get("apply_patch", False) and state.get("run_commands", False)):
        return unified.implement_with_tools(state, client=client)
    # An outer transaction spans HW, SW, the host header, and fixed oracle/C;
    # per-source protocol atomicity alone cannot undo an accepted HW half when
    # the later SW/model/oracle delivery fails.
    transaction = None
    toolbox = None
    if state.get("run_root") and state.get("apply_patch") and not state.get("dry_run"):
        toolbox = stage_services._toolbox_for_stage(state, "implementation")
        transaction = delivery.DeliveryTransaction(state, toolbox)
    update = {}
    try:
        need_hw = need_sw = True
        if state.get("repair_scope") == "conformance" and toolbox is not None:
            need_hw, need_sw = repair_scope.required_domains(
                state, toolbox, observations.failure_index(state))
            if not need_sw:
                exercises, _why = intrinsics._software_exercises_cfu(
                    state, toolbox, stage_services._software_targets(state, toolbox))
                need_sw = not exercises
        if need_hw:
            update = hardware.implement_hw({**state, "_implementation_domain": "hardware"})
            if _rejected(update, "implementation_hw", state):
                return _rollback(update, state, transaction)
        if need_sw:
            # Apply append reducers as well as scalars so SW sees the current
            # hardware plan, gate, changed paths and accumulated evidence.
            current = outcome.merge_updates(state, update)
            sw_update = software.implement_sw({**current, "_implementation_domain": "software"})
            update = outcome.merge_updates(update, sw_update)
            if _rejected(sw_update, "implementation_sw", state):
                return _rollback(update, state, transaction)
        oracle_update = oracle_author.write_conformance(outcome.merge_updates(state, update))
        update = outcome.merge_updates(update, oracle_update)
        if oracle_update.get("errors"):
            return _rollback(update, state, transaction)
        return update
    except Exception:
        if transaction is not None:
            transaction.rollback()
        raise


def _rejected(update, rung, state):
    gate = (update.get("gates") or {}).get(rung)
    return bool(not state.get("dry_run") and state.get("apply_patch", False) and gate is not None
                and (not gate.get("ok") or gate.get("errors")))


def _rollback(update, state, transaction):
    if transaction is None:
        return update
    transaction.rollback()
    update = dict(update)
    update["conformance_delivery"] = state.get("conformance_delivery") or {}
    for key in ("changed_design_files", "changed_sw_files"):
        if key in update:
            update[key] = []
    gates = dict(update.get("gates") or {})
    for rung in ("implementation_hw", "implementation_sw"):
        if rung in gates and gates[rung] != (state.get("gates") or {}).get(rung):
            gate = dict(gates[rung])
            gate.update(ok=False, applied=False, files=[], candidate_hash="", rolled_back=True)
            gates[rung] = gate
    if gates:
        update["gates"] = gates
    return update
