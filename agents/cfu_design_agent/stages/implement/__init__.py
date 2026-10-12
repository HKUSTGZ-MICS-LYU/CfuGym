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

from ...flow import gates, ladder, outcome


def implement(state: CfuDesignState) -> CfuDesignState:
    """Deliver both sides of the accepted ISA contract.

    The mode is a property of the environment, not a user-facing switch: a run
    without tools still has to be able to implement something, and a run with
    tools gets the free loop.
    """
    # A previous rewind is historical, not this attempt's successor (including
    # rejected deliveries). Removing it prevents a free replay of the old route.
    if "implementation_route" in (state.get("gates") or {}):
        state = {**state, "repair_route": {}, "gates": {
            key: value for key, value in state["gates"].items() if key != "implementation_route"}}
    # Legacy/prepared-source corruption is a host failure, not a model repair:
    # refuse it before spending an implementation generation.
    if ((state.get("benchmark_project") or {}).get("kind") == "embench_benchmark"
            or (state.get("workload_spec") or {}).get("source_kind") == "embench"):
        integrity_box = stage_services._toolbox_for_stage(state, "implementation")
        integrity_problems = stage_services.benchmark_integrity_problems(integrity_box)
        if integrity_problems:
            raise ValueError("benchmark integrity: " + "; ".join(integrity_problems))
    client = llm_step._llm(state)
    if (llm_step._tools_enabled(state, client) and not state.get("dry_run")
            and state.get("apply_patch", False) and state.get("run_commands", False)):
        return _reattribute_conformance_round(
            state, unified.implement_with_tools(state, client=client))
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
            if update.get("repair_route"):
                return _rewind(state, update, transaction)
            if _rejected(update, "implementation_hw", state):
                return _rollback(update, state, transaction)
        if need_sw:
            # Apply append reducers as well as scalars so SW sees the current
            # hardware plan, gate, changed paths and accumulated evidence.
            current = outcome.merge_updates(state, update)
            sw_update = software.implement_sw({**current, "_implementation_domain": "software"})
            update = outcome.merge_updates(update, sw_update)
            if sw_update.get("repair_route"):
                return _rewind(state, update, transaction)
            if _rejected(sw_update, "implementation_sw", state):
                return _rollback(update, state, transaction)
        if toolbox is not None:
            problems = stage_services.hardware_integrity_problems(state, toolbox)
            if problems:
                note = "hardware integrity: " + "; ".join(problems)
                current = outcome.merge_updates(state, update)
                update = outcome.merge_updates(update, {
                    "errors": [note], "gates": outcome.gates(current, {
                        "implementation_hw": {"ok": False, "applied": False, "errors": [note]},
                    }),
                })
                return _rollback(update, state, transaction)
        if state.get("require_conformance", True):
            oracle_update = oracle_author.write_conformance(outcome.merge_updates(state, update))
            update = outcome.merge_updates(update, oracle_update)
            if oracle_update.get("errors"):
                return _rollback(update, state, transaction)
        return _reattribute_conformance_round(state, update)
    except Exception:
        if transaction is not None:
            transaction.rollback()
        raise


def _rewind(state, update, transaction):
    if transaction is not None:
        transaction.rollback()
    transition = ladder.repair_route_transition(state, update["repair_route"])
    return {**update, **transition}


def _reattribute_conformance_round(state: CfuDesignState, update: dict) -> dict:
    """Charge an oracle-only conformance round to the answer budget.

    A conformance-scope round is entered before anyone knows which side is
    wrong, so the ladder charges it to the design budget. When the delivered
    change revised only the oracle, the round was answer work: refund the design
    slot and charge the separate answer pool instead. Bounded by that pool, so
    oracle churn never buys unbounded free rounds. Both implement modes funnel
    through this one place, so they cannot disagree about what the round cost.
    """
    if state.get("repair_scope") != "conformance":
        return update
    if update.get("changed_design_files") or update.get("changed_sw_files"):
        return update
    if not update.get("conformance_delivery"):
        return update
    delivered = update.get("conformance_delivery") or {}
    approved = state.get("conformance_delivery") or {}
    if delivered.get("oracle_sha") == approved.get("oracle_sha"):
        return update
    used = int(state.get("answer_repairs_used", 0) or 0)
    if used >= gates.answer_repairs_max(state):
        return update
    update = dict(update)
    update["repairs_used"] = max(0, int(state.get("repairs_used", 0) or 0) - 1)
    update["answer_repairs_used"] = used + 1
    update["iteration_history"] = [
        *(update.get("iteration_history") or []),
        "Oracle-only repair charged to the answer-repair budget, not the design budget.",
    ]
    return update


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
