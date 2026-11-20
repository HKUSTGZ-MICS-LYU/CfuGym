"""Admission and identity for one candidate's sim/cost evaluation attempt.

The ledger is a native replacement state channel. A replay derives the same ID
and restores normal LangGraph task results; it does not create another charge.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from . import outcome
from ..runtime import dependencies, effects


def attempts(state: dict[str, Any]) -> list[dict[str, Any]]:
    if "evaluation_attempts" in state:
        return list(state["evaluation_attempts"] or [])
    # Legacy checkpoints did not count attempts. Never pretend that their known
    # evaluations were free. Submission count is a conservative migration bound.
    measured = sum(bool(item.get("measured")) for item in state.get("optimization_rounds") or [])
    ids = {str(result["evaluation_id"]) for key in ("validation_results", "cost_results")
           for result in state.get(key) or [] if result.get("evaluation_id")}
    observed = bool(measured or ids or (state.get("evaluation") or {}).get("id")
                    or any(result.get("rung") in {"sim", "cost"}
                           and result.get("returncode") is not None
                           for key in ("validation_results", "cost_results") for result in state.get(key) or []))
    count = max(measured, len(ids), int(state.get("verify_submissions", 0) or 0), 1) if observed else 0
    return [{"id": f"legacy-evaluation-{index + 1}", "legacy": True} for index in range(count)]


def used(state: dict[str, Any]) -> int:
    return len(attempts(state))


def remaining(state: dict[str, Any]) -> bool:
    cap = max(0, int(state.get("max_evals", 0) or 0))
    return not cap or used(state) < cap


def enabled(state: dict[str, Any], rung: str) -> bool:
    if state.get("dry_run") or not state.get("run_commands"):
        return False
    if rung == "sim":
        return not state.get("skip_sim")
    if state.get("implementation_design_type") == "draft":
        return False
    return not state.get("skip_cost") and not (
        state.get("skip_yosys") and state.get("cost_backend", "yosys") != "vivado")


def admission(state: dict[str, Any], rung: str = "prepare") -> dict[str, Any] | None:
    """Reserve once for matching candidate inputs; None means budget exhausted."""
    fingerprints = dependencies.input_fingerprints(state)
    identity = {"verify_submission": int(state.get("verify_submissions", 0) or 0),
                "iteration": int(state.get("iteration", 0) or 0),
                "round": int(state.get("optimization_round", 0) or 0),
                "evidence_epoch": dict(state.get("repair_evidence_epoch") or {}),
                "lanes": [lane for lane in ("sim", "cost") if enabled(state, lane)],
                "inputs": {rung: fingerprints.get(rung, "") for rung in ("sim", "cost")}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ledger = attempts(state)
    existing = next((entry for entry in ledger if entry.get("id") == state.get("evaluation_id")
                     and entry.get("key") == key), None)
    completed = (existing or {}).get("completed_lanes") or []
    if existing and rung not in completed and not (rung in {"prepare", "validation_pair"} and completed):
        return {"evaluation_id": existing["id"], "evaluation_attempts": ledger}
    if not remaining(state):
        return None
    evaluation_id = f"eval-{len(ledger) + 1:06d}-{key[:24]}"
    return {"evaluation_id": evaluation_id,
            "evaluation_attempts": [*ledger, {"id": evaluation_id, "key": key, **identity}]}


def prepare(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Persist admission in the existing software-build checkpoint boundary."""
    gate = (update.get("gates") or {}).get("sw_build") or {}
    if gate.get("next", "simulate") != "simulate" or not (gate.get("ok") or gate.get("skipped")):
        return update
    current = outcome.merge_updates(state, update)
    if not any(enabled(current, rung) for rung in ("sim", "cost")):
        return update
    reservation = admission(current)
    if reservation is not None:
        return {**update, **reservation}
    note = f"Evaluation budget exhausted (--max-evals {state.get('max_evals')}); no validation started."
    return {**update, "stopped_at": "evaluation_budget", "termination_reason": "BUDGET_EXHAUSTED",
            "stage_jobs": "review", "errors": [*(update.get("errors") or []), note],
            "gates": {**update.get("gates", {}), "sw_build": {**gate, "next": "review_and_route",
                                                               "evaluation_budget_exhausted": True}}}


def run(state: dict[str, Any], rung: str, execute: bool,
        callback: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Admission must precede any worker/task or external command side effect."""
    if not execute:
        return callback(state)
    reservation = admission(state, rung)
    if reservation is None:
        note = f"Evaluation budget exhausted (--max-evals {state.get('max_evals')}); no validation started."
        return {"stopped_at": "evaluation_budget", "stage_jobs": "review",
                "termination_reason": "BUDGET_EXHAUSTED", "errors": [note],
                **outcome.stage_result(state, rung=rung, ok=False, next="review_and_route",
                                       reason="evaluation_budget", history=note)}
    current = {**state, **reservation}
    with effects.use_state(current):
        update = callback(current)
    ticket = next(entry for entry in reservation["evaluation_attempts"] if entry["id"] == reservation["evaluation_id"])
    finished = {rung} | (set(ticket.get("lanes") or []) if rung == "validation_pair" else set())
    reservation["evaluation_attempts"] = [
        {**entry, "completed_lanes": sorted(set(entry.get("completed_lanes") or []) | finished)}
        if entry["id"] == reservation["evaluation_id"] else entry
        for entry in reservation["evaluation_attempts"]]
    return {**update, **reservation}
