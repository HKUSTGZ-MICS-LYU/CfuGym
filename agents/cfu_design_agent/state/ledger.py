"""Ledger: Gate verdicts, budgets, accumulators and the accounting for both."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict
from .schemas import CommandResult


class Ledger(TypedDict, total=False):
    """Gate verdicts, budgets, accumulators and the accounting for both."""

    termination_reason: str
    input_fingerprints: dict[str, str]
    iteration: int
    validation_results: Annotated[list[CommandResult], operator.add]
    cost_results: Annotated[list[CommandResult], operator.add]
    iteration_history: Annotated[list[str], operator.add]
    # Validation-ladder bookkeeping. ``attempts`` counts repairs per rung (the
    # simulation escalation reads ``attempts["sim"]``), ``repairs_used`` is the
    # design-repair budget consumed, ``answer_repairs_used`` is the separate
    # budget for answer-only repairs (malformed conformance oracle / ISA
    # contract) so they cannot exhaust the design budget, and
    # ``last_repair_target`` records the most recent repair for reporting.
    attempts: dict[str, int]
    repairs_used: int
    answer_repairs_used: int
    # Corrective retries for deterministically rejected deliveries; these do
    # not consume the design-repair budget (``repairs_used``).
    delivery_retries: int
    last_repair_target: str
    # Historical command evidence before an upstream rewind is no longer current.
    repair_evidence_epoch: dict[str, int]
    repair_route: dict[str, Any]
    # Provenance of the current repair entry; a layer choice can reuse one paid
    # design slot, never an answer/delivery retry and never the same slot twice.
    repair_entry: dict[str, Any]
    stopped_at: str
    final_report_path: str
    status: Literal["initialized", "needs_iteration", "failed", "complete", "blocked", "planned"]
    errors: Annotated[list[str], operator.add]
    graph_events: Annotated[list[str], operator.add]
    gates: dict[str, dict[str, Any]]
    token_usage: dict[str, int]
    llm_raw_outputs: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[dict], operator.add]
    # (response_hash, error_hash) per software attempt; two identical consecutive
    # entries mean the repair prompt produced the same answer again.
    sw_attempts: Annotated[list[dict], operator.add]
    # The same ledger for the hardware rung: a byte-identical stuck retry must
    # switch delivery strategy instead of burning the whole repair budget.
    hw_attempts: Annotated[list[dict], operator.add]
    # Rejected conformance-oracle attempts (problems_hash): a rejection that
    # recurs means the repair is oscillating between two validation errors.
    conformance_attempts: Annotated[list[dict], operator.add]
    progress_summary: dict[str, Any]
