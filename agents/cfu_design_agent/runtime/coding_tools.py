"""Shared schemas and rendering for bounded coding capabilities."""

from __future__ import annotations

import json
import re
from functools import wraps
from pathlib import Path
from typing import Any


REPAIR_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "function", "function": {
        "name": "request_repair_route",
        "description": ("After a current repair diagnosis, choose analysis/contract to abandon "
                        "this transaction and rewind, or implement to keep fixing this contract. "
                        "Rewinds reuse an already-paid design repair; otherwise they use the separate pre-VERIFY repair budget."),
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "target": {"type": "string", "enum": ["analysis", "contract", "implement"]},
                "reason": {"type": "string"},
                "failure_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["target", "reason", "failure_ids"],
        },
    },
}


def validate_repair_route(value: Any, *, diagnosis: dict | None,
                          failure_index: dict, repair_mode: bool) -> dict:
    """Validate selection against the accepted diagnosis and live failure index."""
    if not repair_mode or not diagnosis:
        raise ValueError("a repair route requires an accepted current repair diagnosis")
    if not isinstance(value, dict) or set(value) != {"target", "reason", "failure_ids"}:
        raise ValueError("repair route requires only target, reason and failure_ids")
    if value["target"] not in ("analysis", "contract", "implement"):
        raise ValueError("repair target must be analysis, contract or implement")
    reason, ids = value["reason"], value["failure_ids"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("repair route reason must be non-empty")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("repair route must cite current failure_ids")
    failures = failure_index.get("failures") or {}
    cited = diagnosis.get("failure_ids") or []
    if any(i not in cited or i not in failures or failures[i].get("status", "current") != "current"
           for i in ids) or not cited or any(
               i not in failures or failures[i].get("status", "current") != "current" for i in cited):
        raise ValueError("repair route evidence/diagnosis must cite only current diagnosed failure IDs")
    return {"target": value["target"], "reason": reason.strip(), "failure_ids": list(dict.fromkeys(ids)),
            "diagnosis": dict(diagnosis)}


def parse_repair_route(text: str) -> dict | None:
    """Plain delivery: REPAIR_ROUTE followed by a single JSON object."""
    marker = re.search(r"(?m)^REPAIR_ROUTE\s*:?\s*", text or "")
    if marker is None:
        return None
    body = text[marker.end():].lstrip()
    if body.startswith("```json"):
        body = body[len("```json"):].lstrip()
    try:
        value, _end = json.JSONDecoder().raw_decode(body)
    except (ValueError, TypeError) as exc:
        raise ValueError("REPAIR_ROUTE must contain a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("REPAIR_ROUTE must contain a JSON object")
    return value


DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_repair_diagnosis",
        "description": (
            "Submit the structured diagnosis required before a repair edit. "
            "Cite current failure IDs and map observed errors to exact changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "failure_ids": {"type": "array", "items": {"type": "string"}},
                "observed_errors": {"type": "array", "items": {"type": "string"}},
                "root_cause": {"type": "string"},
                "repair_mapping": {"type": "array", "items": {"type": "string"}},
                "regression_risks": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "failure_ids", "observed_errors", "root_cause",
                "repair_mapping", "regression_risks",
            ],
        },
    },
}


def render_check_result(result: dict[str, Any], *, label: str, final_gate: str) -> str:
    observation = result.get("failure_observation") or {}
    log_path = result.get("full_log_path", result.get("log_path", ""))
    if result.get("returncode") == 0:
        return (
            f"isolated {label} check passed (evidence only; {final_gate} will rerun)\n"
            f"full log: {log_path}"
        )
    return (
        f"isolated {label} check failed\n"
        f"failure: {observation.get('id', '')}\n"
        f"category: {observation.get('category', '')}\n"
        f"summary:\n{observation.get('summary', result.get('stderr_tail', ''))}\n"
        f"full log: {log_path}"
    )[:8000]


def audit_hardware_result(state: dict, toolbox, result: dict, rtl_path: Path) -> dict:
    """Evidence-only tools share the final hw_build source/RTL gate."""
    from . import stage_services
    if result.get('returncode') != 0:
        return result
    if not rtl_path.is_file() or not rtl_path.stat().st_size:
        return {**result, 'returncode': 1, 'stderr_tail': 'hardware check produced no non-empty RTL'}
    evidence = {}
    problems = stage_services.hardware_integrity_problems(state, toolbox, evidence=evidence, rtl_path=rtl_path)
    return {**result, 'instruction_encoding': evidence,
            **({'returncode': 1, 'stderr_tail': 'hardware integrity: ' + '; '.join(problems)} if problems else {})}


def build_soc_check(state: dict, toolbox) -> dict:
    """Run the host SoC command and both independent hardware integrity checks."""
    from . import commands, effects, stage_services
    problems = stage_services.hardware_integrity_problems(state, toolbox)
    if problems:
        return {'returncode': 1, 'stderr_tail': 'hardware integrity: ' + '; '.join(problems)}
    box = stage_services._toolbox_for_stage(state, 'validation')
    argv = (commands.agent_cfu_overlay_commands(state, state.get('isa_spec') or {})
            or commands._legacy_ladder_commands(state, box)['generate'])
    if not argv:
        return {'returncode': 1, 'stderr_tail': 'no SoC generation command'}
    results = [effects.run_command(state, box, command, stage='validation', rung='hw_build',
               command_id='soc_generate_agent', timeout=3600, mutate=True) for command in argv]
    result = next((r for r in results if r.get('returncode') != 0), results[-1])
    rtl = commands._soc_rtl_path(state)
    return audit_hardware_result(state, box, result, rtl) if rtl is not None else result


def delivery_failure_gate(function):
    """Catch only model delivery failures AFTER all transaction decorators unwind."""
    from . import failures
    @wraps(function)
    def wrapped(state, *args, **kwargs):
        try:
            return function(state, *args, **kwargs)
        except failures.StageDeliveryFailure as exc:
            return stage_delivery_failure_update(state, exc)
    return wrapped


def stage_delivery_failure_update(state: dict, exc) -> dict:
    """A known model-delivery failure is a failed rung, not infrastructure success."""
    from . import llm_step, stage_services
    from ..flow import outcome
    reason = exc.reason
    integrity = 'hardware_integrity/' in reason or 'hardware integrity' in reason
    terminal = 'hardware_integrity' if integrity else exc.terminal_reason
    failed = {'ok': False, 'applied': False, 'files': [], 'errors': [reason],
              'terminal_reason': terminal, 'failure_kind': exc.failure_kind}
    return {'errors': [reason], 'stage_jobs': 'validation',
            'changed_design_files': [], 'changed_sw_files': [],
            'validation_results': [stage_services.gate_failure_result(state, 'implementation_hw', terminal, [reason])],
            'tool_calls': [llm_step._tool_call_summary(c) for c in exc.tool_calls],
            'gates': outcome.gates(state, {'implementation_route': {'ok': True, 'next': 'generate_soc'},
                                         'implementation_hw': dict(failed), 'implementation_sw': dict(failed)}),
            'iteration_history': [f'Stage 3 delivery rejected ({terminal}); transaction rolled back.']}
