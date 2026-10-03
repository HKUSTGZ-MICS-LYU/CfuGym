"""Shared schemas and rendering for bounded coding capabilities."""

from __future__ import annotations

from typing import Any


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
