"""Typed execution failures and observed partial-state context.

Delivery failures are model outcomes; SDK/transport exceptions retain their
original type and traceback. Context is attached, never wrapped around them.
"""
from __future__ import annotations

from typing import Any


class StageDeliveryFailure(RuntimeError):
    """The bounded interaction ended without an independently accepted delivery."""

    def __init__(self, reason: str, *, stage: str = "", tool_calls=(),
                 terminal_reason: str = "delivery_rejected") -> None:
        super().__init__(f"native stage failed: {reason}")
        self.reason = reason
        self.stage = stage
        self.tool_calls = tuple(tool_calls)
        self.terminal_reason = terminal_reason
        self.failure_kind = "model_delivery_failure"


class StageToolsUnavailable(RuntimeError):
    """The requested execution capability could not be provided."""

    failure_kind = "infrastructure"


def attach_partial_state(exc: Exception, state: dict[str, Any], *, source: str,
                         stage: str = "") -> None:
    """Attach only observed state, preserving exception identity and traceback."""
    exc.partial_state = dict(state)
    exc.partial_state_source = source
    if stage:
        exc.failed_stage = stage


def termination(exc: Exception) -> dict[str, Any]:
    """Machine-readable exception provenance, without inventing a failed gate."""
    kind = getattr(exc, "failure_kind", "")
    name = type(exc).__name__
    if not kind:
        if name in {"ContextBudgetError"}:
            kind = "precondition"
        elif (type(exc).__module__.startswith(("openai", "httpx", "httpcore"))
              or isinstance(exc, (ConnectionError, TimeoutError))):
            kind = "infrastructure"
        else:
            kind = "runtime_exception"
    result = {"kind": kind, "exception_type": name, "exception_module": type(exc).__module__,
              "reason": str(exc), "partial": True}
    stage = getattr(exc, "failed_stage", "") or getattr(exc, "stage", "")
    if stage:
        result["stage"] = stage
    if getattr(exc, "partial_state_source", ""):
        result["state_source"] = exc.partial_state_source
    if exc.__cause__ is not None:
        result["cause"] = {"exception_type": type(exc.__cause__).__name__, "reason": str(exc.__cause__)}
    return result
