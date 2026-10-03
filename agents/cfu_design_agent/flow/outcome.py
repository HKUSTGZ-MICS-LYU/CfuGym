"""One vocabulary for a stage outcome: its gate, its history line, its rewind.

Every rung used to assemble ``"gates": {**state.get("gates", {}), rung: {...}}``
by hand, with a slightly different key set each time, and the "where does the run
go next" decision buried in the same dict as the measurements. Two things were
impossible to check as a result: that ``next`` names a real flow node, and that a
repair carries the evidence it was decided on.

``gate`` / ``stage_result`` keep the payload shape exactly as it was (a field is
present only when the caller passes it, so stored artifacts do not change) while
giving ``next`` a validated home and the rewind a typed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .wiring import NODE_NAMES

#: Nodes a rung may hand control to. ``_route_*`` routers read ``gate["next"]``,
#: so an unknown value would silently strand the run on a nonexistent node.
VALID_NEXT = frozenset(NODE_NAMES)


@dataclass(frozen=True)
class Rewind:
    """A rung's decision to send the run back to an earlier stage.

    The evidence is what the decision was made on — failure ids, log references,
    measured numbers — so a repair prompt (and the artifact trail) can point at
    it instead of re-deriving it from a reason string.
    """

    target: str
    reason: str = ""
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target not in VALID_NEXT:
            raise ValueError(f"rewind target is not a flow node: {self.target!r}")

    def fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"next": self.target}
        if self.reason:
            payload["reason"] = self.reason
        if self.evidence:
            payload["rewind_evidence"] = list(self.evidence)
        return payload


def merge_updates(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Fold two nodes' state updates into one, honouring the append reducers.

    Concatenating the reducer channels is not optional: LangGraph applies a
    reducer once per returned value, so handing back two updates' worth of
    history as a single value would *replace* the accumulated list instead of
    appending to it, silently losing the first node's ledger.
    """
    from ..state import append_keys

    merged = {**first, **second}
    for key in append_keys():
        left, right = first.get(key), second.get(key)
        if left or right:
            merged[key] = [*(left or []), *(right or [])]
    if first.get("gates") or second.get("gates"):
        merged["gates"] = {**(first.get("gates") or {}), **(second.get("gates") or {})}
    return merged


def gate(state: dict[str, Any], rung: str, *, ok: bool, next: str | None = None,
         reason: str | None = None, errors: Iterable[str] | None = None,
         skipped: bool | None = None, rewind: Rewind | None = None,
         **fields: Any) -> dict[str, Any]:
    """The ``gates`` mapping for one rung's verdict, merged onto the state's.

    A field is written only when the caller passes it (``None`` means "omit"), so
    the recorded gate stays byte-identical to the dict the rung used to build by
    hand — including the cases that deliberately record ``next: ""``.
    """
    if not rung:
        raise ValueError("a gate needs its rung name")
    if next is not None:
        if next and next not in VALID_NEXT:
            raise ValueError(f"gate {rung!r} routes to an unknown node: {next!r}")
        fields.setdefault("next", next)
    if rewind is not None:
        for key, value in rewind.fields().items():
            fields.setdefault(key, value)
    payload: dict[str, Any] = {"ok": bool(ok)}
    if skipped is not None:
        payload["skipped"] = bool(skipped)
    if reason is not None:
        payload["reason"] = reason
    if errors is not None:
        payload["errors"] = list(errors)
    payload.update(fields)
    return {**state.get("gates", {}), rung: payload}


def gates(state: dict[str, Any], rungs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge several rung verdicts at once, for a node that reports more than one.

    The payloads are taken as given (a node reporting two gates has already
    decided both), but each ``next`` is still validated.
    """
    merged: dict[str, Any] = {**state.get("gates", {})}
    for rung, payload in rungs.items():
        if not rung:
            raise ValueError("a gate needs its rung name")
        target = payload.get("next")
        if target and target not in VALID_NEXT:
            raise ValueError(f"gate {rung!r} routes to an unknown node: {target!r}")
        merged[rung] = dict(payload)
    return merged


def stage_result(state: dict[str, Any], *, rung: str, ok: bool, history: str = "",
                 next: str | None = None, reason: str | None = None,
                 errors: Iterable[str] | None = None, skipped: bool | None = None,
                 rewind: Rewind | None = None, **fields: Any) -> dict[str, Any]:
    """One node's state update: the rung's gate plus its history line.

    Callers still add their own keys (``validation_results``, ``stopped_at``,
    ``status``, ...) on top; this only owns the part that was pure duplication.
    """
    update: dict[str, Any] = {
        "gates": gate(state, rung, ok=ok, next=next, reason=reason, errors=errors,
                      skipped=skipped, rewind=rewind, **fields),
    }
    if history:
        from ..state import append_history

        update["iteration_history"] = append_history(state, history)
    return update
