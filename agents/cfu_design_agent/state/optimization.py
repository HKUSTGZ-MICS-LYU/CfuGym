"""Optimization: The --loop island: ISA revisions and measured rounds."""

from __future__ import annotations

from typing import Any, TypedDict


class Optimization(TypedDict, total=False):
    """The --loop island: ISA revisions and measured rounds."""

    isa_revision: int
    isa_extension_base: dict[str, Any]
    isa_added_symbols: list[str]
    best_candidate: dict[str, Any]
    delivered_round: int
    optimization_plan: dict[str, Any]
    optimization_round: int
    optimization_rounds: list[dict[str, Any]]
