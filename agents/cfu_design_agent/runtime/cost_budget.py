"""Host-owned budget for a fixed generic Yosys mapping, not FPGA capacity."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_MAX_CELLS = 100_000
VERSION = 1


def max_cells(state: Mapping[str, Any]) -> int:
    """Return the positive run budget; missing legacy state uses the default."""
    value = state.get("max_yosys_cells", DEFAULT_MAX_CELLS)
    if type(value) is not int or value <= 0:
        raise ValueError("max_yosys_cells must be a positive integer")
    return value


def check(state: Mapping[str, Any], metrics: Any, *, top: str,
          summary_path: str) -> dict[str, Any]:
    """Fail closed on missing metrics; equality is within the cell budget.

    Callers run the fixed generic/techmap/ABC flow before passing its parsed
    ``cells`` here. The same basis is used by both cost backends. In particular,
    generic cells must never be advertised as FPGA LUTs, FFs or device capacity.
    """
    limit = max_cells(state)
    cells = metrics.get("cells") if isinstance(metrics, Mapping) else None
    verdict: dict[str, Any] = {
        "version": VERSION, "metric": "yosys_generic_mapped_cells", "top": top,
        "max_cells": limit, "cells": cells, "summary_path": summary_path,
        "ok": False,
    }
    if type(cells) is not int or cells < 0:
        # Do not persist arbitrary/non-serializable malformed metric values.
        verdict["cells"] = None
        verdict.update(code="cost_metrics_invalid", reason=(
            f"cost_metrics_invalid: Yosys cell count is missing or invalid for {top}; "
            f"Cost Wall cannot certify the {limit} generic-cell budget. "
            f"Obtain a valid Yosys cost report before continuing. Report: {summary_path}"))
    elif cells > limit:
        verdict.update(code="cost_budget_exceeded", reason=(
            f"cost_budget_exceeded: {top} uses {cells} mapped generic Yosys cells > limit {limit}. "
            "Reduce design complexity: fewer parallel lanes, share multipliers/ALUs, "
            "and use bounded multi-cycle datapaths instead of full unrolling. "
            "Preserve the required semantics and correctness; do not hardcode fixture results. "
            f"This is an engineering budget, not FPGA LUT capacity. Report: {summary_path}"))
    else:
        verdict.update(ok=True, code="", reason="")
    return verdict
