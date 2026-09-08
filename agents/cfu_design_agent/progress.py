"""Live stage progress reporting for agent runs."""

from __future__ import annotations

from typing import Any

# Human labels for the workflow nodes. Keys are node/function names.
STAGE_LABELS: dict[str, str] = {
    "load_context": "load repo context",
    "ingest_input": "ingest workload",
    "route_input": "route input",
    "route_input_python": "route input",
    "prepare_c_project_profile_node": "prepare C profile",
    "create_benchmark_project_node": "prepare benchmark",
    "instrument_and_profile": "profile (run)",
    "extract_hotspot": "extract hotspot",
    "analyze_workload": "analyze workload",
    "design_contract": "design CFU ISA",
    "implement_hw_sw": "implement HW/SW",
    "validate": "validate (sim)",
    "estimate_cost": "cost (yosys)",
    "review_and_route": "review & route",
    "review_and_route_python": "review & route",
    "final_report": "final report",
}


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


# Map a node/function name to its logical step slot (one of the stage order
# names) so the printed index stays within 1..N even for node names that differ
# from the human-facing stage slots.
NODE_TO_STEP: dict[str, str] = {
    "load_context": "load_context",
    "ingest_input": "ingest_input",
    "route_input": "route_input",
    "route_input_python": "route_input",
    "prepare_c_project_profile_node": "prepare_benchmark",
    "create_benchmark_project_node": "prepare_benchmark",
    "instrument_and_profile": "profile",
    "extract_hotspot": "hotspot",
    "analyze_workload": "analysis",
    "design_contract": "isa",
    "implement_hw_sw": "implementation",
    "validate": "validation",
    "estimate_cost": "cost",
    "review_and_route": "review",
    "review_and_route_python": "review",
    "final_report": "final_report",
}


class StageProgress:
    """Prints one line when a stage starts and one when it finishes.

    Line 1 (stage begins): ``[ 3/13] analyze workload ... running``
    Line 2 (stage ends):   ``[ 3/13] analyze workload ... OK (321 tok, 2.4s)``
    """

    def __init__(self, stages: list[str]) -> None:
        self.stages = list(stages)
        self.slot_by_step = {name: i for i, name in enumerate(self.stages)}
        self.count = len(self.stages)
        self.results: dict[str, dict[str, Any]] = {}
        self._begin_total = 0

    def _index_of(self, name: str) -> int:
        step = NODE_TO_STEP.get(name, name)
        return self.slot_by_step.get(step, len(self.stages))

    def begin(self, name: str) -> None:
        idx = self._index_of(name)
        self._begin_total = self._running_total()
        print(
            f"[{idx + 1:>2}/{self.count:>2}] {stage_label(name)} ... running",
            flush=True,
        )

    def end(self, name: str, *, ok: bool, elapsed: float, tokens: int) -> None:
        idx = self._index_of(name)
        status = "OK" if ok else "FAIL"
        print(
            f"[{idx + 1:>2}/{self.count:>2}] {stage_label(name)} ... "
            f"{status} ({tokens} tok, {elapsed:.1f}s)",
            flush=True,
        )
        self.results[name] = {
            "ok": ok,
            "elapsed": elapsed,
            "tokens": tokens,
        }

    @staticmethod
    def _running_total() -> int:
        # Imported lazily to avoid a circular import at module load.
        from .llm import usage_totals

        return usage_totals()["total"]

    def summary(self) -> dict[str, Any]:
        stages = list(self.results)
        ok = sum(1 for value in self.results.values() if value["ok"])
        tokens = sum(value["tokens"] for value in self.results.values())
        return {
            "stages_run": len(stages),
            "stages_ok": ok,
            "stages_failed": len(stages) - ok,
            "stage_tokens": tokens,
            "stages": {
                s: {"ok": v["ok"], "tokens": v["tokens"], "elapsed": v["elapsed"]}
                for s, v in self.results.items()
            },
        }
