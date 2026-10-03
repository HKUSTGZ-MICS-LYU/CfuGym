"""Live stage progress reporting for agent runs."""

from __future__ import annotations

from typing import Any

# Human labels for the workflow nodes. Keys are node/function names.
STAGE_LABELS: dict[str, str] = {
    "load_context": "load repo context",
    "ingest_input": "ingest workload",
    "route_input": "route input",
    "prepare_c_project_profile_node": "prepare C profile",
    "create_benchmark_project_node": "prepare benchmark",
    "instrument_and_profile": "profile (run)",
    "prepare_embench_benchmark_node": "prepare embench",
    "profile_embench": "profile embench",
    "extract_hotspot": "extract hotspot",
    "collect_ir_evidence": "collect LLVM IR evidence",
    "analyze_workload": "analyze workload",
    "design_contract": "design CFU ISA",
    "extend_isa": "extend CFU ISA",
    "implement": "implement HW/SW",
    "generate_soc": "generate SoC",
    "check_conformance": "check conformance",
    "build_sw": "compile SW",
    "simulate": "simulate",
    "estimate_cost": "cost (yosys)",
    "review_and_route": "review & route",
    "final_report": "final report",
}


# The macro stages of the run (docs/agent_v2_plan.md). The live log reports
# position *within a macro stage* rather than a flat step counter, because "which
# phase is this run in, and how far through it" is the question an operator
# actually has while a long toolchain run is in flight.
STEP_GROUPS: dict[str, str] = {
    "load_context": "DISCOVER",
    "ingest_input": "DISCOVER",
    "route_input": "DISCOVER",
    "prepare_benchmark": "DISCOVER",
    "profile": "DISCOVER",
    "hotspot": "DISCOVER",
    "ir": "DISCOVER",
    "analysis": "DISCOVER",
    "isa": "CONTRACT",
    "implement": "IMPLEMENT",
    "hw_build": "VERIFY",
    "hw_conformance": "VERIFY",
    "sw_build": "VERIFY",
    "sim": "VERIFY",
    "cost": "VERIFY",
    "review": "DECIDE",
    "final_report": "DECIDE",
}

GROUP_ORDER: tuple[str, ...] = ("DISCOVER", "CONTRACT", "IMPLEMENT", "VERIFY", "DECIDE")


def stage_group(step: str) -> str:
    return STEP_GROUPS.get(step, "OTHER")


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


# Map a node/function name to its logical step slot (one of the stage order
# names) so the printed index stays within 1..N even for node names that differ
# from the human-facing stage slots. Every registered graph node maps to exactly
# one slot, and each validation-ladder rung owns its slot, so a clean run prints
# strictly increasing indices (a repair loop legitimately revisits an earlier
# index).
NODE_TO_STEP: dict[str, str] = {
    "load_context": "load_context",
    "ingest_input": "ingest_input",
    "route_input": "route_input",
    "prepare_c_project_profile_node": "prepare_benchmark",
    "create_benchmark_project_node": "prepare_benchmark",
    "instrument_and_profile": "profile",
    "prepare_embench_benchmark_node": "prepare_benchmark",
    "profile_embench": "profile",
    "extract_hotspot": "hotspot",
    "collect_ir_evidence": "ir",
    "analyze_workload": "analysis",
    "design_contract": "isa",
    "extend_isa": "isa",
    "implement": "implement",
    "generate_soc": "hw_build",
    "check_conformance": "hw_conformance",
    "build_sw": "sw_build",
    "simulate": "sim",
    "estimate_cost": "cost",
    "review_and_route": "review",
    "final_report": "final_report",
}


class StageProgress:
    """Prints one line when a stage starts and one when it finishes.

    Line 1 (stage begins): ``[DISCOVER 8/8] analyze workload ... running``
    Line 2 (stage ends):   ``[DISCOVER 8/8] analyze workload ... OK (in=… out=… tok, 2.4s)``

    The counter is the position *within the macro stage* (see ``STEP_GROUPS``), so
    a repair that revisits an earlier rung legitimately moves the counter back
    instead of appearing to restart the run.
    """

    def __init__(self, stages: list[str]) -> None:
        self.stages = list(stages)
        self.slot_by_step = {name: i for i, name in enumerate(self.stages)}
        self.count = len(self.stages)
        self.results: dict[str, dict[str, Any]] = {}
        # Slot order inside each group, taken from the plan order rather than the
        # execution order, so the denominator is stable for the whole run.
        self.group_slots: dict[str, list[str]] = {
            group: [step for step in self.stages if stage_group(step) == group]
            for group in GROUP_ORDER
        }

    def _position(self, name: str) -> tuple[str, int, int]:
        step = NODE_TO_STEP.get(name, name)
        group = stage_group(step)
        slots = self.group_slots.get(group, [])
        index = slots.index(step) + 1 if step in slots else len(slots) + 1
        return group, index, len(slots)

    def _prefix(self, name: str) -> str:
        group, index, total = self._position(name)
        return f"[{group} {index}/{total}]"

    def _index_of(self, name: str) -> int:
        step = NODE_TO_STEP.get(name, name)
        return self.slot_by_step.get(step, len(self.stages))

    def begin(self, name: str) -> None:
        print(f"{self._prefix(name)} {stage_label(name)} ... running", flush=True)

    def end(self, name: str, *, ok: bool, elapsed: float, tokens: int) -> None:
        status = "OK" if ok else "FAIL"
        # Input/output split straight from the usage ledger: the point of the
        # line is to show which step's input is big at a glance.
        from .llm import stage_usage

        usage = stage_usage(name)
        print(
            f"{self._prefix(name)} {stage_label(name)} ... "
            f"{status} (in={usage['input']} out={usage['output']} tok, {elapsed:.1f}s)",
            flush=True,
        )
        self.results[name] = {
            "ok": ok,
            "elapsed": elapsed,
            "tokens": tokens,
            "input_tokens": int(usage.get("input", 0)),
            "output_tokens": int(usage.get("output", 0)),
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
                s: {
                    "ok": v["ok"],
                    "tokens": v["tokens"],
                    "input_tokens": v.get("input_tokens", 0),
                    "output_tokens": v.get("output_tokens", 0),
                    "elapsed": v["elapsed"],
                }
                for s, v in self.results.items()
            },
        }
