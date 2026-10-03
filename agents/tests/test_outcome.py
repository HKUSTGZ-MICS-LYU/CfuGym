"""The gate/rewind vocabulary: one shape, validated targets, typed evidence."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.flow import outcome
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


class GateTest(unittest.TestCase):
    def test_a_minimal_gate_records_only_ok(self) -> None:
        self.assertEqual(outcome.gate({}, "sim", ok=True), {"sim": {"ok": True}})

    def test_a_gate_merges_onto_the_existing_gates(self) -> None:
        state = {"gates": {"isa": {"ok": True}}}
        merged = outcome.gate(state, "sim", ok=False, errors=["boom"])
        self.assertEqual(merged["isa"], {"ok": True})
        self.assertEqual(merged["sim"], {"ok": False, "errors": ["boom"]})

    def test_it_reproduces_the_hand_written_payload_exactly(self) -> None:
        state = {"gates": {}}
        self.assertEqual(
            outcome.gate(state, "cost", ok=True, skipped=False, reason="", next="",
                         rtl="11_cost/cfu_rtl", round=2),
            {**state["gates"], "cost": {
                "ok": True, "skipped": False, "reason": "", "next": "",
                "rtl": "11_cost/cfu_rtl", "round": 2}},
        )

    def test_an_omitted_field_is_absent_not_empty(self) -> None:
        payload = outcome.gate({}, "sim", ok=True)["sim"]
        for absent in ("next", "reason", "errors", "skipped"):
            self.assertNotIn(absent, payload)

    def test_an_unknown_route_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown node"):
            outcome.gate({}, "sim", ok=False, next="implement_everything")

    def test_an_empty_route_is_allowed(self) -> None:
        self.assertEqual(outcome.gate({}, "sim", ok=True, next="")["sim"]["next"], "")

    def test_a_gate_needs_its_rung(self) -> None:
        with self.assertRaises(ValueError):
            outcome.gate({}, "", ok=True)


class RewindTest(unittest.TestCase):
    def test_a_rewind_carries_its_target_reason_and_evidence(self) -> None:
        rewind = outcome.Rewind("implement", "conformance mismatch", ("failure-1", "log-9"))
        self.assertEqual(rewind.fields(), {
            "next": "implement",
            "reason": "conformance mismatch",
            "rewind_evidence": ["failure-1", "log-9"],
        })

    def test_an_unknown_rewind_target_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a flow node"):
            outcome.Rewind("nowhere")

    def test_a_rewind_fills_a_gate_without_overriding_explicit_fields(self) -> None:
        payload = outcome.gate({}, "hw_conformance", ok=False,
                               rewind=outcome.Rewind("implement", "mismatch"))["hw_conformance"]
        self.assertEqual(payload["next"], "implement")
        self.assertEqual(payload["reason"], "mismatch")
        self.assertNotIn("rewind_evidence", payload)


class StageResultTest(unittest.TestCase):
    def test_it_builds_the_gate_and_the_history_line(self) -> None:
        update = outcome.stage_result({}, rung="sim", ok=True, history="Ran the simulation.")
        self.assertEqual(update["gates"]["sim"], {"ok": True})
        self.assertEqual(update["iteration_history"], ["Ran the simulation."])

    def test_it_omits_the_history_key_when_there_is_none(self) -> None:
        self.assertNotIn("iteration_history", outcome.stage_result({}, rung="sim", ok=True))

    def test_extra_measurements_land_in_the_gate(self) -> None:
        update = outcome.stage_result({}, rung="sim", ok=True, speedup_x100=250,
                                      reference_cycles=1000, cfu_cycles=400)
        self.assertEqual(update["gates"]["sim"]["speedup_x100"], 250)
        self.assertEqual(update["gates"]["sim"]["cfu_cycles"], 400)

    def test_it_is_splattable_into_a_node_update(self) -> None:
        state = {"gates": {}}
        update = {"validation_results": [1], "stage_jobs": "review",
                  **outcome.stage_result(state, rung="cost", ok=False, skipped=True,
                                         reason="--skip-yosys", history="Skipped.")}
        self.assertEqual(update["validation_results"], [1])
        self.assertEqual(update["gates"]["cost"],
                         {"ok": False, "skipped": True, "reason": "--skip-yosys"})


if __name__ == "__main__":
    unittest.main()
