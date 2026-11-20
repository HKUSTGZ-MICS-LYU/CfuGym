"""Offline optimization identities and evidence: unit tests, no graph/LLM/toolchain."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

from agents.cfu_design_agent.flow import gates, ladder, optimize
from agents.cfu_design_agent.runtime import dependencies, stage_services
from agents.cfu_design_agent.stages import review
from agents.cfu_design_agent.stages.review import optimize as planner


class OptimizationIdentityTest(unittest.TestCase):
    def setUp(self):
        self.old = {"round": 0, "target": "baseline", "improved": True,
                    "validated": True, "measured": True, "speedup_x100": 200,
                    "reference_cycles": 100, "cfu_cycles": 50, "cost": {"cells": 100}}
        self.incumbent = {"round": 0, "manifest": "immutable-old", "files": []}
        self.state = {"iteration": 2, "max_iters": 5, "verify_submissions": 2,
                      "optimization_round": 1, "optimization_rounds": [self.old],
                      "optimization_plan": {"round": 1, "target": "implement", "reason": "batch"},
                      "best_candidate": self.incumbent, "delivered_round": 0,
                      "repair_entry": {"kind": "design", "target": "implement", "iteration": 2,
                                       "route_used": False}, "gates": {}, "loop_optimize": True}
        self.route = {"target": "contract", "reason": "contract repair",
                      "failure_ids": ["failure-1"], "diagnosis": {}}
        for name in ("_artifacts_for_stage", "_report_review_decision"):
            patcher = mock.patch.object(stage_services, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(planner, "_choose_optimization", return_value=("", {}, ""))
        patcher.start()
        self.addCleanup(patcher.stop)

    def passing(self, state, *, cycles=25):
        current = copy.deepcopy(state)
        current["gates"] = {name: {"ok": True} for name in gates.REQUIRED_GATES}
        current["gates"]["sim"].update(round=1, reference_cycles=100, cfu_cycles=cycles,
                                           speedup_x100=10000 // cycles)
        current["gates"]["cost"].update(round=1)
        current["cost_results"] = [{"returncode": 0, "parsed": {"cost": {"cells": 90}}}]
        return current

    def test_rewind_preserves_incumbent_and_pending_round(self):
        for target in ("analysis", "contract"):
            with self.subTest(target=target):
                extra = ladder.repair_route_transition(self.state, {**self.route, "target": target})
                current = {**self.state, **extra}
                self.assertEqual(current["best_candidate"], self.incumbent)
                self.assertEqual(current["optimization_plan"], self.state["optimization_plan"])
                self.assertEqual(current["delivered_round"], 0)
                self.assertNotIn("sim", current["gates"])

    def test_success_after_rewind_records_new_metrics_not_old_entry(self):
        for last_improved in (True, False):
            with self.subTest(last_improved=last_improved):
                state = copy.deepcopy(self.state)
                if not last_improved:
                    state["optimization_rounds"].append({**self.old, "round": 1, "improved": False})
                    state["optimization_round"] = 2
                    state["optimization_plan"]["round"] = 2
                extra = ladder.repair_route_transition(state, self.route)
                current = self.passing({**state, **extra})
                index = len(state["optimization_rounds"])
                current["gates"]["sim"]["round"] = index
                current["gates"]["cost"]["round"] = index
                with mock.patch.object(planner, "_snapshot_candidate", return_value={"round": index}) as snapshot:
                    result = review.review_and_route(current)
                entry = snapshot.call_args.args[1]
                self.assertEqual(entry["round"], index)
                self.assertEqual(entry["speedup_x100"], 400)
                self.assertEqual(entry["cfu_cycles"], 25)
                self.assertEqual(len(result["optimization_rounds"]), index + 1)
                self.assertEqual(result["delivered_round"], index)
                self.assertEqual(result["optimization_rounds"][:index], state["optimization_rounds"])

    def test_extension_rewind_becomes_full_redesign_without_losing_round(self):
        state = {**self.state, "optimization_plan": {"round": 1, "target": "extend_isa", "reason": "batch"},
                 "isa_added_symbols": ["cfu_batch"]}
        extra = ladder.repair_route_transition(state, self.route)
        self.assertEqual(extra["optimization_plan"]["round"], 1)
        self.assertEqual(extra["optimization_plan"]["target"], "implement")
        self.assertEqual(extra["optimization_plan"]["original_target"], "extend_isa")
        self.assertEqual(extra["isa_added_symbols"], [])

    def test_review_without_new_round_cannot_resnapshot_historical_entry(self):
        state = self.passing({**self.state, "optimization_plan": {}})
        with mock.patch.object(planner, "_snapshot_candidate") as snapshot:
            result = review.review_and_route(state)
        snapshot.assert_not_called()
        self.assertEqual(result["best_candidate"], self.incumbent)
        self.assertEqual(result["optimization_rounds"], state["optimization_rounds"])

    def test_failed_or_dominated_rewind_keeps_previous_delivery(self):
        extra = ladder.repair_route_transition(self.state, self.route)
        for failed in (True, False):
            with self.subTest(failed=failed):
                current = self.passing({**self.state, **extra}, cycles=80)
                current["cost_results"][-1]["parsed"]["cost"]["cells"] = 150
                current["gates"]["cost"]["ok"] = not failed
                with mock.patch.object(planner, "_snapshot_candidate") as snapshot:
                    result = review.review_and_route(current)
                snapshot.assert_not_called()
                self.assertEqual(result["best_candidate"], self.incumbent)
                self.assertEqual(result["delivered_round"], 0)
                self.assertFalse(result["optimization_rounds"][-1]["improved"])


class OptimizationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.state = {"optimization_round": 1, "verify_submissions": 3, "evaluation_id": "eval-new",
                      "input_fingerprints": {"sim": "sim-new", "cost": "cost-new"},
                      "repair_evidence_epoch": {"validation_results": 4},
                      "gates": {rung: {"ok": True} for rung in gates.REQUIRED_GATES}}
        for rung in ("sim", "cost"):
            self.state["gates"][rung].update(round=1, verify_submission=3, evaluation_id="eval-new",
                input_sha256=f"{rung}-new", evidence_epoch={"validation_results": 4})
        self.state["gates"]["sim"].update(reference_cycles=100, cfu_cycles=50, speedup_x100=200,
            correctness_ok=True, timing_ok=True, performance_valid=True, terminal_reason="passed")
        self.state["gates"]["cost"]["metrics"] = {"cells": 90, "wire_bits": 12}
        self.state["cost_results"] = [{"returncode": 0, "parsed": {"cost": {"cells": 999}}}]

    def snapshot(self, state):
        # These are synthetic fingerprints, not files in a runnable workspace.
        with mock.patch.object(dependencies, "completed_gate_inputs", return_value={}):
            return optimize._score_snapshot(state, target="implement", reason="test", source="test", round_index=1)

    def test_gate_local_cost_wins_over_historical_command_results(self):
        entry = self.snapshot(self.state)
        self.assertTrue(entry["validated"])
        self.assertEqual(entry["cost"]["cells"], 90)
        self.assertIsNone(entry["cost"]["memory_bits"])
        self.assertEqual(entry["evaluation_id"], "eval-new")
        self.assertEqual(entry["verify_submission"], 3)

    def test_failed_cost_cannot_fall_back_to_old_success(self):
        state = copy.deepcopy(self.state)
        state["gates"]["cost"].update(ok=False, metrics={})
        state["cost_results"].append({"returncode": 1})
        self.assertEqual(self.snapshot(state)["cost"], {})
        self.assertNotIn("999", optimize.optimization_score(state))
        # Old gate payloads without IDs must not search past the final failure either.
        state = {"gates": {"cost": {"ok": True}}, "cost_results": state["cost_results"]}
        self.assertEqual(optimize._cost_metric(state), {})

    def test_same_round_stale_or_wrong_identity_is_not_measured(self):
        changes = ({"stale": True}, {"skipped": True}, {"cancelled": True},
                   {"verify_submission": 2}, {"evaluation_id": "eval-old"},
                   {"input_sha256": "sim-old"}, {"evidence_epoch": {}}, {"round": 0})
        for change in changes:
            with self.subTest(change=change):
                state = copy.deepcopy(self.state)
                state["gates"]["sim"].update(change)
                entry = self.snapshot(state)
                self.assertFalse(entry["measured"])
                self.assertFalse(entry["validated"])
                self.assertIsNone(entry["cfu_cycles"])
                self.assertIsNone(entry["speedup_x100"])
                self.assertIn("no cycle measurement", optimize.optimization_score(state))

    def test_invalid_performance_is_not_scored_but_speedup_floor_remains_measured(self):
        state = copy.deepcopy(self.state)
        state["gates"]["sim"].update(ok=False, performance_valid=False, correctness_ok=False)
        self.assertFalse(self.snapshot(state)["measured"])
        state["gates"]["sim"].update(correctness_ok=True, performance_valid=True,
            terminal_reason="speedup_floor", speedup_x100=100, cfu_cycles=100)
        entry = self.snapshot(state)
        self.assertTrue(entry["measured"])
        self.assertFalse(entry["validated"])
        self.assertEqual(entry["cfu_cycles"], 100)

    def test_legacy_gate_without_submission_proof_remains_historical(self):
        state = copy.deepcopy(self.state)
        for rung in ("sim", "cost"):
            state["gates"][rung].pop("verify_submission")
            state["gates"][rung].pop("evidence_epoch")
        state["gates"]["cost"].pop("metrics")
        entry = self.snapshot(state)
        self.assertFalse(entry["measured"])
        self.assertFalse(entry["validated"])
        self.assertEqual(entry["cost"], {})
        self.assertEqual(state["gates"]["sim"]["cfu_cycles"], 50)

    def test_stale_or_mismatched_cost_is_absent(self):
        for change in ({"stale": True}, {"verify_submission": 2}, {"evaluation_id": "old"},
                       {"input_sha256": "old"}, {"evidence_epoch": {}}):
            with self.subTest(change=change):
                state = copy.deepcopy(self.state)
                state["gates"]["cost"].update(change)
                self.assertEqual(self.snapshot(state)["cost"], {})
                self.assertFalse(self.snapshot(state)["validated"])


if __name__ == "__main__":
    unittest.main()
