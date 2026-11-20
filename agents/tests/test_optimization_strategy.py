"""Bounded multi-objective search, hypotheses and source parents, all offline."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.cli import build_parser
from agents.cfu_design_agent.flow import instrument, optimize
from agents.cfu_design_agent.runtime import dependencies, llm_step, stage_services
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.stages import review
from agents.cfu_design_agent.stages.review import candidates, optimize as planner
from agents.tests.common import REPO_ROOT


def point(index, cycles, cells, **fields):
    return {"round": index, "target": "baseline" if index == 0 else "implement",
            "validated": True, "measured": True, "reference_cycles": 1000,
            "cfu_cycles": cycles, "speedup_x100": 100000 // cycles,
            "cost": {"cells": cells}, "improved": index == 0, **fields}


class OptimizationStrategyTest(unittest.TestCase):
    def test_frontier_keeps_tradeoffs_and_drops_dominated_and_equal_points(self):
        points = [point(0, 500, 1000), point(1, 300, 1050), point(2, 700, 800),
                  point(3, 600, 1100), point(4, 500, 1000), point(5, 200, 700, validated=False)]
        self.assertEqual([p["round"] for p in optimize.pareto_frontier(points)], [0, 1, 2])
        self.assertEqual(optimize.select_round(points, "performance")["round"], 1)
        self.assertEqual(optimize.select_round(points, "cost")["round"], 2)
        self.assertEqual(optimize.select_round(points, "balanced")["round"], 1)
        self.assertEqual(optimize.select_round(points, "pareto")["round"], 0)

    def test_exact_ratio_and_cost_basis_are_used(self):
        points = [point(0, 500001, 1000, reference_cycles=1000000),
                  point(1, 500000, 1000, reference_cycles=1000000)]
        self.assertEqual(optimize.select_round(points, "performance")["round"], 1)
        points[1]["cost"] = {"backend": "vivado", "metric": "cfu_post_route_lut", "lut": 1,
                             "part": "other", "target_frequency_mhz": 100}
        self.assertEqual(optimize.select_round(points, "performance")["round"], 0)

    def test_zero_area_is_a_valid_frontier_metric_for_every_objective(self):
        for backend in ("yosys", "vivado"):
            with self.subTest(backend=backend):
                points = [point(0, 500, 10), point(1, 250, 0)]
                if backend == "vivado":
                    for entry in points:
                        entry["cost"] = {"backend": "vivado", "metric": "cfu_post_route_lut",
                                         "part": "p", "target_frequency_mhz": 100,
                                         "lut": entry["cost"]["cells"]}
                for objective in ("performance", "cost", "balanced"):
                    self.assertEqual(optimize.select_round(points, objective)["round"], 1)

    def test_new_cli_defaults_are_explicit_and_legacy_mode_remains_available(self):
        args = build_parser().parse_args(["--loop"])
        self.assertEqual(args.loop_objective, "performance")
        self.assertEqual(args.loop_parent, "incumbent")
        args = build_parser().parse_args(["--loop-objective", "pareto", "--loop-parent", "current"])
        self.assertEqual((args.loop_objective, args.loop_parent), ("pareto", "current"))

    def test_hypothesis_is_bounded_and_reason_only_is_marked_incomplete(self):
        parsed = {"target": "implement", "reason": "Batch  inputs", "hypothesis": {
            "bottleneck": "transfers", "change": "Batch  inputs", "evidence": ["cycles=50"],
            "expected_effect": "reduce command overhead", "risks": ["tail correctness"],
            "validation_plan": "compare exact cycles and conformance"}}
        proposal = optimize.proposal(parsed, 0)
        self.assertTrue(proposal["hypothesis_complete"])
        same = optimize.proposal({**parsed, "hypothesis": {"change": "batch inputs"}}, 0)
        self.assertEqual(proposal["hypothesis_id"], same["hypothesis_id"])
        self.assertNotEqual(proposal["hypothesis_id"], optimize.proposal(parsed, 1)["hypothesis_id"])
        self.assertFalse(optimize.proposal({"reason": "try batching"}, 0)["hypothesis_complete"])

    def test_repeated_proposal_is_rejected_before_spending_candidate_budget(self):
        parsed = {"continue": True, "target": "implement", "reason": "batch"}
        proposal = optimize.proposal(parsed, 0)
        rounds = [point(0, 500, 1000), {**point(1, 600, 1200), **proposal}]
        state = {"loop_optimize": True, "max_iters": 5, "best_candidate": {"round": 0},
                 "candidate_snapshots": {"0": {"round": 0}}, "loop_parent": "incumbent"}
        with mock.patch.object(llm_step, "_complete_llm", return_value=(
                LlmResult('{"continue":true,"target":"implement","reason":"batch"}', used_llm=True), "raw")):
            target, decision, _ = planner._choose_optimization(state, rounds)
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "duplicate_proposal")

    def test_review_retains_tradeoff_and_selects_by_objective(self):
        from agents.cfu_design_agent.flow import gates
        baseline = point(0, 500, 1000)
        incumbent = {"round": 0, "manifest": "old", "files": []}
        state = {"loop_optimize": False, "loop_objective": "performance", "max_iters": 3,
                 "optimization_round": 1, "optimization_plan": {"target": "implement", "parent_round": 0},
                 "optimization_rounds": [baseline], "best_candidate": incumbent,
                 "gates": {name: {"ok": True} for name in gates.REQUIRED_GATES}}
        state["gates"]["sim"].update(round=1, reference_cycles=1000, cfu_cycles=300, speedup_x100=333)
        state["gates"]["cost"].update(round=1, metrics={"cells": 1050})
        with mock.patch.object(planner, "_snapshot_candidate", return_value={"round": 1, "manifest": "new"}), \
             mock.patch.object(stage_services, "_artifacts_for_stage"), \
             mock.patch.object(stage_services, "_report_review_decision"):
            result = review.review_and_route(state)
        self.assertEqual(result["delivered_round"], 1)
        self.assertEqual(result["optimization_frontier"], [0, 1])
        self.assertEqual(set(result["candidate_snapshots"]), {"0", "1"})
        self.assertTrue(result["optimization_rounds"][-1]["improved"])
        self.assertFalse(result["optimization_rounds"][-1]["dominates_incumbent"])


class CandidateRestoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        policy = self.root / "agents/specs/agent_access_policy.yaml"
        policy.parent.mkdir(parents=True)
        policy.write_bytes((REPO_ROOT / "agents/specs/agent_access_policy.yaml").read_bytes())
        run = "agents/generated/runs/strategy-unit"
        self.hw = f"{run}/workspace/design/AgentCfu.scala"
        self.sw = f"{run}/workspace/demo_benchmark/kernel.c"
        self.extra = f"{run}/workspace/demo_benchmark/later.h"
        for path, text in ((self.hw, "verified hardware"), (self.sw, "int k() { return 1; }")):
            full = self.root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(text)
        self.state = {"workdir": str(self.root), "run_root": run, "run_id": "strategy-unit",
                      "apply_patch": True, "design_files": [self.hw], "isa_spec": {"identity": "old"},
                      "isa_revision": 1, "workload_analysis": "old analysis", "gates": {},
                      "benchmark_project": {"kind": "generated_benchmark", "workspace": f"{run}/workspace/demo_benchmark",
                                            "source_path": self.sw},
                      "validation_results": [{"rung": "sim"}], "cost_results": [{}]}
        self.targets = mock.patch.object(stage_services, "_software_targets", return_value=[(self.sw, "")])
        self.targets.start()
        self.addCleanup(self.targets.stop)
        self.alias = self.root / run / "08_isa/cfu_isa.yaml"
        self.alias.parent.mkdir(parents=True, exist_ok=True)
        self.alias.write_text("identity: old\n")
        fingerprints = dependencies.input_fingerprints(self.state)
        self.state["gates"] = {key: {"ok": True, "input_sha256": fingerprints[key]} for key in ("analysis", "isa")}
        self.candidate = planner._snapshot_candidate(self.state, point(0, 500, 1000))

    def test_restores_inputs_and_contract_but_invalidates_gates_and_keeps_ledger(self):
        (self.root / self.hw).write_text("later hardware")
        (self.root / self.sw).write_text("later software")
        (self.root / self.extra).write_text("later dependency")
        state = {**self.state, "isa_spec": {"identity": "new"}, "evaluation_attempts": [{"id": "prior"}],
                 "best_candidate": self.candidate, "gates": {"sim": {"ok": True}, "cost": {"ok": True}}}
        restored = candidates.restore_parent(state, self.candidate)
        self.assertEqual((self.root / self.hw).read_text(), "verified hardware")
        self.assertEqual((self.root / self.sw).read_text(), "int k() { return 1; }")
        self.assertFalse((self.root / self.extra).exists())
        self.assertEqual(restored["isa_spec"], {"identity": "old"})
        self.assertNotIn("sim", restored["gates"])
        self.assertNotIn("evaluation_attempts", restored)
        self.assertNotIn("best_candidate", restored)
        self.assertEqual(restored["repair_evidence_epoch"]["validation_results"], 1)

    def test_restores_tool_visible_contract_alias_as_well_as_state(self):
        self.alias.write_text("identity: child\n")
        restored = candidates.restore_parent(self.state, self.candidate)
        self.assertEqual(self.alias.read_text(), "identity: old\n")
        self.assertEqual(restored["isa_spec"], {"identity": "old"})

    def test_node_wrapper_preserves_only_hash_proven_parent_contract_gates(self):
        from contextlib import nullcontext
        for corrupt_proof in (False, True):
            with self.subTest(corrupt_proof=corrupt_proof):
                child = {**deepcopy(self.state), "isa_spec": {"identity": "child"},
                         "workload_analysis": "child analysis"}
                (self.root / self.hw).write_text("child hardware")
                child["input_fingerprints"] = dependencies.input_fingerprints(child)
                child["gates"] = {key: {"ok": True, "input_sha256": value}
                                  for key, value in child["input_fingerprints"].items()}
                def review_and_route(state):
                    update = candidates.restore_parent(state, self.candidate)
                    update["gates"]["review"] = {"ok": True, "restored_parent": 0, "next": "implement"}
                    if corrupt_proof:
                        update["gates"]["isa"]["input_sha256"] = "not-the-parent"
                    return update
                with mock.patch.object(instrument, "_audit_context", return_value=nullcontext()), \
                     mock.patch.object(instrument, "_with_flow_memory", side_effect=lambda s, u, *a, **k: u):
                    update = instrument._wrap_node(review_and_route)(child)
                self.assertEqual(update["gates"]["isa"]["ok"], not corrupt_proof)
                self.assertTrue(update["gates"]["analysis"]["ok"])
                self.assertNotIn("sim", update["gates"])
                self.assertNotIn("cost", update["gates"])
                self.assertNotIn("hw_build", update["gates"])

    def test_corrupted_snapshot_is_rejected_before_any_source_write(self):
        (self.root / self.hw).write_text("later hardware")
        snapshot = self.root / self.candidate["files"][0]["snapshot"]
        snapshot.write_text("corrupted source")
        with self.assertRaisesRegex(ValueError, "hash mismatched"):
            candidates.restore_parent(self.state, self.candidate)
        self.assertEqual((self.root / self.hw).read_text(), "later hardware")

    def test_mid_restore_io_failure_rolls_back_all_inputs(self):
        from agents.cfu_design_agent.runtime import conformance_delivery
        (self.root / self.hw).write_text("later hardware")
        original = conformance_delivery._restore
        calls = []
        def fail_once(toolbox, inputs):
            calls.append(1)
            if len(calls) == 1:
                path, data = next(iter(inputs.items()))
                original(toolbox, {path: data})
                raise OSError("injected write failure")
            return original(toolbox, inputs)
        with mock.patch.object(conformance_delivery, "_restore", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                candidates.restore_parent(self.state, self.candidate)
        self.assertEqual((self.root / self.hw).read_text(), "later hardware")


if __name__ == "__main__":
    unittest.main()
