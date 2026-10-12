"""Offline repair-layer selection: pure routing and scoped transaction fixtures."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.flow import ladder, wiring
from agents.cfu_design_agent.runtime import coding_tools, llm_step, observations, stage_services
from agents.cfu_design_agent.runtime.implementation_tools import ImplementationToolset
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.cfu_design_agent.stages import feedback, implement
from agents.cfu_design_agent.stages.implement import delivery, intrinsics, unified
from agents.tests.common import REPO_ROOT


def diagnosis():
    return {"failure_ids": ["fail_now"], "observed_errors": ["wrong semantics"],
            "root_cause": "contract is unsuitable", "repair_mapping": ["revisit the contract"],
            "regression_risks": ["ABI changes"]}


def route(target="contract"):
    return {"target": target, "reason": "instruction cannot express the required semantics",
            "failure_ids": ["fail_now"], "diagnosis": diagnosis()}


class RepairRouteTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes((REPO_ROOT / "agents/specs/agent_access_policy.yaml").read_bytes())
        self.run_root = "agents/generated/runs/route-unit"
        self.hw = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.sw = f"{self.run_root}/workspace/demo_benchmark/kernel.c"
        self.header = f"{self.run_root}/workspace/demo_benchmark/cfu_intrinsics.h"
        for path, text in ((self.hw, "class AgentCfu {}\n"), (self.sw, "int x = 1;\n"),
                           (self.header, "old header\n")):
            full = self.root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(text)
        self.box = RepoToolbox(self.root, policy=load_policy(self.root), stage="implementation",
                               run_root=self.run_root, apply_patch=True)
        self.index = {"failures": {"fail_now": {"id": "fail_now", "rung": "hw_build", "status": "current"}}}
        self.state = {"workdir": str(self.root), "run_root": self.run_root, "run_id": "route-unit",
                      "task": "accelerate", "max_iters": 4, "repairs_used": 1,
                      "answer_repairs_used": 0, "max_answer_repairs": 99,
                      "apply_patch": True, "run_commands": True, "agent_tools": False,
                      "require_conformance": False, "isa_spec": {"old": "contract"},
                      "design_contract": "old contract", "workload_analysis": "old analysis",
                      "benchmark_project": {"kind": "generated_benchmark",
                                            "workspace": f"{self.run_root}/workspace/demo_benchmark",
                                            "source_path": self.sw},
                      "gates": {"analysis": {"ok": True}, "isa": {"ok": True},
                                "implementation_hw": {"ok": True, "candidate_hash": "old"},
                                "implementation_sw": {"ok": True, "isa_hash": "old"},
                                "sim": {"ok": False}, "cost": {"ok": True}},
                      "validation_results": [{"returncode": 1, "rung": "hw_build",
                                              "stderr_tail": "wrong semantics",
                                              "failure_observation": self.index["failures"]["fail_now"]}],
                      "cost_results": [], "input_fingerprints": {"sim": "old"},
                      "conformance_delivery": {"oracle_sha": "old"},
                      "best_candidate": {"correctness": "passed"}}

    def tool(self, **kwargs):
        return ImplementationToolset(self.box, frozenset({self.hw}), frozenset({self.sw}),
                                     failure_index=self.index, **kwargs)

    def request(self, tool, target="contract"):
        value = route(target)
        value.pop("diagnosis")
        return tool.run("request_repair_route", value)

    def test_tool_schema_and_first_delivery_cannot_rewind(self):
        first = self.tool()
        self.assertNotIn("request_repair_route", {s["function"]["name"] for s in first.schemas()})
        self.assertIn("rejected", self.request(first))
        repaired = self.tool(diagnosis_required=True)
        schema = next(s for s in repaired.schemas() if s["function"]["name"] == "request_repair_route")
        self.assertEqual(schema["function"]["parameters"]["properties"]["target"]["enum"],
                         ["analysis", "contract", "implement"])
        self.assertIn("rejected", self.request(repaired))

    def test_current_diagnosis_and_evidence_required(self):
        tool = self.tool(diagnosis_required=True)
        self.assertIn("accepted", tool.run("submit_repair_diagnosis", diagnosis()))
        for value in ({"target": "software", "reason": "cause", "failure_ids": ["fail_now"]},
                      {"target": "analysis", "reason": "", "failure_ids": ["fail_now"]},
                      {"target": "contract", "reason": "cause", "failure_ids": ["unknown"]},
                      {"target": "analysis", "reason": "cause", "failure_ids": []}):
            self.assertIn("rejected", tool.run("request_repair_route", value))
        self.index["failures"]["fail_now"]["status"] = "resolved"
        self.assertIn("rejected", self.request(tool))
        self.assertFalse(tool.hardware_session.rolled_back)

    def test_upstream_selection_rolls_back_both_sources_and_oracle(self):
        for target in ("analysis", "contract"):
            with self.subTest(target=target):
                oracle = SimpleNamespace(committed=False, rolled_back=False, diagnosis=None)
                oracle.rollback = mock.Mock(side_effect=lambda: setattr(oracle, "rolled_back", True))
                tool = self.tool(diagnosis_required=True, conformance_session=oracle)
                tool.run("submit_repair_diagnosis", diagnosis())
                tool.run("edit_hardware", {"edits": [{"path": self.hw, "search": "class AgentCfu {}",
                                                       "replace": "class AgentCfu { val x = 1 }"}]})
                tool.run("edit_software", {"path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"})
                self.assertIn("rolled back", self.request(tool, target))
                self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
                self.assertEqual((self.root / self.sw).read_text(), "int x = 1;\n")
                oracle.rollback.assert_called_once()
                self.assertIn("closed", tool.run("edit_software", {}))
                with self.assertRaises(RuntimeError):
                    tool.commit()

    def test_implement_choice_stays_in_same_bounded_transaction(self):
        tool = self.tool(diagnosis_required=True, max_software_edits=1)
        tool.run("submit_repair_diagnosis", diagnosis())
        self.assertIn("continue editing", self.request(tool, "implement"))
        tool.run("edit_software", {"path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"})
        self.request(tool, "implement")
        self.assertIn("budget exhausted", tool.run("edit_software", {
            "path": self.sw, "search": "int x = 2;", "replace": "int x = 3;"}))
        self.assertFalse(tool.software_session.rolled_back)
        self.assertNotIn("implement", wiring.ROUTE_TARGETS["implement"])
        tool.rollback()

    def test_rewind_invalidates_dependencies_but_preserves_history(self):
        for target, node in (("analysis", "analyze_workload"), ("contract", "design_contract")):
            with self.subTest(target=target):
                update = ladder.repair_route_transition(self.state, route(target))
                self.assertEqual(ladder._route_after_implement(update), node)
                self.assertEqual(update["repairs_used"], 2)
                self.assertNotIn("answer_repairs_used", update)
                for key in ("isa_spec", "conformance_delivery", "input_fingerprints", "best_candidate"):
                    self.assertFalse(update[key])
                self.assertNotIn("implementation_hw", update["gates"])
                self.assertNotIn("sim", update["gates"])
                self.assertEqual("analysis" in update["gates"], target == "contract")
                self.assertEqual(update.get("workload_analysis", self.state["workload_analysis"]),
                                 "" if target == "analysis" else "old analysis")
                self.assertNotIn("validation_results", update)
                current = {**self.state, **update}
                self.assertEqual(observations.failure_index(current)["failures"]["fail_now"]["status"], "stale")
                self.assertEqual(feedback.all_failure_feedback(current), "")
                current["validation_results"] = [*current["validation_results"], current["validation_results"][0]]
                self.assertEqual(observations.failure_index(current)["failures"]["fail_now"]["status"], "current")

    def test_last_paid_design_slot_can_select_contract_without_double_charge(self):
        state = {**self.state, "max_iters": 2, "repairs_used": 0, "iteration": 0}
        extra, target = ladder._ladder_transition(
            state, "sim", ok=False, default="estimate_cost", repair_target="implement")
        self.assertEqual(target, "implement")
        state = {**state, **extra}
        self.assertEqual(state["repairs_used"], 1)
        update = ladder.repair_route_transition(state, route())
        self.assertEqual(ladder._route_after_implement(update), "design_contract")
        self.assertNotIn("repairs_used", update)
        self.assertEqual(update["repair_entry"]["kind"], "design")
        self.assertTrue(update["repair_entry"]["route_used"])
        self.assertEqual(state["answer_repairs_used"], 0)
        again = ladder.repair_route_transition({**state, **update}, route("analysis"))
        self.assertEqual(ladder._route_after_implement(again), "review_and_route")

    def test_answer_and_delivery_entries_must_pay_design_budget_for_upstream(self):
        for kind in ("answer", "delivery"):
            with self.subTest(kind=kind):
                state = {**self.state, "iteration": 1, "repair_entry": {
                    "kind": kind, "target": "implement", "iteration": 1, "route_used": False}}
                update = ladder.repair_route_transition(state, route())
                self.assertEqual(update["repairs_used"], 2)
                blocked = ladder.repair_route_transition({**state, "max_iters": 2}, route())
                self.assertEqual(ladder._route_after_implement(blocked), "review_and_route")

    def test_old_delivery_fingerprints_stay_historical_after_rewind(self):
        attempts = [{"response_hash": "same", "error_hash": "same", "applied": False}] * 2
        state = {**self.state, "hw_attempts": attempts, "sw_attempts": attempts,
                 "conformance_attempts": [{"problems_hash": "same"}] * 2}
        update = ladder.repair_route_transition(state, route())
        current = {**state, **update}
        self.assertEqual(feedback.hardware_stuck_directive(current), "")
        self.assertEqual(feedback.software_stuck_directive(current), "")
        self.assertEqual(feedback.conformance_oracle_directive(current), "")
        self.assertEqual(current["hw_attempts"], attempts)

    def test_upstream_prompt_feedback_retains_diagnosis_not_current_claims(self):
        self.assertEqual(feedback.upstream_repair_feedback(self.state), "")
        update = ladder.repair_route_transition(self.state, route("analysis"))
        text = feedback.upstream_repair_feedback({**self.state, **update})
        self.assertIn("contract is unsuitable", text)
        self.assertIn("fail_now", text)
        self.assertIn("historical evidence", text)
        self.assertIn("old contract/implementation approvals are invalidated", text)

    def test_answer_retries_cannot_buy_unlimited_upstream_rewinds(self):
        state = self.state
        for _ in range(2):
            state = {**state, **ladder.repair_route_transition(state, route())}
        update = ladder.repair_route_transition(state, route("analysis"))
        self.assertEqual(ladder._route_after_implement(update), "review_and_route")
        self.assertEqual(update["stopped_at"], "implementation_route")
        self.assertEqual(state["repairs_used"], 3)
        self.assertNotIn("answer_repairs_used", update)

    def test_every_implementation_conditional_edge_is_registered_natively(self):
        self.assertEqual(ladder._route_after_implement({}), "generate_soc")
        app = graph.build_graph()
        edges = {(edge.source, edge.target) for edge in app.get_graph().edges}
        for target in wiring.ROUTE_TARGETS["implement"]:
            self.assertIn(("implement", target), edges)
        for node, name in wiring.ROUTER_NAMES.items():
            self.assertTrue(callable(getattr(graph, name)), node)

    def test_plain_route_validates_diagnosis_and_rejects_mixed_delivery(self):
        text = "REPAIR_ROUTE\n" + json.dumps(route())
        self.assertTrue(llm_step._has_delivery_block(text))
        self.assertEqual(delivery.repair_route_output(self.state, text, self.box)["repair_route"]["target"], "contract")
        for invalid in ("REPAIR_ROUTE\nnot json", "REPAIR_ROUTE\n{}",
                        text + f"\nFILE_BEGIN: {self.hw}\nclass X {{}}\nFILE_END",
                        "REPAIR_ROUTE\n" + json.dumps(route("implement"))):
            with self.subTest(text=invalid), self.assertRaises(ValueError):
                delivery.repair_route_output(self.state, invalid, self.box)
        stale = {**self.state, "repair_evidence_epoch": {"validation_results": 1}}
        with self.assertRaises(ValueError):
            delivery.repair_route_output(stale, text, self.box)
        with self.assertRaises(ValueError):
            delivery.repair_route_output({**self.state, "validation_results": []}, text, self.box)

    def test_plain_outer_transaction_rolls_back_previous_hardware_half(self):
        def hardware(_state):
            (self.root / self.hw).write_text("changed hardware")
            (self.root / self.header).write_text("changed header")
            return {"gates": {"implementation_hw": {"ok": True}}, "changed_design_files": [self.hw]}
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=False)), \
                mock.patch.object(implement.hardware, "implement_hw", side_effect=hardware), \
                mock.patch.object(implement.software, "implement_sw", return_value={"repair_route": route()}), \
                mock.patch.object(implement.oracle_author, "write_conformance") as oracle:
            update = implement.implement(self.state)
        oracle.assert_not_called()
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
        self.assertEqual((self.root / self.header).read_text(), "old header\n")
        self.assertEqual(update["changed_design_files"], [])
        self.assertEqual(ladder._route_after_implement(update), "design_contract")

    def test_hardware_half_can_rewind_before_software_model_call(self):
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=False)), \
                mock.patch.object(implement.hardware, "implement_hw", return_value={"repair_route": route("analysis")}), \
                mock.patch.object(implement.software, "implement_sw") as software:
            update = implement.implement(self.state)
        software.assert_not_called()
        self.assertEqual(ladder._route_after_implement(update), "analyze_workload")

    def test_old_route_is_not_replayed_by_rejected_implementation(self):
        state = {**self.state, "gates": {"implementation_route": {"next": "analyze_workload"}}}
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=False)), \
                mock.patch.object(implement.hardware, "implement_hw", return_value={
                    "gates": {"implementation_hw": {"ok": False, "errors": ["bad delivery"]}}}):
            update = implement.implement(state)
        self.assertEqual(ladder._route_after_implement(update), "generate_soc")

    def test_unified_rewind_uses_existing_submit_without_committing_header(self):
        def write_header(_state, _box):
            (self.root / self.header).write_text("new generated header")
            return self.header
        def scripted_step(state, box, **kwargs):
            tool = kwargs["toolset_factory"](box, failure_index=kwargs["failure_observations"],
                                              **kwargs["toolset_kwargs"])
            self.assertIn("accepted", tool.run("submit_repair_diagnosis", diagnosis()))
            self.assertIn("rolled back", self.request(tool, "analysis"))
            self.assertEqual(kwargs["submit_stage"]("revisit analysis"), (True, ""))
            return LlmResult("revisit analysis", used_llm=True), "raw fixture", []
        state = {**self.state, "agent_tools": True}
        with mock.patch.object(intrinsics, "intrinsics_header_path", return_value=self.header), \
                mock.patch.object(intrinsics, "_write_intrinsics_header", side_effect=write_header), \
                mock.patch.object(llm_step, "_run_agent_step", side_effect=scripted_step) as step:
            update = unified.implement_with_tools(state, client=mock.Mock(available=True))
        self.assertEqual(step.call_count, 1)
        self.assertEqual((self.root / self.header).read_text(), "old header\n")
        self.assertEqual(ladder._route_after_implement(update), "analyze_workload")
        self.assertNotIn("implementation_hw", update["gates"])

    def test_plain_integrity_rejection_restores_whole_round(self):
        def hardware(_state):
            (self.root / self.hw).write_text("constant result hardware")
            return {"gates": {"implementation_hw": {"ok": True}}, "changed_design_files": [self.hw]}
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=False)), \
                mock.patch.object(implement.hardware, "implement_hw", side_effect=hardware), \
                mock.patch.object(implement.software, "implement_sw", return_value={}), \
                mock.patch.object(stage_services, "hardware_integrity_problems", return_value=["fixed expected result"]), \
                mock.patch.object(implement.oracle_author, "write_conformance") as oracle:
            update = implement.implement(self.state)
        oracle.assert_not_called()
        self.assertFalse(update["gates"]["implementation_hw"]["ok"])
        self.assertIn("hardware integrity", update["errors"][0])
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")


if __name__ == "__main__":
    unittest.main()
