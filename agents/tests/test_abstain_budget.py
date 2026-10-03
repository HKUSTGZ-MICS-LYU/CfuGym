"""No real model/API: explicit abstention and evaluation-budget contracts."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.flow import gates
from agents.cfu_design_agent.runtime import llm_step
from agents.cfu_design_agent.runtime import stage_services as services
from agents.cfu_design_agent.stages import contract
from agents.cfu_design_agent.stages import review
from agents.cfu_design_agent.stages.review import optimize as review_optimize
from agents.cfu_design_agent.flow import gates as gates_mod


class AbstainBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state = {"workdir": str(self.root), "run_id": "disposable-abstain",
                      "run_root": "agents/generated/runs/disposable-abstain", "task": "nothing useful",
                      "workload_spec": {}, "workload_analysis": "no candidate", "gates": {},
                      "max_iters": 2}

    def test_explicit_abstain_skips_implementation(self) -> None:
        class _Artifacts:
            def write_yaml(self, *args, **kwargs):
                pass
            def write_text(self, *args, **kwargs):
                pass
        with mock.patch.object(llm_step, "_complete_llm", return_value=(LlmResult("ABSTAIN", used_llm=True), "raw")), \
             mock.patch.object(services, "_write_isa_artifact"), \
             mock.patch.object(services, "_artifacts_for_stage", return_value=_Artifacts()):
            update = contract.design_contract(self.state)
            self.assertTrue(update["gates"]["isa"]["abstain"])
            self.assertEqual(update["termination_reason"], "ABSTAIN")
            self.assertEqual(graph._route_after_isa(update), "review_and_route")
            reviewed = review.review_and_route({**self.state, **update, "max_evals": 1})
        self.assertEqual(reviewed["termination_reason"], "ABSTAIN")
        self.assertEqual(reviewed["gates"]["review"]["next"], "final_report")

    def test_eval_budget_stops_before_planner(self) -> None:
        gates = {name: {"ok": True} for name in gates_mod.REQUIRED_GATES}
        gates["sim"].update(round=0, reference_cycles=100, cfu_cycles=50, speedup_x100=200)
        gates["cost"].update(round=0)
        state = {**self.state, "gates": gates, "max_evals": 1, "loop_optimize": True,
                 "optimization_rounds": [{"round": 0, "target": "baseline", "measured": True,
                                          "validated": True, "improved": True}]}
        with mock.patch.object(review_optimize, "_choose_optimization") as choose, \
             mock.patch.object(review_optimize, "_snapshot_candidate", return_value={"round": 0}), \
             mock.patch.object(services, "_artifacts_for_stage") as artifacts, \
             mock.patch.object(services, "_report_review_decision"):
            result = review.review_and_route(state)
        choose.assert_not_called()
        self.assertEqual(result["termination_reason"], "BUDGET_EXHAUSTED")
        self.assertEqual(result["gates"]["review"]["next"], "final_report")


if __name__ == "__main__":
    unittest.main()
