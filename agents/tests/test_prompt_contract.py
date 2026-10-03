"""Prompt-contract wiring: every stage reads files by path, not from the prompt.

The unit-level contract (five-part brief, no inlined bodies) lives in
``test_prompts``. ISA-contract and optimization stages use the read-only tool
loop or plain completion. Formal conformance only inspects Implement's delivery
and routes missing answers back for repair without using a model.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent.runtime import conformance_delivery, effects, llm_step
from agents.cfu_design_agent.runtime import stage_services as services
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.stages import contract
from agents.cfu_design_agent.stages.implement import intrinsics
from agents.cfu_design_agent.stages.review import optimize as review_optimize
from agents.cfu_design_agent.stages.verify import conformance as verify_conformance
from agents.tests.common import REPO_ROOT, VALID_ISA

import yaml


class _AvailableClient:
    available = True


def _base_state(run_id: str, *, workdir: Path = REPO_ROOT, **overrides) -> dict:
    state = {
        "workdir": str(workdir),
        "run_id": run_id,
        "run_root": f"agents/generated/runs/{run_id}",
        "task": "accelerate a kernel",
        "agent_tools": True,
        "dry_run": False,
        "apply_patch": True,
        "run_commands": True,
        "skip_sim": False,
        "max_tool_rounds": 4,
        "gates": {},
        "context": {
            "files": {},
            "workdir": str(workdir),
            "run_root": f"agents/generated/runs/{run_id}",
        },
    }
    state.update(overrides)
    return state


class _DisposableWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = self.root / "agents/specs/agent_access_policy.yaml"
        policy.parent.mkdir(parents=True)
        policy.write_bytes((REPO_ROOT / "agents/specs/agent_access_policy.yaml").read_bytes())
        self.run_id = "prompt-contract-test"

    def base_state(self, **overrides) -> dict:
        return _base_state(self.run_id, workdir=self.root, **overrides)


class ContractToolRoutingTest(_DisposableWorkspaceTest):

    def test_design_contract_uses_the_tool_loop_and_paths(self) -> None:
        analysis_body = "ANALYSIS_BODY_MARKER"
        state = self.base_state(
            workload_spec={"name": "w", "operation": "crc32", "unknowns": []},
            workload_analysis=f"roofline:\n  bottleneck: memory-bound\n{analysis_body}",
        )
        with mock.patch.object(llm_step, "_llm", return_value=_AvailableClient()), \
                mock.patch.object(
                    llm_step, "_run_agent_step",
                    return_value=(LlmResult(VALID_ISA, used_llm=True), "raw", []),
                ) as step, \
                mock.patch.object(llm_step, "_complete_llm") as plain:
            update = contract.design_contract(state)
        self.assertTrue(update["gates"]["isa"]["ok"])
        step.assert_called_once()
        kwargs = step.call_args.kwargs
        self.assertEqual(kwargs["tool_calls_relpath"], "08_isa/isa_tool_calls.yaml")
        self.assertIn("07_analysis/workload_analysis.yaml", kwargs["user"])
        self.assertNotIn(analysis_body, kwargs["user"])
        plain.assert_not_called()

    def test_design_contract_falls_back_without_tools(self) -> None:
        state = self.base_state(
            workload_spec={"name": "w", "operation": "crc32", "unknowns": []},
            workload_analysis="roofline:\n  bottleneck: memory-bound",
        )
        with mock.patch.object(llm_step, "_llm", return_value=object()), \
                mock.patch.object(
                    llm_step, "_complete_llm",
                    return_value=(LlmResult(VALID_ISA, used_llm=True), "raw"),
                ) as plain, \
                mock.patch.object(llm_step, "_run_agent_step") as step:
            update = contract.design_contract(state)
        self.assertTrue(update["gates"]["isa"]["ok"])
        plain.assert_called_once()
        step.assert_not_called()


class ConformanceToolRoutingTest(_DisposableWorkspaceTest):
    def test_missing_delivery_is_inspected_without_model_and_repairs_answer_in_implement(self) -> None:
        state = self.base_state(
            workload_spec={"name": "w"},
            isa_spec=yaml.safe_load(VALID_ISA),
            max_answer_repairs=2,
            max_iters=2,
        )
        with mock.patch.object(conformance_delivery, "inspect_delivery",
                               return_value=({}, ["conformance delivery is missing"])) as inspect, \
                mock.patch.object(llm_step, "_llm") as client, \
                mock.patch.object(llm_step, "_run_agent_step") as step, \
                mock.patch.object(llm_step, "_complete_llm") as plain, \
                mock.patch.object(intrinsics, "_write_intrinsics_header") as header, \
                mock.patch.object(effects, "run_command") as execute:
            update = verify_conformance.check_conformance(state)
        inspect.assert_called_once()
        self.assertEqual(inspect.call_args.args[0], state)
        self.assertEqual(inspect.call_args.args[1].stage, "conformance")
        for forbidden in (client, step, plain, header, execute):
            forbidden.assert_not_called()
        gate = update["gates"]["hw_conformance"]
        self.assertFalse(gate["ok"])
        self.assertTrue(gate["oracle_failure"])
        self.assertFalse(gate.get("model_failure", False))
        self.assertEqual(gate["next"], "implement")
        self.assertEqual(gate["repair_target"], "implement")
        self.assertEqual(update["repair_scope"], "conformance")
        self.assertEqual(update["answer_repairs_used"], 1)
        self.assertNotIn("repairs_used", update)
        self.assertEqual(len(update["conformance_attempts"]), 1)
        self.assertTrue(any("delivery is missing" in str(err) for err in update["errors"]))


class OptimizationToolRoutingTest(_DisposableWorkspaceTest):
    def test_planner_uses_the_tool_loop_when_tools_are_available(self) -> None:
        state = self.base_state()
        with mock.patch.object(services, "_toolbox_for_stage", return_value=object()), \
                mock.patch.object(
                    llm_step, "_run_agent_step",
                    return_value=(LlmResult('{"continue": false}', used_llm=True), "raw", []),
                ) as step, \
                mock.patch.object(llm_step, "_complete_llm") as plain:
            result, _ = review_optimize._planner_step(
                state, _AvailableClient(), True, "prompt", "fallback", "optimize"
            )
        self.assertTrue(result.used_llm)
        step.assert_called_once()
        self.assertEqual(
            step.call_args.kwargs["tool_calls_relpath"], "12_review/optimize_tool_calls.yaml"
        )
        plain.assert_not_called()

    def test_planner_falls_back_to_a_plain_completion(self) -> None:
        state = self.base_state()
        with mock.patch.object(
                    llm_step, "_complete_llm",
                    return_value=(LlmResult('{"continue": false}', used_llm=True), "raw"),
                ) as plain, \
                mock.patch.object(llm_step, "_run_agent_step") as step:
            review_optimize._planner_step(
                state, object(), False, "prompt", "fallback", "optimize"
            )
        plain.assert_called_once()
        step.assert_not_called()
