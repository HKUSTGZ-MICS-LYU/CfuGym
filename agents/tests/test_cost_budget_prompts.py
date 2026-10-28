"""Offline Cost Wall prompt contracts; no model, build, or synthesis calls."""
from __future__ import annotations

import copy
import inspect
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent import prompts
from agents.cfu_design_agent.runtime import commands, cost_budget, effects, llm_step
from agents.cfu_design_agent.runtime import observations, stage_services
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.stages import analysis, contract
from agents.cfu_design_agent.stages.implement import hw, intrinsics, repair_scope, unified
from agents.cfu_design_agent.stages.review import optimize


class _PromptCaptured(Exception):
    """Stop a stage at its model boundary, before delivery or gate effects."""


class CostBudgetPromptTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"name": "cost_prompt", "operation": "dot_product", "unknowns": []}
        self.context = {
            "workdir": "/offline/repo",
            "run_root": "agents/generated/runs/offline-cost-prompt",
            "files": {prompts.HARDWARE_TARGETS[0]: "// SCALA_BODY_MARKER"},
        }

    def render(self, stage, *, state=None, agent_tools=False):
        kwargs = {"agent_tools": agent_tools}
        if state is not None:
            kwargs["state"] = state
        if stage == "analysis":
            return prompts.workload_prompt("accelerate", self.spec, self.context, **kwargs)
        if stage == "isa":
            return prompts.contract_prompt("accelerate", self.spec, "analysis", self.context, **kwargs)
        if stage == "hardware":
            return prompts.hardware_prompt(
                "accelerate", self.spec, "analysis", "contract", self.context, **kwargs)
        return prompts.optimization_prompt(
            "accelerate", self.spec, "measured score", "", {}, "hardware", "software",
            2, "cost", workdir=self.context["workdir"], run_root=self.context["run_root"],
            **kwargs)

    def test_legacy_calls_render_default_cap_in_each_design_builder(self):
        for stage in ("analysis", "isa", "hardware", "optimization"):
            for tools in (False, True):
                with self.subTest(stage=stage, tools=tools):
                    text = self.render(stage, agent_tools=tools)
                    self.assertIn("at most 100000 mapped Yosys generic cells", text)
                    self.assertEqual(text.count("## Cost Wall"), 1)

    def test_custom_cap_is_rendered_with_and_without_tools(self):
        state = {"max_yosys_cells": 23457, "cost_backend": "vivado", "skip_yosys": True}
        for stage in ("analysis", "isa", "hardware", "optimization"):
            for tools in (False, True):
                with self.subTest(stage=stage, tools=tools):
                    text = self.render(stage, state=state, agent_tools=tools)
                    self.assertIn("at most 23457 mapped Yosys generic cells", text)
                    self.assertNotIn("at most 100000", text)
                    self.assertIn("max_yosys_cells", text)

    def test_reminder_preserves_semantics_and_distinguishes_fpga_capacity(self):
        text = prompts.cost_wall_guidance({})
        for phrase in (
            "not an FPGA LUT count or device capacity", "do not equate them",
            "datapath compact", "limit parallel lane count and unrolling",
            "multicycle resource sharing", "required semantic input domain",
            "exact correctness", "widths, signedness, overflow, tails and state effects",
            "Never meet the cap by narrowing semantics or using fixture-specific shortcuts",
            "not measured synthesis evidence",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_guidance_is_early_but_never_changes_the_shared_head(self):
        prefixes = []
        shared = prompts.workload_digest_block(self.spec, None)
        for stage in ("analysis", "isa", "hardware", "optimization"):
            for tools in (False, True):
                with self.subTest(stage=stage, tools=tools):
                    before = self.render(stage, agent_tools=tools)
                    after = self.render(stage, state={"max_yosys_cells": 23457}, agent_tools=tools)
                    self.assertEqual(before.split("## Cost Wall")[0], after.split("## Cost Wall")[0])
                    self.assertLess(after.index("## Stage:"), after.index("## Cost Wall"))
                    self.assertLess(after.index("## Cost Wall"), after.index("## Deliverable"))
                    if stage != "optimization":
                        self.assertLess(after.index(shared), after.index("## Stage:"))
                        prefixes.append(after.split("## Stage:")[0])
                    if not tools and stage != "optimization":
                        self.assertLess(after.index("## Cost Wall"), after.index("SCALA_BODY_MARKER"))
        self.assertTrue(all(prefix == prefixes[0] for prefix in prefixes))

    def test_renderer_uses_owner_helper_without_mutating_state(self):
        state = {"max_yosys_cells": 23457, "gates": {"cost": {"ok": False}}}
        original = copy.deepcopy(state)
        with mock.patch.object(cost_budget, "max_cells", return_value=9876) as maximum:
            text = prompts.cost_wall_guidance(state)
        maximum.assert_called_once_with(state)
        self.assertIn("at most 9876 mapped Yosys generic cells", text)
        self.assertEqual(state, original)

    def stage_state(self, cap=None):
        state = {
            "task": "accelerate", "workdir": self.context["workdir"],
            "run_root": self.context["run_root"], "context": self.context,
            "workload_spec": self.spec, "workload_analysis": "analysis",
            "design_contract": "contract", "isa_spec": {}, "gates": {},
            "apply_patch": True, "run_commands": True, "agent_tools": True,
            "require_conformance": False, "loop_optimize": True, "max_iters": 3,
            "loop_policy": "explore",
        }
        if cap is not None:
            state["max_yosys_cells"] = cap
        return state

    def capture_environment(self, stack):
        """Mock execution services; prompt rendering and stage plumbing stay real."""
        stack.enter_context(mock.patch.object(
            stage_services, "_toolbox_for_stage",
            return_value=SimpleNamespace(workdir=Path(self.context["workdir"]))))
        stack.enter_context(mock.patch.object(stage_services, "existing_evidence", return_value=[]))
        stack.enter_context(mock.patch.object(
            stage_services, "_software_targets", return_value=[("offline_kernel.c", "")]))
        stack.enter_context(mock.patch.object(llm_step, "_llm", return_value=SimpleNamespace(available=True)))
        stack.enter_context(mock.patch.object(llm_step, "_tools_enabled", return_value=True))
        stack.enter_context(mock.patch.object(observations, "failure_index", return_value={"failures": {}}))
        stack.enter_context(mock.patch.object(contract, "_isa_from_template", return_value={}))
        stack.enter_context(mock.patch.object(hw, "_hardware_context", return_value=self.context))
        stack.enter_context(mock.patch.object(commands, "agent_cfu_check_commands", return_value=[]))
        stack.enter_context(mock.patch.object(intrinsics, "_write_intrinsics_header", return_value="offline.h"))
        stack.enter_context(mock.patch.object(repair_scope, "required_domains", return_value=(True, True)))
        execute = stack.enter_context(mock.patch.object(effects, "run_command"))
        plain = stack.enter_context(mock.patch.object(llm_step, "_complete_llm"))
        return execute, plain

    def test_production_stages_pass_actual_state_to_their_prompts(self):
        # Only the unified prompt owner is unwrapped: there is no edit/delivery to
        # transact, so no header snapshot or CodingSession should touch disk.
        stages = {
            "analysis": analysis.analyze_workload,
            "isa": contract.design_contract,
            "isa_extension": contract.extend_isa,
            "hardware": hw.implement_hw,
            "unified": inspect.unwrap(unified.implement_with_tools),
            "optimization": lambda state: optimize._choose_optimization(state, []),
        }
        for cap in (None, 23457):
            for name, run_stage in stages.items():
                with self.subTest(stage=name, cap=cap), ExitStack() as stack:
                    execute, plain = self.capture_environment(stack)
                    step = stack.enter_context(mock.patch.object(
                        llm_step, "_run_agent_step", side_effect=_PromptCaptured))
                    try:
                        run_stage(self.stage_state(cap))
                    except _PromptCaptured:
                        pass
                    step.assert_called_once()
                    text = step.call_args.kwargs["user"]
                    self.assertIn(f"at most {100000 if cap is None else cap} mapped Yosys generic cells", text)
                    self.assertLess(text.index("## Stage:"), text.index("## Cost Wall"))
                    self.assertLess(text.index("## Cost Wall"), text.index("## Deliverable"))
                    if name == "unified":
                        shared = prompts.workload_digest_block(self.spec, None)
                        self.assertLess(text.index(shared), text.index("## Stage:"))
                    execute.assert_not_called()
                    plain.assert_not_called()

    def test_optimization_reconsideration_retains_the_custom_cap(self):
        with ExitStack() as stack:
            execute, plain = self.capture_environment(stack)
            stop = (LlmResult('{"continue": false}', used_llm=True), "", [])
            step = stack.enter_context(mock.patch.object(
                llm_step, "_run_agent_step", side_effect=[stop, _PromptCaptured]))
            optimize._choose_optimization(self.stage_state(23457), [])
            self.assertEqual(step.call_count, 2)
            for call in step.call_args_list:
                self.assertIn("at most 23457 mapped Yosys generic cells", call.kwargs["user"])
            self.assertIn("RECONSIDER EARLY STOP", step.call_args.kwargs["user"])
            execute.assert_not_called()
            plain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
