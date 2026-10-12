"""Stage-local verifications: the agent schedules them, the ladder still decides.

Implement may build the SoC/software and deliver an oracle, but simulation belongs
to Verify. Build verifications take no arguments, are budgeted, and record evidence
rather than gate verdicts. Oracle delivery and syntax-only checking are separate
capabilities, not simulation verifications.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.cfu_design_agent.runtime.implementation_tools import (
    ImplementationToolset,
    Verification,
)
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


class VerificationDescriptorTest(unittest.TestCase):
    def test_it_needs_a_tool_safe_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid verification name"):
            Verification(name="build soc", description="d", label="l", final_gate="g",
                         budget=1, callback=dict)

    def test_it_needs_a_positive_budget(self) -> None:
        for budget in (0, -1, True):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "positive budget"):
                    Verification(name="ok", description="d", label="l", final_gate="g",
                                 budget=budget, callback=dict)


class VerificationToolsetTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = REPO_ROOT / "agents/specs/agent_access_policy.yaml"
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(policy.read_bytes())
        self.run_root = "agents/generated/runs/verification-test"
        self.hw = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.sw = f"{self.run_root}/workspace/demo_benchmark/kernel.c"
        for name in (self.hw, self.sw):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("class AgentCfu {}\n" if name == self.hw else "int x;\n")
        self.toolbox = RepoToolbox(self.root, policy=load_policy(self.root),
                                   stage="implementation", run_root=self.run_root,
                                   apply_patch=True)
        self.calls: list[str] = []

    def _verification(self, name: str, *, budget: int, ok: bool) -> Verification:
        def callback(_args: dict) -> dict:
            self.calls.append(name)
            if ok:
                return {"returncode": 0, "full_log_path": f"09_implementation/{name}.log"}
            return {"returncode": 1, "stderr_tail": f"{name} broke",
                    "failure_observation": {"id": f"fail_{name}", "category": "build",
                                            "summary": f"{name} broke"}}

        return Verification(name=name, description=f"run {name}", label=name,
                            final_gate="hw_build", budget=budget, callback=callback)

    def _tool(self, *verifications: Verification, **kwargs) -> ImplementationToolset:
        return ImplementationToolset(
            self.toolbox, frozenset({self.hw}), frozenset({self.sw}),
            verifications=verifications, **kwargs)

    def test_the_tools_are_offered_and_take_no_arguments(self) -> None:
        tool = self._tool(self._verification("build_soc", budget=2, ok=True))
        names = {s["function"]["name"] for s in tool.schemas()}
        self.assertIn("build_soc", names)
        self.assertIn("usage error", tool.run("build_soc", {"argv": ["rm", "-rf", "/"]}))

    def test_a_passing_result_names_the_rung_that_will_rerun(self) -> None:
        tool = self._tool(self._verification("build_soc", budget=1, ok=True))
        answer = tool.run("build_soc", {})
        self.assertIn("evidence only", answer)
        self.assertIn("hw_build", answer)

    def test_a_failing_result_reports_the_failure_without_claiming_a_verdict(self) -> None:
        """A check never reads as a gate verdict: it reports diagnostics only."""
        tool = self._tool(self._verification("build_soc", budget=1, ok=False))
        answer = tool.run("build_soc", {})
        self.assertIn("failed", answer)
        self.assertIn("fail_build_soc", answer)
        self.assertIn("build_soc broke", answer)
        self.assertNotIn("gate", answer.lower())

    def test_the_budget_is_enforced_and_withdraws_the_tool(self) -> None:
        tool = self._tool(self._verification("build_software", budget=1, ok=True))
        self.assertIn("passed (evidence only", tool.run("build_software", {}))
        self.assertIn("budget exhausted", tool.run("build_software", {}))
        self.assertNotIn("build_software", {s["function"]["name"] for s in tool.schemas()})
        self.assertEqual(self.calls, ["build_software"])

    def test_verifications_may_be_repeated_while_budget_remains(self) -> None:
        tool = self._tool(self._verification("build_software", budget=3, ok=True))
        for _ in range(3):
            self.assertIn("passed", tool.run("build_software", {}))
        self.assertEqual(len(self.calls), 3)

    def test_the_audit_trail_records_every_verification(self) -> None:
        tool = self._tool(self._verification("build_software", budget=2, ok=True))
        tool.run("build_software", {})
        tool.run("build_software", {})
        self.assertEqual(len(tool.checks["build_software"]), 2)

    def test_a_verification_cannot_shadow_a_reserved_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "shadows a reserved tool"):
            self._tool(self._verification("edit_hardware", budget=1, ok=True))

    def test_duplicate_verification_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            self._tool(self._verification("build_soc", budget=1, ok=True),
                       self._verification("build_soc", budget=1, ok=True))


class StageWiringTest(unittest.TestCase):
    """Stage 3 offers builds plus oracle delivery, never simulation."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = self.root / "agents/specs/agent_access_policy.yaml"
        policy.parent.mkdir(parents=True)
        policy.write_bytes((REPO_ROOT / "agents/specs/agent_access_policy.yaml").read_bytes())
        self.run_root = "agents/generated/runs/stage-wiring-test"
        self.header = f"{self.run_root}/workspace/cfu_intrinsics.h"
        self.software = f"{self.run_root}/workspace/demo_benchmark/kernel.c"
        target = self.root / self.software
        target.parent.mkdir(parents=True)
        target.write_text("int x;\n")
        self.state = {
            "task": "t", "workdir": str(self.root), "run_root": self.run_root,
            "apply_patch": True, "run_commands": True, "dry_run": False, "agent_tools": True,
            "isa_spec": {"commands": [{"function_id": 1, "name": "compute"}]},
            "isa_spec_text": "", "workload_analysis": "", "gates": {}, "workload_spec": {},
            "benchmark_project": {"kind": "generated_benchmark", "source_path": self.software},
        }

    def _capture(self) -> dict:
        from unittest import mock

        from agents.cfu_design_agent.runtime import llm_step
        from agents.cfu_design_agent.stages.implement import intrinsics, unified

        captured: dict = {}

        class _Client:
            available = True

        def fake_step(state, toolbox, **kwargs):
            captured.update(kwargs)
            captured["toolbox"] = toolbox
            raise RuntimeError("stop after capturing the toolset")

        with mock.patch.object(llm_step, "_llm", return_value=_Client()), \
                mock.patch.object(llm_step, "_run_agent_step", fake_step), \
                mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=self.header):
            with self.assertRaisesRegex(RuntimeError, "stop after capturing the toolset"):
                unified.implement_stage3(self.state)
        return captured

    def _tool(self, captured: dict) -> ImplementationToolset:
        return captured["toolset_factory"](captured["toolbox"], **captured["toolset_kwargs"])

    def test_the_stage_declares_only_build_verifications_and_separate_oracle_tools(self) -> None:
        from agents.cfu_design_agent.runtime.conformance_delivery import ConformanceDelivery

        captured = self._capture()
        kwargs = captured["toolset_kwargs"]
        verifications = {v.name: v for v in kwargs["verifications"]}
        self.assertIsInstance(kwargs["verifications"], tuple)
        self.assertEqual(set(verifications), {"build_soc", "build_software"})
        self.assertEqual(verifications["build_soc"].final_gate, "hw_build")
        self.assertEqual(verifications["build_software"].final_gate, "sw_build")
        for verification in verifications.values():
            self.assertGreater(verification.budget, 0)
            self.assertIsNone(verification.parameters)
        self.assertIsInstance(kwargs["conformance_session"], ConformanceDelivery)
        self.assertTrue(callable(kwargs["check_conformance_syntax"]))
        tools = {schema["function"]["name"]: schema["function"]
                 for schema in self._tool(captured).schemas()}
        self.assertIn("write_conformance", tools)
        self.assertEqual(tools["write_conformance"]["parameters"]["required"], ["oracle"])
        self.assertEqual(tools["check_conformance_syntax"]["parameters"]["properties"], {})
        for forbidden in ("run_conformance", "run_simulation"):
            self.assertNotIn(forbidden, tools)
            self.assertNotIn(forbidden, captured["user"])

    def test_old_simulation_tools_are_forbidden_and_execute_no_commands(self) -> None:
        from unittest import mock
        from agents.cfu_design_agent.runtime import effects

        tool = self._tool(self._capture())
        with mock.patch.object(effects, "run_command") as execute:
            for forbidden in ("run_conformance", "run_simulation"):
                with self.subTest(tool=forbidden):
                    self.assertEqual(tool.run(forbidden, {}),
                                     "error: simulation is available only in Verify")
            execute.assert_not_called()

    def test_edit_hardware_accepts_the_run_dir_token(self) -> None:
        """The model may edit the path exactly as the prompt rendered it."""
        design = self.root / self.run_root / "workspace/design"
        design.mkdir(parents=True, exist_ok=True)
        target = design / "AgentCfu.scala"
        target.write_text("class AgentCfu {}\n")
        tool = self._tool(self._capture())
        applied = tool.run("edit_hardware", {"edits": [{
            "path": "{{run-dir}}/workspace/design/AgentCfu.scala",
            "search": "class AgentCfu {}", "replace": "class AgentCfu { int changed; }",
        }]})
        self.assertIn("+class AgentCfu { int changed; }", applied)
        self.assertEqual(target.read_text(), "class AgentCfu { int changed; }\n")

    def test_oracle_delivery_is_preflighted_and_keeps_the_session_open(self) -> None:
        """A delivered test that cannot compile is rejected at delivery time.

        Regression (run 20261003-130005): the oracle passed validation, the
        ladder's hw_conformance build then died on ``redefinition of
        'md5_oracle_mem'`` with no answer-repair budget left.
        """
        from unittest import mock

        from agents.cfu_design_agent.runtime import effects

        captured = self._capture()
        tool = self._tool(captured)
        session = captured["toolset_kwargs"]["conformance_session"]
        oracle = ("commands:\n  - name: compute\n    reference: a + b\n"
                  "    data:\n      roles:\n        - kind: value\n          max: 255\n        - kind: value\n          max: 255\n")
        failure = {"returncode": 1,
                   "stderr_tail": "cfu_conformance.c:19:1: error: redefinition of 'md5_oracle_mem'"}
        with mock.patch.object(effects, "run_command", return_value=failure):
            rejected = tool.run("write_conformance", {"oracle": oracle})
        self.assertIn("does not compile", rejected)
        self.assertIn("redefinition", rejected)
        self.assertFalse(session.changed, "a rejected preflight must not keep the pair on disk")
        # The transaction stays open so the model can deliver a corrected oracle.
        with mock.patch.object(effects, "run_command", return_value={"returncode": 0}):
            self.assertIn("delivered", tool.run("write_conformance", {"oracle": oracle}))

    def test_conformance_session_refuses_an_undeclared_command_without_executing(self) -> None:
        from unittest import mock
        from agents.cfu_design_agent.runtime import effects

        captured = self._capture()
        session = captured["toolset_kwargs"]["conformance_session"]
        with mock.patch.object(effects, "run_command") as execute:
            delivery, problems = session.write(
                "commands:\n  - name: not_a_command\n    reference: a + b\n"
                "    data:\n      roles:\n        - kind: value\n          max: 255\n        - kind: value\n          max: 255\n")
            self.assertEqual(delivery, {})
            self.assertTrue(problems)
            self.assertIn("not_a_command", " ".join(problems))
            execute.assert_not_called()
        self.assertFalse(session.changed)

    def test_syntax_callback_requires_delivery_and_runs_only_the_host_syntax_command(self) -> None:
        from unittest import mock
        from agents.cfu_design_agent.runtime import commands, effects

        captured = self._capture()
        session = captured["toolset_kwargs"]["conformance_session"]
        check = captured["toolset_kwargs"]["check_conformance_syntax"]
        argv = ["compiler", "-fsyntax-only", "cfu_conformance.c"]
        with mock.patch.object(commands, "_conformance_syntax_command", return_value=argv) as command, \
                mock.patch.object(effects, "run_command", return_value={"returncode": 0}) as execute:
            self.assertNotEqual(check()["returncode"], 0)
            command.assert_not_called()
            execute.assert_not_called()
            delivery, problems = session.write(
                "commands:\n  - name: compute\n    reference: a + b\n"
                "    data:\n      roles:\n        - kind: value\n          max: 255\n        - kind: value\n          max: 255\n")
            self.assertEqual(problems, [])
            self.assertTrue(delivery["source_path"].endswith("cfu_conformance.c"))
            execute.assert_not_called()
            self.assertEqual(check()["returncode"], 0)
            command.assert_called_once_with(self.state, self.header)
            execute.assert_called_once_with(
                self.state, captured["toolbox"], argv, stage="implementation",
                rung="conformance_syntax", command_id="conformance_syntax", timeout=120,
                mutate=True)


if __name__ == "__main__":
    unittest.main()
