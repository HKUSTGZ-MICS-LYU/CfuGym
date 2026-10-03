"""Formal Verify consumes implementation deliveries without model/source effects."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent import conformance
from agents.cfu_design_agent.runtime import (
    commands, conformance_delivery, effects, llm_step, stage_services,
)
from agents.cfu_design_agent.runtime.observations import build_failure_observation
from agents.cfu_design_agent.runtime.policy import PolicyError
from agents.cfu_design_agent.stages.implement import intrinsics
from agents.cfu_design_agent.stages.verify import conformance as verify_stage
from agents.tests.test_conformance import CPU_COMMAND


class VerifyConformanceStageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = "agents/generated/runs/verify-conformance-test/workspace/cfu_intrinsics.h"
        self.delivery = {"oracle_path": "oracle.yaml", "source_path": "cfu_conformance.c",
                         "header_path": self.header, "oracle_sha": "oracle",
                         "source_sha": "source", "isa_sha": "isa",
                         "generator_sha": "generator"}
        self.state = {
            "workdir": str(self.root), "run_id": "verify-conformance-test",
            "run_root": "agents/generated/runs/verify-conformance-test",
            "task": "verify the CFU", "workload_spec": {},
            "isa_spec": {"commands": [CPU_COMMAND]},
            "gates": {"implementation_sw": {"ok": True, "intrinsics_header": self.header}},
            "conformance_delivery": self.delivery,
            "max_answer_repairs": 2, "max_iters": 2,
        }
        self.box = mock.Mock()
        self.box.read_text.return_value = verify_stage.isa.render_cfu_intrinsics(self.state["isa_spec"])
        self.artifacts = mock.Mock()
        self.pass_verdict = {"ok": True, "status": "pass", "commands": 1, "total": 1,
                             "cases": 2, "failed": [], "details": []}

    def run_stage(self, *, problems=None, verdict=None, build=None, simulation=None,
                  inspection_error=None, reason=""):
        if verdict is None:
            verdict = self.pass_verdict
        if build is None:
            build = {"returncode": 0, "rung": "hw_conformance", "command": "make test"}
        if simulation is None:
            simulation = {"returncode": 0, "rung": "hw_conformance", "command": "sim test.elf",
                          "parsed": {"conformance": verdict}}
        with mock.patch.object(stage_services, "_profile_reason", return_value=reason), \
             mock.patch.object(stage_services, "_artifacts_for_stage", return_value=self.artifacts), \
             mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box), \
             mock.patch.object(stage_services, "_write_ladder_artifact") as artifact, \
             mock.patch.object(conformance_delivery, "inspect_delivery",
                               return_value=(self.delivery, problems or []),
                               side_effect=inspection_error) as inspect, \
             mock.patch.object(intrinsics, "_write_intrinsics_header") as generate_header, \
             mock.patch.object(llm_step, "_llm") as client, \
             mock.patch.object(llm_step, "_run_agent_step") as agent_step, \
             mock.patch.object(llm_step, "_complete_llm") as complete, \
             mock.patch.object(conformance, "render_conformance") as render, \
             mock.patch.object(commands, "_conformance_build_command",
                               return_value=(["make", "test"], "test.elf")) as build_command, \
             mock.patch.object(commands, "agent_cfu_sim_commands", return_value=[["sim", "test.elf"]]), \
             mock.patch.object(effects, "run_command", side_effect=[build, simulation]) as execute:
            update = verify_stage.check_conformance(dict(self.state))
            for forbidden in (generate_header, client, agent_step, complete, render,
                              self.box.write_text, self.box.apply_patch):
                forbidden.assert_not_called()
        return update, execute, inspect, artifact, build_command

    def test_valid_delivery_is_consumed_and_only_host_verdict_artifact_written(self):
        update, execute, inspect, artifact, build_command = self.run_stage()
        inspect.assert_called_once_with(self.state, self.box)
        self.box.read_text.assert_called_once_with(self.header, limit=1_000_000)
        build_command.assert_called_once_with(self.state, self.header)
        self.assertEqual(execute.call_count, 2)
        self.assertTrue(update["gates"]["hw_conformance"]["ok"])
        self.assertEqual(update["gates"]["hw_conformance"]["next"], "build_sw")
        artifact.assert_called_once()
        self.artifacts.write_yaml.assert_called_once()
        self.assertEqual(self.artifacts.write_yaml.call_args.args[1],
                         "09_implementation/conformance/result.yaml")
        self.assertNotIn("conformance_delivery", update)

    def test_stale_missing_or_invalid_delivery_never_builds_and_repairs_answer(self):
        for problem in ("conformance delivery is missing", "oracle artifact is missing",
                        "source artifact is missing", "oracle hash mismatch",
                        "source hash mismatch", "ISA hash mismatch", "generator hash mismatch",
                        "oracle schema rejected"):
            with self.subTest(problem=problem):
                update, execute, inspect, _, build_command = self.run_stage(problems=[problem])
                inspect.assert_called_once()
                execute.assert_not_called()
                build_command.assert_not_called()
                gate = update["gates"]["hw_conformance"]
                self.assertFalse(gate["ok"])
                self.assertTrue(gate["oracle_failure"])
                self.assertEqual(gate["next"], "implement")
                self.assertEqual(update["repair_scope"], "conformance")
                self.assertEqual(update["answer_repairs_used"], 1)
                self.assertNotIn("repairs_used", update)
                self.assertEqual(len(update["conformance_attempts"]), 1)

    def test_malformed_state_approval_is_rejected_without_build(self):
        self.state["conformance_delivery"] = ["invalid approval"]
        self.state["gates"] = {"implementation_sw": {"ok": True}}
        update, execute, _, _, _ = self.run_stage(problems=["host approval must be a mapping"])
        execute.assert_not_called()
        self.assertEqual(update["gates"]["hw_conformance"]["next"], "implement")
        self.assertEqual(update["repair_scope"], "conformance")

    def test_tampered_intrinsics_header_never_builds(self):
        self.box.read_text.return_value = "/* wrong encoding or bogus header */"
        update, execute, _, _, _ = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(update["repair_scope"], "conformance")
        self.assertTrue(any("differs from the current ISA" in item for item in update["errors"]))

    def test_missing_or_empty_header_never_builds(self):
        for mode in ("path", "file"):
            with self.subTest(mode=mode):
                self.state["gates"] = {"implementation_sw": {"ok": True}}
                self.delivery["header_path"] = "" if mode == "path" else self.header
                self.box.read_text.return_value = ""
                update, execute, _, _, build_command = self.run_stage()
                execute.assert_not_called()
                build_command.assert_not_called()
                self.assertEqual(update["gates"]["hw_conformance"]["next"], "implement")
                self.assertEqual(update["repair_scope"], "conformance")
                self.assertEqual(update["answer_repairs_used"], 1)

    def test_gate_cannot_substitute_another_header_path(self):
        self.state["gates"]["implementation_sw"]["intrinsics_header"] = "another_header.h"
        update, execute, _, _, _ = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(update["repair_scope"], "conformance")
        self.assertTrue(any("path differs" in error for error in update["errors"]))

    def test_delivery_header_path_is_available_as_fallback(self):
        self.state["gates"] = {"implementation_sw": {"ok": True}}
        update, execute, _, _, build_command = self.run_stage()
        build_command.assert_called_once_with(self.state, self.header)
        self.assertTrue(update["gates"]["hw_conformance"]["ok"])
        self.assertEqual(execute.call_count, 2)

    def test_read_policy_or_environment_failure_stops_without_repair(self):
        for failure in (PolicyError("read denied"), OSError("host read failed")):
            for where in ("delivery", "header"):
                with self.subTest(where=where, failure=type(failure).__name__):
                    self.box.read_text.side_effect = failure if where == "header" else None
                    update, execute, _, _, _ = self.run_stage(
                        inspection_error=failure if where == "delivery" else None)
                    execute.assert_not_called()
                    self.assertEqual(update["gates"]["hw_conformance"]["next"], "review_and_route")
                    self.assertEqual(update["stage_jobs"], "review")
                    self.assertNotIn("answer_repairs_used", update)
                    self.assertNotIn("repairs_used", update)

    def test_oracle_compile_failure_repairs_answer_and_preserves_exit(self):
        build = {"returncode": 1, "rung": "hw_conformance", "command": "make test",
                 "stderr_tail": "undefined reference to `invented_intrinsic'"}
        update, execute, _, _, _ = self.run_stage(build=build)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(update["validation_results"][0]["returncode"], 1)
        self.assertEqual(update["answer_repairs_used"], 1)
        self.assertEqual(update["repair_scope"], "conformance")
        self.assertNotIn("repairs_used", update)
        self.assertTrue(any("invented_intrinsic" in e for e in update["errors"]))

    def test_environment_or_policy_build_failure_does_not_retry_answer(self):
        for code, category in ((-1, "environment"), (-2, "policy")):
            with self.subTest(category=category):
                build = {"returncode": code, "rung": "hw_conformance", "command": "make test",
                         "stderr_tail": "host tool unavailable or denied",
                         "failure_observation": {"id": "fail-host", "category": category}}
                update, execute, _, _, _ = self.run_stage(build=build)
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(update["gates"]["hw_conformance"]["next"], "review_and_route")
                self.assertNotIn("answer_repairs_used", update)
                self.assertNotIn("repairs_used", update)
                self.assertEqual(update["conformance_attempts"], [])

    def test_simulation_environment_policy_or_harness_failure_stops_without_repair(self):
        for category in ("environment", "policy", "sim_harness_crash"):
            with self.subTest(category=category):
                simulation = {"returncode": 1, "rung": "hw_conformance", "command": "sim test.elf",
                              "stderr_tail": "host simulation failure",
                              "failure_observation": {"id": "fail-host", "category": category}}
                update, execute, _, _, _ = self.run_stage(simulation=simulation)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(update["gates"]["hw_conformance"]["next"], "review_and_route")
                self.assertEqual(update["stage_jobs"], "review")
                self.assertNotIn("answer_repairs_used", update)
                self.assertNotIn("repairs_used", update)

    def test_formal_conformance_preserves_failed_exit_after_pass_verdict(self):
        for returncode in (1, -9, 0):
            with self.subTest(returncode=returncode):
                simulation = {"returncode": returncode, "rung": "hw_conformance",
                              "command": "sim test.elf", "stderr_tail": "teardown diagnostics",
                              "parsed": {"conformance": self.pass_verdict}}
                update, execute, _, _, _ = self.run_stage(simulation=simulation)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(update["validation_results"][-1]["returncode"], returncode)
                self.assertEqual(update["gates"]["hw_conformance"]["ok"], returncode == 0)
                if returncode:
                    self.assertTrue(update["errors"])

    def test_commit_watchdog_returns_to_design_implementation_not_review(self):
        text = ('[error] Exception in thread "main" spinal.sim.SimFailure: '
                "Vexii hasn't committed anything for too long, waiting on uop 0x6FD6\n"
                '[error] at vexiiriscv.test.VexiiRiscvProbe.checkCommits(VexiiRiscvProbe.scala:580)')
        simulation = {"returncode": 1, "rung": "hw_conformance", "command": "sim test.elf",
                      "stderr_tail": text}
        simulation["failure_observation"] = build_failure_observation(
            "hw_conformance", simulation, stderr=text)
        self.state["repair_scope"] = "conformance"
        update, execute, _, _, _ = self.run_stage(simulation=simulation)
        self.assertEqual(execute.call_count, 2)
        gate = update["gates"]["hw_conformance"]
        self.assertFalse(gate["ok"])
        self.assertTrue(gate["repairable"])
        self.assertEqual(gate["repair_target"], "implement")
        self.assertEqual(gate["next"], "implement")
        self.assertEqual(update["repair_scope"], "design")
        self.assertEqual(update["repairs_used"], 1)
        self.assertNotIn("answer_repairs_used", update)
        self.assertTrue(any("commit-progress watchdog" in error for error in update["errors"]))

    def test_completed_differential_failure_routes_to_design_repair(self):
        verdict = {**self.pass_verdict, "ok": False, "status": "fail", "failed": ["compute"]}
        update, execute, _, _, _ = self.run_stage(verdict=verdict)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(update["gates"]["hw_conformance"]["next"], "implement")
        self.assertFalse(update["gates"]["hw_conformance"]["oracle_failure"])
        self.assertEqual(update["repairs_used"], 1)
        self.assertEqual(update["repair_scope"], "design")

    def test_exhausted_answer_budget_stops_without_using_design_budget(self):
        self.state["answer_repairs_used"] = 2
        update, execute, _, _, _ = self.run_stage(problems=["delivery missing"])
        execute.assert_not_called()
        self.assertEqual(update["gates"]["hw_conformance"]["next"], "review_and_route")
        self.assertEqual(update["stopped_at"], "hw_conformance")
        self.assertNotIn("repairs_used", update)
        self.assertNotIn("repair_scope", update)

    def test_skipped_stage_does_not_inspect_or_build(self):
        update, execute, inspect, artifact, build_command = self.run_stage(reason="dry-run")
        execute.assert_not_called()
        inspect.assert_not_called()
        build_command.assert_not_called()
        self.assertTrue(update["gates"]["hw_conformance"]["skipped"])
        artifact.assert_called_once()
        self.artifacts.write_yaml.assert_not_called()

    def test_verify_has_no_model_or_oracle_generation_imports(self):
        tree = ast.parse(Path(verify_stage.__file__).read_text())
        forbidden = {"llm", "llm_step", "prompts", "intrinsics", "LlmResult", "SYSTEM_PROMPT"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertFalse(forbidden.intersection(alias.name for alias in node.names))


if __name__ == "__main__":
    unittest.main()
