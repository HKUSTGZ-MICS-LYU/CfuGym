"""Structured deterministic failure evidence and stable signatures."""

from __future__ import annotations

import shutil
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime.agent_tools import ReadOnlyToolset
from agents.cfu_design_agent.runtime.observations import (
    build_failure_observation,
    failure_index,
    failure_is_design_actionable,
    record_failure_result,
)
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime import checks
from agents.tests.common import REPO_ROOT


class FailureObservationTest(unittest.TestCase):
    def test_actionable_width_error_survives_spinal_phase_banner(self) -> None:
        noise = "\n".join(f"[error] at Misc.scala:{index}" for index in range(12))
        observation = build_failure_observation("hw_build", {"returncode": 1}, stdout=(
            '[info] [Warning] Elaboration failed (2 errors).\n'
            '[error] Exception in thread main spinal.core.SpinalExit\n'
            '[error] Error detected in phase PhaseNormalizeNodeInputs\n'
            '[error] ****************************************\n'
            + noise + '\n[error] WIDTH MISMATCH (32 bits <- 24 bits)\n'
            '[error] at vexiiriscv.soc.mico.AgentCfu(AgentCfu.scala:170)\n'))
        self.assertIn("WIDTH MISMATCH", observation["summary"])
        self.assertIn("AgentCfu.scala:170", observation["source_locations"])
        self.assertFalse(any("Misc.scala" in item for item in observation["source_locations"]))

    def test_signature_ignores_run_path_address_and_source_line(self) -> None:
        result = {
            "command": "sbt runMain Test",
            "cwd": str(REPO_ROOT),
            "returncode": 1,
        }
        first = build_failure_observation(
            "hw_build",
            result,
            stdout=(
                f"[error] {REPO_ROOT}/agents/generated/runs/run-a/workspace/design/"
                "AgentCfu.scala:155:12: WIDTH MISMATCH at 0xabc123 time=421\n"
            ),
            workdir=str(REPO_ROOT),
        )
        second = build_failure_observation(
            "hw_build",
            result,
            stdout=(
                f"[error] {REPO_ROOT}/agents/generated/runs/run-b/workspace/design/"
                "AgentCfu.scala:201:9: WIDTH MISMATCH at 0xdef999 time=999\n"
            ),
            workdir=str(REPO_ROOT),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first["category"], "elaboration")
        self.assertEqual(first["signature"], second["signature"])
        self.assertTrue(first["signature"].startswith("failure-v1:"))
        self.assertEqual(first["id"], second["id"])

    def test_timeout_and_policy_have_distinct_categories(self) -> None:
        timeout = build_failure_observation(
            "sim", {"command": "sim", "returncode": -9, "timed_out": True},
            stderr="command timed out after 10s",
        )
        policy = build_failure_observation(
            "sw_build", {"command": "make", "returncode": -2},
            stderr="command rejected by the access policy",
        )
        self.assertEqual(timeout["category"], "timeout")
        self.assertEqual(policy["category"], "policy")
        self.assertNotEqual(timeout["signature"], policy["signature"])

    def test_success_and_skips_do_not_create_failures(self) -> None:
        self.assertIsNone(build_failure_observation("sw_build", {"returncode": 0}))
        self.assertIsNone(build_failure_observation(
            "sw_build", {"returncode": None, "skipped": True}
        ))

    def test_failure_index_reads_validation_and_cost_results(self) -> None:
        first = {"id": "fail_sw_build_a", "rung": "sw_build"}
        second = {"id": "fail_cost_b", "rung": "cost"}
        index = failure_index({
            "validation_results": [{"failure_observation": first}],
            "cost_results": [{"failure_observation": second}],
        })
        self.assertEqual(set(index["failures"]), {"fail_sw_build_a", "fail_cost_b"})

    def test_commit_watchdog_is_repairable_despite_probe_stack(self) -> None:
        text = ('[error] Exception in thread "main" spinal.sim.SimFailure: '
                "Vexii hasn't committed anything for too long, waiting on uop 0x6FD6\n"
                '[error] at vexiiriscv.test.VexiiRiscvProbe.checkCommits(VexiiRiscvProbe.scala:580)\n')
        for rung in ("hw_conformance", "sim"):
            with self.subTest(rung=rung):
                observation = build_failure_observation(rung, {"returncode": 1}, stderr=text)
                self.assertEqual(observation["category"], "liveness_timeout")
                self.assertTrue(failure_is_design_actionable(observation["category"]))
                self.assertIn("hasn't committed", observation["summary"])

    def test_sim_harness_crash_is_separate_from_design_evidence(self) -> None:
        crash = build_failure_observation(
            "hw_conformance",
            {"command": "sbt runMain MiCoSocSim", "returncode": 1},
            stderr=(
                '[info] [Error] Simulation failed at time=10000\n'
                '[error] Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: '
                'Index -1 out of bounds for length 1024\n'
                '[error] \tat vexiiriscv.test.VexiiRiscvProbe.$anonfun$checkCommits$2'
                '(VexiiRiscvProbe.scala:595)\n'
                '[error] \tat scala.collection.TraversableLike$WithFilter.foreach'
                '(TraversableLike.scala:984)\n'
            ),
        )
        self.assertEqual(crash["category"], "sim_harness_crash")
        self.assertFalse(failure_is_design_actionable(crash["category"]))
        locations = crash["source_locations"]
        self.assertIn("VexiiRiscvProbe.scala:595", locations)
        self.assertFalse(any(loc.startswith("scala.c") for loc in locations))
        self.assertNotIn("TraversableLike.scala:984", locations)

    def test_fail_symbol_and_hdl_assertion_are_design_actionable(self) -> None:
        fail_symbol = build_failure_observation(
            "sim", {"command": "sbt runMain MiCoSocSim", "returncode": 1},
            stderr="[info] FAILURE Software reached the fail symbol :(\n",
        )
        assertion = build_failure_observation(
            "hw_conformance",
            {"command": "sbt runMain MiCoSocSim", "returncode": 1},
            stderr=(
                "[info] FAILURE config: defer requires a tick handler\n"
                '[error] Exception in thread "main" spinal.sim.SimFailure: HDL assertion failure\n'
            ),
        )
        self.assertEqual(fail_symbol["category"], "fail_symbol")
        self.assertEqual(assertion["category"], "assertion")
        self.assertTrue(failure_is_design_actionable(fail_symbol["category"]))
        self.assertTrue(failure_is_design_actionable(assertion["category"]))

    def test_failure_index_marks_resolved_after_rung_success(self) -> None:
        stale = {"id": "fail_hw_build_x", "rung": "hw_build"}
        live = {"id": "fail_hw_conformance_y", "rung": "hw_conformance"}
        index = failure_index({
            "validation_results": [
                {"failure_observation": stale},
                {"rung": "hw_build", "returncode": 0},
                {"failure_observation": live},
            ],
            "cost_results": [],
        })
        self.assertEqual(index["failures"]["fail_hw_build_x"]["status"], "resolved")
        self.assertEqual(index["failures"]["fail_hw_conformance_y"]["status"], "current")
        # A later failure of the same rung makes the observation current again.
        again = failure_index({
            "validation_results": [
                {"failure_observation": stale},
                {"rung": "hw_build", "returncode": 0},
                {"failure_observation": stale},
            ],
            "cost_results": [],
        })
        self.assertEqual(again["failures"]["fail_hw_build_x"]["status"], "current")

    def test_live_internal_failure_resolves_only_after_explicit_same_rung_success(self) -> None:
        index = failure_index({})
        failed = record_failure_result(index, {
            "returncode": 1, "command": "sbt isolated-check",
            "stderr_tail": "AgentCfu.scala:7: error: WIDTH MISMATCH",
        }, rung="hw_check")
        ident = failed["failure_observation"]["id"]
        self.assertEqual(index["failures"][ident]["status"], "current")
        self.assertEqual(index["failures"][ident]["rung"], "hw_check")
        for result in ({"returncode": 0, "rung": "sw_check"},
                       {"returncode": 0, "rung": "hw_build"},
                       {"returncode": None, "rung": "hw_check"},
                       {"returncode": 0, "rung": "hw_check", "skipped": True}):
            with self.subTest(result=result):
                record_failure_result(index, result)
                self.assertEqual(index["failures"][ident]["status"], "current")
        succeeded = record_failure_result(index, {
            "returncode": 0, "failure_observation": failed["failure_observation"],
        }, rung="hw_check")
        self.assertNotIn("failure_observation", succeeded)
        self.assertEqual(index["failures"][ident]["status"], "resolved")
        failed_again = record_failure_result(index, failed, rung="hw_check")
        self.assertEqual(failed_again["failure_observation"]["id"], ident)
        self.assertEqual(index["failures"][ident]["status"], "current")


class RunCheckPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"run-check-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = REPO_ROOT / self.run_root
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        self.state = {
            "workdir": str(REPO_ROOT),
            "out_dir": "agents/generated",
            "run_id": self.run_id,
            "run_root": self.run_root,
            "validation_results": [],
            "cost_results": [],
        }
        self.toolbox = RepoToolbox(
            REPO_ROOT, dry_run=False, stage="validation", run_root=self.run_root
        )

    def test_run_check_persists_full_output_but_returns_only_tails(self) -> None:
        full_stdout = "HEAD_MARKER\n" + ("filler\n" * 1200) + "main.c:77: error: TAIL_MARKER\n"
        compact = {
            "command": "make compile",
            "cwd": str(REPO_ROOT),
            "returncode": 1,
            "stdout_tail": full_stdout[-6000:],
            "stderr_tail": "",
            "parsed": {},
            "_stdout_full": full_stdout,
            "_stderr_full": "",
        }
        with mock.patch.object(self.toolbox, "run_command", return_value=compact):
            result = checks._run_check(
                self.state, self.toolbox, ["make", "compile"],
                rung="sw_build", command_id="project_test",
            )
        self.assertNotIn("_stdout_full", result)
        self.assertNotIn("HEAD_MARKER", result["stdout_tail"])
        self.assertIn("TAIL_MARKER", result["stdout_tail"])
        log_path = REPO_ROOT / result["log_path"]
        self.assertIn("HEAD_MARKER", log_path.read_text())
        self.assertIn("TAIL_MARKER", log_path.read_text())
        self.assertTrue(result["stdout_truncated"])
        self.assertFalse(result["stderr_truncated"])
        self.assertEqual(result["full_log_path"], result["log_path"])
        self.assertEqual(result["failure_observation"]["category"], "compile")
        self.assertEqual(result["failure_observation"]["log_path"], result["log_path"])

    def test_policy_inference_rejection_still_persists_observation(self) -> None:
        result = checks._run_check(
            self.state, self.toolbox, ["definitely-not-allowlisted", "x"], rung="sw_build"
        )
        self.assertEqual(result["returncode"], -2)
        self.assertEqual(result["failure_observation"]["category"], "policy")
        self.assertTrue((REPO_ROOT / result["full_log_path"]).is_file())


class FailureToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"failure-tools-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = REPO_ROOT / self.run_root
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        self.log = f"{self.run_root}/10_validation/logs/sw_build_000.log"
        path = REPO_ROOT / self.log
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("line one\ncompile error marker\nline three\n")
        self.failure = {
            "schema_version": "1",
            "id": "fail_sw_build_deadbeef0000",
            "rung": "sw_build",
            "category": "compile",
            "command": "make compile",
            "returncode": 1,
            "summary": "compile error marker",
            "source_locations": ["main.c:7"],
            "signature": "deadbeef" * 8,
            "log_path": self.log,
        }
        self.index = {"failures": {self.failure["id"]: self.failure}}

    def _tools(self, stage: str = "implementation") -> ReadOnlyToolset:
        return ReadOnlyToolset(
            RepoToolbox(REPO_ROOT, dry_run=False, stage=stage, run_root=self.run_root),
            failure_index=self.index,
        )

    def test_failure_tool_is_exposed_only_during_implementation(self) -> None:
        implementation = [item["function"]["name"] for item in self._tools().schemas()]
        analysis = [item["function"]["name"] for item in self._tools("analysis").schemas()]
        self.assertIn("inspect_failure", implementation)
        self.assertNotIn("inspect_failure", analysis)

    def test_failure_summary_and_log_are_bounded_and_policy_checked(self) -> None:
        tools = self._tools()
        summary = tools.run("inspect_failure", {"id": self.failure["id"]})
        self.assertIn("category: compile", summary)
        self.assertIn("main.c:7", summary)
        log = tools.run(
            "inspect_failure", {"id": self.failure["id"], "view": "log", "max_lines": 8}
        )
        self.assertIn("compile error marker", log)
        self.assertIn("log_path:", log)
        self.assertIn("unknown failure ID", tools.run("inspect_failure", {"id": "fail_missing"}))


if __name__ == "__main__":
    unittest.main()
