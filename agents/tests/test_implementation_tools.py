"""Isolated Stage 3 capability tests: no model, compiler, or real run artifacts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent.runtime.implementation_tools import ImplementationToolset, Verification
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.cfu_design_agent.runtime.policy import load_policy


class ImplementationToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Policy is copied into a disposable workdir, never modified by the test.
        real = Path(__file__).resolve().parents[1] / "specs/agent_access_policy.yaml"
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(real.read_bytes())
        self.run_root = "agents/generated/runs/test-1"
        self.hw = f"{self.run_root}/workspace/design/AgentCfu.scala"
        self.sw = f"{self.run_root}/workspace/demo_benchmark/kernel.c"
        for name, text in ((self.hw, "class AgentCfu {}\n"), (self.sw, "int x = 1;\n")):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        self.toolbox = RepoToolbox(self.root, policy=load_policy(self.root),
                                   stage="implementation", run_root=self.run_root,
                                   apply_patch=True)
        self.hw_calls = 0
        self.sw_calls = 0

    def hardware_check(self):
        self.hw_calls += 1
        return {"returncode": 0, "full_log_path": "09_implementation/checks/hw.log"}

    def software_check(self):
        self.sw_calls += 1
        return {"returncode": 1, "stderr_tail": "compile failure", "failure_observation": {
            "id": "fail_compile", "category": "compile", "summary": "compile failure"}}

    def make_tool(self, **kwargs):
        return ImplementationToolset(self.toolbox, frozenset({self.hw}),
                                     frozenset({self.sw}), check_elaboration=self.hardware_check,
                                     check_compile=self.software_check,
                                     **kwargs)

    def test_checks_before_edits_are_evidence_only_and_budgeted(self):
        tool = self.make_tool(max_hardware_checks=1, max_software_checks=1)
        names = {s["function"]["name"] for s in tool.schemas()}
        self.assertTrue({"read_file", "search_text", "edit_hardware", "edit_software",
                         "check_hardware_elaboration", "check_software_compile"} <= names)
        self.assertIn("passed (evidence only", tool.run("check_hardware_elaboration", {}))
        self.assertIn("failed", tool.run("check_software_compile", {}))
        self.assertEqual((self.hw_calls, self.sw_calls), (1, 1))
        self.assertEqual(tool.checks["hardware"][0]["candidate_hash"], tool.hardware_session.candidate_hash())
        self.assertEqual(tool.checks["software"][0]["candidate_hash"], tool.software_session.candidate_hash())
        self.assertIn("budget exhausted", tool.run("check_software_compile"))
        self.assertIn("no arguments", tool.run("check_hardware_elaboration", {"argv": ["sh"]}))
        self.assertEqual(self.hw_calls, 1)
        old_hash = tool.checks["hardware"][0]["candidate_hash"]
        self.assertNotIn("rejected", tool.run("edit_hardware", {"edits": [
            {"path": self.hw, "search": "class AgentCfu {}", "replace": "class Accel {}"}]}))
        self.assertNotEqual(old_hash, tool.hardware_session.candidate_hash())
        self.assertIsNone(tool.hardware_session.current_check())

    def test_lightweight_checks_are_unlimited_by_default(self):
        tool = self.make_tool()
        self.assertIsNone(tool.max_hardware_checks)
        self.assertIsNone(tool.max_software_checks)
        hw_text = "class AgentCfu {}\n"
        sw_text = "int x = 1;\n"
        for index in range(3):
            next_hw = f"class AgentCfu {{ val n{index} = {index} }}\n"
            self.assertNotIn("rejected", tool.run("edit_hardware", {"edits": [{
                "path": self.hw, "search": hw_text, "replace": next_hw}]}))
            self.assertIn("passed (evidence only", tool.run("check_hardware_elaboration", {}))
            hw_text = next_hw

            next_sw = f"int x = {index + 2};\n"
            self.assertNotIn("rejected", tool.run("edit_software", {
                "path": self.sw, "search": sw_text, "replace": next_sw}))
            self.assertIn("failed", tool.run("check_software_compile", {}))
            sw_text = next_sw
        self.assertEqual((self.hw_calls, self.sw_calls), (3, 3))
        self.assertEqual((len(tool.checks["hardware"]), len(tool.checks["software"])), (3, 3))

    def test_first_internal_check_failure_is_inspectable_and_resolved_in_place(self):
        for kind, name in (("hardware", "check_hardware_elaboration"),
                           ("software", "check_software_compile")):
            with self.subTest(kind=kind):
                index = {"failures": {}}
                callback = mock.Mock(side_effect=[
                    {"returncode": 1, "stderr_tail": "kernel.c:7: error: internal marker"},
                    {"returncode": 0},
                ])
                kwargs = {"check_elaboration" if kind == "hardware" else "check_compile": callback}
                tool = ImplementationToolset(
                    self.toolbox, frozenset({self.hw}), frozenset({self.sw}),
                    failure_index=index, **kwargs)
                self.assertIn("inspect_failure", {s["function"]["name"] for s in tool.schemas()})
                output = tool.run(name, {})
                ident = tool.checks[kind][0]["failure_id"]
                self.assertTrue(ident.startswith("fail_"))
                self.assertEqual(index["failures"][ident]["rung"],
                                 "hw_check" if kind == "hardware" else "sw_check")
                self.assertIn(ident, output)
                self.assertIs(tool.readonly.failure_index, index)
                self.assertIs(tool.hardware_session.failure_index, index)
                self.assertIs(tool.software_session.failure_index, index)
                summary = tool.run("inspect_failure", {"id": ident})
                self.assertIn("status: current", summary)
                self.assertIn("internal marker", summary)
                self.assertIn("passed (evidence only", tool.run(name, {}))
                self.assertIn("status: resolved", tool.run("inspect_failure", {"id": ident}))
                self.assertEqual(callback.call_count, 2)
                tool.rollback()

    def test_first_verification_failure_is_inspectable_and_resolved_by_same_rung(self):
        callback = mock.Mock(side_effect=[
            {"returncode": 1, "stderr_tail": "AgentCfu.scala:7: error: verification marker"},
            {"returncode": 0},
        ])
        tool = self.make_tool(verifications=(Verification(
            name="build_soc", description="isolated test build", label="SoC generation",
            final_gate="hw_build", budget=2, callback=callback),))
        output = tool.run("build_soc", {})
        failure = tool.checks["build_soc"][0]["failure_observation"]
        ident = failure["id"]
        self.assertIn(ident, output)
        self.assertEqual(failure["rung"], "hw_build")
        self.assertIn("status: current", tool.run("inspect_failure", {"id": ident}))
        # An isolated elaboration check is not the same rung as SoC generation.
        tool.run("check_hardware_elaboration", {})
        self.assertIn("status: current", tool.run("inspect_failure", {"id": ident}))
        tool.run("build_soc", {})
        self.assertIn("status: resolved", tool.run("inspect_failure", {"id": ident}))
        self.assertEqual(callback.call_count, 2)
        tool.rollback()

    def test_edit_budgets_bound_thrash_not_a_legitimate_patch(self):
        # Observed run-0003: a correct kernel patch needed three software edits
        # (include guard, kernel body, boundary fix) and the third was refused
        # with "budget exhausted" — right after the gate demanded that change.
        tool = self.make_tool()
        for search, replace in (("int x = 1;", "int x = 2;"),
                                ("int x = 2;", "int x = 3;"),
                                ("int x = 3;", "int x = 4;")):
            out = tool.run("edit_software", {"path": self.sw, "search": search, "replace": replace})
            self.assertNotIn("budget exhausted", out, out)
        exhausted = self.make_tool(max_software_edits=1)
        # A refused edit does not consume the budget...
        self.assertIn("rejected", exhausted.run(
            "edit_software", {"path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"}))
        self.assertNotIn("budget exhausted", exhausted.run(
            "edit_software", {"path": self.sw, "search": "int x = 4;", "replace": "int x = 5;"}))
        # ...and the second successful edit is where a 1-edit budget stops.
        self.assertIn(
            "budget exhausted",
            exhausted.run("edit_software", {"path": self.sw, "search": "int x = 5;", "replace": "int x = 6;"}),
        )

    def test_a_gate_rejection_reopens_the_edit_and_check_budgets(self):
        tool = self.make_tool(max_software_edits=1, max_software_checks=1)
        tool.run("edit_software", {"path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"})
        self.assertIn(
            "budget exhausted",
            tool.run("edit_software", {"path": self.sw, "search": "int x = 2;", "replace": "int x = 3;"}),
        )
        tool.grant_correction()
        self.assertNotIn(
            "budget exhausted",
            tool.run("edit_software", {"path": self.sw, "search": "int x = 2;", "replace": "int x = 3;"}),
        )
        self.assertNotIn("budget exhausted", tool.run("check_software_compile", {}))

    def test_edit_order_cross_target_atomicity_and_rollback(self):
        tool = self.make_tool()
        self.assertNotIn("rejected", tool.run("edit_software", {
            "path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"}))
        bad = tool.run("edit_hardware", {"edits": [
            {"path": self.hw, "search": "class AgentCfu {}", "replace": "class Accel {}"},
            {"path": self.hw, "search": "missing", "replace": "x"}]})
        self.assertIn("rejected", bad)
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
        self.assertNotIn("rejected", tool.run("edit_hardware", {"edits": [
            {"path": "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
             "search": "class AgentCfu {}", "replace": "class Accel {}"}]}))
        self.assertEqual(set(tool.touched_paths), {self.hw, self.sw})
        tool.rollback()
        self.assertEqual(tool.touched_paths, [])
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
        self.assertEqual((self.root / self.sw).read_text(), "int x = 1;\n")
        self.assertIn("closed", tool.run("edit_software", {}))

    def test_commit_preserves_both_target_groups(self):
        tool = self.make_tool()
        tool.run("edit_hardware", {"edits": [{"path": self.hw, "search": "class AgentCfu {}", "replace": "class Accel {}"}]})
        tool.run("edit_software", {"path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"})
        tool.commit()
        with self.assertRaises(RuntimeError):
            tool.rollback()
        self.assertIn("Accel", (self.root / self.hw).read_text())
        self.assertIn("2", (self.root / self.sw).read_text())

    def test_missing_callback_not_exposed_and_diagnosis_unlocks_both(self):
        tool = ImplementationToolset(self.toolbox, frozenset({self.hw}), frozenset({self.sw}),
                                     diagnosis_required=True,
                                     failure_index={"failures": {"fail_now": {"status": "current"}}})
        names = {s["function"]["name"] for s in tool.schemas()}
        self.assertNotIn("check_hardware_elaboration", names)
        self.assertNotIn("edit_dev_test", names)
        self.assertIn("rejected", tool.run("edit_hardware", {"edits": [
            {"path": self.hw, "search": "class AgentCfu {}", "replace": "class Accel {}"}]}))
        self.assertIn("accepted", tool.run("submit_repair_diagnosis", {
            "failure_ids": ["fail_now"], "observed_errors": ["compiler error"],
            "root_cause": "bad implementation", "repair_mapping": ["fix both"],
            "regression_risks": ["none"]}))
        self.assertNotIn("rejected", tool.run("edit_software", {
            "path": self.sw, "search": "int x = 1;", "replace": "int x = 2;"}))
        tool.rollback()

    def test_reject_bad_target_groups_and_policy_confinement(self):
        self.assertTrue(self.toolbox.is_write_allowed(self.sw))
        oracle = f"{self.run_root}/workspace/formal_conformance/test.c"
        self.assertFalse(self.toolbox.is_write_allowed(oracle))
        with self.assertRaises(ValueError):
            ImplementationToolset(self.toolbox, frozenset({"src/main/scala/AgentCfu.scala"}),
                                  frozenset({self.sw}))


if __name__ == "__main__":
    unittest.main()
