"""Agent-authored dev tests: ownership boundary, transaction and runner verdicts.

The point of a dev test is that the model owns the *scenario* while the host owns
the *verdict*. These tests pin that split: what the model may change, what the
host refuses, and which verdict lines are accepted.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent import conformance_dev_runner
from agents.cfu_design_agent.flow import gates
from agents.cfu_design_agent.runtime import dev_tests
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


class RegionValidationTest(unittest.TestCase):
    def test_the_pristine_template_is_a_valid_starting_point(self) -> None:
        self.assertEqual(
            dev_tests.region_problems(dev_tests.template_text(), allow_stub=True), [])

    def test_the_pristine_template_is_not_runnable(self) -> None:
        problems = dev_tests.region_problems(dev_tests.template_text())
        self.assertTrue(any("cfu_conformance_dut calls no cfu_*" in item for item in problems))

    def test_a_filled_template_is_valid(self) -> None:
        text = dev_tests.template_text().replace(
            "return 0xFFFFFFFFu;", "return cfu_add(1u, 2u);")
        self.assertEqual(dev_tests.region_problems(text), [])

    def test_the_reference_may_not_call_the_device(self) -> None:
        text = dev_tests.template_text().replace(
            "return 0u;", "return cfu_sneaky(1u);")
        problems = dev_tests.region_problems(text)
        self.assertTrue(any("cfu_conformance_ref calls the device" in item for item in problems))

    def test_the_device_side_may_not_return_the_reference(self) -> None:
        text = dev_tests.template_text().replace(
            "return 0xFFFFFFFFu;", "return cfu_conformance_ref();")
        problems = dev_tests.region_problems(text)
        self.assertTrue(any("vacuous" in item for item in problems))

    def test_main_is_rejected_inside_the_model_region(self) -> None:
        text = dev_tests.template_text().replace(
            "return 0u;", "return 0u; }\nint main(void) { return 0; }\nuint32_t unused_ref(void) {")
        problems = dev_tests.region_problems(text)
        self.assertTrue(any("must not define main" in item for item in problems))

    def test_a_modified_header_is_rejected(self) -> None:
        text = dev_tests.template_text().replace(
            '#include "cfu_conf.h"', '#include "cfu_conf.h"\n#define SNEAKY 1')
        problems = dev_tests.region_problems(text, allow_stub=True)
        self.assertTrue(any("host-owned header was modified" in item for item in problems))

    def test_content_after_the_end_marker_is_rejected(self) -> None:
        text = dev_tests.template_text() + "\nint extra;\n"
        problems = dev_tests.region_problems(text, allow_stub=True)
        self.assertTrue(any("after the MODEL REGION end marker" in item for item in problems))

    def test_missing_markers_are_reported_once(self) -> None:
        text = dev_tests.template_text().replace(dev_tests.MODEL_REGION_END, "")
        problems = dev_tests.region_problems(text, allow_stub=True)
        self.assertTrue(any("markers are missing" in item for item in problems))


class TestNameTest(unittest.TestCase):
    def test_valid_names_are_normalised(self) -> None:
        self.assertEqual(dev_tests.validate_name("  crc32_roundtrip "), "crc32_roundtrip")

    def test_path_like_and_empty_names_are_refused(self) -> None:
        for bad in ("", "   ", "a/b", "../escape", ".hidden", "1leading", "a b",
                    "x" * 40, "_host", "name.c", None, 7):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    dev_tests.validate_name(bad)


class WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = REPO_ROOT / "agents/specs/agent_access_policy.yaml"
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(policy.read_bytes())
        self.run_root = "agents/generated/runs/devtest-test"
        self.toolbox = RepoToolbox(self.root, policy=load_policy(self.root),
                                   stage="implementation", run_root=self.run_root,
                                   apply_patch=True)
        self.workspace = dev_tests.DevTestWorkspace(
            self.toolbox, dev_tests.root_for({"run_root": self.run_root}))

    def _path(self, name: str) -> Path:
        return self.root / self.workspace.path(name)

    def test_create_writes_the_template_and_the_host_files(self) -> None:
        answer = self.workspace.create("crc32")
        self.assertIn("created", answer)
        text = self._path("crc32").read_text()
        self.assertTrue(text.startswith(dev_tests.TEMPLATE_HEADER))
        host = self.root / self.workspace.host_dir
        self.assertTrue((host / dev_tests.RUNNER_SOURCE).is_file())
        self.assertTrue((host / dev_tests.HOST_HEADER).is_file())
        self.assertTrue((host / "crc32_runner.c").is_file())

    def test_a_second_create_is_refused(self) -> None:
        self.workspace.create("crc32")
        self.assertIn("already exists", self.workspace.create("crc32"))

    def test_the_test_budget_is_enforced(self) -> None:
        small = dev_tests.DevTestWorkspace(self.toolbox, self.workspace.root, max_tests=1)
        small.create("one")
        self.assertIn("budget exhausted", small.create("two"))

    def test_edit_inside_the_region_is_applied(self) -> None:
        self.workspace.create("crc32")
        answer = self.workspace.edit(
            "crc32", "  return 0xFFFFFFFFu;", "  return cfu_crc32_step(1u);")
        self.assertIn("edited", answer)
        self.assertIn("cfu_crc32_step", self._path("crc32").read_text())

    def test_an_edit_outside_the_region_restores_the_file(self) -> None:
        self.workspace.create("crc32")
        before = self._path("crc32").read_text()
        answer = self.workspace.edit("crc32", '#include "cfu_conf.h"', '#include "other.h"')
        self.assertIn("header was modified", answer)
        self.assertEqual(self._path("crc32").read_text(), before)

    def test_edit_of_an_unknown_test_is_refused(self) -> None:
        answer = self.workspace.edit("nope", "a", "b")
        self.assertIn("does not exist", answer)

    def test_rollback_removes_created_tests_and_keeps_the_host_files(self) -> None:
        self.workspace.create("crc32")
        created = self._path("crc32")
        host = self.root / self.workspace.host_dir
        self.workspace.rollback()
        self.assertFalse(created.exists())
        self.assertTrue((host / dev_tests.RUNNER_SOURCE).is_file())
        self.assertTrue(host.is_dir())

    def test_commit_closes_the_transaction(self) -> None:
        self.workspace.create("crc32")
        self.workspace.commit()
        self.assertIn("transaction is closed", self.workspace.create("other"))

    def test_the_host_directory_is_unreachable_by_name(self) -> None:
        for name in ("_host", "_host/runner", "../_host/runner"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.workspace.path(name)


class RunFromStateTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = REPO_ROOT / "agents/specs/agent_access_policy.yaml"
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(policy.read_bytes())
        self.run_root = "agents/generated/runs/devtest-run"
        self.toolbox = RepoToolbox(self.root, policy=load_policy(self.root),
                                   stage="implementation", run_root=self.run_root,
                                   apply_patch=True)
        self.workspace = dev_tests.DevTestWorkspace(
            self.toolbox, dev_tests.root_for({"run_root": self.run_root}))

    def test_no_harness_refuses_and_runs_nothing(self) -> None:
        from agents.cfu_design_agent.runtime import effects

        with mock.patch.object(effects, "run_command") as execute:
            answer = dev_tests.run_from_state({}, None, "", {"test": "x"})
            self.assertIn("disabled", answer["stderr_tail"])
            execute.assert_not_called()

    def test_a_missing_test_lists_what_exists(self) -> None:
        self.workspace.create("known")
        answer = dev_tests.run_from_state(
            {"run_root": self.run_root}, self.workspace, "", {"test": "absent"})
        self.assertIn("does not exist", answer["stderr_tail"])
        self.assertIn("known", answer["stderr_tail"])

    def test_a_pristine_template_is_not_run(self) -> None:
        from agents.cfu_design_agent.runtime import effects

        self.workspace.create("stub")
        with mock.patch.object(effects, "run_command") as execute:
            answer = dev_tests.run_from_state(
                {"run_root": self.run_root, "isa_spec": {}},
                self.workspace, "header.h", {"test": "stub"})
            self.assertIn("not runnable", answer["stderr_tail"])
            execute.assert_not_called()


class RunnerTest(unittest.TestCase):
    def test_the_runner_owns_the_entry_point_and_the_verdict(self) -> None:
        text = conformance_dev_runner.render_runner()
        self.assertIn("int main(void)", text)
        self.assertIn("cfu_conformance_dut();", text)
        self.assertIn("cfu_conformance_ref();", text)
        self.assertIn("cfu_enable();", text)
        self.assertIn("CFU_CONF_NONCE", text)

    def test_the_header_declares_the_harness_data(self) -> None:
        text = conformance_dev_runner.render_header()
        self.assertIn("extern uint8_t cfu_conf_data", text)
        self.assertIn("extern uint32_t cfu_conf_len", text)

    def test_this_runs_verdict_is_accepted(self) -> None:
        nonce = 0x1A2B3C4D
        line = f"CFU_CONFORMANCE_{nonce:08x} PASS cases=33 failed=0"
        verdict = conformance_dev_runner.parse_verdict(line, nonce, test="t")
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["cases"], 33)

    def test_the_sbt_prefix_is_tolerated(self) -> None:
        nonce = 0x0000BEEF
        line = f"[info] CFU_CONFORMANCE_{nonce:08x} PASS cases=33 failed=0"
        self.assertTrue(conformance_dev_runner.parse_verdict(line, nonce)["ok"])

    def test_a_foreign_nonce_cannot_forge_a_pass(self) -> None:
        line = "CFU_CONFORMANCE_00000001 PASS cases=33 failed=0"
        verdict = conformance_dev_runner.parse_verdict(line, 0xDEADBEEF, test="t")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["status"], "missing")

    def test_a_failing_verdict_is_not_ok(self) -> None:
        nonce = 7
        line = f"CFU_CONFORMANCE_{nonce:08x} FAIL cases=33 failed=2"
        verdict = conformance_dev_runner.parse_verdict(line, nonce)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed"], 2)
        self.assertIn("2 of 33", verdict["reason"])

    def test_a_pass_with_failures_is_rejected(self) -> None:
        nonce = 7
        line = f"CFU_CONFORMANCE_{nonce:08x} PASS cases=33 failed=1"
        self.assertFalse(conformance_dev_runner.parse_verdict(line, nonce)["ok"])

    def test_silence_is_missing_not_a_pass(self) -> None:
        verdict = conformance_dev_runner.parse_verdict("no output at all", 5)
        self.assertEqual(verdict["status"], "missing")
        self.assertFalse(verdict["ok"])

    def test_a_mismatch_is_a_verdict_not_a_crash(self) -> None:
        """Returning non-zero traps the simulator on `fail` and hides got/want."""
        text = conformance_dev_runner.render_runner()
        self.assertIn("return 0;", text)
        self.assertNotIn("failed ? 1 : 0", text)

    def test_the_case_count_matches_the_sweep(self) -> None:
        self.assertEqual(
            conformance_dev_runner.expected_cases(),
            len(conformance_dev_runner.BOUNDARY_LENGTHS)
            * len(conformance_dev_runner.SELECTOR_VALUES),
        )


class ConformanceModeGateTest(unittest.TestCase):
    def test_the_gate_is_required_for_the_legacy_mode_only(self) -> None:
        self.assertTrue(gates.conformance_gates({"conformance_mode": "gate"}))
        self.assertFalse(gates.conformance_gates({"conformance_mode": "dev"}))
        self.assertTrue(gates.conformance_gates(
            {"conformance_mode": "dev", "conformance_promote": True}))
        self.assertFalse(gates.conformance_gates({"conformance_mode": "off"}))

    def test_older_states_keep_the_historical_meaning(self) -> None:
        self.assertEqual(gates.conformance_mode({"require_conformance": True}), "gate")
        self.assertEqual(gates.conformance_mode({"require_conformance": False}), "off")
        self.assertEqual(gates.conformance_mode({}), "gate")

    def test_hw_conformance_is_not_required_in_dev_mode(self) -> None:
        self.assertNotIn("hw_conformance", gates.required_gates({"conformance_mode": "dev"}))
        self.assertIn("hw_conformance", gates.required_gates({"conformance_mode": "gate"}))


if __name__ == "__main__":
    unittest.main()


class DevCommandTest(unittest.TestCase):
    """The command a dev test compiles through must be both buildable and allowed.

    Two regressions from run 20261031-043206 are pinned here. (1) ``MAIN`` dropped
    the ``_host/`` segment, so make looked for a file that did not exist and every
    test died with "No rule to make target". (2) The build used the legacy
    ``conformance_build`` command id, whose policy requires the fixed
    ``cfu_conformance`` bin name, so any test named differently was rejected
    before it compiled.
    """

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = REPO_ROOT / "agents/specs/agent_access_policy.yaml"
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(policy.read_bytes())
        self.run_root = "agents/generated/runs/devcmd-test"
        self.state = {"run_root": self.run_root, "isa_spec": {}}
        self.header = f"{self.run_root}/workspace/bench/cfu/include/cfu_intrinsics.h"

    def _argv(self, name: str) -> list[str]:
        from agents.cfu_design_agent.runtime import dev_commands

        argv, _elf = dev_commands.dev_conformance_build_command(
            self.state, name, self.header, 0x1234)
        return argv

    def test_main_and_elf_live_in_the_host_directory(self) -> None:
        argv = self._argv("md5_block")
        main = next(item for item in argv if item.startswith("MAIN=")).split("=", 1)[1]
        self.assertIn("/_host/", main)
        self.assertTrue(main.endswith("md5_block_runner"))
        from agents.cfu_design_agent.runtime import dev_commands

        _argv, elf = dev_commands.dev_conformance_build_command(
            self.state, "md5_block", self.header, 1)
        self.assertIn("/_host/", elf)

    def test_the_argv_declares_the_test_as_extra_sources(self) -> None:
        argv = self._argv("md5_block")
        extra = next(item for item in argv if item.startswith("EXTRA_SOURCES="))
        self.assertIn("workspace/dev_tests/md5_block.c", extra)
        # The model's translation unit must never be MAIN: the host owns main().
        main = next(item for item in argv if item.startswith("MAIN="))
        self.assertNotIn("md5_block.c", main)

    def test_the_policy_accepts_a_test_named_anything(self) -> None:
        from agents.cfu_design_agent.runtime.policy import load_policy

        policy = load_policy(self.root)
        for name in ("md5_block", "md5_boundary", "cfu_conformance"):
            with self.subTest(name=name):
                policy.check_command(
                    "conformance", "dev_conformance_build", self._argv(name), self.root)

    def test_the_legacy_command_id_still_requires_its_fixed_bin(self) -> None:
        from agents.cfu_design_agent.runtime.policy import PolicyError, load_policy

        policy = load_policy(self.root)
        with self.assertRaisesRegex(PolicyError, "cfu_conformance"):
            policy.check_command(
                "conformance", "conformance_build", self._argv("md5_block"), self.root)


class DevConformanceModeTest(unittest.TestCase):
    def test_dev_without_promotion_stays_off_the_happy_path(self) -> None:
        from agents.cfu_design_agent.flow import ladder

        for state, expected in (
            ({"conformance_mode": "dev", "gates": {}}, "build_sw"),
            ({"conformance_mode": "dev", "conformance_promote": True, "gates": {}},
             "check_conformance"),
            ({"conformance_mode": "gate", "gates": {}}, "check_conformance"),
            ({"conformance_mode": "off", "gates": {}}, "build_sw"),
        ):
            with self.subTest(state=state):
                self.assertEqual(ladder._route_after_hw_build(state), expected)


class EvidenceOnlyRungTest(unittest.TestCase):
    def test_a_non_promoted_rung_records_checks_without_failing_the_run(self) -> None:
        from unittest import mock

        from agents.cfu_design_agent.runtime import stage_services
        from agents.cfu_design_agent.stages.verify import conformance as verify_conformance

        workspace = mock.Mock()
        workspace.names.return_value = ["md5_block"]
        workspace.runs = [{"test": "md5_block", "verdict": {"ok": False, "status": "fail"}}]

        state = {"run_root": "agents/generated/runs/x", "conformance_mode": "dev",
                 "gates": {}, "isa_spec": {}}
        with mock.patch.object(stage_services, "_toolbox_for_stage", return_value=mock.Mock()), \
             mock.patch.object(verify_conformance.dev_tests, "DevTestWorkspace",
                               return_value=workspace), \
             mock.patch.object(verify_conformance.dev_tests, "tree_digest",
                               return_value={"md5_block": "deadbeef"}), \
             mock.patch.object(verify_conformance.dev_tests, "run_one",
                               return_value={"returncode": 1, "parsed": {"dev_conformance": {
                                   "status": "fail", "ok": False, "cases": 1, "failed": 1}}}), \
             mock.patch.object(stage_services, "_artifacts_for_stage") as artifacts, \
             mock.patch.object(stage_services, "_write_ladder_artifact"):
            artifacts.return_value = mock.Mock()
            update = verify_conformance._check_dev_tests(state, "hw_conformance")

        gate = update["gates"]["hw_conformance"]
        self.assertTrue(gate["ok"], "evidence-only checks must not fail the run")
        self.assertTrue(gate["evidence_only"])
        self.assertEqual(gate["verdict"], "fail")
        self.assertEqual(gate["next"], "build_sw")
        self.assertEqual(update["errors"], [])


class DevTestResultFeedbackTest(unittest.TestCase):
    """A failed dev test must hand the model its cases, not a stack trace."""

    def _run(self, stdout: str, nonce: int = 0x2A) -> dict:
        import pathlib
        from unittest import mock

        from agents.cfu_design_agent import conformance_dev_runner
        from agents.cfu_design_agent.runtime import dev_commands, dev_tests, effects

        workspace = mock.Mock()
        workspace.exists.return_value = True
        workspace.read.return_value = dev_tests.template_text().replace(
            "  return 0xFFFFFFFFu;", "  return cfu_x();")
        workspace.toolbox = mock.Mock()
        workspace.toolbox.workdir = pathlib.Path(".")
        with mock.patch.object(conformance_dev_runner, "new_nonce", return_value=nonce), \
             mock.patch.object(dev_commands, "dev_conformance_build_command",
                               return_value=(["make"], "x.elf")), \
             mock.patch.object(effects, "run_command",
                               side_effect=[{"returncode": 0},
                                            {"returncode": 0, "parsed": {}, "stdout_tail": stdout}]):
            return dev_tests.run_one({"run_root": "x", "isa_spec": {}}, workspace, "t", "h.h",
                                     mock.Mock())

    def test_a_failing_verdict_becomes_a_readable_check_failure(self) -> None:
        result = self._run(
            "CFU_CONFORMANCE_0000002a FAIL cases=33 failed=33\n"
            "CFU_CONFORMANCE_CASE nonce=0000002a len=0 index=0 got=7d839644 want=d75c2e52\n"
            "CFU_CONFORMANCE_CASE nonce=0000002a len=0 index=1 got=7d839644 want=637d4c57")
        self.assertEqual(result["returncode"], 1, "a mismatch must fail the check, not crash it")
        self.assertIn("FAIL cases=33 failed=33", result["stderr_tail"])
        self.assertIn("got=7d839644 want=d75c2e52", result["stderr_tail"])
        self.assertFalse(result["parsed"]["dev_conformance"]["ok"])

    def test_a_passing_verdict_leaves_a_zero_status(self) -> None:
        result = self._run("CFU_CONFORMANCE_0000002a PASS cases=33 failed=0")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["parsed"]["dev_conformance"]["ok"])

    def test_a_missing_verdict_keeps_the_simulation_evidence(self) -> None:
        result = self._run("trap at 0x80000100: no verdict printed")
        self.assertEqual(result["parsed"]["dev_conformance"]["status"], "missing")
        self.assertIn("no verdict printed", result["stdout_tail"])


class StaleTestDetectionTest(unittest.TestCase):
    """A test written for an earlier ISA must not silently describe the delivery."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_root = "agents/generated/runs/stale-test"
        self.dir = self.root / self.run_root / "workspace/dev_tests"
        self.dir.mkdir(parents=True)
        self.isa = {"commands": [{
            "name": "md5_round", "function_id": 1, "subfunction_id": 0, "semantics": "round",
        }]}

    def _state(self, isa=None) -> dict:
        return {"run_root": self.run_root, "workdir": str(self.root),
                "isa_spec": self.isa if isa is None else isa}

    def test_an_intrinsic_the_isa_removed_is_reported(self) -> None:
        (self.dir / "old.c").write_text(
            '#include "cfu_intrinsics.h"\n'
            "uint32_t cfu_conformance_dut(void) { return cfu_md5_step_r7(1u, 2u); }\n"
            "uint32_t cfu_conformance_ref(void) { return 0u; }\n")
        self.assertEqual(dev_tests.stale_tests_from_state(self._state()),
                         {"old": ["cfu_md5_step_r7"]})

    def test_a_function_pointer_reference_is_reported(self) -> None:
        """Run 20261031-082724 called the intrinsics through a table, not directly."""
        (self.dir / "table.c").write_text(
            "static const uint32_t (*ops[2])(uint32_t, uint32_t) = "
            "{ cfu_md5_step_r7, cfu_md5_step_r12 };\n")
        self.assertEqual(dev_tests.stale_tests_from_state(self._state()),
                         {"table": ["cfu_md5_step_r12", "cfu_md5_step_r7"]})

    def test_a_current_test_and_harness_names_are_not_stale(self) -> None:
        (self.dir / "ok.c").write_text(
            "uint32_t d(void) { cfu_enable(); return cfu_md5_round(cfu_conf_len, "
            "cfu_conf_data[0]); }\n")
        self.assertEqual(dev_tests.stale_tests_from_state(self._state()), {})

    def test_no_contract_means_no_staleness_claim(self) -> None:
        (self.dir / "any.c").write_text("uint32_t f(void) { return cfu_whatever(1u); }\n")
        self.assertEqual(dev_tests.stale_tests_from_state(self._state(isa={})), {})

    def test_the_report_says_the_runs_describe_an_earlier_revision(self) -> None:
        from agents.cfu_design_agent.stages.review import report

        (self.dir / "old.c").write_text(
            "uint32_t cfu_conformance_dut(void) { return cfu_md5_step_r7(1u, 2u); }\n")
        state = {**self._state(), "conformance_mode": "dev",
                 "conformance_tests": [{"name": "old"}],
                 "conformance_runs": [{"test": "old", "verdict": {
                     "ok": True, "status": "pass", "cases": 33, "failed": 0}}]}
        lines = "\n".join(report.conformance_lines(state))
        self.assertIn("stale:", lines)
        self.assertIn("cfu_md5_step_r7", lines)
        self.assertIn("no runnable dev test covers the delivered ISA", lines)
