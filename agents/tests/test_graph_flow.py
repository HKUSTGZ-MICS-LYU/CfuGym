"""Deterministic dry-run flow and prompt visibility."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.graph import run_agent_python
from agents.cfu_design_agent.prompts import context_digest
from agents.tests.common import REPO_ROOT


class GraphFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "flow-test-00000000-0000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"

    def tearDown(self) -> None:
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_dry_run_plans_without_completing(self) -> None:
        state = {
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": True,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": True,
            "skip_yosys": True,
            "max_iters": 1,
            "run_id": self.run_id,
        }
        final = run_agent_python(state)
        self.assertEqual(final["status"], "planned")  # forced by dry_run
        self.assertTrue(final["final_report_path"])
        self.assertTrue((REPO_ROOT / final["final_report_path"]).exists())
        self.assertTrue((self.run_dir / "run.yaml").exists())

    def test_ir_auto_dry_run_does_not_execute_and_does_not_block(self) -> None:
        state = {
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": True,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": True,
            "skip_yosys": True,
            "ir_features": "auto",
            "max_iters": 1,
            "run_id": self.run_id,
        }
        final = run_agent_python(state)
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final["ir_result"]["status"], "unavailable")
        self.assertTrue(final["gates"]["ir"]["ok"])
        self.assertTrue((self.run_dir / "07_analysis/ir_features.yaml").exists())

    def test_context_digest_omits_sensitive_and_other_runs(self) -> None:
        context = {
            "files": {
                "skills/cfu-designer/SKILL.md": "allowed",
                ".git/config": "secret",
                "agents/generated/runs/20260729-162009/report.md": "other run",
            },
            "git_status": "",
            "cfu_entrypoints": [],
            "software_tests": [],
        }
        digest = context_digest(context)
        self.assertIn("allowed", digest)
        self.assertNotIn(".git/config", digest)
        self.assertNotIn("other run", digest)

class OverlayCommandTest(unittest.TestCase):
    """The overlay command swaps sources without touching the repository."""

    def test_overlay_command_swaps_scaffolds_and_scopes_output(self) -> None:
        from agents.cfu_design_agent.graph import agent_cfu_overlay_commands
        from agents.cfu_design_agent.runtime.policy import load_policy

        run_root = "agents/generated/runs/overlay-test"
        state = {"run_root": run_root, "workdir": str(REPO_ROOT)}
        commands = agent_cfu_overlay_commands(
            state, {"datapath": {"vlen": 128, "xlen": 32, "regDepth": 4}}
        )
        self.assertEqual(len(commands), 1)
        argv = commands[0]
        self.assertEqual(argv[0], "sbt")
        self.assertTrue(any(f"{run_root}/workspace/design" in token for token in argv))
        self.assertTrue(any("AgentCfuSourceFilter.sources" in token for token in argv))
        self.assertTrue(any(f"{run_root}/workspace/soc" in token for token in argv))
        run_main = argv[-1]
        self.assertIn("runMain vexiiriscv.soc.mico.MiCoSocGen", run_main)
        self.assertIn("--mico-agent-cfu", run_main)
        self.assertIn("--agent-cfu-vlen 128", run_main)
        self.assertIn("--agent-cfu-reg-depth 4", run_main)
        # No absolute path or shell metacharacter may appear in the argv.
        for token in argv:
            self.assertFalse(token.startswith("/"), token)
            for bad in (";", "&", "|", "`", "$("):
                self.assertNotIn(bad, token, token)
        load_policy(REPO_ROOT).check_command(
            "validation", "soc_generate_agent", argv, REPO_ROOT
        )

    def test_dry_run_seeds_and_reports_design_workspace(self) -> None:
        run_id = "flow-overlay-test-00000000"
        run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            final = run_agent_python({
                "workdir": str(REPO_ROOT),
                "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
                "input_mode": "natural_language",
                "dry_run": True,
                "run_commands": False,
                "apply_patch": False,
                "skip_sim": True,
                "skip_yosys": True,
                "max_iters": 1,
                "run_id": run_id,
            })
            design = final.get("design_workspace", {})
            self.assertTrue(design.get("root", "").endswith("workspace/design"))
            self.assertEqual(len(final.get("design_files", [])), 2)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


class RepairFeedbackTest(unittest.TestCase):
    """A hardware repair must see every failing rung's exact error log."""

    def test_hardware_repair_feedback_labels_each_source(self) -> None:
        from agents.cfu_design_agent.graph import hardware_repair_feedback

        state = {
            "validation_results": [
                {
                    "rung": "hw_build",
                    "returncode": 1,
                    "command": "sbt ... MiCoSocGen",
                    "stdout_tail": "BUILD_ERROR_MARKER",
                    "stderr_tail": "",
                },
                {
                    "rung": "sim",
                    "returncode": 1,
                    "command": "sbt ... MiCoSocSim",
                    "stdout_tail": "SIM_ERROR_MARKER",
                    "stderr_tail": "",
                },
                {
                    "rung": "hw_build",
                    "returncode": 0,
                    "command": "sbt ... MiCoSocGen",
                    "stdout_tail": "IGNORED_SUCCESS",
                    "stderr_tail": "",
                },
            ],
            "cost_results": [
                {
                    "returncode": 1,
                    "command": "python3 yosys_cost_report.py",
                    "stdout_tail": "COST_ERROR_MARKER",
                    "stderr_tail": "",
                }
            ],
            "gates": {"implementation_hw": {"errors": ["no usable patch"]}},
        }
        feedback = hardware_repair_feedback(state)
        self.assertNotIn("hw_build (SoC generation) errors", feedback)
        self.assertIn("sim (simulation) errors", feedback)
        self.assertIn("cost (Yosys) errors", feedback)
        self.assertIn("implementation_hw rejection", feedback)
        self.assertNotIn("BUILD_ERROR_MARKER", feedback)
        self.assertIn("SIM_ERROR_MARKER", feedback)
        self.assertIn("COST_ERROR_MARKER", feedback)
        self.assertIn("no usable patch", feedback)
        self.assertNotIn("IGNORED_SUCCESS", feedback)

    def test_hardware_repair_feedback_is_empty_without_failures(self) -> None:
        from agents.cfu_design_agent.graph import hardware_repair_feedback

        self.assertEqual(hardware_repair_feedback({}), "")


if __name__ == "__main__":
    unittest.main()
