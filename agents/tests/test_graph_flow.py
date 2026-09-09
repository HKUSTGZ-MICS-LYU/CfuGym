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
        from agents.cfu_design_agent.policy import load_policy

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


if __name__ == "__main__":
    unittest.main()
