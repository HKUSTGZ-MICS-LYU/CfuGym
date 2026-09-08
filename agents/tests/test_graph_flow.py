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


if __name__ == "__main__":
    unittest.main()
