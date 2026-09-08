"""RepoToolbox read/write/copy policy enforcement."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.policy import PolicyError
from agents.cfu_design_agent.tools import RepoToolbox
from agents.tests.common import REPO_ROOT


class ToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.toolbox = RepoToolbox(REPO_ROOT, dry_run=True, stage="load_context")

    def test_read_allows_declared_context(self) -> None:
        text = self.toolbox.read_text("skills/cfu-designer/SKILL.md", limit=2000)
        self.assertTrue(text)

    def test_read_denies_generated_run(self) -> None:
        with self.assertRaises(PolicyError):
            self.toolbox.read_text("agents/generated/runs/20260729-162009/report.md")

    def test_read_denies_git(self) -> None:
        with self.assertRaises(PolicyError):
            self.toolbox.read_text(".git/config")

    def test_write_dry_run_records_without_writing_source(self) -> None:
        with self.assertRaises(PolicyError):
            self.toolbox.write_text("src/main/scala/vexiiriscv/soc/mico/NewCfu.scala", "// x")

    def test_git_status_only_exposes_visible_paths(self) -> None:
        status = self.toolbox.git_status()
        self.assertNotIn(".git/", status)

    def test_command_allowed_matches_policy(self) -> None:
        self.assertTrue(
            self.toolbox.command_allowed(
                ["git", "status", "--short"],
                command_id="git_status",
            )
        )
        self.assertFalse(
            self.toolbox.command_allowed(["rm", "-rf", "/"], command_id="git_status"),
        )


if __name__ == "__main__":
    unittest.main()
