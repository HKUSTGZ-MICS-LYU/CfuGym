"""Machine-readable verdicts are parsed before command output truncation."""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT


class FullOutputParsingTest(unittest.TestCase):
    def _run(self, output: str) -> dict:
        toolbox = RepoToolbox(REPO_ROOT, dry_run=False, stage="validation")
        completed = subprocess.CompletedProcess(["make"], 0, stdout=output, stderr="")
        with mock.patch(
            "agents.cfu_design_agent.runtime.tools.subprocess.run", return_value=completed
        ):
            return toolbox.run_command(
                ["make", "-C", "sw", "compile", "MAIN=tests/a", "TARGET=vexii_soc"],
                command_id="project_test",
                capture_full=True,
            )

    def test_embench_profile_before_long_epilogue_is_preserved(self) -> None:
        output = (
            "CFU_AGENT_PROFILE crc32=100 calls=3\n"
            "EMBENCH_VERIFY_PASS benchmark=crc32 verify=1 result=42\n"
            + "late chatter\n" * 1000
        )
        result = self._run(output)
        self.assertNotIn("CFU_AGENT_PROFILE", result["stdout_tail"])
        profile = result["parsed"]["embench_profile"]
        self.assertEqual(profile["crc32"]["cycles"], 100)
        self.assertEqual(profile["crc32"]["result"], 42)

    def test_conformance_verdict_before_long_epilogue_is_preserved(self) -> None:
        output = (
            "CFU_CONFORMANCE PASS commands=2/2 cases=10 failed=none\n"
            + "late chatter\n" * 1000
        )
        result = self._run(output)
        self.assertNotIn("CFU_CONFORMANCE", result["stdout_tail"])
        verdict = result["parsed"]["conformance"]
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["commands"], 2)


if __name__ == "__main__":
    unittest.main()
