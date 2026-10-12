"""Security and authority boundaries of the software coding-loop pilot."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.runtime.software_tools import SoftwareCodingToolset
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT


class SoftwareCodingToolsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"software-tools-{self._testMethodName}"[:60]
        self.run_root = f"agents/generated/runs/{self.run_id}"
        self.run_dir = REPO_ROOT / self.run_root
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        self.target = f"{self.run_root}/workspace/cfu_benchmark/main.c"
        self.other = f"{self.run_root}/workspace/cfu_benchmark/other.c"
        for path, text in ((self.target, "int x = 1;\n"), (self.other, "int y = 1;\n")):
            full = REPO_ROOT / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(text)
        self.checks = []

        def check():
            self.checks.append(True)
            return {
                "returncode": 1,
                "stderr_tail": "compile error",
                "full_log_path": f"{self.run_root}/10_validation/logs/sw_check_000.log",
                "failure_observation": {
                    "id": "fail_sw_check_abc",
                    "category": "compile",
                    "summary": "main.c:1: error: bad value",
                },
            }

        toolbox = RepoToolbox(
            REPO_ROOT, dry_run=False, stage="implementation", run_root=self.run_root
        )
        self.tools = SoftwareCodingToolset(
            toolbox,
            allowed_targets=frozenset({self.target}),
            check_compile=check,
        )

    def test_schema_never_accepts_argv_or_cwd(self) -> None:
        check = next(
            item for item in self.tools.schemas()
            if item["function"]["name"] == "check_software_compile"
        )
        self.assertEqual(check["function"]["parameters"]["properties"], {})

    def test_only_exact_graph_bound_target_can_be_edited(self) -> None:
        denied = self.tools.run(
            "edit_software", {"path": self.other, "search": "1", "replace": "2"}
        )
        self.assertIn("not an approved", denied)
        self.assertEqual((REPO_ROOT / self.other).read_text(), "int y = 1;\n")
        ok = self.tools.run(
            "edit_software", {"path": self.target, "search": "int x = 1;", "replace": "int x = 2;"}
        )
        self.assertIn(self.target, ok)
        self.assertEqual((REPO_ROOT / self.target).read_text(), "int x = 2;\n")
        self.assertEqual(self.tools.touched_paths, [self.target])

    def test_check_requires_edit_rejects_arguments_and_never_sets_gate(self) -> None:
        self.assertIn("edit software", self.tools.run("check_software_compile", {}))
        self.tools.run(
            "edit_software", {"path": self.target, "search": "int x = 1;", "replace": "int x = 2;"}
        )
        self.assertIn(
            "accepts no arguments",
            self.tools.run("check_software_compile", {"argv": ["rm", "-rf", "/"]}),
        )
        result = self.tools.run("check_software_compile", {})
        self.assertIn("fail_sw_check_abc", result)
        self.assertNotIn("gate", result.lower())
        self.assertEqual(len(self.checks), 1)

    def test_compile_checks_are_unlimited_by_default(self) -> None:
        self.assertIsNone(self.tools.max_checks)
        previous = "int x = 1;\n"
        for value in range(2, 5):
            current = f"int x = {value};\n"
            self.assertNotIn("rejected", self.tools.run("edit_software", {
                "path": self.target, "search": previous, "replace": current}))
            self.assertIn("failed", self.tools.run("check_software_compile", {}))
            previous = current
        self.assertEqual(len(self.checks), 3)

    def test_edit_and_check_budgets_are_enforced(self) -> None:
        self.tools.max_edits = 1
        self.tools.run(
            "edit_software", {"path": self.target, "search": "int x = 1;", "replace": "int x = 2;"}
        )
        self.assertIn(
            "budget exhausted",
            self.tools.run(
                "edit_software", {"path": self.target, "search": "int x = 2;", "replace": "int x = 3;"}
            ),
        )
        self.tools.max_checks = 1
        self.tools.run("check_software_compile", {})
        self.tools.edited_since_check = True
        self.assertIn("budget exhausted", self.tools.run("check_software_compile", {}))


if __name__ == "__main__":
    unittest.main()
