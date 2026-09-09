"""Isolated hardware design workspace: seeding, patch redirect, and boundaries."""

from __future__ import annotations

import hashlib
import shutil
import unittest
from pathlib import Path

from agents.cfu_design_agent.policy import PolicyError, load_policy
from agents.cfu_design_agent.tools import (
    DESIGN_SOURCE_FILES,
    RepoToolbox,
    design_workspace_rel,
    repair_patch_paths,
)
from agents.tests.common import REPO_ROOT


RUN_ID = "design-ws-test-00000000"
RUN_ROOT = f"agents/generated/runs/{RUN_ID}"


def scaffold_patch(compute_value: int) -> str:
    return "\n".join([
        "--- a/src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
        "+++ b/src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
        "@@ -36,7 +36,7 @@",
        " }",
        " ",
        " object AgentCfuFunction {",
        "-  val compute = 1",
        f"+  val compute = {compute_value}",
        "   val config = 2",
        "   val load = 4",
        "   val store = 5",
        " }",
        "",
    ])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DesignWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(REPO_ROOT)
        self.run_dir = REPO_ROOT / RUN_ROOT
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def toolbox(self, *, apply_patch: bool = False, stage: str = "implementation") -> RepoToolbox:
        return RepoToolbox(
            REPO_ROOT,
            dry_run=False,
            policy=self.policy,
            stage=stage,
            run_root=RUN_ROOT,
            apply_patch=apply_patch,
        )

    def test_seed_copies_both_scaffolds_and_soc_dir(self) -> None:
        seeded = self.toolbox().seed_design_workspace()
        self.assertEqual(len(seeded), 2)
        for source in DESIGN_SOURCE_FILES:
            overlay = REPO_ROOT / design_workspace_rel(RUN_ROOT, Path(source).name)
            self.assertTrue(overlay.exists(), overlay)
            self.assertEqual(overlay.read_text(), (REPO_ROOT / source).read_text())
        self.assertTrue((self.run_dir / "workspace/soc").is_dir())

    def test_policy_rejects_repo_source_writes(self) -> None:
        for rel in (
            "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
            "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala",
            "src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
            "src/main/scala/vexiiriscv/soc/cfu/CfuLib.scala",
        ):
            with self.assertRaises(PolicyError, msg=rel):
                self.policy.check_write("implementation", rel)

    def test_policy_allows_only_design_files_in_workspace(self) -> None:
        for name in ("AgentCfu.scala", "AgentCfuFiber.scala"):
            rel = design_workspace_rel(RUN_ROOT, name)
            self.policy.check_write("implementation", rel)
        with self.assertRaises(PolicyError):
            self.policy.check_write("implementation", f"{RUN_ROOT}/workspace/design/Evil.scala")

    def test_patch_paths_are_redirected_into_workspace(self) -> None:
        rewritten, redirected, rejected = repair_patch_paths(scaffold_patch(6), RUN_ROOT)
        self.assertEqual(rejected, [])
        self.assertIn(design_workspace_rel(RUN_ROOT, "AgentCfu.scala"), redirected)
        self.assertIn(f"+++ b/{RUN_ROOT}/workspace/design/AgentCfu.scala", rewritten)
        self.assertNotIn("+++ b/src/main/scala", rewritten)

    def test_out_of_scope_source_paths_are_rejected(self) -> None:
        patch = "\n".join([
            "--- a/src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
            "+++ b/src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
            "@@ -1,1 +1,1 @@",
            "-x",
            "+y",
            "",
        ])
        _, _, rejected = repair_patch_paths(patch, RUN_ROOT)
        self.assertIn("src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala", rejected)

    def test_apply_patch_writes_workspace_and_leaves_repo_untouched(self) -> None:
        tracked = REPO_ROOT / DESIGN_SOURCE_FILES[0]
        before = sha256(tracked)
        toolbox = self.toolbox(apply_patch=True)
        toolbox.seed_design_workspace()
        result = toolbox.apply_patch_text(scaffold_patch(6))
        self.assertEqual(result["returncode"], 0)
        overlay = REPO_ROOT / design_workspace_rel(RUN_ROOT, "AgentCfu.scala")
        self.assertIn("val compute = 6", overlay.read_text())
        self.assertEqual(sha256(tracked), before)

    def test_apply_patch_rejects_repo_source_target(self) -> None:
        toolbox = self.toolbox(apply_patch=True)
        toolbox.seed_design_workspace()
        patch = "\n".join([
            "--- a/src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
            "+++ b/src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala",
            "@@ -1,1 +1,1 @@",
            "-x",
            "+y",
            "",
        ])
        with self.assertRaises(ValueError):
            toolbox.apply_patch_text(patch)


if __name__ == "__main__":
    unittest.main()
