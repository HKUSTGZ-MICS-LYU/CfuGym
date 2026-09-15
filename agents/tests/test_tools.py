"""RepoToolbox read/write/copy policy enforcement."""

from __future__ import annotations

import shutil
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

    def test_context_includes_agent_cfu_scaffold(self) -> None:
        context = self.toolbox.discover_context()
        self.assertIn("src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala", context["files"])
        self.assertIn("src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala", context["files"])
        self.assertIn("skills/cfu-designer/references/roofline-analysis.md", context["files"])

    def test_read_denies_generated_run(self) -> None:
        with self.assertRaises(PolicyError):
            self.toolbox.read_text("agents/generated/runs/20260729-162009/report.md")

    def test_read_denies_git(self) -> None:
        with self.assertRaises(PolicyError):
            self.toolbox.read_text(".git/config")

    def test_write_dry_run_records_without_writing_source(self) -> None:
        with self.assertRaises(PolicyError):
            self.toolbox.write_text("src/main/scala/vexiiriscv/soc/mico/NewCfu.scala", "// x")

    def test_write_allows_only_the_run_design_workspace(self) -> None:
        run_root = "agents/generated/runs/tools-test-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            workspace = RepoToolbox(
                REPO_ROOT,
                dry_run=True,
                stage="implementation",
                run_root=run_root,
            )
            result = workspace.write_text(f"{run_root}/workspace/design/AgentCfu.scala", "// design")
            self.assertIn("wrote", result)
            self.assertTrue((run_dir / "workspace/design/AgentCfu.scala").exists())
            with self.assertRaises(PolicyError):
                workspace.write_text(f"{run_root}/workspace/design/Extra.scala", "// nope")
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

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

    def test_filter_output_removes_terminal_formatting_and_controls(self) -> None:
        from agents.cfu_design_agent.tools import filter_output

        text = "\x1b[31mERROR\x1b[0m\rprogress\b!\x1b]8;;https://example.test\x07link\x1b]8;;\x07\n"
        self.assertEqual(filter_output(text), "ERROR\nprogress!link\n")

    def test_parse_command_output_filters_ansi_before_parsing(self) -> None:
        from agents.cfu_design_agent.tools import parse_command_output

        parsed = parse_command_output("\x1b[32mCFU_AGENT_PROFILE crc32=42\x1b[0m\n")
        self.assertEqual(parsed["profiles"][0]["crc32"], 42)

    def test_parse_command_output_extracts_the_yosys_cost_line(self) -> None:
        from agents.cfu_design_agent.tools import parse_command_output

        parsed = parse_command_output(
            "Wrote out/summary.md\ncost top=AgentCfu cells=3845 wire_bits=4402 "
            "memories=0 memory_bits=0\n"
        )
        self.assertEqual(
            parsed["cost"],
            {
                "top": "AgentCfu",
                "cells": 3845,
                "wire_bits": 4402,
                "memories": 0,
                "memory_bits": 0,
            },
        )
        self.assertNotIn("cost", parse_command_output("no cost summary here\n"))

    def test_parse_command_output_extracts_the_verify_diagnostic(self) -> None:
        from agents.cfu_design_agent.tools import parse_command_output

        # Embench: verify_benchmark() != 0 means correct; 0 means a mismatch.
        parsed = parse_command_output(
            "EMBENCH_VERIFY_FAIL benchmark=depthconv verify=0 result=0\n"
        )
        self.assertEqual(
            parsed["verify"],
            {"status": "fail", "benchmark": "depthconv", "verify": 0, "result": 0},
        )
        passed = parse_command_output("EMBENCH_VERIFY_PASS benchmark=crc32 verify=1 result=0\n")
        self.assertEqual(passed["verify"]["status"], "pass")
        self.assertNotIn("verify", parse_command_output("no verify line here\n"))

    def test_edit_end_tolerates_trailing_sampler_junk(self) -> None:
        from agents.cfu_design_agent.tools import (
            extract_edit_blocks,
            unterminated_edit_blocks,
        )

        # Observed in a real run: "EDIT_END સલ" discarded an otherwise valid edit.
        text = (
            "REPAIR_DIAGNOSIS\n- Observed errors: x\n\n"
            "EDIT_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n"
            "    val input = a.asSInt\n"
            "=======\n"
            "    val input = a.intoSInt\n"
            ">>>>>>> REPLACE\n"
            "EDIT_END સલ\n"
        )
        blocks = extract_edit_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(
            blocks[0][1], [("    val input = a.asSInt", "    val input = a.intoSInt")]
        )
        self.assertEqual(unterminated_edit_blocks(text), 0)

    def test_unterminated_edit_block_is_counted(self) -> None:
        from agents.cfu_design_agent.tools import (
            extract_edit_blocks,
            unterminated_edit_blocks,
        )

        text = (
            "EDIT_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n    a\n=======\n    b\n>>>>>>> REPLACE\n"
        )
        self.assertEqual(extract_edit_blocks(text), [])
        self.assertEqual(unterminated_edit_blocks(text), 1)

    def test_resolve_target_accepts_run_workspace_software_paths(self) -> None:
        run_root = "agents/generated/runs/tools-target-00000000"
        workspace = RepoToolbox(
            REPO_ROOT,
            dry_run=True,
            stage="implementation",
            run_root=run_root,
        )
        self.assertEqual(
            workspace.resolve_target(f"{run_root}/workspace/cfu_benchmark/main.c"),
            f"{run_root}/workspace/cfu_benchmark/main.c",
        )
        self.assertEqual(
            workspace.resolve_target(
                "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
            ),
            f"{run_root}/workspace/design/AgentCfu.scala",
        )
        with self.assertRaises(ValueError):
            workspace.resolve_target(f"{run_root}/workspace/design/Extra.scala")


class CodeFenceTest(unittest.TestCase):
    def test_strips_markdown_fences_around_a_file_body(self) -> None:
        from agents.cfu_design_agent.tools import extract_file_blocks, strip_code_fences

        self.assertEqual(strip_code_fences("```scala\nobject A\n```"), "object A\n")
        self.assertEqual(strip_code_fences("object A\n"), "object A\n")
        # Only leading/trailing fences are removed; interior content is kept.
        self.assertEqual(strip_code_fences("a\n```\nb\n"), "a\n```\nb\n")
        blocks = extract_file_blocks(
            "FILE_BEGIN path/A.scala\n```scala\nobject A\n```\nFILE_END\n"
        )
        self.assertEqual(blocks, [("path/A.scala", "object A\n")])

    def test_extract_patch_strips_fences(self) -> None:
        from agents.cfu_design_agent.tools import extract_patch

        text = "PATCH_BEGIN\n```diff\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```\nPATCH_END"
        patch = extract_patch(text)
        self.assertTrue(patch.startswith("--- a/x"))
        self.assertNotIn("```", patch)


class EditBlockTest(unittest.TestCase):
    """Targeted SEARCH/REPLACE edits: parsing, matching, and policy reuse."""

    def test_extract_edit_blocks_parses_multiple_pairs(self) -> None:
        from agents.cfu_design_agent.tools import extract_edit_blocks

        text = (
            "prose\n"
            "EDIT_BEGIN path/A.scala\n"
            "<<<<<<< SEARCH\nold one\n=======\nnew one\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nold two\n=======\nnew two\n>>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        blocks = extract_edit_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], "path/A.scala")
        self.assertEqual(blocks[0][1], [("old one", "new one"), ("old two", "new two")])

    def test_apply_edit_pairs_requires_a_unique_match(self) -> None:
        from agents.cfu_design_agent.tools import apply_edit_pairs

        updated, errors = apply_edit_pairs("a\nb\n", [("b", "c")])
        self.assertEqual(errors, [])
        self.assertEqual(updated, "a\nc\n")

        _updated, errors = apply_edit_pairs("a\nb\na\nb\n", [("a\nb", "x")])
        self.assertTrue(any("ambiguous" in problem for problem in errors))

        _updated, errors = apply_edit_pairs("a\nb\n", [("zzz", "x")])
        self.assertTrue(any("not found" in problem for problem in errors))

    def test_apply_edit_pairs_tolerates_trailing_whitespace_only(self) -> None:
        from agents.cfu_design_agent.tools import apply_edit_pairs

        updated, errors = apply_edit_pairs("  keep me   \nnext\n", [("  keep me", "  changed")])
        self.assertEqual(errors, [])
        self.assertEqual(updated, "  changed\nnext\n")

    def test_mismatch_names_the_first_differing_line(self) -> None:
        from agents.cfu_design_agent.tools import apply_edit_pairs

        # A paraphrased comment (the depthconv failure mode) must be localized.
        content = "a\n// is a int8_t\nc\n"
        search = "a\n// is an int8_t\nc"
        _updated, errors = apply_edit_pairs(content, [(search, "x")])
        self.assertEqual(len(errors), 1)
        self.assertIn("First difference: SEARCH line 2 of 3", errors[0])
        self.assertIn("is an int8_t", errors[0])  # the model's line
        self.assertIn("is a int8_t", errors[0])  # the current file line

    def test_mismatch_reports_when_an_earlier_pair_changed_the_region(self) -> None:
        from agents.cfu_design_agent.tools import apply_edit_pairs

        _updated, errors = apply_edit_pairs(
            "alpha\nbeta\n", [("alpha", "ALPHA"), ("alpha", "omega")]
        )
        self.assertIn("an earlier section or pair changed that region", errors[0])

    def test_mismatch_without_an_anchor_says_to_rebuild_from_the_prompt(self) -> None:
        from agents.cfu_design_agent.tools import apply_edit_pairs

        _updated, errors = apply_edit_pairs("a\nb\n", [("totally absent\ntext", "x")])
        self.assertIn("does not appear in the current file", errors[0])

    def test_toolbox_applies_edits_to_a_run_workspace_file(self) -> None:
        run_root = "agents/generated/runs/tools-edit-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            result = toolbox.apply_edit_blocks([(target, [("  return 1;", "  return 0;")])])
            self.assertEqual(result["returncode"], 0, result)
            self.assertEqual(result["parsed"]["touched_paths"], [target])
            self.assertIn("return 0;", toolbox.read_text(target))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_toolbox_rejects_an_unmatched_edit_without_writing(self) -> None:
        run_root = "agents/generated/runs/tools-edit-bad-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            result = toolbox.apply_edit_blocks([(target, [("  return 99;", "  return 0;")])])
            self.assertEqual(result["returncode"], 1)
            self.assertIn("not found", result["stderr_tail"])
            self.assertIn("return 1;", toolbox.read_text(target))
            self.assertEqual(result["parsed"]["touched_paths"], [])
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_apply_edit_blocks_is_atomic_across_targets(self) -> None:
        # When a later target fails, an earlier validated target must not be
        # half-applied: the whole response is rejected together.
        run_root = "agents/generated/runs/tools-edit-atomic-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            good = f"{run_root}/workspace/cfu_benchmark/main.c"
            bad = f"{run_root}/workspace/cfu_benchmark/helper.c"
            toolbox.write_text(good, "int main(void) {\n  return 1;\n}\n")
            toolbox.write_text(bad, "int helper(void) {\n  return 2;\n}\n")
            edits = [
                (good, [("  return 1;", "  return 0;")]),
                (bad, [("  return 99;", "  return 9;")]),  # not present -> rejects
            ]
            result = toolbox.apply_edit_blocks(edits)
            self.assertEqual(result["returncode"], 1, result)
            self.assertEqual(result["parsed"]["touched_paths"], [])
            # Neither file changed: the validated good edit was rolled back.
            self.assertIn("return 1;", toolbox.read_text(good))
            self.assertNotIn("return 0;", toolbox.read_text(good))
            self.assertIn("return 2;", toolbox.read_text(bad))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_multiple_edit_sections_for_one_file_are_composed(self) -> None:
        # Several EDIT_BEGIN sections may target the same file: they apply in
        # order to the evolving content.
        run_root = "agents/generated/runs/tools-edit-multi-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(
                target, "int main(void) {\n  int a = 1;\n  int b = 2;\n  return a + b;\n}\n"
            )
            edits = [
                (target, [("  int a = 1;", "  int a = 10;")]),
                (target, [("  int b = 2;", "  int b = 20;")]),
            ]
            result = toolbox.apply_edit_blocks(edits)
            self.assertEqual(result["returncode"], 0, result)
            self.assertEqual(result["parsed"]["touched_paths"], [target])
            text = toolbox.read_text(target)
            self.assertIn("int a = 10;", text)
            self.assertIn("int b = 20;", text)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_overlapping_edit_sections_fail_atomically(self) -> None:
        # A later section must match the file after the earlier sections; a
        # section that re-searches a region an earlier one replaced is reported
        # with its section number and nothing is written.
        run_root = "agents/generated/runs/tools-edit-overlap-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            edits = [
                (target, [("  return 1;", "  return 0;")]),
                (target, [("  return 1;", "  return 2;")]),
            ]
            result = toolbox.apply_edit_blocks(edits)
            self.assertEqual(result["returncode"], 1, result)
            self.assertIn("section 2/2", result["stderr_tail"])
            self.assertIn("SEARCH block not found", result["stderr_tail"])
            self.assertEqual(result["parsed"]["touched_paths"], [])
            self.assertIn("return 1;", toolbox.read_text(target))
            self.assertNotIn("return 0;", toolbox.read_text(target))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
