"""RepoToolbox read/write/copy policy enforcement."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.runtime.policy import PolicyError
from agents.cfu_design_agent.runtime.tools import RepoToolbox
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

    def test_run_dir_token_translates_and_rejects_misuse(self) -> None:
        run_root = "agents/generated/runs/tools-test-00000000"
        box = RepoToolbox(REPO_ROOT, dry_run=True, stage="implementation", run_root=run_root)
        self.assertEqual(box.expand_run_dir("{{run-dir}}"), run_root)
        self.assertEqual(
            box.normalize("{{run-dir}}/08_isa/cfu_isa.yaml"), f"{run_root}/08_isa/cfu_isa.yaml"
        )
        self.assertEqual(box.normalize("agents/specs/cfu_isa_template.yaml"),
                         "agents/specs/cfu_isa_template.yaml")
        # The edit tools resolve the same token onto the approved overlay.
        self.assertEqual(
            box.resolve_target("{{run-dir}}/workspace/design/AgentCfu.scala"),
            f"{run_root}/workspace/design/AgentCfu.scala",
        )
        with self.assertRaisesRegex(PolicyError, "must be a path prefix"):
            box.normalize("agents/{{run-dir}}/x")
        with self.assertRaisesRegex(PolicyError, "only valid inside a run"):
            self.toolbox.normalize("{{run-dir}}/x")
        # The other direction: output shown to the model renders the token back.
        from agents.cfu_design_agent.runtime.tools import virtualize_run_paths

        self.assertEqual(
            virtualize_run_paths(f"log: {run_root}/10_validation/logs/sim_000.log", run_root),
            "log: {{run-dir}}/10_validation/logs/sim_000.log",
        )
        self.assertEqual(virtualize_run_paths("no run path here", run_root), "no run path here")

    def test_failed_check_results_are_not_ok(self) -> None:
        # Run 20261003-130005: a real "check failed" was journaled ok while the
        # budget-refusal (error:) was the only thing marked ERROR.
        from agents.cfu_design_agent.runtime.tools import tool_result_ok

        self.assertFalse(tool_result_ok("isolated hardware elaboration check failed\nfailure: x"))
        self.assertFalse(tool_result_ok("error: hardware check budget exhausted"))
        self.assertFalse(tool_result_ok("denied by access policy: nope"))
        self.assertFalse(tool_result_ok("usage error: limit must be an integer"))
        self.assertTrue(tool_result_ok(
            "isolated software compile check passed (evidence only)\nfull log: x"))
        self.assertTrue(tool_result_ok("edits applied atomically"))
        self.assertTrue(tool_result_ok(""))
        # Only the first line decides: a log excerpt may mention a failure.
        self.assertTrue(tool_result_ok("Failure fail_sim_1\nsummary:\ncheck failed at boot"))

    def test_command_allowed_matches_policy(self) -> None:
        # Command authority is per stage: load_context may run nothing, while the
        # profile stage may run the allowlisted embench build.
        self.assertFalse(
            self.toolbox.command_allowed(
                ["make", "TARGET=vexii_soc", "compile"], command_id="make_profile"
            )
        )
        profile = RepoToolbox(REPO_ROOT, dry_run=True, stage="profile")
        self.assertTrue(
            profile.command_allowed(
                ["make", "TARGET=vexii_soc", "compile"], command_id="make_profile"
            )
        )
        self.assertFalse(
            profile.command_allowed(["rm", "-rf", "/"], command_id="make_profile"),
        )

    def test_filter_output_removes_terminal_formatting_and_controls(self) -> None:
        from agents.cfu_design_agent.runtime.tools import filter_output

        text = "\x1b[31mERROR\x1b[0m\rprogress\b!\x1b]8;;https://example.test\x07link\x1b]8;;\x07\n"
        self.assertEqual(filter_output(text), "ERROR\nprogress!link\n")

    def test_parse_command_output_filters_ansi_before_parsing(self) -> None:
        from agents.cfu_design_agent.runtime.tools import parse_command_output

        parsed = parse_command_output("\x1b[32mCFU_AGENT_PROFILE crc32=42\x1b[0m\n")
        self.assertEqual(parsed["profiles"][0]["crc32"], 42)

    def test_parse_command_output_extracts_the_yosys_cost_line(self) -> None:
        from agents.cfu_design_agent.runtime.tools import parse_command_output

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
        from agents.cfu_design_agent.runtime.tools import parse_command_output

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
        from agents.cfu_design_agent.runtime.tools import (
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
        from agents.cfu_design_agent.runtime.tools import (
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
        from agents.cfu_design_agent.runtime.tools import extract_file_blocks, strip_code_fences

        self.assertEqual(strip_code_fences("```scala\nobject A\n```"), "object A\n")
        self.assertEqual(strip_code_fences("object A\n"), "object A\n")
        # Only leading/trailing fences are removed; interior content is kept.
        self.assertEqual(strip_code_fences("a\n```\nb\n"), "a\n```\nb\n")
        blocks = extract_file_blocks(
            "FILE_BEGIN path/A.scala\n```scala\nobject A\n```\nFILE_END\n"
        )
        self.assertEqual(blocks, [("path/A.scala", "object A\n")])

    def test_extract_patch_strips_fences(self) -> None:
        from agents.cfu_design_agent.runtime.tools import extract_patch

        text = "PATCH_BEGIN\n```diff\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```\nPATCH_END"
        patch = extract_patch(text)
        self.assertTrue(patch.startswith("--- a/x"))
        self.assertNotIn("```", patch)


class EditBlockTest(unittest.TestCase):
    """Targeted SEARCH/REPLACE edits: parsing, matching, and policy reuse."""

    def test_extract_edit_blocks_parses_multiple_pairs(self) -> None:
        from agents.cfu_design_agent.runtime.tools import extract_edit_blocks

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
        from agents.cfu_design_agent.runtime.tools import apply_edit_pairs

        updated, errors = apply_edit_pairs("a\nb\n", [("b", "c")])
        self.assertEqual(errors, [])
        self.assertEqual(updated, "a\nc\n")

        _updated, errors = apply_edit_pairs("a\nb\na\nb\n", [("a\nb", "x")])
        self.assertTrue(any("ambiguous" in problem for problem in errors))

        _updated, errors = apply_edit_pairs("a\nb\n", [("zzz", "x")])
        self.assertTrue(any("not found" in problem for problem in errors))

    def test_apply_edit_pairs_tolerates_trailing_whitespace_only(self) -> None:
        from agents.cfu_design_agent.runtime.tools import apply_edit_pairs

        updated, errors = apply_edit_pairs("  keep me   \nnext\n", [("  keep me", "  changed")])
        self.assertEqual(errors, [])
        self.assertEqual(updated, "  changed\nnext\n")

    def test_mismatch_names_the_first_differing_line(self) -> None:
        from agents.cfu_design_agent.runtime.tools import apply_edit_pairs

        # A paraphrased comment (the depthconv failure mode) must be localized.
        content = "a\n// is a int8_t\nc\n"
        search = "a\n// is an int8_t\nc"
        _updated, errors = apply_edit_pairs(content, [(search, "x")])
        self.assertEqual(len(errors), 1)
        self.assertIn("First difference: SEARCH line 2 of 3", errors[0])
        self.assertIn("is an int8_t", errors[0])  # the model's line
        self.assertIn("is a int8_t", errors[0])  # the current file line

    def test_mismatch_reports_when_an_earlier_pair_changed_the_region(self) -> None:
        from agents.cfu_design_agent.runtime.tools import apply_edit_pairs

        _updated, errors = apply_edit_pairs(
            "alpha\nbeta\n", [("alpha", "ALPHA"), ("alpha", "omega")]
        )
        self.assertIn("an earlier section or pair changed that region", errors[0])

    def test_mismatch_without_an_anchor_says_to_rebuild_from_the_prompt(self) -> None:
        from agents.cfu_design_agent.runtime.tools import apply_edit_pairs

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

    def test_edit_result_carries_the_applied_diff(self) -> None:
        # The model must see what its edit did without re-reading the file.
        run_root = "agents/generated/runs/tools-edit-diff-00000000"
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
            diff = result["stdout_tail"]
            self.assertIn(f"--- a/{target}", diff)
            self.assertIn(f"+++ b/{target}", diff)
            self.assertIn("-  return 1;", diff)
            self.assertIn("+  return 0;", diff)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_edit_diff_is_capped_with_a_truncation_marker(self) -> None:
        run_root = "agents/generated/runs/tools-edit-diffcap-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            original = "".join(f"line {i}\n" for i in range(300))
            toolbox.write_text(target, original)
            replaced = "".join(f"LINE {i}\n" for i in range(300))
            result = toolbox.apply_edit_blocks([(target, [(original.rstrip("\n"), replaced.rstrip("\n"))])])
            self.assertEqual(result["returncode"], 0, result)
            self.assertIn("[diff truncated at 4000 chars]", result["stdout_tail"])
            self.assertLessEqual(len(result["stdout_tail"]), 4000 + 200)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_noop_edit_is_success_with_note(self) -> None:
        # A replayed/echo delivery (SEARCH == REPLACE, or an already-applied
        # edit) must not churn the repair loop with "edits produced no change".
        run_root = "agents/generated/runs/tools-edit-noop-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            result = toolbox.apply_edit_blocks([(target, [("  return 1;", "  return 1;")])])
            self.assertEqual(result["returncode"], 0, result)
            self.assertIn("edit was already applied (no content change)", result["stdout_tail"])
            self.assertEqual(result["parsed"]["touched_paths"], [])
            self.assertIn("return 1;", toolbox.read_text(target))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_noop_pair_beside_a_real_change_still_applies(self) -> None:
        run_root = "agents/generated/runs/tools-edit-noopmix-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            edits = [(target, [("  return 1;", "  return 0;"), ("}", "}")])]
            result = toolbox.apply_edit_blocks(edits)
            self.assertEqual(result["returncode"], 0, result)
            self.assertEqual(result["parsed"]["touched_paths"], [target])
            self.assertIn("return 0;", toolbox.read_text(target))
            self.assertIn("+  return 0;", result["stdout_tail"])
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_unmatched_search_stays_an_error(self) -> None:
        # The success-with-note path must not swallow genuine mismatches.
        run_root = "agents/generated/runs/tools-edit-mismatch-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            result = toolbox.apply_edit_blocks([(target, [("  return 9;", "  return 0;")])])
            self.assertEqual(result["returncode"], 1)
            self.assertIn("SEARCH block not found", result["stderr_tail"])
            self.assertEqual(result["parsed"]["touched_paths"], [])
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_replace_with_empty_deletes_without_a_blank_line(self) -> None:
        from agents.cfu_design_agent.runtime.tools import apply_edit_pairs

        updated, errors = apply_edit_pairs("a\nb\nc\n", [("b", "")])
        self.assertEqual(errors, [])
        self.assertEqual(updated, "a\nc\n")

    def test_conflict_marker_content_rejects_the_whole_response(self) -> None:
        # A git-conflict region inside SEARCH used to silently mis-parse (the
        # `=======` line flipped the parser into replace mode).
        run_root = "agents/generated/runs/tools-edit-conflict-00000000"
        run_dir = REPO_ROOT / run_root
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            toolbox = RepoToolbox(
                REPO_ROOT, dry_run=False, stage="implementation", run_root=run_root
            )
            target = f"{run_root}/workspace/cfu_benchmark/main.c"
            toolbox.write_text(target, "int main(void) {\n  return 1;\n}\n")
            text = (
                f"EDIT_BEGIN {target}\n"
                "<<<<<<< SEARCH\n"
                "<<<<<<< HEAD\n"
                "  return 1;\n"
                "=======\n"
                "  return 2;\n"
                ">>>>>>> branch\n"
                "=======\n"
                "  return 0;\n"
                ">>>>>>> REPLACE\n"
                "EDIT_END\n"
            )
            from agents.cfu_design_agent.runtime.tools import extract_edit_blocks

            result = toolbox.apply_edit_blocks(extract_edit_blocks(text))
            self.assertEqual(result["returncode"], 1, result)
            self.assertIn("conflict-marker-like line", result["stderr_tail"])
            self.assertIn("return 1;", toolbox.read_text(target))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
