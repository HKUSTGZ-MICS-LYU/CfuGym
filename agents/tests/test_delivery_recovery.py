"""Recovery from degenerate LLM deliveries and stuck identical retries.

Covers failures observed in real runs:

* a sampler restart emits the whole answer twice with the second copy glued to
  the first terminator (``EDIT_ENDREPAIR_DIAGNOSIS ...``), which used to be
  rejected as "missing EDIT_END"; and
* the model repeats one byte-identical broken answer across repairs -- both
  implementation rungs now carry the same stuck-attempt directive; and
* the conformance oracle repair oscillates between two validation errors until
  the budget is gone -- a recurring rejection now pins the correct shape.
"""

from __future__ import annotations

import unittest

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime import tools
from agents.cfu_design_agent.stages import feedback


# One completion containing the same edit twice, second copy glued to the first
# terminator (verbatim shape of the 2026-09-26 matmult-int run failure).
DOUBLE_ANSWER = (
    "REPAIR_DIAGNOSIS\n"
    "\n"
    "EDIT_BEGIN design/AgentCfu.scala\n"
    "<<<<<<< SEARCH\n"
    "        ctx.complete(sum(31 downto 0))\n"
    "=======\n"
    "        ctx.complete(sum(31 downto 0)).asBits\n"
    ">>>>>>> REPLACE\n"
    "EDIT_ENDREPAIR_DIAGNOSIS\n"
    "\n"
    "EDIT_BEGIN design/AgentCfu.scala\n"
    "<<<<<<< SEARCH\n"
    "        ctx.complete(sum(31 downto 0))\n"
    "=======\n"
    "        ctx.complete(sum(31 downto 0)).asBits\n"
    ">>>>>>> REPLACE\n"
    "EDIT_END\n"
)


class GluedTerminatorTest(unittest.TestCase):
    def test_normalize_splits_glued_terminator(self) -> None:
        self.assertEqual(
            tools.normalize_delivery_terminators("EDIT_ENDREPAIR_DIAGNOSIS"),
            "EDIT_END\nREPAIR_DIAGNOSIS",
        )
        self.assertEqual(
            tools.normalize_delivery_terminators("FILE_ENDMORE"),
            "FILE_END\nMORE",
        )

    def test_identifier_continuation_stays_content(self) -> None:
        self.assertEqual(
            tools.normalize_delivery_terminators("EDIT_END_foo = 1"),
            "EDIT_END_foo = 1",
        )

    def test_duplicated_glued_answer_parses_once(self) -> None:
        blocks = tools.extract_edit_blocks(DOUBLE_ANSWER)
        # The duplicate copy collapses to the first: after it applies, the
        # second SEARCH would no longer match and reject the whole response.
        self.assertEqual(len(blocks), 1)
        path, pairs = blocks[0]
        self.assertEqual(path, "design/AgentCfu.scala")
        self.assertEqual(len(pairs), 1)
        self.assertIn(".asBits", pairs[0][1])
        self.assertEqual(tools.unterminated_edit_blocks(DOUBLE_ANSWER), 0)

    def test_distinct_sections_are_not_deduplicated(self) -> None:
        text = (
            "EDIT_BEGIN a.scala\n"
            "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
            "EDIT_END\n"
            "EDIT_BEGIN b.scala\n"
            "<<<<<<< SEARCH\nx\n=======\nz\n>>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        self.assertEqual(len(tools.extract_edit_blocks(text)), 2)

    def test_genuinely_unterminated_block_still_detected(self) -> None:
        text = "EDIT_BEGIN p\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n"
        self.assertEqual(tools.unterminated_edit_blocks(text), 1)

    def test_duplicated_file_blocks_collapse(self) -> None:
        text = (
            "FILE_BEGIN p\ncontent\nFILE_ENDMORE\n"
            "FILE_BEGIN p\ncontent\nFILE_END\n"
        )
        blocks = tools.extract_file_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("content", blocks[0][1])


class ConflictMarkerContentTest(unittest.TestCase):
    """A git-conflict region inside a SEARCH/REPLACE block used to silently
    mis-parse: the content `=======` line flipped the parser into replace mode
    and the wrong pair was produced (2026-09-28 audit, verified by experiment).
    The parser now rejects marker-shaped content with an actionable message,
    routed through apply_edit_pairs as feedback instead of raising into the run.
    """

    @staticmethod
    def _single_pair(text: str) -> tuple[str, list[tuple[str, str]]]:
        blocks = tools.extract_edit_blocks(text)
        assert len(blocks) == 1, blocks
        return blocks[0]

    def test_conflict_region_in_search_is_rejected(self) -> None:
        text = (
            "EDIT_BEGIN design/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n"
            "<<<<<<< HEAD\n"
            "old line\n"
            "=======\n"
            "new line\n"
            ">>>>>>> branch\n"
            "=======\n"
            "replacement\n"
            ">>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        path, pairs = self._single_pair(text)
        self.assertEqual(path, "design/AgentCfu.scala")
        self.assertEqual(pairs[0][0], tools.EDIT_PARSE_ERROR)
        _updated, errors = tools.apply_edit_pairs("anything\n", pairs)
        self.assertEqual(len(errors), 1)
        self.assertIn("conflict-marker-like line", errors[0])
        self.assertIn("'<<<<<<< HEAD'", errors[0])
        self.assertIn("edit the lines around it instead", errors[0])

    def test_conflict_marker_in_replace_is_rejected(self) -> None:
        text = (
            "EDIT_BEGIN design/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n"
            "old line\n"
            "=======\n"
            "new line\n"
            ">>>>>>> branch\n"
            ">>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        _path, pairs = self._single_pair(text)
        self.assertEqual(pairs[0][0], tools.EDIT_PARSE_ERROR)
        _updated, errors = tools.apply_edit_pairs("x\n", pairs)
        self.assertIn("conflict-marker-like line", errors[0])
        self.assertIn("'>>>>>>> branch'", errors[0])

    def test_divider_like_line_in_replace_content_is_rejected(self) -> None:
        text = (
            "EDIT_BEGIN design/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n"
            "old line\n"
            "=======\n"
            "new line\n"
            "=======\n"
            ">>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        _path, pairs = self._single_pair(text)
        self.assertEqual(pairs[0][0], tools.EDIT_PARSE_ERROR)

    def test_parse_errors_never_raise_out_of_extract(self) -> None:
        # An unterminated pair inside a closed EDIT block used to raise
        # ValueError from the parser; it now becomes a sentinel pair whose
        # message reaches the model as ordinary repair feedback.
        text = "EDIT_BEGIN p\n<<<<<<< SEARCH\na\nEDIT_END\n"
        path, pairs = self._single_pair(text)
        self.assertEqual(path, "p")
        self.assertEqual(pairs[0][0], tools.EDIT_PARSE_ERROR)
        _updated, errors = tools.apply_edit_pairs("a\n", pairs)
        self.assertIn("unterminated SEARCH/REPLACE block", errors[0])

    def test_nested_search_marker_routed_through_the_error_channel(self) -> None:
        text = (
            "EDIT_BEGIN p\n"
            "<<<<<<< SEARCH\na\n<<<<<<< SEARCH\n=======\nb\n>>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        _path, pairs = self._single_pair(text)
        self.assertEqual(pairs[0][0], tools.EDIT_PARSE_ERROR)
        _updated, errors = tools.apply_edit_pairs("a\n", pairs)
        self.assertIn("nested SEARCH marker", errors[0])

    def test_normal_content_still_parses(self) -> None:
        text = (
            "EDIT_BEGIN p\n"
            "<<<<<<< SEARCH\nval x = 1\n=======\nval x = 2\n>>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        _path, pairs = self._single_pair(text)
        self.assertEqual(pairs, [("val x = 1", "val x = 2")])


class StuckAttemptDirectiveTest(unittest.TestCase):
    @staticmethod
    def _attempt(tag: str) -> dict:
        return {"response_hash": tag, "error_hash": "err", "applied": False}

    def test_identical_hardware_attempts_trigger_directive(self) -> None:
        state = {"hw_attempts": [self._attempt("a"), self._attempt("a")]}
        self.assertIn("REPEATED IDENTICAL ATTEMPT", feedback.hardware_stuck_directive(state))

    def test_changed_hardware_attempt_stays_quiet(self) -> None:
        state = {"hw_attempts": [self._attempt("a"), self._attempt("b")]}
        self.assertEqual(feedback.hardware_stuck_directive(state), "")
        self.assertEqual(feedback.hardware_stuck_directive({"hw_attempts": []}), "")

    def test_software_directive_unchanged(self) -> None:
        state = {"sw_attempts": [self._attempt("a"), self._attempt("a")]}
        self.assertIn("REPEATED IDENTICAL ATTEMPT", feedback.software_stuck_directive(state))


class ConformanceOracleDirectiveTest(unittest.TestCase):
    """The 2026-09-26 matmult-int run oscillated between two oracle rejections
    (arg count -> arg literal -> arg count) until the budget was gone."""

    @staticmethod
    def _attempt(tag: str) -> dict:
        return {"problems_hash": tag, "response_hash": tag}

    def test_recurrence_triggers_directive(self) -> None:
        a, b = self._attempt("count-error"), self._attempt("literal-error")
        state = {"conformance_attempts": [a, b, a]}
        directive = feedback.conformance_oracle_directive(state)
        self.assertIn("REPEATED ORACLE REJECTION", directive)
        self.assertIn("setup", directive)
        self.assertIn("args: [0]", directive)
        self.assertIn("undefined reference", directive)  # invented intrinsic names

    def test_first_rejection_stays_quiet(self) -> None:
        a, b = self._attempt("count-error"), self._attempt("literal-error")
        self.assertEqual(feedback.conformance_oracle_directive({"conformance_attempts": [a, b]}), "")
        self.assertEqual(feedback.conformance_oracle_directive({"conformance_attempts": []}), "")


class ConformanceBuildFeedbackTest(unittest.TestCase):
    """A build failure must show the compiler/linker output (2026-09-26: five
    blind repairs of `undefined reference to cfu_load` because it was hidden)."""

    def test_build_failure_carries_linker_diagnostics(self) -> None:
        build = {
            "returncode": 2,
            "stderr_tail": (
                "cfu_conformance.c:(.text.main): undefined reference to `cfu_load'\n"
                "collect2: error: ld returned 1 exit status"
            ),
        }
        message = feedback.conformance_build_error(build, ["make", "-C", "sw", "compile"])
        self.assertIn("conformance test build failed", message)
        self.assertIn("build diagnostics:", message)
        self.assertIn("undefined reference to `cfu_load'", message)

    def test_build_failure_without_output_stays_compact(self) -> None:
        message = feedback.conformance_build_error({"returncode": 2}, ["make"])
        self.assertIn("conformance test build failed", message)
        self.assertNotIn("diagnostics", message)


class RejectionAttemptTest(unittest.TestCase):
    def test_recurrence_hash_ignores_incidental_chatter(self) -> None:
        common = [
            "conformance test build failed: make ...",
            "undefined reference to `cfu_load'",
        ]
        a = feedback._rejection_attempt(list(common), "")
        b = feedback._rejection_attempt(
            common + ["ld: warning: LOAD segment with RWX permissions"], ""
        )
        self.assertIsNotNone(a)
        self.assertEqual(a["problems_hash"], b["problems_hash"])

    def test_no_problems_no_entry(self) -> None:
        self.assertIsNone(feedback._rejection_attempt([]))


if __name__ == "__main__":
    unittest.main()
