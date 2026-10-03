"""One implementation stage must see every failing rung, exactly once.

The free Implement stage absorbs repairs from both sides, so its prompt has to
carry the complete failure picture: the hardware sources (hw_build,
hw_conformance, sim, cost, the hardware rung's own rejection) *and* the software
ones (sw_build, the software rung's rejection). Two properties matter:

* every failing rung is attributed to the rung that produced it, and
* nothing is inlined whole — a 50 KB tool log contributes an excerpt plus a log
  path, not 50 KB of prompt.
"""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.stages import feedback
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)

# The excerpt keeps a head and a tail window, so the proof of truncation has to
# sit in the middle of the log.
LONG_LOG = ("compiler noise\n" * 4000 + "MIDDLE_MARKER_SHOULD_NOT_BE_INLINED\n"
            + "compiler noise\n" * 4000 + "final diagnostic line")
assert len(LONG_LOG) > 100_000


def failed(rung: str) -> dict:
    return {"rung": rung, "returncode": 1, "command": f"make {rung}",
            "stdout_tail": LONG_LOG, "stderr_tail": ""}


class AllFailureFeedbackTest(unittest.TestCase):
    def _state(self, **overrides) -> dict:
        state = {
            "validation_results": [failed("hw_build"), failed("hw_conformance"),
                                   failed("sw_build"), failed("sim")],
            "cost_results": [failed("cost")],
            "gates": {
                "hw_conformance": {"ok": False, "errors": ["conformance verdict failed: 0/1"]},
                "implementation_hw": {"ok": False, "errors": ["hw delivery rejected"]},
                "implementation_sw": {"ok": False, "errors": ["sw delivery rejected"]},
            },
        }
        state.update(overrides)
        return state

    def test_every_failing_rung_is_attributed(self) -> None:
        text = feedback.all_failure_feedback(self._state())
        for label in ("hw_build", "hw_conformance", "sim", "cost",
                      "sw_build", "implementation_hw", "implementation_sw"):
            with self.subTest(label=label):
                self.assertIn(label, text)

    def test_the_software_side_is_not_dropped(self) -> None:
        text = feedback.all_failure_feedback(self._state())
        self.assertIn("sw delivery rejected", text)
        self.assertIn("make sw_build", text)

    def test_a_long_log_contributes_an_excerpt_not_the_whole_log(self) -> None:
        text = feedback.all_failure_feedback(self._state())
        self.assertNotIn("MIDDLE_MARKER_SHOULD_NOT_BE_INLINED", text)
        self.assertIn("...[truncated]...", text)
        self.assertLess(len(text), 40_000, "the failure digest must stay bounded")
        self.assertLess(len(text), len(LONG_LOG) // 4, "no single log may dominate the prompt")

    def test_the_simulation_evidence_appears_once(self) -> None:
        """`hardware_repair_feedback` already includes it; it was duplicated."""
        text = feedback.all_failure_feedback(self._state())
        self.assertEqual(text.count("make sim"), 1)

    def test_a_clean_run_produces_no_feedback(self) -> None:
        state = {"validation_results": [{"rung": "hw_build", "returncode": 0}],
                 "cost_results": [{"rung": "cost", "returncode": 0}], "gates": {}}
        self.assertEqual(feedback.all_failure_feedback(state), "")

    def test_a_resolved_rung_drops_out(self) -> None:
        """A later passing recheck resolves an earlier failure."""
        state = self._state(validation_results=[
            failed("hw_build"), failed("sw_build"),
            {"rung": "sw_build", "returncode": 0, "command": "make sw_build"}])
        text = feedback.all_failure_feedback(state)
        self.assertNotIn("make sw_build", text)
        self.assertIn("make hw_build", text)


if __name__ == "__main__":
    unittest.main()
