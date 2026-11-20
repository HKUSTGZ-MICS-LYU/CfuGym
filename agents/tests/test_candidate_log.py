"""Offline durable candidate logging contracts; no graph, snapshots or toolchain."""
from __future__ import annotations

import copy
import io
import unittest
from contextlib import ExitStack
from unittest import mock

from agents.cfu_design_agent.flow import evaluations, gates, instrument
from agents.cfu_design_agent.flow import optimize as flow_optimize, outcome
from agents.cfu_design_agent.runtime import candidate_log, stage_services
from agents.cfu_design_agent.stages import review
from agents.cfu_design_agent.stages.review import optimize as planner


def _entry(**overrides):
    entry = {
        "round": 2, "target": "implement", "isa_revision": 3,
        "evaluation_id": "eval-17", "verify_submission": 4,
        "measured": True, "validated": True,
        "reference_cycles": 1200, "cfu_cycles": 600, "speedup_x100": 200,
        "cost": {"backend": "vivado", "metric": "vivado_cfu_lut",
                 "lut": 123, "ff": 456, "dsp": 7, "bram_36k": 1.5,
                 "part": "xc7a100tcsg324-1", "target_frequency_mhz": 100},
    }
    entry.update(overrides)
    return entry


def _state(**overrides):
    state = {"max_iters": 8, "loop_objective": "pareto", "optimization_frontier": [2]}
    state.update(overrides)
    return state


def _lines(entry, *, rounds=None, selected=None, candidate=None, state=None):
    return candidate_log.validated_candidate_lines(
        _state() if state is None else state, entry,
        [entry] if rounds is None else rounds,
        entry if selected is None else selected,
        {"round": entry["round"], "manifest": "memory-only/candidate-manifest.yaml"}
        if candidate is None else candidate,
    )


class CandidateFormatterTest(unittest.TestCase):
    def test_vivado_complete_metrics_identity_source_and_no_mutation(self):
        entry, state = _entry(), _state()
        rounds = [entry]
        candidate = {"round": 2, "manifest": "memory-only/candidate-manifest.yaml"}
        before = copy.deepcopy((state, entry, rounds, candidate))
        lines = _lines(entry, rounds=rounds, candidate=candidate, state=state)
        text = "\n".join(lines)
        for fragment in (
            "[candidate] VALIDATED round=2 target=implement isa_revision=3",
            "evaluation=eval-17 verify=4/8",
            "reference_cycles=1200 cfu_cycles=600 speedup=2.00x",
            "backend=vivado metric=vivado_cfu_lut CFU_LUT=123 FF=456 DSP=7 BRAM36=1.5",
            "part=xc7a100tcsg324-1 target_MHz=100", "vs_parent=N/A",
            "[best] objective=pareto round=2 cfu_cycles=600 speedup=2.00x CFU_LUT=123 current_selected=yes frontier=[2]",
            "[source] memory-only/candidate-manifest.yaml",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        self.assertEqual(len(lines), 6)
        self.assertEqual((state, entry, rounds, candidate), before)
        self.assertNotIn("cells=", text)

    def test_yosys_legacy_cost_has_backend_and_explicit_units(self):
        entry = _entry(cost={"cells": 81, "wire_bits": 1234,
                             "memories": 2, "memory_bits": 128})
        text = "\n".join(_lines(entry))
        self.assertIn("backend=yosys metric=yosys_generic_mapped_cells", text)
        self.assertIn("cells=81 wire_bits=1234 memories=2 memory_bits=128", text)
        self.assertNotIn("CFU_LUT", text)
        self.assertNotIn("BRAM36", text)

    def test_missing_metrics_are_na_but_real_zero_is_preserved(self):
        for backend, keys in (
            ("vivado", {"lut": "CFU_LUT", "ff": "FF", "dsp": "DSP",
                        "bram_36k": "BRAM36", "target_frequency_mhz": "target_MHz"}),
            ("yosys", {"cells": "cells", "wire_bits": "wire_bits",
                       "memories": "memories", "memory_bits": "memory_bits"}),
        ):
            for value, expected in ((None, "N/A"), (0, "0")):
                with self.subTest(backend=backend, value=value):
                    entry = _entry(reference_cycles=value, cfu_cycles=value,
                                   speedup_x100=value, cost={"backend": backend})
                    if value is not None:
                        entry["cost"].update(dict.fromkeys(keys, value))
                    text = "\n".join(_lines(entry))
                    self.assertIn(f"reference_cycles={expected} cfu_cycles={expected}", text)
                    self.assertIn("speedup=" + ("N/A" if value is None else "0.00x"), text)
                    for label in keys.values():
                        self.assertIn(f"{label}={expected}", text)

    def test_nonfinite_and_boolean_metrics_are_not_reported_as_measurements(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, False):
            with self.subTest(value=value):
                text = "\n".join(_lines(_entry(cfu_cycles=value, speedup_x100=value,
                                                cost={"cells": value})))
                self.assertIn("cfu_cycles=N/A speedup=N/A", text)
                self.assertIn("cells=N/A", text)

    def test_nonvalidated_unmeasured_and_snapshot_error_emit_nothing(self):
        for change in ({"validated": False}, {"validated": None}, {"validated": 1},
                       {"measured": False}, {"measured": None}, {"measured": 1},
                       {"snapshot_error": "source snapshot unavailable"}):
            with self.subTest(change=change):
                self.assertEqual(_lines(_entry(**change)), [])
        for key in ("validated", "measured"):
            entry = _entry()
            del entry[key]
            with self.subTest(missing=key):
                self.assertEqual(_lines(entry), [])

    def test_regression_is_current_while_best_and_source_remain_incumbent(self):
        old = _entry(round=0, cfu_cycles=400, speedup_x100=300)
        current = _entry(round=1, cfu_cycles=800, speedup_x100=150, improved=False)
        text = "\n".join(_lines(current, rounds=[old, current], selected=old,
                                 candidate={"round": 0, "manifest": "incumbent-manifest.yaml"},
                                 state=_state(optimization_frontier=[0])))
        self.assertIn("[candidate] VALIDATED round=1", text)
        self.assertIn("[performance] reference_cycles=1200 cfu_cycles=800 speedup=1.50x", text)
        self.assertIn("[best] objective=pareto round=0 cfu_cycles=400 speedup=3.00x", text)
        self.assertIn("CFU_LUT=123 current_selected=no frontier=[0]", text)
        self.assertIn("[source] incumbent-manifest.yaml", text)
        self.assertNotIn("current_selected=yes", text)

    def test_delta_uses_declared_parent_not_previous_round(self):
        parent = _entry(round=0, cfu_cycles=400)
        previous = _entry(round=1, cfu_cycles=1000, cost={**parent["cost"], "lut": 999})
        current = _entry(round=2, parent_round=0, cfu_cycles=600,
                         cost={**parent["cost"], "lut": 143})
        text = "\n".join(_lines(current, rounds=[parent, previous, current]))
        self.assertIn("[delta] vs_round=0 cycles=+200 (+50.0%) lut=+20", text)
        self.assertNotIn("vs_round=1", text)
        missing = {**current, "parent_round": 99}
        self.assertIn("[delta] vs_parent=N/A", "\n".join(_lines(missing, rounds=[parent, previous, missing])))

    def test_cost_delta_is_not_compared_across_eda_bases(self):
        parent = _entry(round=0, cfu_cycles=400)
        for changed in (
            {"backend": "yosys", "cells": 100},
            {**parent["cost"], "metric": "different_utilization_metric"},
            {**parent["cost"], "part": "another-part"},
            {**parent["cost"], "target_frequency_mhz": 200},
        ):
            with self.subTest(cost=changed):
                current = _entry(parent_round=0, cost=changed)
                delta = next(line for line in _lines(current, rounds=[parent, current]) if "[delta]" in line)
                self.assertIn("cycles=+200 (+50.0%)", delta)
                self.assertIn("cost=N/A (EDA basis differs)", delta)
                self.assertNotIn("lut=", delta)
                self.assertNotIn("cells=", delta)

    def test_reference_basis_change_does_not_claim_cycle_delta(self):
        parent = _entry(round=0, reference_cycles=999)
        current = _entry(parent_round=0)
        text = "\n".join(_lines(current, rounds=[parent, current]))
        self.assertIn("cycles=N/A (measurement basis differs)", text)


class _FlushBuffer(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = []

    def flush(self):
        self.flushes.append(self.getvalue())
        super().flush()


class CandidateProgressTest(unittest.TestCase):
    def setUp(self):
        instrument.disable_progress()
        self.addCleanup(instrument.disable_progress)

    def test_disabled_progress_has_no_stdout_or_formatter_work(self):
        out = _FlushBuffer()
        with mock.patch.object(instrument.sys, "stdout", out), \
                mock.patch.object(candidate_log, "validated_candidate_lines") as formatter:
            instrument.report_validated_candidate(_state(), _entry(), [], None, {})
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(out.flushes, [])
        formatter.assert_not_called()

    def test_enabled_non_tty_is_durable_multiline_clears_live_line_and_flushes(self):
        out = _FlushBuffer()
        entry = _entry()
        candidate = {"round": 2, "manifest": "memory-only/manifest.yaml"}
        instrument.enable_progress()
        with mock.patch.object(instrument.sys, "stdout", out), \
                mock.patch.object(instrument._LIVE_LINE, "clear", wraps=instrument._LIVE_LINE.clear) as clear:
            self.assertFalse(out.isatty())
            instrument._LIVE_LINE.active, instrument._LIVE_LINE.width = True, 30
            instrument.report_validated_candidate(_state(), entry, [entry], entry, candidate)
            clear.assert_called_once_with()
            self.assertFalse(instrument._LIVE_LINE.active)
            self.assertEqual(instrument._LIVE_LINE.width, 0)
        expected = "\n" + "\n".join(_lines(entry, candidate=candidate)) + "\n"
        self.assertEqual(out.getvalue(), expected)
        self.assertTrue(out.flushes)
        self.assertEqual(out.flushes[-1], expected)
        self.assertNotIn("\r", out.getvalue())
        self.assertNotIn("\x1b", out.getvalue())

    def test_enabled_invalid_candidate_does_not_clear_or_flush(self):
        instrument.enable_progress()
        out = _FlushBuffer()
        with mock.patch.object(instrument.sys, "stdout", out), \
                mock.patch.object(instrument._LIVE_LINE, "clear") as clear:
            instrument.report_validated_candidate(_state(), _entry(validated=False), [], None, {})
            clear.assert_not_called()
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(out.flushes, [])


class ReviewCandidateLogTest(unittest.TestCase):
    """Call only the review node; mock all file/model/control-plane seams at owners."""

    def setUp(self):
        instrument.disable_progress()
        self.addCleanup(instrument.disable_progress)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.old = _entry(round=0, cfu_cycles=800, speedup_x100=150)
        self.new = _entry(round=1, cfu_cycles=600, parent_round=0)
        self.candidate = {"round": 1, "manifest": "memory-only/new.yaml"}
        self.state = _state(
            optimization_rounds=[self.old], optimization_frontier=[0],
            optimization_plan={"target": "implement", "parent_round": 0},
            best_candidate={"round": 0, "manifest": "memory-only/old.yaml"},
            gates={}, loop_optimize=True,
        )
        self.decision = self.patch(gates, "_review_decision", return_value=("final_report", "complete"))
        self.patch(gates, "_rung_gate", return_value={})
        self.patch(gates, "_verify_submissions_left", return_value=True)
        self.patch(evaluations, "used", return_value=0)
        self.patch(evaluations, "remaining", return_value=True)
        self.score = self.patch(flow_optimize, "_score_snapshot", side_effect=lambda *a, **kw: copy.deepcopy(self.new))
        self.patch(flow_optimize, "_round_improved", return_value=True)
        self.patch(flow_optimize, "_best_round", return_value=self.old)
        self.patch(flow_optimize, "select_round", side_effect=lambda rounds, objective: next(
            (row for row in reversed(rounds) if not row.get("snapshot_error")), None))
        self.patch(flow_optimize, "pareto_frontier", side_effect=lambda rounds: [
            row for row in rounds if not row.get("snapshot_error")][-1:])
        self.snapshot = self.patch(planner, "_snapshot_candidate", return_value=self.candidate)
        self.choose = self.patch(planner, "_choose_optimization", return_value=("", {}, ""))
        self.patch(outcome, "stage_result", side_effect=lambda state, **kw: {
            "gates": {"review": {"decision": kw["decision"]}}, "iteration_history": [],
        })
        self.artifacts = self.patch(stage_services, "_artifacts_for_stage")
        self.patch(stage_services, "_report_review_decision")
        self.report = self.patch(instrument, "report_validated_candidate", wraps=instrument.report_validated_candidate)
        self.out = _FlushBuffer()
        self.stack.enter_context(mock.patch.object(instrument.sys, "stdout", self.out))
        instrument.enable_progress()

    def patch(self, owner, name, **kwargs):
        return self.stack.enter_context(mock.patch.object(owner, name, **kwargs))

    def test_complete_logs_and_flushes_new_frontier_before_planner(self):
        def choose(state, rounds):
            self.report.assert_called_once()
            self.assertIn("[candidate] VALIDATED round=1", self.out.getvalue())
            self.assertIn("frontier=[1]", self.out.getvalue())
            self.assertTrue(self.out.flushes)
            self.assertEqual(self.out.flushes[-1], self.out.getvalue())
            self.assertEqual(state["optimization_frontier"], [1])
            self.assertEqual(state["best_candidate"], self.candidate)
            self.assertEqual(rounds[-1]["round"], 1)
            return "", {}, ""

        self.choose.side_effect = choose
        result = review.review_and_route(self.state)
        self.choose.assert_called_once()
        self.snapshot.assert_called_once()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["optimization_rounds"]), 2)
        self.assertIn("[source] memory-only/new.yaml", self.out.getvalue())
        logged_state, logged_entry, rounds, selected, candidate = self.report.call_args.args
        self.assertEqual(logged_state["optimization_frontier"], [1])
        self.assertEqual((logged_entry["round"], selected["round"], candidate["round"]), (1, 1, 1))
        self.assertEqual(rounds[-1]["parent_round"], 0)

    def test_validated_regression_logs_current_without_replacing_best(self):
        self.new.update(cfu_cycles=1000, speedup_x100=120)
        incumbent = self.state["best_candidate"]
        self.patch(flow_optimize, "_round_improved", return_value=False)
        self.patch(flow_optimize, "select_round", return_value=self.old)
        self.patch(flow_optimize, "pareto_frontier", return_value=[self.old])

        def choose(state, rounds):
            text = self.out.getvalue()
            self.assertIn("[candidate] VALIDATED round=1", text)
            self.assertIn("[performance] reference_cycles=1200 cfu_cycles=1000 speedup=1.20x", text)
            self.assertIn("[best] objective=pareto round=0 cfu_cycles=800 speedup=1.50x", text)
            self.assertIn("current_selected=no frontier=[0]", text)
            self.assertEqual(state["best_candidate"], incumbent)
            self.assertFalse(rounds[-1]["improved"])
            return "", {}, ""

        self.choose.side_effect = choose
        result = review.review_and_route(self.state)
        self.report.assert_called_once()
        self.choose.assert_called_once()
        self.snapshot.assert_not_called()
        self.assertEqual(result["best_candidate"], incumbent)
        self.assertEqual(result["optimization_rounds"][-1]["cfu_cycles"], 1000)
        self.assertEqual(result["status"], "complete")

    def test_failed_and_planned_reviews_never_log_success_or_call_planner(self):
        for status in ("failed", "planned"):
            with self.subTest(status=status):
                self.decision.return_value = "final_report", status
                result = review.review_and_route(copy.deepcopy(self.state))
                self.assertEqual(result["status"], status)
        self.report.assert_not_called()
        self.choose.assert_not_called()
        self.snapshot.assert_not_called()
        self.assertEqual(self.out.getvalue(), "")

    def test_draft_cannot_emit_success_even_if_gate_review_says_complete(self):
        result = review.review_and_route({**self.state, "implementation_design_type": "draft"})
        self.assertEqual(result["status"], "blocked")
        self.score.assert_not_called()
        self.snapshot.assert_not_called()
        self.report.assert_not_called()
        self.choose.assert_not_called()
        self.assertEqual(self.out.getvalue(), "")

    def test_snapshot_error_prevents_success_log_and_planner(self):
        self.snapshot.side_effect = OSError("synthetic snapshot unavailable")
        result = review.review_and_route(self.state)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["optimization_rounds"][-1]["snapshot_error"], "synthetic snapshot unavailable")
        self.report.assert_not_called()
        self.choose.assert_not_called()
        self.assertEqual(self.out.getvalue(), "")

    def test_without_new_record_historical_success_is_not_relogged(self):
        result = review.review_and_route({**self.state, "optimization_plan": {}})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["optimization_rounds"], [self.old])
        self.score.assert_not_called()
        self.snapshot.assert_not_called()
        self.report.assert_not_called()
        self.choose.assert_called_once()
        self.assertEqual(self.out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
