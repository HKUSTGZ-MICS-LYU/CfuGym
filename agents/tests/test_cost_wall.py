"""Offline Cost Wall contracts: mapped cells, fail-fast and actionable repair."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import (
    commands, cost_budget, effects, rtl_snapshot, stage_services, tools, validation_parallel, vivado,
)
from agents.cfu_design_agent.stages import feedback
from agents.cfu_design_agent.stages.verify import cost


class CostBudgetCheckTest(unittest.TestCase):
    def check(self, cells, *, limit=100_000):
        return cost_budget.check({"max_yosys_cells": limit}, {"cells": cells},
                                 top="AgentCfu", summary_path="yosys/summary.json")

    def test_at_or_below_limit_passes_including_zero(self):
        for cells in (0, 1, 99_999, 100_000):
            with self.subTest(cells=cells):
                wall = self.check(cells)
                self.assertTrue(wall["ok"])
                self.assertEqual(wall["cells"], cells)
                self.assertEqual(wall["max_cells"], 100_000)

    def test_over_budget_has_actionable_error_not_fpga_capacity_claim(self):
        wall = self.check(444_032)
        self.assertFalse(wall["ok"])
        self.assertEqual(wall["code"], "cost_budget_exceeded")
        for phrase in ("444032", "100000", "Reduce design complexity", "share multipliers",
                       "multi-cycle", "Preserve", "not FPGA LUT capacity", "yosys/summary.json"):
            self.assertIn(phrase, wall["reason"])

    def test_real_yosys_output_parser_feeds_the_host_wall(self):
        result = {"returncode": 0, "parsed": tools.parse_command_output(
            "cost top=AgentCfu cells=444032 wire_bits=461470 memories=0 memory_bits=0\n")}
        wall = cost._check_cost_wall({}, result, top="AgentCfu", summary_path="yosys/summary.json")
        self.assertEqual(wall["cells"], 444_032)
        self.assertEqual(wall["code"], "cost_budget_exceeded")
        self.assertEqual(result["returncode"], 0)
        self.assertIs(result["parsed"]["cost_wall"], wall)

    def test_custom_limit_is_used_without_rounding(self):
        self.assertTrue(self.check(123, limit=123)["ok"])
        self.assertFalse(self.check(124, limit=123)["ok"])

    def test_invalid_or_missing_measurement_fails_closed(self):
        for cells in (None, -1, "100", 1.0, True, {}, float("nan")):
            with self.subTest(cells=cells):
                wall = self.check(cells)
                self.assertFalse(wall["ok"])
                self.assertEqual(wall["code"], "cost_metrics_invalid")
                self.assertNotIn("Reduce design complexity", wall["reason"])
                self.assertIsNone(wall["cells"])
        for metrics in (None, {}, [], "not metrics"):
            with self.subTest(metrics=metrics):
                self.assertFalse(cost_budget.check({}, metrics, top="AgentCfu", summary_path="s")["ok"])


class CostWallWorkerTest(unittest.TestCase):
    def call(self, *, cells=123, backend="vivado", limit=100_000,
             missing=False, malformed=False, cancelled=False, dry_run=False, skip=False,
             transition=False, verify_submissions=1):
        with tempfile.TemporaryDirectory(prefix="cost-wall-test-") as root, ExitStack() as stack:
            state = {"workdir": root, "run_root": "agents/generated/runs/cost-wall-test",
                     "agent_cfu": True, "cost_backend": backend, "run_commands": True,
                     "evaluation_id": "wall-eval", "max_yosys_cells": limit,
                     "dry_run": dry_run, "skip_cost": skip, "max_iters": 3,
                     "verify_submissions": verify_submissions}
            toolbox = mock.Mock()
            toolbox.run_command.return_value = {"returncode": None, "skipped": True}
            stack.enter_context(mock.patch.object(stage_services, "_toolbox_for_stage", return_value=toolbox))
            write = stack.enter_context(mock.patch.object(stage_services, "_write_cost_artifact"))
            stack.enter_context(mock.patch.object(rtl_snapshot, "validate", return_value={"root": "snapshot", "sha256": "rtl"}))
            stack.enter_context(mock.patch.object(commands, "agent_cfu_cost_commands", return_value=[["sbt", "fake-generator"]]))
            stack.enter_context(mock.patch.object(cost.software, "_rtl_ready", return_value=True))
            build = stack.enter_context(mock.patch.object(vivado, "build_command", return_value=["python3", "fake-vivado"]))
            load = stack.enter_context(mock.patch.object(vivado, "load_summary", return_value={"status": "passed"}))
            notify = stack.enter_context(mock.patch.object(validation_parallel, "command_failed"))

            def run_command(_state, _toolbox, argv, **kwargs):
                if "yosys_cost_report.py" in " ".join(argv):
                    if cancelled:
                        return {"returncode": -15, "cancelled": True}
                    if malformed:
                        return {"returncode": 0, "parsed": {"cost": "broken metrics"}}
                    return {"returncode": 0, "parsed": {} if missing else {
                        "cost": {"top": "AgentCfu", "cells": cells, "wire_bits": 456}}}
                return {"returncode": 0}

            run = stack.enter_context(mock.patch.object(effects, "run_command", side_effect=run_command))
            update = cost.estimate_cost(state, transition=transition)
            return update, run, build, load, notify, write

    def test_both_backends_enforce_budget_and_preserve_successful_yosys(self):
        for backend in ("yosys", "vivado"):
            with self.subTest(backend=backend):
                update, run, build, load, notify, write = self.call(cells=444_032, backend=backend)
                gate = update["gates"]["cost"]
                self.assertFalse(gate["ok"])
                self.assertEqual(gate["cost_wall"]["code"], "cost_budget_exceeded")
                self.assertEqual(gate["cost_wall"]["cells"], 444_032)
                self.assertEqual(run.call_count, 2)  # standalone generator + Yosys; never Vivado
                build.assert_not_called()
                load.assert_not_called()
                notify.assert_called_once()
                self.assertEqual(write.call_args.args[2], "failed")
                successful = update["cost_results"][-2]
                self.assertEqual(successful["returncode"], 0)
                self.assertEqual(successful["parsed"]["cost"]["cells"], 444_032)
                failure = update["cost_results"][-1]
                self.assertEqual(failure["kind"], "host_gate_check")
                self.assertEqual(failure["failure_observation"]["category"], "cost_budget_exceeded")
                self.assertTrue(failure["failure_observation"]["id"].startswith("fail_cost_"))
                self.assertTrue(failure["log_path"].endswith("/11_cost/result.yaml"))
                if backend == "vivado":
                    self.assertIn("Vivado was not started", gate["reason"])
                    self.assertNotIn("Passed Yosys", " ".join(update.get("iteration_history", [])))

    def test_exact_limit_allows_vivado_and_keeps_wall_evidence(self):
        update, run, build, load, notify, _ = self.call(cells=100_000)
        gate = update["gates"]["cost"]
        self.assertTrue(gate["ok"])
        self.assertEqual(run.call_count, 3)
        build.assert_called_once()
        load.assert_called_once()
        notify.assert_not_called()
        self.assertTrue(gate["ppa"]["yosys_precheck"]["cost_wall"]["ok"])

    def test_exact_limit_allows_yosys_only(self):
        update, _, _, _, notify, _ = self.call(cells=100_000, backend="yosys")
        self.assertTrue(update["gates"]["cost"]["ok"])
        notify.assert_not_called()

    def test_custom_limit_blocks_one_above_and_allows_equality(self):
        for backend in ("yosys", "vivado"):
            for cells, ok in ((9, True), (10, False)):
                with self.subTest(backend=backend, cells=cells):
                    update, *_ = self.call(cells=cells, backend=backend, limit=9)
                    self.assertEqual(update["gates"]["cost"]["ok"], ok)

    def test_missing_or_invalid_metrics_never_start_vivado(self):
        for options in ({"missing": True}, {"malformed": True}, {"cells": None}, {"cells": "123"}, {"cells": True}):
            for backend in ("yosys", "vivado"):
                with self.subTest(backend=backend, options=options):
                    update, _, build, load, notify, _ = self.call(backend=backend, **options)
                    self.assertEqual(update["gates"]["cost"]["cost_wall"]["code"], "cost_metrics_invalid")
                    build.assert_not_called()
                    load.assert_not_called()
                    notify.assert_called_once()

    def test_skipped_dry_run_and_cancelled_do_not_claim_or_spend_wall_failure(self):
        for backend in ("yosys", "vivado"):
            for options in ({"skip": True}, {"dry_run": True}, {"cancelled": True}):
                with self.subTest(backend=backend, options=options):
                    update, _, build, load, notify, _ = self.call(backend=backend, transition=True, **options)
                    gate = update["gates"]["cost"]
                    self.assertEqual(cost.ladder._route_after_cost(update), "review_and_route")
                    self.assertFalse(update.get("attempts"))
                    self.assertFalse(gate["ok"])
                    self.assertFalse(gate.get("cost_wall"))
                    self.assertEqual(update.get("errors", []), [])
                    self.assertNotIn("repairs_used", update)
                    build.assert_not_called()
                    load.assert_not_called()
                    notify.assert_not_called()

    def test_repair_feedback_includes_id_measurements_and_complexity_reduction(self):
        update, *_ = self.call(cells=444_032)
        text = feedback.all_failure_feedback({"cost_backend": "vivado", **update})
        for phrase in ("fail_cost_", "cost_budget_exceeded", "444032", "100000",
                       "Reduce design complexity", "share multipliers", "multi-cycle", "/summary.json"):
            self.assertIn(phrase, text)

    def test_repaired_cost_clears_old_failure_feedback(self):
        failed, *_ = self.call(cells=444_032)
        passed, *_ = self.call(cells=123)
        state = {"cost_backend": "vivado", "cost_results": failed["cost_results"] + passed["cost_results"]}
        self.assertEqual(feedback.cost_feedback(state), "")

    def test_serial_failure_uses_existing_repair_budget_and_target(self):
        update, _, _, _, notify, _ = self.call(cells=444_032, transition=True)
        self.assertEqual(update["gates"]["cost"]["next"], "implement")
        self.assertEqual(update["repairs_used"], 1)
        self.assertEqual(update["last_repair_target"], "implement")
        notify.assert_not_called()
        self.assertIn("cost_budget_exceeded", ";".join(update["errors"]))

    def test_exhausted_budget_stops_instead_of_repairing_forever(self):
        update, *_ = self.call(cells=444_032, transition=True, verify_submissions=3)
        self.assertEqual(update["gates"]["cost"]["next"], "review_and_route")
        self.assertEqual(update["stopped_at"], "cost")
        self.assertNotIn("repairs_used", update)


if __name__ == "__main__":
    unittest.main()
