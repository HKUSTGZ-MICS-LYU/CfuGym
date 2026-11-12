"""Offline, temporary, synthetic experiment-history reporting tests."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from agents.experiments.harness import collection, history, metrics, reporting


class HistoryReportingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fw = self.root / "framework"
        h = "a" * 64
        self.sim = {"terminal_reason": "passed", "correctness_ok": True, "timing_ok": True,
                    "performance_valid": True, "design_changed": True,
                    "reference_cycles": 1000, "cfu_cycles": 500, "speedup_x100": 200,
                    "measurement_protocol": "p1", "benchmark_integrity": {"ok": True},
                    "commands": [{"returncode": 0}, {"kind": "host_gate_check", "returncode": 1}],
                    "performance_provenance": {"input_sha256": h,
                        "candidate_hashes": {"hardware": h, "software": h, "isa": "b" * 16},
                        "evidence_epoch": {}, "compiled_products": {
                            k: {"path": k, "sha256": h} for k in ("elf", "rtl", "asm")}}}
        self.record = {"run_name": "synthetic", "model_config": "fake", "benchmark": "example",
                       "status": "failed", "stopped_at": "sim", "sim_terminal_reason": "speedup_floor",
                       "measurement_protocol": "p1", "started_at": "2000-01-01T00:00:00+00:00",
                       "finished_at": "2999-01-01T00:00:00+00:00", "reference_cycles": 1000,
                       "terminal": True, "framework_run_dir": str(self.fw),
                       "speedup_x100": 90, "performance_valid": True,
                       "failure": {"failure_class": "sim_speedup_floor", "failure_rung": "sim"}}

    def save(self, sim, *, bad_name=False):
        raw = yaml.safe_dump(sim).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = self.fw / "10_validation/versions" / ("sim.yaml." + ("0" * 64 if bad_name else digest))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def test_highest_verified_not_last_or_final_and_reported(self):
        self.save(self.sim)
        self.save({**self.sim, "cfu_cycles": 800, "speedup_x100": 125})
        self.save({**self.sim, "cfu_cycles": 100, "speedup_x100": 1000, "correctness_ok": False})
        archive = self.root / "runs/synthetic/record.json"
        archive.parent.mkdir(parents=True)
        archive.write_text(json.dumps(self.record))
        original = archive.read_bytes()
        result = reporting.aggregate(self.root, {"exp_id": "fixture"}, bootstrap_iters=2)
        self.assertEqual(archive.read_bytes(), original)
        with (self.root / "results/failures.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["historical_best_speedup_x100"], "200")
        summary = Path(result["summary"]).read_text()
        self.assertIn("| 0.90 | 2.00 |", summary)
        with (self.root / "results/runs.csv").open() as f:
            run_row = next(csv.DictReader(f))
        self.assertEqual(run_row["speedup_x100"], "90")
        self.assertEqual(run_row["display_speedup_x100"], "200")
        self.assertEqual(run_row["display_speedup_source"], "historical_best_candidate")
        record = reporting.load_records(self.root)[0]
        self.assertEqual(record["speedup_x100"], 90)
        self.assertEqual(record["historical_best_performance"]["speedup_x100"], 200)
        self.assertEqual(metrics.cell_metrics([record], bootstrap_iters=2)["speedup_samples"], 0)
        self.assertIsNot(metrics._pass_flag(record, 1), True)

    def test_untrusted_evidence_is_excluded(self):
        variants = [
            {**self.sim, "measurement_protocol": "other"},
            {**self.sim, "commands": [{"kind": "host_gate_check", "returncode": 0}]},
            {**self.sim, "commands": [{"returncode": 0, "skipped": True}]},
            {**self.sim, "timing_ok": False},
            {**self.sim, "benchmark_integrity": {}},
            {**self.sim, "speedup_x100": 201},
            {**self.sim, "performance_provenance": {}},
            {**self.sim, "performance_provenance": {**self.sim["performance_provenance"], "compiled_products": {}}},
        ]
        for value in variants:
            self.save(value)
        self.save(self.sim, bad_name=True)
        self.assertEqual(history.best_speedup(self.record, self.fw), {})
        self.save(self.sim)
        self.assertEqual(history.best_speedup({**self.record, "started_at": "2999-01-01T00:00:00+00:00"}, self.fw), {})
        self.assertEqual(history.best_speedup({**self.record, "measurement_protocol": "legacy"}, self.fw), {})
        self.assertEqual(history.best_speedup({**self.record, "status": "complete"}, self.fw), {})
        self.assertEqual(history.best_speedup({**self.record, "sim_terminal_reason": "verify_failed"}, self.fw), {})
        for change in ({"finished_at": None}, {"finished_at": "2001-01-01T00:00:00+00:00"},
                       {"reference_cycles": 999}, {"stopped_at": "hw_build"}, {"stopped_at": "cost"},
                       {"failure": {"failure_class": "sim_correctness"}}):
            self.assertEqual(history.best_speedup({**self.record, **change}, self.fw), {})

    def test_display_current_requires_matching_current_provenance(self):
        path = self.root / "runs.csv"
        record = {**self.record, "status": "complete", "speedup_x100": 150,
                  "current_performance": {"performance_valid": True, "candidate_matches": True, "speedup_x100": 150}}
        reporting.write_runs_csv([record, {**record, "performance_valid": False},
                                  {**record, "current_performance": {"candidate_matches": False}}], path)
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["display_speedup_source"] for r in rows],
                         ["current_candidate", "unavailable", "unavailable"])
        self.assertEqual([r["display_speedup_x100"] for r in rows], ["150", "", ""])

    def test_floor_fallback_to_verified_current_and_zero_history(self):
        path = self.root / "runs.csv"
        record = {**self.record, "current_performance": {"performance_valid": True,
                  "candidate_matches": True, "speedup_x100": 90}}
        reporting.write_runs_csv([record], path)
        with path.open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual((row["display_speedup_x100"], row["display_speedup_source"]), ("90", "current_candidate"))
        self.save({**self.sim, "reference_cycles": 1, "cfu_cycles": 101, "speedup_x100": 0})
        self.assertEqual(history.best_speedup({**self.record, "reference_cycles": 1}, self.fw)["speedup_x100"], 0)

    def test_each_snapshot_type_is_independent_and_legacy_is_not_complete(self):
        self.save({**self.sim, "implementation_design_type": "draft", "cfu_cycles": 100, "speedup_x100": 1000})
        for value, cfu in ((None, 200), ("bad", 250), (False, 400)):
            self.save({**self.sim, "implementation_design_type": value, "cfu_cycles": cfu, "speedup_x100": 100000 // cfu})
        legacy = self.save(self.sim)
        best = history.best_speedup({**self.record, "implementation_design_type": "draft"}, self.fw)
        self.assertEqual(best["speedup_x100"], 200)
        self.assertEqual(best["evidence"]["snapshot"], str(legacy))
        self.assertNotIn("implementation_design_type", best)
        self.save({**self.sim, "implementation_design_type": "complete", "cfu_cycles": 400, "speedup_x100": 250})
        self.assertEqual(history.best_speedup(self.record, self.fw)["speedup_x100"], 250)
        self.save({**self.sim, "implementation_design_type": "legacy", "cfu_cycles": 250, "speedup_x100": 400})
        self.assertEqual(history.best_speedup(self.record, self.fw)["speedup_x100"], 400)
        self.assertTrue(metrics.successful({"status": "complete", "implementation_design_type": "legacy"}))

    def test_draft_denominator_and_csv_diagnostic(self):
        base = {"status": "complete", "attempt_index": 1, "speedup_x100": 200,
                "performance_valid": True, "current_performance": {"performance_valid": True,
                "candidate_matches": True, "speedup_x100": 200}}
        draft = {**base, "implementation_design_type": "draft"}
        invalid = {**base, "implementation_design_type": None}
        complete = {**base, "implementation_design_type": "complete"}
        cell = metrics.cell_metrics([draft, invalid, complete], bootstrap_iters=4)
        self.assertEqual((cell["runs"], cell["complete_runs"], cell["speedup_samples"]), (3, 1, 1))
        self.assertAlmostEqual(cell["pass_at_1_budget"], 1 / 3)
        self.assertAlmostEqual(cell["pass_at_1_sample"], 1 / 3)
        path = self.root / "runs.csv"
        reporting.write_runs_csv([draft], path)
        with path.open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual((row["status"], row["speedup_x100"], row["observed_speedup_x100"],
                          row["display_speedup_source"], row["pass_at_1"]),
                         ("draft", "", "200", "draft_diagnostic", ""))

    def test_archive_reaggregation_does_not_promote_draft(self):
        archive = self.root / "runs/a/record.json"
        archive.parent.mkdir(parents=True)
        archive.write_text(json.dumps({**self.record, "status": "complete", "attempt_index": 1,
            "gates": {"sim": {"implementation_design_type": "draft"}}, "speedup_x100": 200}))
        original = archive.read_bytes()
        result = reporting.aggregate(self.root, {}, bootstrap_iters=2)
        self.assertEqual(archive.read_bytes(), original)
        with (self.root / "results/runs.csv").open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual((row["status"], row["implementation_design_type"], row["pass_at_1"]), ("draft", "draft", ""))
        with (self.root / "results/pass_at_k.csv").open() as f:
            cell = next(csv.DictReader(f))
        self.assertEqual((cell["runs"], cell["complete_runs"], cell["pass_at_1_budget"]), ("1", "0", "0.00"))
        self.assertTrue(result["records"])

    def test_collection_type_precedence_and_invalid_not_fallthrough(self):
        run = self.root / "run"
        run.mkdir()
        (run / "stdout.log").write_text("status=complete\n")
        def write(name, obj):
            path = self.fw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(obj))
        write("run.yaml", {"implementation_design_type": "complete"})
        write("10_validation/embench.yaml", {"implementation_design_type": "complete"})
        write("10_validation/sim.yaml", {**self.sim, "implementation_design_type": "draft"})
        write("12_review/decision.yaml", {"gates": {"sim": {"implementation_design_type": None}}})
        record = collection.collect_record("exp", {}, "name", run, self.fw)
        self.assertIsNone(record["implementation_design_type"])
        self.assertEqual(record["implementation_design_type_source"], "sim_gate")
        self.assertEqual(record["status"], "invalid_implementation_design_type")
        write("12_review/decision.yaml", {"gates": {"sim": {}}})
        record = collection.collect_record("exp", {}, "name", run, self.fw)
        self.assertEqual((record["implementation_design_type"], record["status"]), ("draft", "draft"))
        self.assertEqual(record["observed_speedup_x100"], 200)
        self.assertIsNone(record["speedup_x100"])
        self.assertIsNone(record["pass_at_1"])
        (run / "stdout.log").write_text("status=failed\nstopped_at=hw_build\n")
        write("12_review/decision.yaml", {"gates": {"sim": {"implementation_design_type": "draft"},
            "implementation_hw": {"ok": True, "design_type": "complete"}}})
        record = collection.collect_record("exp", {}, "name", run, self.fw)
        self.assertEqual((record["implementation_design_type"], record["implementation_design_type_source"]),
                         ("complete", "implementation_hw"))
        write("12_review/decision.yaml", {"gates": {"sim": {"implementation_design_type": "draft"}}})
        write("run.yaml", {})
        record = collection.collect_record("exp", {}, "name", run, self.fw)
        self.assertNotIn("implementation_design_type", record)

    def test_equal_rounded_speedup_uses_exact_ratio(self):
        self.save({**self.sim, "cfu_cycles": 498, "speedup_x100": 200})
        self.save({**self.sim, "cfu_cycles": 499, "speedup_x100": 200})
        self.assertEqual(history.best_speedup(self.record, self.fw)["cfu_cycles"], 498)
