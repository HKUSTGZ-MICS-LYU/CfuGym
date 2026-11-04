"""Offline experiment ratio/resource fixtures; no API or EDA tools are invoked."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from agents.experiments.harness import collection, layers, metrics, parsing, reporting


def cost_fixture() -> tuple[dict, dict]:
    ppa = {
        "backend": "vivado", "status": "passed", "phase": "post_route", "part": "xc7a100t-csg324-1",
        "soc": {"area": {"lut": 100, "ff": 200, "bram_36k": 0.5, "dsp": 2}},
        "cfu": {"area": {"lut": 10, "ff": 20, "bram_36k": 0, "dsp": 0}},
        "evidence": {"input_sha256": "cost-input", "evaluation_id": "eval-1", "rtl_sha256": "rtl",
                     "summary_path": "11_cost/evaluations/eval-1/vivado/summary.json"},
    }
    gate = {"backend": "vivado", "ok": True, "skipped": False, "cancelled": False,
            "input_sha256": "cost-input", "evaluation_id": "eval-1", "ppa": copy.deepcopy(ppa)}
    cost = {"status": "passed", "commands": [
        {"command": "python3 skills/cfu-designer/scripts/yosys_cost_report.py", "returncode": 0,
         "evaluation_id": "eval-1", "parsed": {"cost": {"cells": 9999}}},
        {"command": "python3 skills/cfu-designer/scripts/vivado_ppa_report.py", "returncode": 0,
         "evaluation_id": "eval-1", "parsed": {"ppa": copy.deepcopy(ppa)}},
    ]}
    return cost, {"cost": gate}


def successful(speedup: object = 105, **updates) -> dict:
    return {"status": "complete", "performance_valid": True, "speedup_x100": speedup,
            "attempt_index": 1, "measurement_protocol": "cold_single_invocation_v1", **updates}


def write_fixture(root: Path, *, conformance: dict | None = None) -> tuple[Path, Path, dict]:
    run = root / "runs" / "r1"
    framework = root / "framework"
    run.mkdir(parents=True)
    framework.mkdir(parents=True)
    cost, gates = cost_fixture()
    proof = {"input_sha256": "sim-input"}
    gates["sim"] = {"ok": True, "terminal_reason": "passed", "performance_valid": True,
                    "correctness_ok": True, "timing_ok": True, "reference_cycles": 1050,
                    "cfu_cycles": 1000, "speedup_x100": 105, "performance_provenance": proof}
    manifest = {"status": "complete", "visibility": {"level": "extended"}}
    if conformance is not None:
        manifest["conformance"] = conformance
    (framework / "run.yaml").write_text(yaml.safe_dump(manifest))
    (framework / "11_cost").mkdir()
    (framework / "11_cost" / "result.yaml").write_text(yaml.safe_dump(cost))
    report = ["# Fixture", "Status: complete", "## Gates"]
    report += [f"- {key}: {value!r}" for key, value in gates.items()]
    report += ["## Validation Ladder", "- verify submissions: 1/5", "- repairs used: 0/4 (iteration=0)",
               "## Candidate", f"- current_performance: {{'candidate_provenance': {proof!r}, 'candidate_matches': True, 'performance_valid': True}}",
               "## Ablation", "- `--no-llvm-ir`", "## Hardware Cost",
               "- current validated FPGA PPA; implementation status: passed", ""]
    (framework / "report.md").write_text("\n".join(report))
    (run / "stdout.log").write_text("status=complete\nvisibility=extended\nsummary:\n  speedup = 1.05x\n")
    job = {"model": "fixture", "benchmark": "crc32", "repeat": 1, "max_iters": 5,
           "layer": "l3-devconf", "visibility": "extended", "ablation_flags": ["--no-llvm-ir"],
           "ablation_scheme": layers.ABLATION_SCHEME, "conformance_mode": "dev",
           "conformance_promote": False, "max_conformance_runs": 2}
    return run, framework, job


class SpeedRatioTest(unittest.TestCase):
    def test_ratio_statistics_exclude_failed_infra_invalid_and_nonfinite(self):
        records = [successful(105), successful(250), successful(900, performance_valid=False),
                   successful(900, status="failed"), successful(900, infra={"infra": True})]
        records += [successful(value) for value in (None, True, False, 0, -5, "200", float("nan"), float("inf"))]
        row = metrics.cell_metrics(records, bootstrap_iters=20)
        self.assertEqual(row["speedup_samples"], 2)
        self.assertEqual(row["speedup_ratio_x_min"], 1.05)
        self.assertEqual(row["speedup_ratio_x_max"], 2.5)
        self.assertAlmostEqual(row["speedup_ratio_x_mean"], 1.775)
        self.assertAlmostEqual(row["speedup_ratio_x_median"], 1.775)
        self.assertAlmostEqual(row["speedup_x100_median"], 177.5)

    def test_empty_and_single_sample(self):
        for records, expected in (([], None), ([successful()], 1.05)):
            with self.subTest(records=records):
                row = metrics.cell_metrics(records, bootstrap_iters=20)
                for stat in metrics.STAT_NAMES:
                    self.assertEqual(row[f"speedup_ratio_x_{stat}"], expected)
                self.assertEqual(row["speedup_samples"], len(records))

    def test_protocol_mismatch_never_pools_ratios(self):
        row = metrics.cell_metrics([successful(), successful(200, measurement_protocol="legacy")], bootstrap_iters=20)
        self.assertEqual(row["speedup_valid_runs"], 2)
        self.assertEqual(row["speedup_samples"], 0)
        self.assertTrue(row["speedup_warning"])
        for stat in metrics.STAT_NAMES:
            self.assertIsNone(row[f"speedup_ratio_x_{stat}"])

    def test_all_complete_protocols_must_agree_even_when_speedup_missing(self):
        row = metrics.cell_metrics([successful(), successful(None, performance_valid=False,
                                                            measurement_protocol="future_v2")], bootstrap_iters=20)
        self.assertEqual(row["speedup_samples"], 0)
        self.assertEqual(row["speedup_valid_runs"], 1)
        self.assertIn("Speedup median unavailable", row["speedup_warning"])


class VivadoResourceTest(unittest.TestCase):
    def test_real_cost_artifact_shape_and_scope_units(self):
        cost, gates = cost_fixture()
        warnings = []
        row = collection.collect_resources(cost, gates, {}, warnings)
        self.assertTrue(row["resource_valid"])
        self.assertEqual(row["resource_status"], "valid")
        self.assertEqual(row["cost_backend"], "vivado")
        self.assertEqual(row["vivado_soc_lut"], 100)
        self.assertEqual(row["vivado_cfu_lut"], 10)
        self.assertEqual(row["vivado_soc_bram_36k"], 0.5)
        self.assertEqual(row["vivado_cfu_dsp"], 0)
        self.assertEqual(warnings, [])

    def test_gate_only_and_command_surface_fallback(self):
        cost, gates = cost_fixture()
        self.assertTrue(collection.collect_resources({}, gates, {}, [])["resource_valid"])
        gates["cost"].pop("ppa")
        self.assertTrue(collection.collect_resources(cost, gates, {}, [])["resource_valid"])

    def test_failed_stale_unbound_or_mismatched_evidence_is_rejected(self):
        for variant in ("cost_failed", "gate_failed", "stale", "cancelled", "skipped", "wrong_input",
                        "wrong_eval", "missing_eval", "command_failed", "command_eval", "history",
                        "candidate_mismatch", "wrong_phase", "wrong_backend", "command_disagree", "stale_run"):
            with self.subTest(variant=variant):
                cost, gates = cost_fixture()
                gate = gates["cost"]
                report = {}
                if variant == "cost_failed":
                    cost["status"] = "failed"
                elif variant == "gate_failed":
                    gate["ok"] = False
                elif variant in {"stale", "cancelled", "skipped"}:
                    gate[variant] = True
                elif variant == "wrong_input":
                    gate["input_sha256"] = "other-candidate"
                elif variant == "wrong_eval":
                    gate["evaluation_id"] = "eval-other"
                elif variant == "missing_eval":
                    gate.pop("evaluation_id")
                elif variant == "command_failed":
                    cost["commands"][-1]["returncode"] = 1
                elif variant == "command_eval":
                    cost["commands"][-1]["evaluation_id"] = "eval-other"
                elif variant == "history":
                    report["ppa_current_valid"] = False
                elif variant == "candidate_mismatch":
                    report["current_performance"] = {"candidate_provenance": {}, "candidate_matches": False}
                elif variant == "wrong_phase":
                    gate["ppa"]["phase"] = "post_synth"
                elif variant == "wrong_backend":
                    gate["ppa"]["backend"] = "yosys"
                elif variant == "command_disagree":
                    cost["commands"][-1]["parsed"]["ppa"]["soc"]["area"]["lut"] = 999
                warnings = []
                row = collection.collect_resources(cost, gates, report, warnings, stale=variant == "stale_run")
                self.assertFalse(row["resource_valid"])
                self.assertTrue(row["resource_warning"])
                self.assertTrue(warnings)
                self.assertTrue(all(row[field] is None for field in metrics.RESOURCE_FIELDS))

    def test_yosys_is_not_fpga_resources(self):
        warnings = []
        row = collection.collect_resources({"status": "passed", "commands": [
            {"parsed": {"cost": {"cells": 1234, "wire_bits": 5000}}, "returncode": 0}]}, {}, {}, warnings)
        self.assertEqual(warnings, [])
        self.assertEqual(row["cost_backend"], "yosys")
        self.assertEqual(row["resource_status"], "unsupported")
        self.assertFalse(row["resource_valid"])
        self.assertTrue(all(row[field] is None for field in metrics.RESOURCE_FIELDS))
        self.assertIn("not FPGA", row["resource_warning"])

    def test_partial_and_missing_resources_never_zero_filled(self):
        cost, gates = cost_fixture()
        gates["cost"]["ppa"]["cfu"]["area"]["ff"] = None
        gates["cost"]["ppa"]["soc"]["area"]["lut"] = float("nan")
        cost["commands"][-1]["parsed"]["ppa"] = copy.deepcopy(gates["cost"]["ppa"])
        row = collection.collect_resources(cost, gates, {}, [])
        self.assertTrue(row["resource_valid"])
        self.assertEqual(row["resource_status"], "partial")
        self.assertIsNone(row["vivado_cfu_ff"])
        self.assertIsNone(row["vivado_soc_lut"])
        self.assertEqual(row["vivado_cfu_dsp"], 0)
        row = collection.collect_resources({"status": "planned"}, {"cost": {"backend": "vivado"}}, {}, [])
        self.assertFalse(row["resource_valid"])
        self.assertTrue(row["resource_warning"])

    def test_resource_statistics_and_sample_counts(self):
        cost, gates = cost_fixture()
        resource = collection.collect_resources(cost, gates, {}, [])
        rows = [successful(**resource), successful(**{**resource, "vivado_soc_lut": 300, "vivado_cfu_ff": None}),
                successful(**{**resource, "vivado_soc_lut": 900}, status="failed"),
                successful(**{**resource, "vivado_soc_lut": 900}, infra={"infra": True}),
                successful(**{**resource, "resource_valid": False, "vivado_soc_lut": 900})]
        row = metrics.cell_metrics(rows, bootstrap_iters=20)
        self.assertEqual(row["resource_samples"], 2)
        self.assertEqual(row["vivado_soc_lut_samples"], 2)
        self.assertEqual(row["vivado_cfu_ff_samples"], 1)
        self.assertEqual([row[f"vivado_soc_lut_{stat}"] for stat in metrics.STAT_NAMES], [100, 300, 200, 200])
        self.assertEqual(row["vivado_cfu_dsp_mean"], 0)
        empty = metrics.cell_metrics([], bootstrap_iters=20)
        self.assertEqual(empty["resource_samples"], 0)
        self.assertIsNone(empty["vivado_soc_lut_mean"])


class CollectionAndReportingTest(unittest.TestCase):
    def test_record_csv_summary_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, framework, job = write_fixture(root, conformance={"mode": "dev", "promote": False, "max_runs": 2})
            row = collection.collect_record("fixture", job, "r1", run, framework, {"exit_code": 0})
            self.assertEqual(row["conformance_mode"], "dev")
            self.assertEqual(row["conformance_mode_observed"], "dev")
            self.assertIs(row["conformance_promote_observed"], False)
            self.assertEqual(row["max_conformance_runs_observed"], 2)
            self.assertTrue(row["performance_valid"])
            self.assertEqual(row["speedup_ratio_x"], 1.05)
            self.assertTrue(row["resource_valid"])
            self.assertEqual(row["vivado_cfu_lut"], 10)
            collection.save_record(row, run)
            reporting.aggregate(root, {"ablation_scheme": layers.ABLATION_SCHEME}, bootstrap_iters=20)
            with (root / "results" / "runs.csv").open() as handle:
                run_csv = next(csv.DictReader(handle))
            self.assertEqual(run_csv["speedup_x100"], "105")
            self.assertEqual(run_csv["speedup_ratio_x"], "1.05")
            self.assertEqual(run_csv["vivado_soc_bram_36k"], "0.50")
            self.assertEqual(run_csv["conformance_mode"], "dev")
            self.assertEqual(json.loads(run_csv["resource_provenance"])["evaluation_id"], "eval-1")
            with (root / "results" / "pass_at_k.csv").open() as handle:
                cell_csv = next(csv.DictReader(handle))
            self.assertEqual(cell_csv["speedup_ratio_x_mean"], "1.05")
            self.assertEqual(cell_csv["speedup_samples"], "1")
            self.assertEqual(cell_csv["vivado_soc_lut_mean"], "100")
            self.assertEqual(cell_csv["vivado_soc_lut_samples"], "1")
            summary = (root / "results" / "summary.md").read_text()
            self.assertIn("ratio×", summary)
            self.assertIn("min (ratio×) | max (ratio×) | mean (ratio×) | median (ratio×)", summary)
            self.assertIn("Vivado resources", summary)
            self.assertIn("BRAM36 mean (n)", summary)
            self.assertIn("| soc | 1 | 100 (1)", summary)
            self.assertIn("| cfu | 1 | 10 (1)", summary)

    def test_legacy_metadata_does_not_acquire_v3_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, framework, job = write_fixture(Path(tmp))
            for key in ("conformance_mode", "conformance_promote", "max_conformance_runs", "ablation_scheme"):
                job.pop(key)
            job.update(layer="l2-vis", ablation_flags=["--no-conformance", "--no-llvm-ir"])
            row = collection.collect_record("old", job, "r1", run, framework)
            for key in ("conformance_mode", "conformance_promote", "max_conformance_runs"):
                self.assertEqual(row[key], "unknown")
                self.assertEqual(row[f"{key}_observed"], "unknown")

    def test_scheduled_and_observed_remain_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, framework, job = write_fixture(Path(tmp), conformance={"mode": "gate", "promote": True, "max_runs": 7})
            row = collection.collect_record("fixture", job, "r1", run, framework, {"exit_code": 0})
            self.assertEqual(row["conformance_mode"], "dev")
            self.assertEqual(row["conformance_mode_observed"], "gate")
            self.assertEqual(row["max_conformance_runs"], 2)
            self.assertEqual(row["max_conformance_runs_observed"], 7)
            self.assertIn("conformance_mode mismatch", row["parse_warnings"])

    def test_explicit_cli_and_report_conformance_parsing(self):
        cli = parsing.parse_cli_stdout("conformance_mode=off\nconformance_promote=False\nmax_conformance_runs=2\n")
        self.assertEqual(cli["conformance_mode"], "off")
        self.assertIs(cli["conformance_promote"], False)
        self.assertEqual(cli["max_conformance_runs"], 2)
        report = parsing.parse_report("## Conformance\n- mode: dev\n- authority: EVIDENCE only — `sim` owns the correctness verdict\n"
                                      "- max_conformance_runs: 2\n## Hardware Cost\n- historical FPGA evidence; does not certify the current candidate\n", [])
        self.assertEqual(report["conformance_mode"], "dev")
        self.assertIs(report["conformance_promote"], False)
        self.assertIs(report["ppa_current_valid"], False)
        self.assertEqual(report["max_conformance_runs"], 2)

    def test_v3_v2_v1_archived_ladder_semantics(self):
        cell = metrics.cell_metrics([], bootstrap_iters=20)
        v3 = "\n".join(reporting._ablation_summary_lines(
            [(name, "m", "b", cell) for name in layers.ABLATION_LADDER],
            {"ablation_scheme": layers.ABLATION_SCHEME,
             "layers": {"l3-devconf": {"adds": "stored wording", "max_conformance_runs": 7}}}))
        self.assertIn("dev conformance -> LLVM IR", v3)
        self.assertIn("L0-2 use conformance off, L3-4 use dev", v3)
        self.assertIn("no promotion", v3)
        self.assertIn("stored wording | extended | dev | False | 7", v3)
        v2 = "\n".join(reporting._ablation_summary_lines(
            [(name, "m", "b", cell) for name in layers.V2_ABLATION_LAYERS],
            {"ablation_scheme": layers.V2_ABLATION_SCHEME,
             "layers": {"l2-vis": {"adds": "archived visibility", "flags": ["--no-conformance"]}}}))
        self.assertIn("conformance is OFF throughout", v2)
        self.assertNotIn("dev conformance ->", v2)
        self.assertIn("archived visibility", v2)
        self.assertIn("| off | unknown | unknown |", v2)
        v1 = "\n".join(reporting._ablation_summary_lines([("l2-conf", "m", "b", cell)]))
        self.assertIn("legacy required YAML-oracle gate", v1)


if __name__ == "__main__":
    unittest.main()
