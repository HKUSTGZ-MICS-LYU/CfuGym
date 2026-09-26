"""Tests for the pass@1/3/5 experiment matrix harness (agents/experiments/run_matrix.py).

Everything here is free of model/API calls: parsing, classification, metrics,
aggregation and command construction against synthetic run artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from agents.cfu_design_agent.artifacts import _safe_run_id
from agents.experiments import run_matrix as rm


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


REPORT_PASS = """# CFU Design Agent Report
Task: Accelerate the embench-iot benchmark crc32 with a CFU.
Status: complete
Run root: agents/generated/runs/x

## Gates
- sim: {'ok': True, 'skipped': False, 'round': 0, 'reference_cycles': 15503563, 'cfu_cycles': 14636223, 'speedup_x100': 105, 'attribution': '', 'repair_decision': {}, 'next': 'estimate_cost'}
- cost: {'ok': True, 'skipped': False, 'reason': '', 'round': 0, 'next': 'review_and_route'}

## Validation Ladder
- isa: pass
- sim: pass
- repairs used: 2/4 (iteration=2)
"""

RUN_YAML_PASS = """status: complete
llm_request: {config: glm-5.2, model: GLM-5.2, base_url: 'https://gw/v1'}
visibility: {level: standard}
profile_cache: {status: hit, key: abc}
"""

EMBENCH_PASS = """benchmark: crc32
verification: bench_verify
reference_cycles: 15503563
cfu_cycles: 14636223
speedup_x100: 105
min_speedup_x100: 105
attribution: ''
status: passed
"""

STDOUT_PASS = """status=complete
visibility=standard
report=agents/generated/runs/x/report.md
graph_mermaid=a.mmd
graph_ascii=a.txt

summary:
  cycles_before = 15503563
  cycles_after  = 14636223
  speedup       = 1.05x
  yosys_cost    = top=AgentCfu cells=2750 wire_bits=4402 memories=0 memory_bits=0

token_usage:
  calls       = 42
  input_tokens= 100
  output_tokens= 200
  total_tokens= 300
"""


class NamingTest(unittest.TestCase):
    def test_run_name_is_explicit_and_id_safe(self) -> None:
        name = rm.make_run_name("pk5-20260212", "glm-5.2", "matmult-int", 3, 5)
        self.assertEqual(name, "pk5-20260212-glm52-matmultint-r03-mi5")
        self.assertTrue(_safe_run_id(name))
        self.assertEqual(
            rm.make_run_name("pk5-20260212", "glm-5.2", "matmult-int", 3, 5, infra_retry=1),
            "pk5-20260212-glm52-matmultint-r03-mi5-infra1",
        )

    def test_run_name_is_unique_per_coordinate(self) -> None:
        names = {
            rm.make_run_name("e", model, bench, repeat, 5)
            for model in ("deepseek-pro", "glm-5.2", "gpt-5.6-terra")
            for bench in ("crc32", "matmult-int", "md5sum")
            for repeat in (1, 2)
        }
        self.assertEqual(len(names), 18)

    def test_slugs(self) -> None:
        self.assertEqual(rm.model_slug("deepseek-pro"), "deepseekpro")
        self.assertEqual(rm.model_slug("mystery-model 9"), "mystery-model-9")
        self.assertEqual(rm.benchmark_slug("matmult-int"), "matmultint")


class MatrixTest(unittest.TestCase):
    def test_expand_matrix_shape(self) -> None:
        jobs = rm.expand_matrix(("a", "b"), ("x", "y", "z"), 3)
        self.assertEqual(len(jobs), 18)
        self.assertEqual({j["repeat"] for j in jobs}, {1, 2, 3})

    def test_schedule_is_repeat_major_round_robin(self) -> None:
        jobs = rm.schedule_jobs(rm.expand_matrix(("a", "b", "c"), ("x", "y"), 3), seed=7)
        self.assertEqual(len(jobs), 18)
        # Blocks of one full cell sweep per repeat: replicates land in different
        # time slots instead of one cell hogging a whole period.
        repeats = [j["repeat"] for j in jobs]
        self.assertEqual(repeats, [1] * 6 + [2] * 6 + [3] * 6)
        for repeat in (1, 2, 3):
            sweep = {(j["model"], j["benchmark"]) for j in jobs if j["repeat"] == repeat}
            self.assertEqual(len(sweep), 6)

    def test_schedule_is_deterministic(self) -> None:
        base = rm.expand_matrix(("a", "b", "c"), ("x", "y"), 3)
        self.assertEqual(rm.schedule_jobs(base, seed=7), rm.schedule_jobs(base, seed=7))

    def test_filter_jobs_requires_all_substrings(self) -> None:
        jobs = rm.schedule_jobs(rm.expand_matrix(("glm-5.2", "kimi-k3"), ("crc32", "md5sum"), 3), seed=1)
        for job in jobs:
            job["exp_id"] = "pk5"
            job["max_iters"] = 5
        kept = rm.filter_jobs(jobs, ["glm52", "crc32", "r02"])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["repeat"], 2)


class ConfigTest(unittest.TestCase):
    def test_materialize_configs_forces_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            cfg_dir = workdir / "agents" / "llm_configs"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "glm-5.2.json").write_text(json.dumps({
                "name": "glm-5.2", "model": "GLM-5.2", "base_url": "https://gw/v1",
                "api_mode": "chat", "thinking": True, "reasoning_effort": "high",
            }))
            out = rm.materialize_configs("pk5-test", ["glm-5.2"], 0.7, workdir)
            self.assertEqual(out["glm-5.2"]["config"], "pk5-test-glm-5.2")
            generated = json.loads((cfg_dir / "pk5-test-glm-5.2.json").read_text())
            self.assertEqual(generated["temperature"], 0.7)
            self.assertEqual(generated["model"], "GLM-5.2")
            self.assertEqual(generated["name"], "pk5-test-glm-5.2")
            # The base config is never modified.
            self.assertNotIn("temperature", json.loads((cfg_dir / "glm-5.2.json").read_text()))
            self.assertTrue(out["glm-5.2"]["sha256"])

    def test_missing_base_config_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                rm.materialize_configs("e", ["nope"], 0.7, Path(tmp))


class SbtLockTest(unittest.TestCase):
    def test_shim_serializes_and_resolves_real_sbt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shim = rm.write_sbt_shim(Path(tmp) / "bin")
            text = shim.read_text()
            self.assertIn("exec flock -x -w", text)
            self.assertIn("CFU_EXP_SBT_LOCK_FILE", text)
            self.assertIn("CFU_EXP_SBT_SHIM_ACTIVE", text)  # recursion guard
            self.assertIn('ENTRY="${REMAINING%%:*}"', text)  # entry-wise PATH filter
            self.assertIn('"$SHIM_DIR"/*', text)  # refuse to resolve back to the shim
            self.assertIn('REAL_SBT="$(command -v sbt || true)"', text)
            self.assertTrue(shim.stat().st_mode & stat.S_IXUSR)

    @unittest.skipUnless(shutil.which("flock") and shutil.which("bash"), "flock/bash required")
    def test_shim_survives_adjacent_duplicate_path_entries(self) -> None:
        """Regression: <shim>:<shim>:PATH used to defeat ${PATH//...} cleaning.

        The harness prepends the shim dir to PATH and the conda wrapper prepends
        it again; with the old pattern substitution the second adjacent copy
        survived, ``command -v sbt`` resolved back to the shim, and the
        recursion guard aborted every sbt/sim call (exit 97).
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shim = rm.write_sbt_shim(root / "bin")
            fake_dir = root / "fake"
            fake_dir.mkdir()
            fake = fake_dir / "sbt"
            fake.write_text("#!/bin/bash\necho REAL-SBT-RAN\n")
            fake.chmod(0o755)
            env = {
                "PATH": os.pathsep.join(
                    [str(root / "bin"), str(root / "bin"), str(fake_dir), os.environ.get("PATH", "")]
                ),
                "CFU_EXP_SBT_LOCK_FILE": str(root / "sbt.lock"),
            }
            result = subprocess.run(
                ["bash", str(shim), "--version"], env=env, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("REAL-SBT-RAN", result.stdout)

    def test_launcher_reprepends_shim_after_conda_activation(self) -> None:
        # conda run activation puts <env>/bin first on PATH and would shadow the
        # shim; the wrapper must re-prepend it *inside* the environment.
        argv = rm.launcher_argv(["python3", "-B", "x.py"], "mico_env", Path("/exp/bin"))
        self.assertEqual(argv[:4], ["conda", "run", "-n", "mico_env"])
        self.assertIn("--no-capture-output", argv)
        self.assertIn(rm.SHIM_PATH_WRAPPER, argv)
        wrapper_at = argv.index(rm.SHIM_PATH_WRAPPER)
        self.assertEqual(argv[wrapper_at + 1:wrapper_at + 3], ["cfu-exp-sbt-shim", "/exp/bin"])
        self.assertEqual(argv[wrapper_at + 3:], ["python3", "-B", "x.py"])

    def test_launcher_variants(self) -> None:
        self.assertEqual(rm.launcher_argv(["python3"], "mico_env", None)[:5],
                         ["conda", "run", "-n", "mico_env", "--no-capture-output"])
        self.assertEqual(rm.launcher_argv(["python3"], "", None), ["python3"])
        self.assertEqual(rm.launcher_argv(["python3"], "", Path("/exp/bin")), ["python3"])

    def test_lock_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = rm.sbt_lock_env(Path(tmp) / "locks" / "sbt.lock", 60)
            self.assertEqual(env["CFU_EXP_SBT_LOCK_TIMEOUT"], "60")
            self.assertTrue(Path(env["CFU_EXP_SBT_LOCK_FILE"]).parent.exists())


class ParseTest(unittest.TestCase):
    def test_parse_cli_stdout(self) -> None:
        out = rm.parse_cli_stdout(STDOUT_PASS)
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["report_path"], "agents/generated/runs/x/report.md")
        self.assertEqual(out["speedup_x100"], 105)
        self.assertEqual(out["yosys_cost"]["cells"], 2750)
        self.assertEqual(out["token_calls"], 42)
        self.assertEqual(out["token_total"], 300)

    def test_parse_cli_stdout_handles_na_speedup(self) -> None:
        out = rm.parse_cli_stdout("status=failed\nsummary:\n  speedup       = n/a (skipped)\n")
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("speedup_x100", out)

    def test_parse_report(self) -> None:
        warnings: list[str] = []
        out = rm.parse_report(REPORT_PASS, warnings)
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["repairs_used"], 2)
        self.assertEqual(out["max_repairs"], 4)
        self.assertTrue(out["gates"]["sim"]["ok"])
        self.assertEqual(out["gates"]["sim"]["speedup_x100"], 105)
        self.assertEqual(warnings, [])


def _failure_record(rung: str, gate: dict, **extra) -> dict:
    record = {
        "status": "failed",
        "stopped_at": rung,
        "repairs_used": 4,
        "max_repairs": 4,
        "max_iters": 5,
        "gates": {rung: gate},
    }
    record.update(extra)
    return record


class ClassifyTest(unittest.TestCase):
    def test_complete_is_not_a_failure(self) -> None:
        self.assertIsNone(rm.classify_failure({"status": "complete", "gates": {}}))

    def test_sim_subclasses(self) -> None:
        cases = [
            (["simulation timed out after 3600s"], {}, "sim_timeout"),
            (["the CFU build is not meaningfully faster than the reference"], {}, "sim_speedup_floor"),
            (["Embench verify failed for crc32 (wrong-result failure)"], {}, "sim_correctness"),
            (["no design change was applied; the cfu build equals the reference"], {}, "sim_no_change"),
        ]
        for errors, extra, expected in cases:
            record = _failure_record("sim", {"ok": False, "errors": errors, **extra})
            result = rm.classify_failure(record)
            self.assertEqual(result["failure_class"], expected, errors)
            self.assertEqual(result["failure_rung"], "sim")
            self.assertTrue(result["budget_exhausted"])

    def test_speedup_floor_from_measured_values(self) -> None:
        record = _failure_record(
            "sim", {"ok": False, "errors": ["too slow"]},
            speedup_x100=101, min_speedup_x100=105,
        )
        self.assertEqual(rm.classify_failure(record)["failure_class"], "sim_speedup_floor")

    def test_hw_conformance_keeps_oracle_vs_design(self) -> None:
        record = _failure_record(
            "hw_conformance", {"ok": False, "oracle_failure": True, "errors": ["verdict fail"]}
        )
        result = rm.classify_failure(record)
        self.assertEqual(result["failure_class"], "hw_conformance")
        self.assertIn("oracle:", result["failure_detail"])
        record = _failure_record(
            "hw_conformance", {"ok": False, "oracle_failure": False, "errors": ["commands=3/5"]}
        )
        self.assertIn("design:", rm.classify_failure(record)["failure_detail"])

    def test_other_rungs(self) -> None:
        cases = [
            ("prepare_benchmark", {"ok": False, "errors": ["denied"]}, "input_or_policy"),
            ("isa", {"ok": False, "problems": ["bad"]}, "isa_contract"),
            ("implementation_hw", {"ok": False, "errors": ["no EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN deliverable"]}, "no_deliverable"),
            ("hw_build", {"ok": False, "errors": ["scala compile error"]}, "hw_build"),
            ("sw_build", {"ok": False, "errors": ["make error"]}, "sw_build"),
            ("cost", {"ok": False, "errors": ["yosys failed"]}, "cost"),
        ]
        for rung, gate, expected in cases:
            result = rm.classify_failure(_failure_record(rung, gate))
            self.assertEqual(result["failure_class"], expected, rung)

    def test_blocked_run_maps_to_contract_or_input(self) -> None:
        record = {"status": "blocked", "stopped_at": "", "gates": {"isa": {"ok": False, "problems": ["x"]}}}
        self.assertEqual(rm.classify_failure(record)["failure_class"], "isa_contract")
        record = {"status": "blocked", "stopped_at": "", "gates": {}}
        self.assertEqual(rm.classify_failure(record)["failure_class"], "spec_invalid")


class InfraTest(unittest.TestCase):
    def test_timeout_is_infra(self) -> None:
        result = rm.detect_infra("status=failed", "", True, None)
        self.assertTrue(result["infra"])
        self.assertEqual(result["infra_kind"], "infra_timeout")

    def test_api_transport_is_infra(self) -> None:
        result = rm.detect_infra("api-error: connection reset", "", False, 1)
        self.assertTrue(result["infra"])
        self.assertEqual(result["infra_kind"], "infra_error")

    def test_framework_crash_is_infra(self) -> None:
        result = rm.detect_infra("Traceback (most recent call last):\n  ...", "", False, 1)
        self.assertTrue(result["infra"])

    def test_ordinary_model_failure_is_not_infra(self) -> None:
        result = rm.detect_infra("status=failed\nstopped_at=hw_build\n", "scala boom", False, 1)
        self.assertFalse(result["infra"])


class CollectTest(unittest.TestCase):
    def _fixture(self, tmp: Path, stdout: str, report: str, run_yaml: str, embench: str):
        run_dir = _write(tmp / "harness" / "run_name" / "stdout.log", stdout).parent
        _write(run_dir / "stderr.log", "")
        fw = tmp / "agents" / "generated" / "runs" / "run_name"
        _write(fw / "report.md", report)
        _write(fw / "run.yaml", run_yaml)
        _write(fw / "10_validation" / "embench.yaml", embench)
        return run_dir, fw

    def test_collect_passing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = self._fixture(tmp, STDOUT_PASS, REPORT_PASS, RUN_YAML_PASS, EMBENCH_PASS)
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 3, "max_iters": 5, "infra_retry": 0}
            record = rm.collect_record(
                "pk5-test", job, "pk5-test-glm52-crc32-r03-mi5", run_dir, fw,
                {"exit_code": 0, "wall_seconds": 12.5, "started_at": "t0", "finished_at": "t1"},
            )
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["repairs_used"], 2)
            self.assertEqual(record["attempt_index"], 3)  # repairs_used + 1
            self.assertEqual(record["speedup_x100"], 105)
            self.assertEqual(record["token_total"], 300)
            self.assertEqual(record["wall_seconds"], 12.5)
            self.assertEqual(record["model"], "GLM-5.2")  # from run.yaml llm_request
            self.assertEqual(record["pass_at_1"], 0)   # needed 2 repairs
            self.assertEqual(record["pass_at_3"], 1)
            self.assertEqual(record["pass_at_5"], 1)
            self.assertIsNone(record["failure"])
            self.assertEqual(record["parse_warnings"], "")
            self.assertFalse(record["infra"]["infra"])

    def test_collect_failing_run_writes_flags(self) -> None:
        report = REPORT_PASS.replace("Status: complete", "Status: failed").replace(
            "- sim: {'ok': True", "- sim: {'ok': False"
        ).replace("repairs used: 2/4 (iteration=2)", "repairs used: 4/4 (iteration=4)")
        stdout = STDOUT_PASS.replace("status=complete", "status=failed").replace("stopped", "stopped")
        stdout = "status=failed\nstopped_at=sim\n" + stdout.split("\n", 1)[1]
        embench = EMBENCH_PASS.replace("status: passed", "status: failed").replace(
            "attribution: ''", "attribution: 'Embench verify failed for crc32'"
        ).replace("speedup_x100: 105", "speedup_x100: 105")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = self._fixture(tmp, stdout, report, RUN_YAML_PASS.replace("complete", "failed"), embench)
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1, "max_iters": 5, "infra_retry": 0}
            record = rm.collect_record("pk5-test", job, "run_name", run_dir, fw, {"exit_code": 1, "wall_seconds": 5})
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["failure_class"], "sim_correctness")
            self.assertEqual(record["attempt_index"], 5)
            self.assertIsNone(record["pass_at_5"])  # failures stay empty in CSV
            path = rm.write_failure_json(record, run_dir)
            self.assertTrue(path and path.exists())
            payload = json.loads(path.read_text())
            self.assertEqual(payload["terminal"]["failure_class"], "sim_correctness")
            self.assertEqual(payload["terminal"]["stopped_at"], "sim")


class MetricsTest(unittest.TestCase):
    @staticmethod
    def _records(spec):
        return [{"status": "complete", "attempt_index": t} if t else {"status": "failed"} for t in spec]

    def test_budget_estimator(self) -> None:
        records = self._records([1, 3, 5, 0, 0])  # 0 = failed within budget
        self.assertAlmostEqual(rm.budget_pass_at_k(records, 1), 0.2)
        self.assertAlmostEqual(rm.budget_pass_at_k(records, 3), 0.4)
        self.assertAlmostEqual(rm.budget_pass_at_k(records, 5), 0.6)

    def test_sample_estimator(self) -> None:
        records = self._records([1, 0, 0, 0, 0])  # n=5, c=1
        self.assertAlmostEqual(rm.sample_pass_at_k(records, 1), 0.2)
        self.assertAlmostEqual(rm.sample_pass_at_k(records, 3), 1 - 4 / 10)
        self.assertAlmostEqual(rm.sample_pass_at_k(records, 5), 1.0)
        self.assertIsNone(rm.sample_pass_at_k(records, 6))  # k > n

    def test_bootstrap_ci_deterministic_and_bracketed(self) -> None:
        records = self._records([1, 1, 0, 0, 0])
        lo1, hi1 = rm.bootstrap_budget_ci(records, 5, iters=2000)
        lo2, hi2 = rm.bootstrap_budget_ci(records, 5, iters=2000)
        self.assertEqual((lo1, hi1), (lo2, hi2))
        self.assertLessEqual(lo1, 0.4)
        self.assertGreaterEqual(hi1, 0.4)

    def test_cell_metrics(self) -> None:
        records = self._records([1, 3, 0])
        records[0].update({"token_total": 100, "wall_seconds": 3600, "speedup_x100": 150, "yosys_cells": 2000})
        records[1].update({"token_total": 300, "wall_seconds": 7200, "speedup_x100": 250, "yosys_cells": 3000})
        records[2]["failure"] = {"failure_class": "hw_build"}
        metrics = rm.cell_metrics(records, bootstrap_iters=500)
        self.assertEqual(metrics["runs"], 3)
        self.assertEqual(metrics["complete_runs"], 2)
        self.assertEqual(metrics["first_try_rate"], 1 / 3)
        self.assertEqual(metrics["attempt_index_median"], 2.0)
        self.assertEqual(metrics["tokens_per_complete_mean"], 200)
        self.assertEqual(metrics["speedup_x100_median"], 200)
        self.assertEqual(metrics["failure_classes"], "hw_build:1")


class AggregateTest(unittest.TestCase):
    def _record(self, **over):
        record = {
            "exp_id": "pk5-test", "run_name": "r", "run_id": "r",
            "model_config": "glm-5.2", "model": "GLM-5.2", "benchmark": "crc32",
            "suite": "embench-iot", "repeat": 1, "max_iters": 5,
            "status": "complete", "exit_code": 0, "stopped_at": "",
            "failure_class": "", "failure_rung": "", "failure_detail": "",
            "repairs_used": 1, "attempt_index": 2,
            "pass_at_1": 0, "pass_at_3": 1, "pass_at_5": 1,
            "sim_ok": True, "reference_cycles": 10, "cfu_cycles": 5, "speedup_x100": 200,
            "yosys_cells": 2750, "yosys_wire_bits": 4402,
            "token_calls": 40, "token_in": 10, "token_out": 20, "token_total": 30,
            "wall_seconds": 60, "profile_cache_status": "hit", "visibility": "standard",
            "infra_retries": 0, "parse_warnings": "", "run_dir": "a", "framework_run_dir": "b",
            "report_path": "c", "started_at": "t0", "finished_at": "t1",
            "failure": None, "episodes": [], "infra": {"infra": False},
            "terminal": True,
        }
        record.update(over)
        return record

    def test_aggregate_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            exp_dir = Path(tmp_str) / "pk5-test"
            good = self._record(run_name="good")
            bad = self._record(
                run_name="bad", status="failed", attempt_index=5, pass_at_1=None,
                pass_at_3=None, pass_at_5=None,
                failure={"failure_class": "hw_build", "failure_rung": "hw_build",
                         "failure_detail": "scala error", "budget_exhausted": True},
                episodes=[{"kind": "repair", "trigger_rung": "hw_build", "repair_target": "implement_hw",
                           "decision_source": "fixed", "detail": "d", "evidence": "e"}],
            )
            for name, record in (("good", good), ("bad", bad)):
                run_dir = exp_dir / "runs" / record["run_name"]
                run_dir.mkdir(parents=True)
                (run_dir / "record.json").write_text(json.dumps(record))
            result = rm.aggregate(exp_dir, {"exp_id": "pk5-test", "git_commit": "abc"}, bootstrap_iters=200)
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["cells"], 1)

            results = exp_dir / "results"
            with (results / "runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(rows[0].keys()), sorted(rm.RUNS_COLUMNS))

            with (results / "failures.csv").open() as handle:
                failure_rows = list(csv.DictReader(handle))
            kinds = sorted(row["kind"] for row in failure_rows)
            self.assertEqual(kinds, ["repair", "terminal"])
            self.assertEqual(failure_rows[0]["model"] or failure_rows[1]["model"], "glm-5.2")

            with (results / "pass_at_k.csv").open() as handle:
                pass_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pass_rows), 1)
            self.assertEqual(pass_rows[0]["runs"], "2")
            self.assertEqual(pass_rows[0]["pass_at_5_budget"], "0.50")

            summary = (results / "summary.md").read_text()
            self.assertIn("## pass@5 (budget estimator)", summary)
            self.assertIn("hw_build:1", summary)


@unittest.skipUnless(shutil.which("flock"), "flock(1) required for the sbt lock")
class DryRunTest(unittest.TestCase):
    def test_dry_run_writes_experiment_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            workdir = Path(tmp_str)
            cfg_dir = workdir / "agents" / "llm_configs"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "test-a.json").write_text(json.dumps({"name": "test-a", "model": "TEST-A"}))
            out_root = workdir / "out"
            code = rm.main([
                "--exp-id", "pk5-test",
                "--models", "test-a",
                "--benchmarks", "crc32",
                "--repeats", "2",
                "--max-iters", "5",
                "--workdir", str(workdir),
                "--out-root", str(out_root),
                "--dry-run",
            ])
            self.assertEqual(code, 0)
            exp_dir = out_root / "pk5-test"
            self.assertTrue((exp_dir / "experiment.yaml").exists())
            shim = exp_dir / "bin" / "sbt"
            self.assertTrue(shim.exists())  # default: sbt lock enabled
            self.assertTrue(shim.stat().st_mode & stat.S_IXUSR)
            self.assertTrue((out_root / ".locks").exists())
            # Experiment config with forced temperature was generated.
            self.assertTrue((cfg_dir / "pk5-test-test-a.json").exists())


if __name__ == "__main__":
    unittest.main()
