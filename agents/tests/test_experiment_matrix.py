"""Tests for the pass@1/3/5 experiment harness (agents/experiments/harness/ + presets).

Everything here is free of model/API calls: parsing, classification, metrics,
aggregation and command construction against synthetic run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from agents.cfu_design_agent.runtime.artifacts import _safe_run_id
from agents.experiments import run_ablation, run_matrix as rm, run_sweep
from agents.experiments.harness import cli as harness_cli
from agents.experiments.harness import execution
from agents.experiments.harness import layers as harness_layers


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
- verify submissions: 2/5
- answer repairs used: 0/2
- repairs used: 2/4 (iteration=2)
- pre-verify repairs used: 0/2
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
            (["the CFU build is not meaningfully faster than the reference"],
             {"reference_cycles": 1000, "cfu_cycles": 2000}, "sim_speedup_floor"),
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
            reference_cycles=1010, cfu_cycles=1000, correctness_ok=True,
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

    def test_guarded_build_failure_points_at_implementation(self) -> None:
        # hw_build refuses to run sbt when implementation_hw applied nothing
        # (gate commands == 0): the failure stage is the implementation
        # rejection, not the build. Regression for the matmult-int run where a
        # missing-EDIT_END rejection was recorded as stopped_at=hw_build.
        record = _failure_record(
            "hw_build",
            {"ok": False, "commands": [], "errors": ["the hardware rung did not apply a design change"]},
        )
        record["gates"]["implementation_hw"] = {
            "ok": False,
            "applied": False,
            "errors": ["the response opened an EDIT_BEGIN section but the block was not closed (missing EDIT_END)"],
        }
        result = rm.classify_failure(record)
        self.assertEqual(result["failure_class"], "no_deliverable")
        self.assertEqual(result["failure_rung"], "implementation_hw")
        self.assertIn("missing EDIT_END", result["failure_detail"])
        self.assertIn("hw_build never ran", result["failure_detail"])

        # A real build failure (sbt actually ran) stays a build failure.
        record = _failure_record(
            "hw_build", {"ok": False, "commands": [{"command": "sbt ..."}], "errors": ["scala boom"]}
        )
        record["gates"]["implementation_hw"] = {"ok": True, "applied": True}
        result = rm.classify_failure(record)
        self.assertEqual(result["failure_class"], "hw_build")
        self.assertEqual(result["failure_rung"], "hw_build")


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

    def test_conformance_endpoint_failure_from_structured_gate(self) -> None:
        for error, expected in (
            ("the conformance oracle request got no model answer (TimeoutError: timed out)", "infra_timeout"),
            ("the conformance oracle request got no model answer (upstream HTTP 500)", "infra_error"),
        ):
            with self.subTest(error=error):
                gate = {"ok": False, "model_failure": True, "errors": [error]}
                result = rm.detect_infra("status=failed\n", "", False, 1,
                                         {"hw_conformance": gate})
                self.assertTrue(result["infra"])
                self.assertEqual(result["infra_kind"], expected)
                self.assertIn(error, result["infra_reason"])

    def test_conformance_endpoint_error_without_model_failure_flag(self) -> None:
        gate = {"ok": False, "errors": ["oracle endpoint upstream HTTP 500"]}
        self.assertEqual(rm.detect_infra("status=failed\n", "", False, 1,
                                         {"hw_conformance": gate})["infra_kind"], "infra_error")
        gate = {"ok": False, "errors": ["conformance oracle request got no model answer (timeout)"]}
        self.assertEqual(rm.detect_infra("status=failed\n", "", False, 1,
                                         {"hw_conformance": gate})["infra_kind"], "infra_timeout")
        gate = {"ok": False, "oracle_failure": True, "errors": ["oracle C compile failed"]}
        self.assertFalse(rm.detect_infra("status=failed\n", "", False, 1,
                                         {"hw_conformance": gate})["infra"])

    def test_endpoint_log_fallback_without_parseable_gate(self) -> None:
        for log in ("HTTP Error 500: Internal Server Error", "upstream 500 Internal Server Error",
                    "conformance oracle request got no model answer (TimeoutError)"):
            with self.subTest(log=log):
                self.assertTrue(rm.detect_infra("status=failed\n" + log, "", False, 1)["infra"])
        self.assertFalse(rm.detect_infra(
            "status=failed\n", "conformance verdict fail commands=0/1 cases=1", False, 1,
            {"hw_conformance": {"ok": False, "oracle_failure": False, "errors": ["verdict fail"]}},
        )["infra"])


class RateLimitTest(unittest.TestCase):
    def test_upstream_429_is_a_rate_limit_not_a_crash(self) -> None:
        # 30 of the 43 attempts in the 2026-10-04 ablation matrix died on this
        # and were reported as "framework crash or launcher failure".
        result = rm.detect_infra("", "openai.APIError: upstream error: 429", False, 1)
        self.assertTrue(result["infra"])
        self.assertEqual(result["infra_kind"], "infra_rate_limit")
        self.assertIn("429", result["infra_reason"])
        self.assertEqual(rm.detect_infra("", "HTTP Error 429: Too Many Requests", False, 1)
                         ["infra_kind"], "infra_rate_limit")

    def test_a_benchmark_number_is_not_a_rate_limit(self) -> None:
        # A cycle count or a result value must never look like endpoint trouble.
        result = rm.detect_infra("CFU_AGENT_PROFILE md5sum=429\nstatus=failed\n", "", False, 1)
        self.assertFalse(result["infra"])

    def test_request_timeout_is_infra_timeout(self) -> None:
        # The SDK's wording carries no role word, so the older rule missed it and
        # eight 2026-10-04 ablation runs read as "framework crash".
        result = rm.detect_infra("", "openai.APITimeoutError: Request timed out.", False, 1)
        self.assertTrue(result["infra"])
        self.assertEqual(result["infra_kind"], "infra_timeout")
        self.assertEqual(rm.detect_infra("", "Read timed out.", False, 1)["infra_kind"],
                         "infra_timeout")

    def test_a_completed_run_is_never_infra(self) -> None:
        # One complete l1 run was marked infra by a stray 500 echoed into its
        # output and burned a redundant retry.
        self.assertFalse(rm.detect_infra("status=complete\n", "upstream error: 500",
                                         False, 0)["infra"])
        # A killed run that only *looks* complete is still infra.
        self.assertTrue(rm.detect_infra("status=complete\n", "upstream error: 500",
                                        True, 1)["infra"])

    def test_stage_gate_rejection_is_a_model_failure_not_infra(self) -> None:
        stderr = ("RuntimeError: native stage failed: independent submit_stage gate rejected "
                  "delivery (conformance oracle is missing/stale/invalid: call write_conformance)")
        result = rm.detect_infra("", stderr, False, 1)
        self.assertFalse(result["infra"])
        self.assertIn("stage gate", result["infra_reason"])
        record = rm.classify_failure({"status": "initialized", "gates": {}, "stopped_at": "",
                                      "stage_gate_rejected": True})
        self.assertEqual(record["failure_class"], "no_deliverable")
        self.assertEqual(record["failure_rung"], "implementation")

    def test_stale_framework_artifacts_never_decide_a_dead_attempt(self) -> None:
        """A reused run dir must not report the previous round's outcome.

        2026-10-04: 29 of 30 ablation records mixed a new attempt's stdout with
        an older run.yaml, so attempts that died in analysis were collected as
        ``complete`` (which inflated every layer's pass@k).
        """
        fixture = CollectTest("test_collect_passing_run")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = fixture._fixture(tmp, STDOUT_PASS, REPORT_PASS, RUN_YAML_PASS, EMBENCH_PASS)
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 3, "max_iters": 5,
                   "infra_retry": 0}
            meta = {"exit_code": 0, "wall_seconds": 1.0,
                    "started_at": "2026-10-04T15:00:00+00:00",
                    "finished_at": "2026-10-04T15:01:00+00:00"}
            fresh = rm.collect_record("e", job, "run_name", run_dir, fw, meta)
            self.assertFalse(fresh["stale_framework_artifacts"])
            self.assertEqual(fresh["status"], "complete")
            # The attempt itself never reported an outcome, and the artifacts on
            # disk were last written before it started: they are foreign evidence.
            (run_dir / "stdout.log").write_text(
                "[DISCOVER 8/8] analyze workload ... running\n"
                "[DISCOVER 8/8] analyze workload ... FAIL (in=0 out=0 tok, 67.5s)\n")
            (run_dir / "stderr.log").write_text("openai.APITimeoutError: Request timed out.\n")
            from datetime import datetime, timezone

            old = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc).timestamp()
            for name in ("run.yaml", "report.md"):
                os.utime(fw / name, (old, old))
            stale = rm.collect_record("e", job, "run_name", run_dir, fw,
                                      {**meta, "exit_code": 1, "wall_seconds": 67.5})
            self.assertTrue(stale["stale_framework_artifacts"])
            self.assertTrue(stale["stale_evidence_ignored"])
            self.assertNotEqual(stale["status"], "complete")
            self.assertIn("stale framework artifacts", stale["parse_warnings"])
            # A genuine completion in a reused dir still trusts its own status line.
            (run_dir / "stdout.log").write_text(STDOUT_PASS)
            self.assertEqual(rm.collect_record("e", job, "run_name", run_dir, fw, meta)["status"],
                             "complete")

    def test_a_stale_run_dir_is_archived_before_an_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            path = Path(tmp_str) / "agents/generated/runs/run_name"
            _write(path / "run.yaml", "status: complete\n")
            archived = rm.archive_stale_run_dir(path)
            self.assertIsNotNone(archived)
            self.assertFalse(path.exists())
            self.assertTrue((archived / "run.yaml").is_file())
            self.assertIn(".superseded-", archived.name)
            self.assertIsNone(rm.archive_stale_run_dir(path))
            path.mkdir(parents=True)
            self.assertIsNone(rm.archive_stale_run_dir(path), "an empty dir needs no archive")

    def test_a_rate_limited_retry_backs_off_longer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            opts = argparse.Namespace(
                exp_id="pk5-test", max_iters=5, workdir=root, python="python3", conda_env="",
                visibility="standard", repair_routing="auto", software_change="auto",
                profile_cache="on", timeout=1, force=False, infra_retries=1,
                infra_backoff=10.0, rate_limit_backoff=120.0,
            )
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1}
            exp_dir = root / "experiment"
            calls = []
            # Patch the seams on their OWNING module: execute_one reads
            # execution.run_logged / execution.time.sleep, so assignments on the
            # run_matrix facade would not be observed.
            original_run_logged, original_sleep = execution.run_logged, execution.time.sleep

            def fake_run_logged(argv, out_path, err_path, cwd, env, timeout):
                calls.append(out_path.parent.name)
                _write(out_path, "status=failed\n")
                _write(err_path, "openai.APIError: upstream error: 429\n")
                return {"exit_code": 1, "wall_seconds": 1}

            delays = []
            try:
                execution.run_logged = fake_run_logged
                execution.time.sleep = delays.append
                records = rm.execute_one(job, opts, exp_dir, {"glm-5.2": {"config": "test"}}, None, {})
            finally:
                execution.run_logged, execution.time.sleep = original_run_logged, original_sleep
            self.assertEqual(len(records), 2)
            self.assertEqual([r["infra"]["infra_kind"] for r in records],
                             ["infra_rate_limit", "infra_rate_limit"])
            self.assertEqual(len(delays), 1, delays)
            self.assertGreaterEqual(delays[0], 120.0)
            self.assertLessEqual(delays[0], 180.0)


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
            self.assertEqual(record["verify_submissions"], 2)
            self.assertEqual(record["max_verify_submissions"], 5)
            self.assertEqual(record["answer_repairs_used"], 0)
            self.assertEqual(record["max_answer_repairs"], 2)
            self.assertEqual(record["preverify_repairs_used"], 0)
            self.assertEqual(record["max_preverify_repairs"], 2)
            self.assertEqual(record["attempt_index"], 2)
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

    def test_collect_legacy_report_falls_back_to_repair_attempt_index(self) -> None:
        report = REPORT_PASS.replace("- verify submissions: 2/5\n", "")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = self._fixture(tmp, STDOUT_PASS, report, RUN_YAML_PASS, EMBENCH_PASS)
            record = rm.collect_record(
                "pk5-test", {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1,
                             "max_iters": 5, "infra_retry": 0},
                "legacy-run", run_dir, fw, {"exit_code": 0, "wall_seconds": 5},
            )
            self.assertIsNone(record["verify_submissions"])
            self.assertEqual(record["attempt_index"], 3)

    def test_collect_failing_run_writes_flags(self) -> None:
        report = REPORT_PASS.replace("Status: complete", "Status: failed").replace(
            "- sim: {'ok': True", "- sim: {'ok': False"
        ).replace("verify submissions: 2/5", "verify submissions: 5/5").replace(
            "repairs used: 2/4 (iteration=2)", "repairs used: 4/4 (iteration=4)")
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
            self.assertEqual(payload["terminal"]["verify_submissions"], 5)


class VerifyStatsTest(unittest.TestCase):
    """平均 calls / 平均 tokens / 平均 Verify 失败次数 的采集与口径。"""

    ATTEMPT_VERIFY_FAIL = {
        "rung": "sim", "status": "failed",
        "commands": [{
            "command": "sbt sim", "returncode": 1,
            "parsed": {"verify": {"status": "fail", "benchmark": "crc32", "verify": 0, "result": 123},
                       "embench_profile": {"crc32": {"cycles": 5, "result": 123}}},
            "failure_observation": {"category": "fail_symbol"},
        }],
    }
    ATTEMPT_TIMEOUT = {
        "rung": "sim", "status": "failed",
        "commands": [{
            "command": "sbt sim", "returncode": -9, "timed_out": True,
            "parsed": {"verify": {"status": "pass", "benchmark": "crc32", "verify": 1, "result": 999}},
            "failure_observation": {"category": "timeout"},
        }],
    }
    ATTEMPT_HARNESS_MISMATCH = {
        "rung": "sim", "status": "failed",
        "commands": [{
            "command": "sbt sim", "returncode": 0,
            # verify_benchmark passed but the observable result differs from the
            # scalar reference (05_profile: result 999): still a verify failure.
            "parsed": {"verify": {"status": "pass", "benchmark": "crc32", "verify": 1, "result": 123},
                       "embench_profile": {"crc32": {"cycles": 5, "result": 123}}},
            "failure_observation": {},
        }],
    }

    def _fixture(self, tmp: Path, attempts: list[dict], versions: bool = True):
        run_dir, fw = CollectTest()._fixture(tmp, STDOUT_PASS, REPORT_PASS, RUN_YAML_PASS, EMBENCH_PASS)
        import yaml as _yaml

        for index, attempt in enumerate(attempts):
            sha = f"{'ab' * 30}{index:02d}"
            if versions:
                _write(fw / "10_validation" / "versions" / f"sim.yaml.{sha}", _yaml.safe_dump(attempt))
            elif index == len(attempts) - 1:
                _write(fw / "10_validation" / "sim.yaml", _yaml.safe_dump(attempt))
        _write(fw / "05_profile" / "embench.yaml",
               "benchmark: crc32\nprofiles:\n  profile:\n    crc32: {cycles: 10, result: 999, verify: 1}\n")
        return run_dir, fw

    def test_collect_counts_attempts_and_verify_failures(self) -> None:
        attempts = [self.ATTEMPT_VERIFY_FAIL, self.ATTEMPT_TIMEOUT, self.ATTEMPT_HARNESS_MISMATCH]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = self._fixture(tmp, attempts)
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1, "max_iters": 5, "infra_retry": 0}
            record = rm.collect_record("pk5-test", job, "run_name", run_dir, fw, {"exit_code": 0})
            self.assertEqual(record["sim_attempts"], 3)
            # verify oracle rejected one; timeout is NOT a verify failure; the
            # harness-side result mismatch is.
            self.assertEqual(record["verify_failures"], 2)
            self.assertEqual(record["parse_warnings"], "")

    def test_latest_sim_yaml_is_the_fallback(self) -> None:
        # Stores without version files still report the final attempt.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = self._fixture(tmp, [self.ATTEMPT_VERIFY_FAIL, self.ATTEMPT_TIMEOUT],
                                        versions=False)
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1, "max_iters": 5, "infra_retry": 0}
            record = rm.collect_record("pk5-test", job, "run_name", run_dir, fw, {"exit_code": 0})
            self.assertEqual(record["sim_attempts"], 1)
            self.assertEqual(record["verify_failures"], 0)  # the timeout attempt only

    def test_token_usage_falls_back_to_report(self) -> None:
        report = REPORT_PASS.replace(
            "## Gates",
            "## Token Usage\n- llm_calls: 55\n- input tokens: 111\n- output tokens: 222\n- total tokens: 333\n\n## Gates",
        )
        stdout = "\n".join(line for line in STDOUT_PASS.splitlines()
                           if not re.search(r"calls\s+=|tokens", line))
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir, fw = CollectTest()._fixture(tmp, stdout, report, RUN_YAML_PASS, EMBENCH_PASS)
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1, "max_iters": 5, "infra_retry": 0}
            record = rm.collect_record("pk5-test", job, "run_name", run_dir, fw, {"exit_code": 0})
            self.assertEqual(record["token_calls"], 55)
            self.assertEqual(record["token_total"], 333)

    def test_cell_metrics_effort_means(self) -> None:
        records = [
            {"status": "complete", "attempt_index": 1,
             "token_calls": 40, "token_total": 300, "verify_failures": 0},
            {"status": "failed", "token_calls": 60, "token_total": 500, "verify_failures": 2},
            # Older record: values unknown are skipped, never counted as zero.
            {"status": "failed", "token_calls": None, "token_total": None, "verify_failures": None},
        ]
        metrics = rm.cell_metrics(records, bootstrap_iters=50)
        self.assertEqual(metrics["calls_mean"], 50)
        self.assertEqual(metrics["tokens_mean"], 400)
        self.assertEqual(metrics["verify_failures_mean"], 1)


class MeasurementProtocolTest(unittest.TestCase):
    PROTOCOL = "cold_single_invocation_v1"
    MEASUREMENT = {"protocol": PROTOCOL, "local_scale_factor": 1,
                   "global_scale_factor": 1, "warmup": False}
    INTEGRITY = {"version": 1, "ok": True, "problems": []}

    def test_collection_preserves_validation_protocol_and_integrity_metadata(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, fw = CollectTest()._fixture(
                Path(tmp), STDOUT_PASS, REPORT_PASS, RUN_YAML_PASS,
                EMBENCH_PASS + yaml.safe_dump({"measurement_protocol": self.PROTOCOL,
                                              "measurement": self.MEASUREMENT,
                                              "benchmark_integrity": self.INTEGRITY}),
            )
            record = rm.collect_record("e", {"benchmark": "crc32"}, "run_name", run_dir, fw)
        self.assertEqual(record["measurement_protocol"], self.PROTOCOL)
        self.assertEqual(record["measurement"], self.MEASUREMENT)
        self.assertEqual(record["benchmark_integrity"], self.INTEGRITY)
        self.assertIn("not pure math-kernel latency", record["measurement_notice"])
        self.assertIn("not proof of true operation equivalence", record["measurement_notice"])

    def test_collection_uses_sim_artifact_protocol_when_embench_evidence_is_absent(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, fw = CollectTest()._fixture(Path(tmp), STDOUT_PASS, REPORT_PASS, RUN_YAML_PASS, "")
            _write(fw / "10_validation/sim.yaml", yaml.safe_dump({
                "measurement_protocol": self.PROTOCOL, "measurement": self.MEASUREMENT,
                "benchmark_integrity": self.INTEGRITY,
            }))
            record = rm.collect_record("e", {"benchmark": "crc32"}, "run_name", run_dir, fw)
        self.assertEqual(record["measurement_protocol"], self.PROTOCOL)
        self.assertEqual(record["benchmark_integrity"], self.INTEGRITY)

    def test_collection_accepts_nested_protocol_and_marks_absent_evidence_legacy(self) -> None:
        import yaml

        for metadata in ({}, {"measurement": self.MEASUREMENT}):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as tmp:
                run_dir, fw = CollectTest()._fixture(
                    Path(tmp), STDOUT_PASS, REPORT_PASS, RUN_YAML_PASS,
                    EMBENCH_PASS + yaml.safe_dump(metadata),
                )
                record = rm.collect_record("e", {"benchmark": "crc32"}, "run_name", run_dir, fw)
                self.assertEqual(record["measurement_protocol"], self.PROTOCOL if metadata else "legacy")
                if not metadata:
                    self.assertIn("protocol evidence absent", record["measurement_notice"])
                    self.assertEqual(record["benchmark_integrity"], {})
                    self.assertIn("not a legacy resume", record["measurement_notice"])

    def test_collection_can_use_sim_gate_protocol(self) -> None:
        gate = {"ok": True, "measurement_protocol": self.PROTOCOL,
                "measurement": self.MEASUREMENT, "benchmark_integrity": self.INTEGRITY}
        report = REPORT_PASS.replace(
            REPORT_PASS.split("## Gates\n", 1)[1].splitlines()[0], f"- sim: {gate!r}")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, fw = CollectTest()._fixture(Path(tmp), STDOUT_PASS, report, RUN_YAML_PASS, "")
            record = rm.collect_record("e", {"benchmark": "crc32"}, "run_name", run_dir, fw)
        self.assertEqual(record["measurement_protocol"], self.PROTOCOL)
        self.assertEqual(record["measurement"], self.MEASUREMENT)

    def test_mixed_complete_protocols_suppress_only_speedup_median(self) -> None:
        records = [
            {"status": "complete", "attempt_index": 1, "measurement_protocol": self.PROTOCOL,
             "speedup_x100": 150, "yosys_cells": 2000, "token_total": 100, "wall_seconds": 3600},
            {"status": "complete", "attempt_index": 3, "speedup_x100": 250,
             "yosys_cells": 3000, "token_total": 300, "wall_seconds": 7200},
        ]
        row = rm.cell_metrics(records, bootstrap_iters=20)
        self.assertIsNone(row["speedup_x100_median"])
        self.assertIn("different measurement protocols", row["speedup_warning"])
        self.assertIn("legacy", row["measurement_protocols"])
        self.assertEqual(row["complete_runs"], 2)
        self.assertEqual(row["pass_at_3_budget"], 1)
        self.assertEqual(row["tokens_per_complete_mean"], 200)
        self.assertEqual(row["yosys_cells_median"], 2500)
        self.assertEqual(row["wall_h_per_complete_mean"], 1.5)

    def test_any_different_complete_protocols_are_noncomparable(self) -> None:
        row = rm.cell_metrics([
            {"status": "complete", "measurement_protocol": self.PROTOCOL, "speedup_x100": 150},
            {"status": "complete", "measurement_protocol": "future_v2", "speedup_x100": None},
        ], bootstrap_iters=20)
        self.assertIsNone(row["speedup_x100_median"])
        self.assertIn("future_v2", row["speedup_warning"])

    def test_single_protocol_median_and_noncomplete_records_remain_compatible(self) -> None:
        for protocol in (None, self.PROTOCOL):
            with self.subTest(protocol=protocol):
                records = [
                    {"status": "complete", "measurement_protocol": protocol, "speedup_x100": 150},
                    {"status": "complete", "measurement_protocol": protocol, "speedup_x100": 250},
                    {"status": "failed", "measurement_protocol": "other", "speedup_x100": 900},
                    {"status": "complete", "measurement_protocol": "infra", "speedup_x100": 900,
                     "infra": {"infra": True}},
                ]
                row = rm.cell_metrics(records, bootstrap_iters=20)
                self.assertEqual(row["speedup_x100_median"], 200)
                self.assertEqual(row["speedup_warning"], "")

    def test_csv_and_summary_preserve_protocol_metadata_and_mixed_notice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            cold = AggregateTest()._record(
                run_name="cold", measurement_protocol=self.PROTOCOL,
                measurement=self.MEASUREMENT, benchmark_integrity=self.INTEGRITY,
                measurement_notice="Not pure math-kernel latency.",
            )
            old = AggregateTest()._record(run_name="old", repeat=2)
            for record in (cold, old):
                _write(exp_dir / "runs" / record["run_name"] / "record.json", json.dumps(record))
            rm.aggregate(exp_dir, {"exp_id": "e"}, bootstrap_iters=20)
            with (exp_dir / "results/runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            by_name = {row["run_name"]: row for row in rows}
            self.assertEqual(by_name["old"]["measurement_protocol"], "legacy")
            self.assertEqual(by_name["cold"]["measurement_protocol"], self.PROTOCOL)
            self.assertEqual(json.loads(by_name["cold"]["measurement"]), self.MEASUREMENT)
            self.assertEqual(json.loads(by_name["cold"]["benchmark_integrity"]), self.INTEGRITY)
            self.assertEqual(by_name["cold"]["measurement_notice"], cold["measurement_notice"])
            with (exp_dir / "results/pass_at_k.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["speedup_x100_median"], "")
            self.assertIn("different measurement protocols", row["speedup_warning"])
            self.assertEqual(row["yosys_cells_median"], "2750.00")
            summary = (exp_dir / "results/summary.md").read_text()
            self.assertIn("## Embench measurement protocols", summary)
            self.assertIn("Speedup median unavailable", summary)
            self.assertIn("not pure math-kernel latency", summary)
            self.assertIn("legacy", summary)


class EndpointRetryTest(unittest.TestCase):
    def test_endpoint_failure_is_collected_classified_and_retried(self) -> None:
        error = "the conformance oracle request got no model answer (TimeoutError: timed out)"
        failed_report = REPORT_PASS.replace("Status: complete", "Status: failed").replace(
            "- sim: {'ok': True,", f"- hw_conformance: {{'ok': False, 'model_failure': True, 'errors': [{error!r}]}}\n- sim: {{'ok': True,"
        ).replace("repairs used: 2/4 (iteration=2)", "repairs used: 0/4 (iteration=0)")
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            opts = argparse.Namespace(
                exp_id="pk5-test", max_iters=5, workdir=root, python="python3", conda_env="",
                visibility="standard", repair_routing="auto", software_change="auto",
                profile_cache="on", timeout=1, force=False, infra_retries=1,
                infra_backoff=0.0, rate_limit_backoff=0.0,
            )
            job = {"model": "glm-5.2", "benchmark": "crc32", "repeat": 1}
            exp_dir = root / "experiment"
            calls = []
            original_run_logged = execution.run_logged

            def fake_run_logged(argv, out_path, err_path, cwd, env, timeout):
                calls.append(out_path.parent.name)
                fw = root / "agents" / "generated" / "runs" / calls[-1]
                if len(calls) == 1:
                    _write(out_path, "status=failed\nstopped_at=hw_conformance\n")
                    _write(fw / "report.md", failed_report)
                    _write(fw / "run.yaml", RUN_YAML_PASS.replace("complete", "failed"))
                    code = 1
                else:
                    _write(out_path, STDOUT_PASS)
                    _write(fw / "report.md", REPORT_PASS)
                    _write(fw / "run.yaml", RUN_YAML_PASS)
                    code = 0
                _write(err_path, "")
                return {"exit_code": code, "wall_seconds": 1}

            try:
                execution.run_logged = fake_run_logged
                records = rm.execute_one(job, opts, exp_dir, {"glm-5.2": {"config": "test"}}, None, {})
            finally:
                execution.run_logged = original_run_logged
            self.assertEqual(len(records), 2)
            self.assertTrue(calls[1].endswith("-infra1"), calls)
            first = records[0]
            self.assertEqual(first["infra"]["infra_kind"], "infra_timeout")
            self.assertEqual(first["failure"]["failure_class"], "infra_timeout")
            self.assertEqual(first["failure"]["failure_rung"], "hw_conformance")
            self.assertIsNone(first["failure"]["budget_exhausted"])
            self.assertEqual(json.loads((exp_dir / "runs" / calls[0] / "failure.json").read_text())
                             ["terminal"]["failure_class"], "infra_timeout")
            self.assertFalse(records[1]["infra"]["infra"])
            self.assertEqual(rm.cell_metrics(records, bootstrap_iters=100)["runs"], 1)
            self.assertEqual(rm.cell_metrics(records, bootstrap_iters=100)["pass_at_5_budget"], 1)
            rm.aggregate(exp_dir, {"exp_id": "pk5-test"}, bootstrap_iters=100)
            with (exp_dir / "results" / "runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)  # retain incident and retry for auditing
            self.assertEqual(rows[0]["failure_class"], "infra_timeout")
            with (exp_dir / "results" / "failures.csv").open() as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual(failures[0]["failure_class"], "infra_timeout")
            with (exp_dir / "results" / "pass_at_k.csv").open() as handle:
                metrics = list(csv.DictReader(handle))[0]
            self.assertEqual(metrics["runs"], "1")
            self.assertEqual(metrics["failure_classes"], "")


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
                episodes=[{"kind": "repair", "trigger_rung": "hw_build", "repair_target": "implement",
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
            self.assertIn("## Run effort & verification load", summary)
            with (results / "pass_at_k.csv").open() as handle:
                metric_keys = list(csv.DictReader(handle))[0].keys()
            for key in ("calls_mean", "tokens_mean", "verify_failures_mean"):
                self.assertIn(key, metric_keys)


class AblationLayerTest(unittest.TestCase):
    """The layered ladder: one added capability per layer, ISA Contract -> Conformance -> LLVM IR."""

    def test_ladder_adds_one_capability_per_layer(self) -> None:
        self.assertEqual(len(rm.ABLATION_LADDER), 4)
        disabled = [
            set(rm.ABLATION_LAYERS[name]["flags"]) for name in rm.ABLATION_LADDER
        ]
        # l0 ablates everything; each next layer re-enables exactly one
        # capability and the last layer is the un-ablated reference flow.
        self.assertEqual(disabled[0],
                         {"--no-isa-contract", "--no-conformance", "--no-llvm-ir"})
        self.assertEqual(disabled[-1], set())
        for prev, nxt in zip(disabled, disabled[1:]):
            self.assertEqual(len(prev - nxt), 1, (prev, nxt))
            self.assertFalse(nxt - prev, (prev, nxt))  # never re-disable anything
        # The order of re-enabling is ISA Contract, then Conformance, then LLVM IR.
        reenabled = [next(iter(a - b)) for a, b in zip(disabled, disabled[1:])]
        self.assertEqual(reenabled, ["--no-isa-contract", "--no-conformance", "--no-llvm-ir"])
        # --no-isa-contract implies --no-conformance in the CLI: layer 0 spells
        # both out so the recorded argv is honest about what was removed.
        self.assertIn("--no-conformance", rm.ABLATION_LAYERS["l0-none"]["flags"])
        # Every flag is a real agent CLI switch.
        for name in rm.ABLATION_LADDER:
            for flag in rm.ABLATION_LAYERS[name]["flags"]:
                self.assertRegex(flag, r"^--no-(isa-contract|conformance|llvm-ir)$")
            self.assertTrue(rm.ABLATION_LAYERS[name]["adds"])

    def test_resolve_layers(self) -> None:
        self.assertEqual(rm.resolve_layers("ladder"), list(rm.ABLATION_LADDER))
        self.assertEqual(rm.resolve_layers("all"), list(rm.ABLATION_LADDER))
        self.assertEqual(rm.resolve_layers("full"), ["full"])
        self.assertEqual(rm.resolve_layers("none"), ["l0-none"])
        self.assertEqual(rm.resolve_layers("conf,isa"), ["l1-isa", "l2-conf"])  # ladder order
        self.assertEqual(rm.resolve_layers("l3"), ["full"])
        with self.assertRaises(SystemExit):
            rm.resolve_layers("nope")
        with self.assertRaises(SystemExit):
            rm.resolve_layers("")

    def test_layer_flags(self) -> None:
        self.assertEqual(rm.layer_flags("l2-conf"), ("--no-llvm-ir",))
        self.assertEqual(rm.layer_flags("full"), ())
        self.assertEqual(rm.layer_flags(""), ())  # legacy jobs = full flow

    def test_layer_suffix_naming(self) -> None:
        name = rm.make_run_name("e", "glm-5.2", "crc32", 1, 5, layer="l0-none")
        self.assertEqual(name, "e-glm52-crc32-r01-mi5-l0-none")
        self.assertTrue(_safe_run_id(name))
        # The full flow keeps the canonical name; layer token lands before -infraN.
        self.assertEqual(rm.make_run_name("e", "glm-5.2", "crc32", 1, 5, layer="full"),
                         "e-glm52-crc32-r01-mi5")
        self.assertEqual(
            rm.make_run_name("e", "glm-5.2", "crc32", 1, 5, infra_retry=2, layer="l1-isa"),
            "e-glm52-crc32-r01-mi5-l1-isa-infra2",
        )
        # Run names stay unique per (layer, model, benchmark, repeat).
        names = {
            rm.make_run_name("e", model, bench, repeat, 5, layer=layer)
            for model in ("glm-5.2", "kimi-k3")
            for bench in ("crc32", "md5sum")
            for repeat in (1, 2)
            for layer in rm.ABLATION_LADDER
        }
        self.assertEqual(len(names), 32)


class LayerMatrixTest(unittest.TestCase):
    def test_expand_matrix_with_layers(self) -> None:
        layers = rm.resolve_layers("ladder")
        jobs = rm.expand_matrix(("a",), ("x", "y"), 2, layers)
        self.assertEqual(len(jobs), 2 * 2 * 4)
        self.assertEqual([j["layer"] for j in jobs[:4]], list(rm.ABLATION_LADDER))
        for job in jobs:
            self.assertEqual(tuple(job["ablation_flags"]), rm.layer_flags(job["layer"]))
        # Default: a single full-flow layer (the classic pass@k matrix shape).
        self.assertEqual(len(rm.expand_matrix(("a", "b"), ("x", "y", "z"), 3)), 18)

    def test_filter_jobs_can_select_a_layer(self) -> None:
        jobs = rm.schedule_jobs(
            rm.expand_matrix(("glm-5.2",), ("crc32",), 1, rm.resolve_layers("ladder")), seed=1
        )
        for job in jobs:
            job["exp_id"] = "abl"
            job["max_iters"] = 5
        kept = rm.filter_jobs(jobs, ["l2-conf"])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["layer"], "l2-conf")


class AblationParseTest(unittest.TestCase):
    def test_parse_report_reads_ablation_section(self) -> None:
        text = REPORT_PASS.replace(
            "## Gates",
            "## Ablation\n"
            "- `--no-llvm-ir`: LLVM IR content removed (ir_features forced off)\n"
            "- `--no-conformance`: conformance requirement dropped\n"
            "\n## Gates",
        )
        warnings: list[str] = []
        out = rm.parse_report(text, warnings)
        self.assertEqual(out["ablation_flags"], ["--no-llvm-ir", "--no-conformance"])
        self.assertEqual(warnings, [])
        # A full-flow report records no ablation.
        self.assertEqual(rm.parse_report(REPORT_PASS, [])[ "ablation_flags"], [])

    def test_collect_cross_checks_ablation(self) -> None:
        report = REPORT_PASS.replace(
            "## Gates",
            "## Ablation\n- `--no-llvm-ir`: LLVM IR content removed\n\n## Gates",
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)

            def collect(job, report_text):
                run_dir, fw = CollectTest()._fixture(tmp, STDOUT_PASS, report_text, RUN_YAML_PASS, EMBENCH_PASS)
                return rm.collect_record("pk5-test", job, "run_name", run_dir, fw, {"exit_code": 0})

            # Scheduled flags match what the framework recorded: no warning.
            record = collect({"model": "glm-5.2", "benchmark": "crc32", "repeat": 1,
                              "max_iters": 5, "infra_retry": 0,
                              "layer": "l2-conf", "ablation_flags": ("--no-llvm-ir",)}, report)
            self.assertEqual(record["parse_warnings"], "")
            self.assertEqual(record["layer"], "l2-conf")
            self.assertEqual(record["ablation_flags"], "--no-llvm-ir")
            self.assertEqual(record["ablation_observed"], "--no-llvm-ir")

            # The framework recorded something different: loud, not silent.
            record = collect({"model": "glm-5.2", "benchmark": "crc32", "repeat": 1,
                              "max_iters": 5, "infra_retry": 0,
                              "layer": "l0-none",
                              "ablation_flags": rm.layer_flags("l0-none")}, report)
            self.assertIn("ablation mismatch", record["parse_warnings"])
            self.assertIn("--no-isa-contract", record["parse_warnings"])

    def test_aggregate_keeps_layers_apart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            exp_dir = Path(tmp_str) / "abl-test"
            for layer in rm.ABLATION_LADDER:
                for attempt in (2, 5):
                    record = AggregateTest()._record(
                        run_name=f"r-{layer}-{attempt}",
                        layer=layer,
                        ablation_flags=" ".join(rm.layer_flags(layer)),
                        ablation_observed=" ".join(rm.layer_flags(layer)),
                        status="complete" if attempt == 2 else "failed",
                        attempt_index=attempt,
                        pass_at_1=(0 if attempt == 2 else None),
                        pass_at_3=(1 if attempt == 2 else None),
                        pass_at_5=(1 if attempt == 2 else None),
                        failure=(None if attempt == 2 else
                                 {"failure_class": "sim_correctness", "failure_rung": "sim",
                                  "failure_detail": "wrong", "budget_exhausted": True}),
                    )
                    run_dir = exp_dir / "runs" / record["run_name"]
                    run_dir.mkdir(parents=True)
                    (run_dir / "record.json").write_text(json.dumps(record))
            result = rm.aggregate(exp_dir, {"exp_id": "abl-test"}, bootstrap_iters=100)
            self.assertEqual(result["cells"], 4)  # layers never pool into one cell

            with (exp_dir / "results" / "pass_at_k.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["layer"] for row in rows], list(rm.ABLATION_LADDER))
            self.assertTrue(all(row["runs"] == "2" for row in rows))

            summary = (exp_dir / "results" / "summary.md").read_text()
            self.assertIn("## Ablation ladder", summary)
            self.assertIn("--no-isa-contract --no-conformance --no-llvm-ir", summary)
            self.assertIn("### Layer comparison (pass@k budget", summary)
            self.assertIn("| model | benchmark | l0-none | l1-isa | l2-conf | full |", summary)

            with (exp_dir / "results" / "runs.csv").open() as handle:
                run_rows = list(csv.DictReader(handle))
            self.assertEqual({row["layer"] for row in run_rows}, set(rm.ABLATION_LADDER))


@unittest.skipUnless(shutil.which("flock"), "flock(1) required for the sbt lock")
class AblationDryRunTest(unittest.TestCase):
    def test_run_ablation_schedules_the_whole_ladder(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp_str:
            workdir = Path(tmp_str)
            cfg_dir = workdir / "agents" / "llm_configs"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "test-a.json").write_text(json.dumps({"name": "test-a", "model": "TEST-A"}))
            out_root = workdir / "out"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = run_ablation.main([
                    "--exp-id", "abl-test",
                    "--models", "test-a",
                    "--benchmarks", "crc32",
                    "--repeats", "2",
                    "--max-iters", "5",
                    "--workdir", str(workdir),
                    "--out-root", str(out_root),
                    "--dry-run",
                ])
            self.assertEqual(code, 0)
            text = stdout.getvalue()
            # 1 model x 1 benchmark x 2 repeats x 4 layers = 8 scheduled runs.
            self.assertEqual(text.count("[dry ]"), 9)  # + the schedule summary line
            for layer in rm.ABLATION_LADDER:
                token = "" if layer == "full" else f"-{layer}"
                self.assertIn(f"abl-test-{rm.model_slug('test-a')}-crc32-r01-mi5{token}", text)
            self.assertIn("--no-isa-contract --no-conformance --no-llvm-ir", text)
            self.assertIn("--no-conformance --no-llvm-ir", text)
            # The layer metadata is frozen into experiment.yaml.
            import yaml
            meta = yaml.safe_load((out_root / "abl-test" / "experiment.yaml").read_text())
            self.assertEqual(list(meta["layers"]), list(rm.ABLATION_LADDER))
            self.assertEqual(meta["layers"]["l1-isa"]["flags"], ["--no-conformance", "--no-llvm-ir"])
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


class AgentArgPassthroughTest(unittest.TestCase):
    """The ``--`` tail is forwarded verbatim to run_cfu_agent; owned flags never."""

    def test_split_agent_args_at_the_first_bare_double_dash(self) -> None:
        self.assertEqual(harness_cli.split_agent_args(["--exp-id", "x"]),
                         (["--exp-id", "x"], []))
        self.assertEqual(
            harness_cli.split_agent_args(["--exp-id", "x", "--", "--skip-sim", "--task", "t"]),
            (["--exp-id", "x"], ["--skip-sim", "--task", "t"]),
        )

    def test_harness_owned_flags_are_rejected_in_the_passthrough(self) -> None:
        for arg in ("--run-id", "--max-iters=5", "--llm-config", "--out-dir",
                    "--workdir", "--profile-cache=off"):
            with self.assertRaises(SystemExit, msg=arg):
                harness_cli.validate_agent_args([arg])

    def test_agent_side_flags_and_values_pass_validation(self) -> None:
        # Ablation switches are agent-side parameters too: they pass through
        # and get folded into the run's recorded ablation flags.
        harness_cli.validate_agent_args([
            "--skip-sim", "--skip-yosys", "--task", "accelerate",
            "--no-conformance", "--no-isa-contract", "--no-llvm-ir",
        ])

    def test_ablation_switches_carry_their_implications(self) -> None:
        # --no-isa-contract implies --no-conformance and the report records
        # BOTH, so the expected set must be closed the same way.
        self.assertEqual(
            harness_layers.effective_ablation_flags(("--no-isa-contract",)),
            ("--no-isa-contract", "--no-conformance"),
        )
        self.assertEqual(
            harness_layers.effective_ablation_flags(("--no-conformance", "--no-isa-contract")),
            ("--no-isa-contract", "--no-conformance"),
        )
        self.assertEqual(
            harness_layers.effective_ablation_flags(("--no-conformance", "--no-llvm-ir")),
            ("--no-conformance", "--no-llvm-ir"),
        )
        self.assertEqual(harness_layers.effective_ablation_flags(()), ())

    def test_build_argv_appends_agent_args_last(self) -> None:
        argv = rm.build_argv(
            "python3", "", None, "cfg", "crc32", "run", 5, "standard", "on",
            ("--no-llvm-ir",), ("--skip-sim", "--task", "accelerate"),
        )
        self.assertEqual(argv[-3:], ["--skip-sim", "--task", "accelerate"])
        self.assertEqual(argv[-4], "--no-llvm-ir")


class SweepPresetTest(unittest.TestCase):
    """run_sweep.py: every suite benchmark once, in order, with -- passthrough."""

    def _fixture(self, root: Path, benchmarks=("beta", "alpha", "gamma")) -> None:
        cfg = root / "agents" / "llm_configs"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "test-a.json").write_text(json.dumps({"name": "test-a", "model": "TEST-A"}))
        (cfg / "deepseek-pro.json").write_text(json.dumps({"name": "deepseek-pro", "model": "PRO"}))
        for name in benchmarks:
            (root / "benchmarks" / "embench-iot" / "src" / name).mkdir(parents=True, exist_ok=True)

    def _sweep(self, root: Path, *extra: str):
        import contextlib
        import io

        out_root = root / "out"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = run_sweep.main([
                "--exp-id", "sweep-test",
                "--workdir", str(root), "--out-root", str(out_root),
                "--dry-run", "--no-sbt-lock",
                *extra,
            ])
        return code, stdout.getvalue(), out_root

    @staticmethod
    def _run_lines(text: str) -> list[str]:
        return [ln for ln in text.splitlines() if ln.startswith("[dry ] sweep-test-")]

    def test_sweeps_all_suite_benchmarks_in_order_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            self._fixture(root)
            code, text, out_root = self._sweep(
                root, "--models", "test-a", "--", "--skip-sim", "--skip-yosys")
            self.assertEqual(code, 0)
            lines = self._run_lines(text)
            # Suite (sorted) order, exactly one run per benchmark; the summary
            # line ("[dry ] experiment.yaml -> ...") is not a run line.
            self.assertEqual([re.search(r"\[dry \] (\S+):", ln).group(1) for ln in lines], [
                "sweep-test-test-a-alpha-r01-mi5",
                "sweep-test-test-a-beta-r01-mi5",
                "sweep-test-test-a-gamma-r01-mi5",
            ])
            self.assertTrue(all(ln.endswith("--skip-sim --skip-yosys") for ln in lines))
            import yaml
            meta = yaml.safe_load((out_root / "sweep-test" / "experiment.yaml").read_text())
            self.assertEqual(meta["repeats"], 1)
            self.assertEqual(meta["agent_args"], ["--skip-sim", "--skip-yosys"])
            self.assertEqual(meta["benchmarks"], ["alpha", "beta", "gamma"])

    def test_explicit_benchmarks_keep_their_given_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            self._fixture(root)
            code, text, _ = self._sweep(root, "--models", "test-a",
                                        "--benchmarks", "gamma,alpha")
            self.assertEqual(code, 0)
            lines = self._run_lines(text)
            self.assertEqual(len(lines), 2)
            self.assertIn("-gamma-r01-mi5", lines[0])
            self.assertIn("-alpha-r01-mi5", lines[1])

    def test_default_model_is_a_single_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            self._fixture(root)
            code, _, out_root = self._sweep(root)
            self.assertEqual(code, 0)
            import yaml
            meta = yaml.safe_load((out_root / "sweep-test" / "experiment.yaml").read_text())
            self.assertEqual(meta["models"], ["deepseek-pro"])
            self.assertEqual(meta["benchmarks"], ["alpha", "beta", "gamma"])

    def test_empty_suite_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            self._fixture(root, benchmarks=())
            with self.assertRaises(SystemExit):
                self._sweep(root, "--models", "test-a")

    def test_reserved_passthrough_flag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            self._fixture(root)
            with self.assertRaises(SystemExit):
                self._sweep(root, "--models", "test-a", "--", "--run-id", "x")

    def test_passthrough_ablation_switch_is_folded_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            self._fixture(root, benchmarks=("alpha",))
            code, text, out_root = self._sweep(
                root, "--models", "test-a", "--", "--no-isa-contract", "--skip-sim")
            self.assertEqual(code, 0)
            (line,) = self._run_lines(text)
            # --no-isa-contract implies --no-conformance: both land exactly once
            # (folded into the job's ablation flags), the plain tail follows.
            for flag in ("--no-isa-contract", "--no-conformance"):
                self.assertEqual(line.count(flag), 1, line)
            self.assertTrue(line.endswith("--skip-sim"), line)
            import yaml
            meta = yaml.safe_load((out_root / "sweep-test" / "experiment.yaml").read_text())
            self.assertEqual(meta["agent_args"], ["--no-isa-contract", "--skip-sim"])


class SimulationClassificationRegressionTest(unittest.TestCase):
    def test_legacy_wrong_result_precedes_measured_floor(self):
        for text in ("Embench verify failed (wrong-result failure)",
                     "the CFU build computed result=1 but the reference run computed result=2: observable result",
                     "result_mismatch"):
            record = _failure_record("sim", {"ok": False, "errors": [text,
                "the CFU build is not meaningfully faster than the reference"]},
                reference_cycles=100, cfu_cycles=200, speedup_x100=50, min_speedup_x100=105)
            self.assertEqual(rm.classify_failure(record)["failure_class"], "sim_correctness")

    def test_typed_terminal_reason_ignores_historical_error_strings(self):
        cases = (("result_mismatch", "sim_correctness"), ("assertion", "sim_assertion"),
                 ("no_timing", "sim_no_timing"), ("speedup_floor", "sim_speedup_floor"))
        for reason, expected in cases:
            with self.subTest(reason=reason):
                record = _failure_record("sim", {"ok": False, "terminal_reason": reason,
                    "errors": ["OLD timeout; verify failed; not meaningfully faster"]},
                    speedup_x100=0, min_speedup_x100=105)
                self.assertEqual(rm.classify_failure(record)["failure_class"], expected)

    def test_numeric_floor_without_valid_measurements_is_not_performance(self):
        for cycles in (None, 0):
            record = _failure_record("sim", {"ok": False, "errors": ["too slow"]},
                reference_cycles=100, cfu_cycles=cycles, speedup_x100=0, min_speedup_x100=105)
            self.assertNotEqual(rm.classify_failure(record)["failure_class"], "sim_speedup_floor")

    def test_hardware_integrity_has_its_own_class(self):
        record = _failure_record("hw_build", {"ok": False, "terminal_reason": "hardware_integrity",
                                              "errors": ["forged pass/fail output"]})
        self.assertEqual(rm.classify_failure(record)["failure_class"], "hardware_integrity")

    def test_explicit_repair_route_stop_is_classified_without_using_historical_route_gate(self):
        record = _failure_record("implementation_route", {"ok": False,
            "reason": "repair_route_budget_exhausted", "errors": ["implementation route budget exhausted"]})
        self.assertEqual(rm.classify_failure(record)["failure_class"], "repair_route_budget")
        record["gates"]["implementation_route"]["reason"] = "contract invariant contradicts the measurement"
        record["gates"]["implementation_route"]["errors"] = []
        self.assertEqual(rm.classify_failure(record)["failure_class"], "repair_route_budget")
        record = _failure_record("sim", {"ok": False, "terminal_reason": "result_mismatch"})
        record["gates"]["implementation_route"] = {"ok": False, "reason": "old route budget exhausted"}
        self.assertEqual(rm.classify_failure(record)["failure_class"], "sim_correctness")
        record["stopped_at"] = ""
        self.assertEqual(rm.classify_failure(record)["failure_class"], "sim_correctness")

    def test_errors_only_hardware_gate_does_not_become_no_deliverable(self):
        record = _failure_record("implementation_hw", {"ok": False,
            "errors": ["hardware integrity: design:17: hardware_integrity/fixed_compute_result"]})
        self.assertEqual(rm.classify_failure(record)["failure_class"], "hardware_integrity")

    def test_native_hardware_rejection_fallback_uses_current_rejection_only(self):
        for final, expected in (("hardware integrity: hardware_integrity/fixed_compute_result", "hardware_integrity"),
                                ("missing EDIT_END", "no_deliverable")):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as tmp:
                run_dir, fw = CollectTest()._fixture(Path(tmp), "", "", RUN_YAML_PASS, "")
                _write(run_dir / "stderr.log", "OLD hardware_integrity/fixed_compute_result\n"
                    "RuntimeError: native stage failed: independent submit_stage gate rejected delivery (" + final + ")\n")
                record = rm.collect_record("e", {"benchmark": "xgboost"}, "r", run_dir, fw, {"exit_code": 1})
                self.assertEqual(record["failure_class"], expected)
                self.assertEqual(record["status"], "failed")
                self.assertFalse(record["infra"]["infra"])
                self.assertIn(final, record["stage_gate_detail"])

    def test_typed_collection_does_not_backfill_null_from_cli_or_old_embench(self):
        import yaml
        for reason, expected in (("assertion", "sim_assertion"), ("result_mismatch", "sim_correctness")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                report = REPORT_PASS.replace("Status: complete", "Status: failed").replace(
                    "- sim: {'ok': True", "- sim: {'ok': False")
                run_dir, fw = CollectTest()._fixture(Path(tmp), "status=failed\nstopped_at=sim\n" + STDOUT_PASS,
                    report, RUN_YAML_PASS, EMBENCH_PASS)
                _write(fw / "10_validation/sim.yaml", yaml.safe_dump({"status": "failed",
                    "terminal_reason": reason, "correctness_ok": False, "timing_ok": False,
                    "performance_valid": False, "reference_cycles": 100, "cfu_cycles": None,
                    "speedup_x100": None, "errors": ["current failure"], "commands": []}))
                record = rm.collect_record("e", {"benchmark": "nsichneu"}, "r", run_dir, fw,
                                           {"exit_code": 1})
                self.assertEqual(record["failure_class"], expected)
                self.assertIsNone(record["cfu_cycles"])
                self.assertIsNone(record["speedup_x100"])
                self.assertFalse(record["performance_valid"])
                self.assertEqual(record["sim_terminal_reason"], reason)
                rm.write_runs_csv([record], Path(tmp) / "runs.csv")
                with (Path(tmp) / "runs.csv").open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["cfu_cycles"], "")
                self.assertEqual(row["speedup_x100"], "")
                self.assertEqual(row["sim_terminal_reason"], reason)

    def test_legacy_collector_wrong_and_slow_nulls_invalid_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = REPORT_PASS.replace("Status: complete", "Status: failed").replace(
                "- sim: {'ok': True", "- sim: {'ok': False")
            embench = EMBENCH_PASS.replace("attribution: ''", "attribution: 'Embench verify failed'").replace(
                "speedup_x100: 105", "speedup_x100: 50")
            run_dir, fw = CollectTest()._fixture(Path(tmp), "status=failed\nstopped_at=sim\n",
                                                report, RUN_YAML_PASS, embench)
            record = rm.collect_record("e", {"benchmark": "picojpeg"}, "r", run_dir, fw, {"exit_code": 1})
            self.assertEqual(record["failure_class"], "sim_correctness")
            self.assertIsNone(record["speedup_x100"])
            self.assertEqual(record["observed_speedup_x100"], 50)

    def test_host_only_gate_failure_does_not_count_as_simulation_attempt(self):
        import yaml
        from agents.experiments.harness import collection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "10_validation/sim.yaml", yaml.safe_dump({"terminal_reason": "benchmark_integrity",
                "commands": [{"kind": "host_gate_check", "returncode": 1, "rung": "sim"}]}))
            self.assertEqual(collection.collect_sim_attempts(root, []), [])
            _write(root / "10_validation/sim.yaml", yaml.safe_dump({"terminal_reason": "result_mismatch",
                "commands": [{"command": "mock-sim", "returncode": 0},
                             {"kind": "host_gate_check", "returncode": 1}]}))
            self.assertEqual(len(collection.collect_sim_attempts(root, [])), 1)

    def test_metrics_excludes_invalid_performance_even_if_status_is_complete(self):
        records = [{"status": "complete", "speedup_x100": 200, "performance_valid": True},
                   {"status": "complete", "speedup_x100": 900, "performance_valid": False},
                   {"status": "complete", "speedup_x100": 700, "correctness_ok": False}]
        self.assertEqual(rm.cell_metrics(records, bootstrap_iters=10)["speedup_x100_median"], 200)


if __name__ == "__main__":
    unittest.main()
