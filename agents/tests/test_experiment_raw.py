"""Tests for the sweep's ``--raw`` cache-warm mode (harness/raw.py + preset).

Free of builds, simulations and model calls: the runner seam
(``harness.execution.run_logged``) is replaced, so these tests only assert the
command that would be launched, the parsing of its summary and the bookkeeping
around it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from agents.experiments import run_ablation, run_matrix, run_sweep
from agents.experiments.harness import cli as harness_cli
from agents.experiments.harness import execution, raw


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _summary_line(benchmark: str = "crc32", *, ok: bool = True, saved: bool = True) -> str:
    return (
        f"profile_only benchmark={benchmark} ok={str(ok).lower()} cache=refresh "
        f"saved={str(saved).lower()} key={'a' * 64} reference_cycles=91262 "
        f"run_root=agents/generated/runs/x\n"
    )


class RawArgvTest(unittest.TestCase):
    def test_the_command_rebuilds_the_measurement_and_stops_before_the_design(self) -> None:
        argv = raw.build_argv("python3 -B", "", None, "crc32", "sweep-raw-crc32")
        self.assertIn("--profile-only", argv)
        self.assertEqual(argv[argv.index("--embench") + 1], "crc32")
        self.assertEqual(argv[argv.index("--run-id") + 1], "sweep-raw-crc32")
        self.assertNotIn("--out-dir", argv, "the agent's own default is the allowlisted tree")
        # A warm-up must not be served by the stale entry it means to replace.
        self.assertEqual(argv[argv.index("--profile-cache") + 1], "refresh")
        # No model and no ladder: those flags would make this an experiment run.
        self.assertNotIn("--llm-config", argv)
        self.assertNotIn("--max-iters", argv)

    def test_pass_through_flags_survive_because_they_shape_the_cache_key(self) -> None:
        argv = raw.build_argv(
            "python3", "", None, "crc32", "r", ["--target-frequency-mhz", "200"],
        )
        self.assertEqual(argv[-2:], ["--target-frequency-mhz", "200"])

    def test_flags_that_would_skip_the_measured_build_are_reported(self) -> None:
        self.assertEqual(raw.blocked_agent_args(["--skip-sim", "--task", "x"]), ["--skip-sim"])
        self.assertEqual(raw.blocked_agent_args(["--skip-sim=1"]), ["--skip-sim"])
        self.assertEqual(raw.blocked_agent_args(["--no-dry-run"]), ["--no-dry-run"])
        self.assertEqual(raw.blocked_agent_args(["--skip-yosys"]), [])

    def test_the_run_id_never_looks_like_a_scheduled_run(self) -> None:
        self.assertEqual(raw.raw_run_name("sweep-full", "matmult-int"),
                         "sweep-full-raw-matmultint")


class SummaryParseTest(unittest.TestCase):
    def test_reads_the_last_profile_only_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stdout.log"
            _write(log, "noise\n" + _summary_line("aha-mont64"))
            fields = raw.parse_summary(log)
            self.assertEqual(fields["benchmark"], "aha-mont64")
            self.assertEqual(fields["saved"], "true")
            self.assertEqual(fields["key"], "a" * 64)
            self.assertEqual(fields["reference_cycles"], "91262")

    def test_a_missing_line_is_an_empty_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stdout.log"
            _write(log, "status=blocked\n")
            self.assertEqual(raw.parse_summary(log), {})
            self.assertEqual(raw.parse_summary(Path(tmp) / "missing.log"), {})


class WarmTest(unittest.TestCase):
    def _opts(self, root: Path, benchmarks, **overrides) -> argparse.Namespace:
        values = dict(
            exp_id="rawtest", benchmarks=list(benchmarks), workdir=root,
            python="python3", conda_env="", timeout=60, agent_forward=(),
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def _runner(self, calls, *, exit_code=0, summary=None):
        def run(argv, out_path, err_path, cwd, env, timeout):
            calls.append({"name": out_path.parent.name, "argv": argv})
            _write(out_path, "noise\n" + (summary or _summary_line()))
            _write(err_path, "")
            return {"exit_code": exit_code, "wall_seconds": 1.5, "timed_out": False}
        return run

    def test_every_benchmark_is_warmed_in_order_with_its_own_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "agents/generated/experiments/rawtest"
            calls: list[dict] = []
            original = execution.run_logged
            try:
                execution.run_logged = self._runner(calls)
                code = raw.warm(self._opts(root, ["crc32", "aha-mont64"]), exp_dir, None, {})
            finally:
                execution.run_logged = original
            self.assertEqual(code, 0)
            self.assertEqual(
                [call["name"] for call in calls],
                [raw.raw_run_name("rawtest", "crc32"), raw.raw_run_name("rawtest", "aha-mont64")],
            )
            first = exp_dir / "raw" / calls[0]["name"]
            self.assertTrue((first / "cmd.txt").is_file())
            self.assertTrue((first / "stdout.log").is_file())
            self.assertIn("--profile-only", (first / "cmd.txt").read_text())
            # Cache maintenance must not create or overwrite an experiment.
            self.assertFalse((exp_dir / "experiment.yaml").exists())
            self.assertFalse((exp_dir / "runs").exists())

    def test_one_failure_does_not_hide_the_rest_and_sets_the_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "agents/generated/experiments/rawtest"
            calls: list[dict] = []
            ok_runner = self._runner(calls)
            bad_runner = self._runner(calls, exit_code=1, summary=_summary_line(ok=False, saved=False))

            def runner(argv, out_path, err_path, cwd, env, timeout):
                run = bad_runner if "crc32" in out_path.parent.name else ok_runner
                return run(argv, out_path, err_path, cwd, env, timeout)

            original = execution.run_logged
            stdout = io.StringIO()
            try:
                execution.run_logged = runner
                with contextlib.redirect_stdout(stdout):
                    code = raw.warm(self._opts(root, ["crc32", "aha-mont64"]), exp_dir, None, {})
            finally:
                execution.run_logged = original
            self.assertEqual(code, 1)
            self.assertEqual(len(calls), 2, "the second benchmark is still warmed")
            self.assertIn("FAILED crc32", stdout.getvalue())
            self.assertIn("cached 1/2", stdout.getvalue())

    def test_a_timeout_counts_as_a_failed_warm_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "agents/generated/experiments/rawtest"
            calls: list[dict] = []

            def runner(argv, out_path, err_path, cwd, env, timeout):
                calls.append({"name": out_path.parent.name})
                _write(out_path, _summary_line())
                _write(err_path, "")
                return {"exit_code": None, "wall_seconds": 60.0, "timed_out": True}

            original = execution.run_logged
            try:
                execution.run_logged = runner
                with contextlib.redirect_stdout(io.StringIO()):
                    code = raw.warm(self._opts(root, ["crc32"]), exp_dir, None, {})
            finally:
                execution.run_logged = original
            self.assertEqual(code, 1)
            self.assertEqual(len(calls), 1)


class RawCliTest(unittest.TestCase):
    def test_raw_modes_are_rejected_where_they_make_no_sense(self) -> None:
        base = dict(
            aggregate_only=False, collect_only=False, dry_run=False, agent_forward=(),
        )
        for broken in ({"aggregate_only": True}, {"collect_only": True}, {"dry_run": True},
                       {"agent_forward": ("--skip-sim",)}):
            with self.subTest(broken=broken), self.assertRaises(SystemExit):
                harness_cli._validate_raw(argparse.Namespace(**{**base, **broken}))
        harness_cli._validate_raw(argparse.Namespace(**base))

    def test_only_presets_that_opt_in_accept_raw(self) -> None:
        self.assertIn("--raw", harness_cli.build_parser(supports_raw=True)._option_string_actions)
        self.assertNotIn("--raw", harness_cli.build_parser()._option_string_actions)
        for preset in (run_matrix, run_ablation):
            argv = ["--exp-id", "raw-parser", "--raw"]
            with self.subTest(preset=preset.__name__), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    preset.main(argv)

    def test_scheduling_knobs_cannot_reject_a_cache_rebuild(self) -> None:
        # --layers/--visibility describe a schedule this mode never builds, so a
        # ladder-style value must not turn a warm-up into a usage error.
        warmed: dict[str, list[str]] = {}
        original = raw.warm

        def fake_warm(opts, exp_dir, shim_dir, lock_env):
            warmed["benchmarks"] = list(opts.benchmarks)
            return 0

        try:
            raw.warm = fake_warm
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_sweep.main([
                    "--exp-id", "raw-knobs", "--raw", "--benchmarks", "crc32",
                    "--layers", "ladder", "--no-sbt-lock",
                ])
        finally:
            raw.warm = original
        self.assertEqual(code, 0)
        self.assertEqual(warmed.get("benchmarks"), ["crc32"])

    def test_the_sweep_preset_opts_in_and_the_others_do_not(self) -> None:
        captured: dict[str, dict] = {}
        original = harness_cli.main

        def fake(argv=None, **kwargs):
            captured[str(argv)] = kwargs
            return 0

        try:
            harness_cli.main = fake
            run_sweep.main(["--exp-id", "raw-preset", "--raw", "--benchmarks", "crc32"])
            run_matrix.main(["--exp-id", "raw-preset", "--benchmarks", "crc32"])
        finally:
            harness_cli.main = original
        sweep_kwargs = captured[str(["--exp-id", "raw-preset", "--raw", "--benchmarks", "crc32"])]
        matrix_kwargs = captured[str(["--exp-id", "raw-preset", "--benchmarks", "crc32"])]
        self.assertTrue(sweep_kwargs.get("supports_raw"))
        self.assertFalse(matrix_kwargs.get("supports_raw", False))


if __name__ == "__main__":
    unittest.main()
