"""Tests for the profile-only partial run (``--profile-only``).

The mode exists so a repository change can rebuild the Embench measurement
cache without spending a model call: these tests pin the prefix that runs, the
fail-closed stop and the machine-readable line the harness parses. No build,
simulation or model call happens here.
"""

from __future__ import annotations

import contextlib
import io
import unittest

from agents.cfu_design_agent import cli as agent_cli
from agents.cfu_design_agent import graph


class _FakeFlow:
    """A stand-in for ``python_nodes`` that records the prefix it is asked for."""

    def __init__(self, *, fail_at: str = "") -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.state_seen: list[dict] = []

    def __call__(self) -> dict:
        return {name: self._node(name) for name in graph.PROFILE_ONLY_NODES}

    def _node(self, name: str):
        def node(state):
            self.calls.append(name)
            self.state_seen.append(dict(state))
            update: dict = {"stage_jobs": "next"}
            rung = graph._PROFILE_ONLY_RUNGS.get(name)
            if rung:
                update["gates"] = {
                    rung: {"ok": name != self.fail_at, "skipped": False, "next": "extract_hotspot"},
                }
            if name == "load_context":
                # The real node creates the run root; later nodes read it from
                # the state, so the fake has to seed it here.
                update["run_id"] = "rawtest-raw-crc32"
                update["run_root"] = "agents/generated/raw/runs/rawtest-raw-crc32"
            if name == "ingest_input":
                update["errors"] = ["workload-spec-note"]
            if name == "profile_embench":
                update["gates"]["profile"]["profile_cache"] = {
                    "status": "refresh", "saved": True, "key": "a" * 64,
                }
                update["embench_profiles"] = {"reference": {"crc32": {"cycles": 91262}}}
            return update
        return node


class ProfileOnlyRunTest(unittest.TestCase):
    def _run(self, flow: _FakeFlow, state: dict | None = None) -> dict:
        original = graph.python_nodes
        try:
            graph.python_nodes = flow
            return graph.run_profile_only(state if state is not None else {
                "workdir": ".", "benchmark_name": "crc32", "errors": ["preexisting"],
            })
        finally:
            graph.python_nodes = original

    def test_runs_the_measurement_prefix_in_flow_order(self) -> None:
        flow = _FakeFlow()
        final = self._run(flow)
        self.assertEqual(flow.calls, list(graph.PROFILE_ONLY_NODES))
        # Nothing that needs a model may run.
        self.assertNotIn("design_contract", flow.calls)
        self.assertNotIn("implement", flow.calls)
        self.assertNotIn("final_report", flow.calls)
        self.assertTrue(final.get("profile_only"))
        self.assertEqual(final.get("stopped_at"), "")
        self.assertEqual(final["gates"]["profile"]["profile_cache"]["key"], "a" * 64)

    def test_the_state_carries_earlier_updates_into_later_nodes(self) -> None:
        flow = _FakeFlow()
        final = self._run(flow)
        profile_state = flow.state_seen[-1]
        self.assertEqual(profile_state.get("run_root"), "agents/generated/raw/runs/rawtest-raw-crc32")
        # Append-reducer channels accumulate instead of replacing (the merge
        # must mirror LangGraph, or a node's ledger silently drops).
        self.assertIn("workload-spec-note", final["errors"])
        self.assertIn("preexisting", final["errors"])

    def test_a_failed_precondition_stops_before_the_measurement(self) -> None:
        flow = _FakeFlow(fail_at="prepare_embench_benchmark")
        final = self._run(flow)
        self.assertEqual(flow.calls[-1], "prepare_embench_benchmark")
        self.assertNotIn("profile_embench", flow.calls)
        self.assertEqual(final.get("stopped_at"), "prepare_benchmark")

    def test_a_benchmark_is_required(self) -> None:
        with self.assertRaises(ValueError):
            self._run(_FakeFlow(), {"workdir": "."})


class ProfileOnlyCliTest(unittest.TestCase):
    def _state(self, *, ok: bool = True) -> dict:
        return {
            "benchmark_name": "crc32",
            "run_root": "agents/generated/raw/runs/rawtest-raw-crc32",
            "embench_profiles": {"reference": {"crc32": {"cycles": 91262}}},
            "errors": [],
            "gates": {"profile": {
                "ok": ok, "skipped": False,
                "profile_cache": {"status": "refresh", "saved": ok, "key": "b" * 64},
            }},
        }

    def _call(self, state: dict) -> tuple[int, str]:
        original = agent_cli.run_profile_only
        out = io.StringIO()
        try:
            agent_cli.run_profile_only = lambda s: state
            with contextlib.redirect_stdout(out):
                code = agent_cli._run_profile_only({"benchmark_name": "crc32"})
        finally:
            agent_cli.run_profile_only = original
        return code, out.getvalue()

    def test_prints_the_line_the_harness_parses(self) -> None:
        code, text = self._call(self._state())
        self.assertEqual(code, 0)
        line = next(line for line in text.splitlines() if line.startswith("profile_only "))
        fields = dict(token.split("=", 1) for token in line.split()[1:])
        self.assertEqual(fields["benchmark"], "crc32")
        self.assertEqual(fields["ok"], "true")
        self.assertEqual(fields["cache"], "refresh")
        self.assertEqual(fields["saved"], "true")
        self.assertEqual(fields["key"], "b" * 64)
        self.assertEqual(fields["reference_cycles"], "91262")

    def test_a_failed_profile_exits_nonzero(self) -> None:
        code, text = self._call(self._state(ok=False))
        self.assertEqual(code, 1)
        self.assertIn("ok=false", text)

    def test_an_exception_is_reported_as_an_agent_error(self) -> None:
        original = agent_cli.run_profile_only
        err = io.StringIO()
        try:
            def boom(state):
                raise RuntimeError("sbt exploded")
            agent_cli.run_profile_only = boom
            with contextlib.redirect_stderr(err):
                code = agent_cli._run_profile_only({"benchmark_name": "crc32"})
        finally:
            agent_cli.run_profile_only = original
        self.assertEqual(code, 1)
        self.assertIn("agent-error: RuntimeError: sbt exploded", err.getvalue())


class ProfileOnlyCliValidationTest(unittest.TestCase):
    def test_requires_a_benchmark_and_rejects_dry_run(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                agent_cli.main(["--profile-only"])
            with self.assertRaises(SystemExit):
                agent_cli.main(["--profile-only", "--embench", "crc32", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
