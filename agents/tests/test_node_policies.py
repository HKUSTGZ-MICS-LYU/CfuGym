"""The declarative per-node policy table is actually applied, and it is safe.

A policy that is declared but not registered is worse than no policy: it reads
as protection that does not exist. These tests assert the registration and the
one behaviour that matters, that an *unexpected* exception is not silently
converted into a repairable rung.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph

from agents.cfu_design_agent.state import CfuDesignState

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.flow import ladder, policies
from agents.cfu_design_agent.runtime.policy import PolicyError
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


class PolicyRegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = graph.build_graph()

    def test_retry_policy_is_registered_only_where_declared(self) -> None:
        declared = {name for name, options in policies.NODE_OPTIONS.items() if "retry_policy" in options}
        for name in graph.NODE_NAMES:
            with self.subTest(node=name):
                effective = bool(getattr(self.app.nodes[name], "retry_policy", None))
                self.assertEqual(effective, name in declared)

    def test_no_model_or_eda_node_carries_a_retry_policy(self) -> None:
        """A node retry re-runs the whole node, so it must not re-spend budget."""
        expensive = {
            "analyze_workload", "design_contract", "implement",
            "generate_soc", "check_conformance", "build_sw",
            "simulate", "estimate_cost", "review_and_route", "profile_embench",
            "instrument_and_profile",
        }
        for name in sorted(expensive):
            with self.subTest(node=name):
                self.assertFalse(bool(getattr(self.app.nodes[name], "retry_policy", None)), name)

    def test_error_handler_is_registered_only_for_the_prepare_rungs(self) -> None:
        with_handler = {
            name for name in graph.NODE_NAMES
            if getattr(self.app.nodes[name], "error_handler_node", None)
        }
        self.assertEqual(
            with_handler,
            {"prepare_c_project_profile", "create_benchmark_project", "prepare_embench_benchmark"},
        )

    def test_an_undeclared_flow_node_is_rejected(self) -> None:
        import agents.cfu_design_agent.flow.wiring as wiring

        with self.assertRaisesRegex(ValueError, "unknown flow nodes"):
            wiring.register_langgraph(
                _Recorder(), graph.python_nodes(),
                {name: getattr(graph, symbol) for name, symbol in wiring.ROUTER_NAMES.items()},
                wrap=lambda fn: fn, start=None, end=None,
                node_options={"not_a_node": {"retry_policy": None}},
            )


class _Recorder:
    """Minimal stand-in for a StateGraph: the options check must run first."""

    nodes: dict = {}

    def add_node(self, *args, **kwargs):
        raise AssertionError("registration must not be reached")


class PrepareFailureHandlerTest(unittest.TestCase):
    """The handler keeps the narrow caught set the nodes used to have."""

    def _handle(self, state, error) -> dict:
        command = ladder.prepare_failure_handler(state, error)
        self.assertEqual(command.goto, "review_and_route")
        self.assertIsInstance(command.update, dict)
        return command.update

    @staticmethod
    def _state() -> dict:
        return {"gates": {}, "run_root": "agents/generated/runs/does-not-exist"}

    def test_an_accepted_failure_becomes_the_prepare_gate(self) -> None:
        update = self._handle(
            self._state(), NodeError("prepare_embench_benchmark", PolicyError("denied read")),
        )
        gate = update["gates"]["prepare_benchmark"]
        self.assertFalse(gate["ok"])
        self.assertEqual(gate["next"], "review_and_route")
        self.assertIn("denied read", gate["errors"][0])
        self.assertEqual(update["stopped_at"], "prepare_benchmark")
        self.assertEqual(update["stage_jobs"], "review")

    def test_a_value_error_is_also_an_attributable_failure(self) -> None:
        update = self._handle(
            self._state(), NodeError("create_benchmark_project", ValueError("bad spec")),
        )
        self.assertFalse(update["gates"]["prepare_benchmark"]["ok"])

    def test_a_failure_report_never_mints_a_run_identity(self) -> None:
        runs = REPO_ROOT / "agents/generated/runs"
        before = {path.name for path in runs.glob("*")} if runs.is_dir() else set()
        self._handle(
            self._state(), NodeError("prepare_embench_benchmark", PolicyError("denied read")),
        )
        after = {path.name for path in runs.glob("*")} if runs.is_dir() else set()
        self.assertEqual(before, after)

    def test_a_named_run_records_the_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "agents/specs/agent_access_policy.yaml"
            policy.parent.mkdir(parents=True)
            shutil.copyfile(REPO_ROOT / "agents/specs/agent_access_policy.yaml", policy)
            run_id = "prepare-failure-00000000"
            run_dir = root / f"agents/generated/runs/{run_id}"
            update = self._handle(
                {
                    "gates": {},
                    "workdir": str(root),
                    "run_id": run_id,
                    "run_root": run_dir.relative_to(root).as_posix(),
                },
                NodeError("create_benchmark_project", ValueError("bad spec")),
            )
            self.assertFalse(update["gates"]["prepare_benchmark"]["ok"])
            manifest = run_dir / "04_benchmark/manifest.yaml"
            self.assertTrue(manifest.is_file())
            self.assertIn("bad spec", manifest.read_text())

    def test_an_unexpected_exception_is_re_raised_not_repaired(self) -> None:
        for exc in (TypeError("bug"), AttributeError("bug"), KeyError("bug")):
            with self.subTest(exc=type(exc).__name__):
                with self.assertRaises(type(exc)):
                    self._handle(
                        self._state(), NodeError("create_benchmark_project", exc),
                    )


class PrepareFailureRuntimeTest(unittest.TestCase):
    """Exercise real LangGraph injection/routing, not a direct callback call."""

    @staticmethod
    def _graph(name: str, exc: Exception):
        def prepare(state):
            raise exc

        def review(state):
            return {"status": "failed"}

        builder = StateGraph(CfuDesignState)
        builder.add_node(name, prepare, **policies.node_options()[name])
        builder.add_node("review_and_route", review)
        builder.add_edge(START, name)
        builder.add_conditional_edges(name, ladder._route_after_prepare,
                                      {"review_and_route": "review_and_route"})
        builder.add_edge("review_and_route", END)
        return builder.compile()

    def test_accepted_failures_in_all_prepare_nodes_reach_review(self) -> None:
        for name, options in policies.node_options().items():
            if "error_handler" not in options:
                continue
            for exc in (PolicyError("denied source"), ValueError("bad spec"), OSError("missing input")):
                with self.subTest(node=name, error=type(exc).__name__):
                    result = self._graph(name, exc).invoke({"gates": {}})
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["stopped_at"], "prepare_benchmark")
                    self.assertFalse(result["gates"]["prepare_benchmark"]["ok"])
                    self.assertEqual(result["gates"]["prepare_benchmark"]["node"], name)
                    self.assertIn(str(exc), result["errors"][0])
                    self.assertIn(name, result["errors"][0])

    def test_unexpected_exceptions_are_not_masked_by_the_handler(self) -> None:
        for name, options in policies.node_options().items():
            if "error_handler" not in options:
                continue
            for exc in (TypeError("node bug"), AttributeError("node bug"), KeyError("node bug")):
                with self.subTest(node=name, error=type(exc).__name__):
                    with self.assertRaises(type(exc)) as caught:
                        self._graph(name, exc).invoke({"gates": {}})
                    self.assertIs(caught.exception, exc)


if __name__ == "__main__":
    unittest.main()
