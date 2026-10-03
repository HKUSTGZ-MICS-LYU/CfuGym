"""Fast, table-driven coverage of the ladder's routing decisions.

A router is a pure ``state -> node name`` function. Testing the decision table
directly is both cheaper and sharper than driving the whole graph: a full-graph
test can only observe the route it happened to take, while these cases pin every
branch, including the ones a happy-path run never reaches.

The end-to-end smoke test in ``test_graph_smoke`` covers wiring and artifacts.
"""

from __future__ import annotations

import unittest
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.flow import outcome, wiring
from agents.cfu_design_agent.flow import ladder
from agents.cfu_design_agent.runtime import llm_step
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


def state(**overrides) -> dict:
    base = {"dry_run": True, "gates": {}, "max_iters": 1, "attempts": {}}
    base.update(overrides)
    return base


class RoutingTableTest(unittest.TestCase):
    """Each rung either honors its own ``next`` or falls back to the default."""

    RUNGS = {
        "hw_build": (graph._route_after_hw_build, "check_conformance"),
        "sw_build": (graph._route_after_sw_build, "simulate"),
        "sim": (graph._route_after_sim, "estimate_cost"),
        "cost": (graph._route_after_cost, "review_and_route"),
        "review": (graph._route_after_review, "final_report"),
    }

    def test_gate_next_wins_over_the_default(self) -> None:
        for rung, (router, _default) in self.RUNGS.items():
            with self.subTest(rung=rung):
                self.assertEqual(router(state(gates={rung: {"next": "implement_hw"}})), "implement_hw")

    def test_default_is_used_without_a_gate(self) -> None:
        for rung, (router, default) in self.RUNGS.items():
            with self.subTest(rung=rung):
                self.assertEqual(router(state()), default)
                self.assertEqual(router(state(gates={rung: {}})), default)

    def test_empty_next_falls_back(self) -> None:
        self.assertEqual(graph._route_after_sim(state(gates={"sim": {"next": ""}})), "estimate_cost")

    def test_isa_default_is_the_single_implementation_stage(self) -> None:
        self.assertEqual(graph._route_after_isa(state()), "implement")
        self.assertEqual(graph._route_after_isa(state(stage3_autonomy=True)), "implement")
        self.assertEqual(
            graph._route_after_isa(state(stage3_autonomy=True, gates={"isa": {"next": "extend_isa"}})),
            "extend_isa",
        )


class PrepareRoutingTest(unittest.TestCase):
    def test_input_kind_selects_the_preparation_node(self) -> None:
        for kind, target in (
            ("embench", "prepare_embench_benchmark"),
            ("c_project", "prepare_c_project_profile"),
            ("natural_language", "create_benchmark_project"),
            ("unknown-kind", "create_benchmark_project"),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(graph._route_input_target(state(input_kind=kind)), target)

    def test_input_mode_is_used_when_no_kind_was_resolved(self) -> None:
        self.assertEqual(graph._route_input_target(state(input_mode="embench")), "prepare_embench_benchmark")

    def test_prepare_router_defaults_to_review(self) -> None:
        self.assertEqual(graph._route_after_prepare(state()), "review_and_route")
        self.assertEqual(
            graph._route_after_prepare(state(gates={"prepare_benchmark": {"next": "instrument_and_profile"}})),
            "instrument_and_profile",
        )


class FlowLayoutTest(unittest.TestCase):
    """The graph the runner compiles is the workflow specification."""

    def test_compiled_graph_has_exactly_the_specified_nodes(self) -> None:
        app = graph.build_graph()
        nodes = set(app.get_graph().nodes)
        self.assertTrue(set(wiring.NODE_NAMES) <= nodes, sorted(set(wiring.NODE_NAMES) - nodes))
        self.assertIn("__start__", nodes)
        self.assertIn("__end__", nodes)

    def test_every_router_and_target_resolves(self) -> None:
        """A wiring entry naming a missing symbol must fail here, not at runtime."""
        for node, symbol in wiring.ROUTER_NAMES.items():
            with self.subTest(node=node):
                self.assertTrue(callable(getattr(graph, symbol, None)), f"{node} -> {symbol}")

    def test_final_report_terminates_the_flow(self) -> None:
        self.assertIsNone(wiring.SUCCESSORS["final_report"])


if __name__ == "__main__":
    unittest.main()


class SingleImplementationStageTest(unittest.TestCase):
    """The collapse: one implementation node, and every repair routes into it."""

    def test_the_graph_has_exactly_one_implementation_node(self) -> None:
        impl = sorted(name for name in graph.NODE_NAMES if "implement" in name)
        self.assertEqual(impl, ["implement"],
                         "the hardware/software rungs must not come back as separate nodes")

    def test_an_implementation_stage_cannot_route_to_a_retired_node(self) -> None:
        """A stale target must fail loudly instead of stranding the run."""
        with self.assertRaisesRegex(ValueError, "unknown node"):
            outcome.gate({}, "sim", ok=False, next="implement_hw")

    def test_a_failing_rung_with_budget_repairs_the_implementation_stage(self) -> None:
        from agents.cfu_design_agent.flow import ladder

        for rung in ("isa", "hw_build", "hw_conformance", "sw_build", "sim", "cost"):
            with self.subTest(rung=rung):
                current = state(gates={rung: {"ok": False, "skipped": False}},
                                repairs_used=0, max_iters=3)
                extra, target = ladder._ladder_transition(
                    current, rung, ok=False, default="review_and_route",
                    repair_target="implement",
                )
                self.assertEqual(target, "implement")
                self.assertEqual(extra["stage_jobs"], "implement")

    def test_a_failing_rung_without_budget_stops_instead(self) -> None:
        from agents.cfu_design_agent.flow import ladder

        current = state(gates={"sim": {"ok": False}}, repairs_used=1, max_iters=2)
        _extra, target = ladder._ladder_transition(
            current, "sim", ok=False, default="estimate_cost", repair_target="implement",
        )
        self.assertEqual(target, "review_and_route")

    def test_a_passing_rung_keeps_its_static_successor(self) -> None:
        from agents.cfu_design_agent.flow import ladder

        current = state(gates={"sim": {"ok": True}})
        _extra, target = ladder._ladder_transition(
            current, "sim", ok=True, default="estimate_cost", repair_target="implement",
        )
        self.assertEqual(target, "estimate_cost")
