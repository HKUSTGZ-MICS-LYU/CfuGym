"""The single Implement stage: mode dispatch, update merging, two gates.

The stage is one node with two ways to work (a tool loop, or the delivery
protocol when tools are unavailable). These tests pin the properties the rest of
the ladder relies on:

* the mode is chosen by capability, not by a flag;
* the two modes write the *same* two gates, so routing is unaffected;
* merging two nodes' updates concatenates the reducer channels instead of
  replacing them;
* a node reporting two gates has both of them re-stamped rather than reported
  stale.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent.flow import instrument, outcome
from agents.cfu_design_agent.runtime import llm_step
from agents.cfu_design_agent.stages import implement as implement_stage
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


class MergeUpdatesTest(unittest.TestCase):
    def test_reducer_channels_concatenate(self) -> None:
        first = {"errors": ["a"], "iteration_history": ["h1"], "llm_raw_outputs": ["r1"]}
        second = {"errors": ["b"], "iteration_history": ["h2"], "llm_raw_outputs": ["r2"]}
        merged = outcome.merge_updates(first, second)
        self.assertEqual(merged["errors"], ["a", "b"])
        self.assertEqual(merged["iteration_history"], ["h1", "h2"])
        self.assertEqual(merged["llm_raw_outputs"], ["r1", "r2"])

    def test_every_accumulator_is_covered(self) -> None:
        """Derived from the state annotations, so a new one cannot be missed."""
        from agents.cfu_design_agent.state import append_keys

        first = {key: [f"{key}-1"] for key in append_keys()}
        second = {key: [f"{key}-2"] for key in append_keys()}
        merged = outcome.merge_updates(first, second)
        for key in append_keys():
            with self.subTest(key=key):
                self.assertEqual(merged[key], [f"{key}-1", f"{key}-2"])

    def test_gates_merge_and_scalars_take_the_later_value(self) -> None:
        merged = outcome.merge_updates(
            {"gates": {"isa": {"ok": True}}, "status": "initialized", "iteration": 1},
            {"gates": {"sim": {"ok": False}}, "status": "failed", "iteration": 2},
        )
        self.assertEqual(set(merged["gates"]), {"isa", "sim"})
        self.assertEqual(merged["status"], "failed")
        self.assertEqual(merged["iteration"], 2)

    def test_a_missing_channel_is_not_invented(self) -> None:
        merged = outcome.merge_updates({"errors": ["a"]}, {"stage_jobs": "x"})
        self.assertEqual(merged["errors"], ["a"])
        self.assertNotIn("iteration_history", merged)


class ModeDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.state = {"workdir": temp.name, "agent_tools": True,
                      "apply_patch": True, "run_commands": True}

    def test_tool_mode_is_used_when_the_model_can_call_tools(self) -> None:
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=True)), \
                mock.patch.object(implement_stage.unified, "implement_with_tools",
                                  return_value={"stage_jobs": "tools"}) as tool_mode:
            self.assertEqual(implement_stage.implement(self.state)["stage_jobs"], "tools")
        self.assertTrue(tool_mode.called)

    def test_delivery_mode_is_used_without_tools(self) -> None:
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=True)), \
                mock.patch.object(implement_stage.hardware, "implement_hw",
                                  return_value={"gates": {"implementation_hw": {"ok": True}},
                                                "errors": ["hw"]}), \
                mock.patch.object(implement_stage.software, "implement_sw",
                                  return_value={"gates": {"implementation_sw": {"ok": True}},
                                                "errors": ["sw"]}), \
                mock.patch.object(implement_stage.oracle_author, "write_conformance", return_value={}), \
                mock.patch.object(implement_stage.unified, "implement_with_tools",
                                  side_effect=AssertionError("tool mode must not run")):
            update = implement_stage.implement({**self.state, "agent_tools": False})
        self.assertEqual(set(update["gates"]), {"implementation_hw", "implementation_sw"})
        self.assertEqual(update["errors"], ["hw", "sw"], "both sides' ledgers must survive")

    def test_delivery_mode_falls_back_when_the_model_is_unavailable(self) -> None:
        with mock.patch.object(llm_step, "_llm", return_value=mock.Mock(available=False)), \
                mock.patch.object(implement_stage.hardware, "implement_hw", return_value={}), \
                mock.patch.object(implement_stage.software, "implement_sw", return_value={}), \
                mock.patch.object(implement_stage.oracle_author, "write_conformance", return_value={}), \
                mock.patch.object(implement_stage.unified, "implement_with_tools",
                                  side_effect=AssertionError("tool mode must not run")):
            self.assertEqual(implement_stage.implement(self.state), {})

    def test_both_modes_are_reachable_from_the_compiled_graph(self) -> None:
        from agents.cfu_design_agent import graph

        self.assertIn("implement", graph.NODE_NAMES)
        self.assertIn("implement", graph.python_nodes())


class TwoGateNodeTest(unittest.TestCase):
    """`implement` writes both implementation gates; both must stay fresh."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workdir = Path(temp.name).resolve()

    @staticmethod
    def implement(state: dict) -> dict:
        # The wrapper derives the gate mapping from fn.__name__, so the node
        # function must carry the node's name.
        return {"errors": [], "gates": {"implementation_hw": {"ok": True},
                                        "implementation_sw": {"ok": True}}}

    def test_both_gates_are_re_stamped_when_an_upstream_input_changes(self) -> None:
        base = {"workdir": str(self.workdir), "gates": {}, "errors": []}
        first = instrument._wrap_node(self.implement)(base)
        second = instrument._wrap_node(self.implement)({
            **base, "workload_analysis": "changed",
            "input_fingerprints": first["input_fingerprints"],
        })
        for gate in ("implementation_hw", "implementation_sw"):
            with self.subTest(gate=gate):
                self.assertNotIn("stale", second["gates"][gate])
                self.assertTrue(second["gates"][gate]["ok"])
                self.assertTrue(second["gates"][gate].get("input_sha256"))

    def test_a_failing_half_makes_the_node_not_ok(self) -> None:
        def implement(state: dict) -> dict:
            return {"gates": {"implementation_hw": {"ok": True},
                              "implementation_sw": {"ok": False, "errors": ["no sw change"]}}}

        self.assertFalse(instrument._node_ok(implement({}), "implement"))
        self.assertTrue(instrument._node_ok(self.implement({}), "implement"))

    def test_an_unmapped_node_is_ok_without_a_gate(self) -> None:
        self.assertEqual(instrument.checked_gates("route_input"), ())
        self.assertTrue(instrument._node_ok({"gates": {}}, "route_input"))


if __name__ == "__main__":
    unittest.main()
