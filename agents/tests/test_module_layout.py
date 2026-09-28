"""Responsibility boundaries use the actual benchmark and runtime modules."""

from __future__ import annotations

import importlib
import unittest

from agents.cfu_design_agent import graph, profile_cache
from agents.cfu_design_agent.benchmark import generic, embench
from agents.cfu_design_agent.flow import wiring
from agents.cfu_design_agent.flow.state_updates import append_state_keys
from agents.cfu_design_agent.runtime import tools
from agents.cfu_design_agent.runtime.policy import PolicyError
from agents.cfu_design_agent.state import CfuDesignState
from agents.tests.common import REPO_ROOT


class ModuleLayoutTest(unittest.TestCase):
    def test_implementation_modules_have_own_boundaries_without_legacy_shims(self) -> None:
        moved = {
            "benchmarks": "benchmark.generic",
            "embench": "benchmark.embench",
            "embench_instrument": "benchmark.embench_instrument",
            "embench_loops": "benchmark.embench_loops",
            "policy": "runtime.policy",
            "tools": "runtime.tools",
            "artifacts": "runtime.artifacts",
            "agent_tools": "runtime.agent_tools",
            "llm": "runtime.llm",
            "llm_config": "runtime.llm_config",
            "progress": "runtime.progress",
        }
        base = REPO_ROOT / "agents/cfu_design_agent"
        prefix = "agents.cfu_design_agent."
        for legacy, implementation in moved.items():
            with self.subTest(module=legacy):
                self.assertFalse((base / f"{legacy}.py").exists())
                module = importlib.import_module(prefix + implementation)
                self.assertEqual(module.__name__, prefix + implementation)
                self.assertEqual(module.__file__, str(base / (implementation.replace(".", "/") + ".py")))

        # The public graph uses the same concrete runtime types as the test imports.
        self.assertIs(graph.RepoToolbox, tools.RepoToolbox)
        self.assertIs(graph.PolicyError, PolicyError)

    def test_both_runners_share_node_and_edge_registry(self) -> None:
        self.assertEqual(set(graph.python_nodes()), set(wiring.NODE_NAMES))
        self.assertEqual(set(graph.PY_ROUTERS), set(wiring.ROUTER_NAMES))
        self.assertEqual(graph.PY_SUCCESSORS, wiring.SUCCESSORS)
        self.assertEqual(set(wiring.ROUTE_TARGETS), set(wiring.ROUTER_NAMES))
        ascii_graph = graph.fallback_ascii_graph()
        for node in wiring.NODE_NAMES:
            self.assertIn(f"{node} -> ", ascii_graph)
        if graph._LANGRAPH_AVAILABLE:
            self.assertIsNotNone(graph.build_graph())

    def test_reducers_come_from_state_schema(self) -> None:
        self.assertEqual(graph.APPEND_STATE_KEYS, append_state_keys(CfuDesignState))
        self.assertIn("conformance_attempts", graph.APPEND_STATE_KEYS)
        self.assertNotIn("observation_report", graph.APPEND_STATE_KEYS)
        state = {"errors": ["before"], "gates": {"profile": {"ok": True}}}
        graph._merge_state_update(state, {"errors": ["after"], "gates": {"profile": {"ok": False}}})
        self.assertEqual(state, {"errors": ["before", "after"], "gates": {"profile": {"ok": False}}})

    def test_profile_parsers_do_not_duplicate_cycles_or_treat_calls_as_cycles(self) -> None:
        text = ("\x1b[31mCFU_AGENT_PROFILE kernel=42 calls=2 repeat=5 bytes=16\x1b[0m\n"
                "CFU_AGENT_PROFILE kernel=75 calls=3\nOTHER_PROFILE cost=9\n")
        regions = generic.parse_profile_regions(text)
        self.assertEqual([(item["name"], item["cycles"]) for item in regions],
                         [("kernel", 42), ("kernel", 75), ("cost", 9)])
        self.assertEqual([item["source"] for item in regions],
                         ["CFU_AGENT_PROFILE", "CFU_AGENT_PROFILE", "OTHER_PROFILE"])
        self.assertEqual(embench.parse_profile(text)["kernel"], {"cycles": 75, "calls": 3})
        self.assertNotIn("calls", embench.parse_profile(text).keys())

    def test_cache_fingerprint_tracks_moved_implementations(self) -> None:
        paths = (
            "agents/cfu_design_agent/benchmark/generic.py",
            "agents/cfu_design_agent/benchmark/embench.py",
            "agents/cfu_design_agent/benchmark/embench_instrument.py",
            "agents/cfu_design_agent/benchmark/embench_loops.py",
        )
        for name in paths:
            with self.subTest(path=name):
                self.assertTrue((REPO_ROOT / name).is_file())
                self.assertTrue(profile_cache.tree_hash(REPO_ROOT / name))
