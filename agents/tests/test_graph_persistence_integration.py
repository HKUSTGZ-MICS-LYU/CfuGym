"""Disposable graph-level persistence and side-effect replay checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from langgraph.graph import START, StateGraph

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime.persistence import PendingOperationError, PersistenceRiskError
from agents.cfu_design_agent.flow import instrument


class GraphPersistenceIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_id = "disposable-graph-001"
        self.run_root = self.root / "agents/generated/runs" / self.run_id
        self.initial = {"workdir": str(self.root), "run_id": self.run_id, "max_iters": 1}
        self.calls: list[str] = []

    def _builder(self, *, interrupt: bool):
        def build_graph(*, checkpointer=None):
            def first(state):
                self.calls.append("first")
                (self.run_root / "touch.txt").write_text("first")
                return {"run_id": state["run_id"], "workdir": state["workdir"]}

            def second(state):
                self.calls.append("second")
                return {"finished": True}

            builder = StateGraph(dict)
            builder.add_node("first", instrument._wrap_node(first))
            builder.add_node("second", instrument._wrap_node(second))
            builder.add_edge(START, "first")
            builder.add_edge("first", "second")
            return builder.compile(checkpointer=checkpointer, interrupt_after=["first"] if interrupt else None)

        return build_graph

    def test_resume_does_not_replay_completed_node(self) -> None:
        with mock.patch.object(graph, "build_graph", self._builder(interrupt=True)):
            first = graph.run_agent(self.initial, checkpoint=True)
        self.assertNotIn("finished", first)
        self.assertEqual(self.calls, ["first"])
        with mock.patch.object(graph, "build_graph", self._builder(interrupt=False)):
            resumed = graph.run_agent(self.initial, checkpoint=True, resume=True)
        self.assertTrue(resumed["finished"])
        self.assertEqual(self.calls, ["first", "second"])
        journal = json.loads((self.run_root / "operation_journal.json").read_text())
        self.assertTrue(all(item["status"] == "committed" for item in journal["operations"].values()))

    def test_source_change_invalidates_checkpoint_gate(self) -> None:
        from agents.cfu_design_agent.runtime.dependencies import input_fingerprints

        def build_graph(*, checkpointer=None):
            def seed(state):
                source = self.run_root / "workspace/design/AgentCfu.scala"
                source.parent.mkdir(parents=True)
                source.write_text("verified")
                values = {**state, "run_root": f"agents/generated/runs/{self.run_id}",
                          "gates": {"hw_build": {"ok": True}}}
                return {**values, "input_fingerprints": input_fingerprints(values)}

            builder = StateGraph(dict)
            builder.add_node("seed", seed)
            builder.add_node("finish", lambda state: {"finished": True})
            builder.add_edge(START, "seed")
            builder.add_edge("seed", "finish")
            return builder.compile(checkpointer=checkpointer, interrupt_after=["seed"])

        with mock.patch.object(graph, "build_graph", build_graph):
            graph.run_agent(self.initial, checkpoint=True)
        (self.run_root / "workspace/design/AgentCfu.scala").write_text("changed")
        with mock.patch.object(graph, "build_graph", build_graph):
            with self.assertRaisesRegex(PersistenceRiskError, "source or contract changed"):
                graph.run_agent(self.initial, checkpoint=True, resume=True)

    def test_committed_unfinished_node_refuses_replay(self) -> None:
        with mock.patch.object(graph, "build_graph", self._builder(interrupt=True)):
            graph.run_agent(self.initial, checkpoint=True)
        journal_path = self.run_root / "operation_journal.json"
        journal = json.loads(journal_path.read_text())
        operation = next(iter(journal["operations"].values()))
        operation["node"] = "second"  # checkpoint.next is second, journal says it already ran
        journal_path.write_text(json.dumps(journal))
        with mock.patch.object(graph, "build_graph", self._builder(interrupt=False)):
            with self.assertRaisesRegex(PersistenceRiskError, "checkpoint may lag"):
                graph.run_agent(self.initial, checkpoint=True, resume=True)
        self.assertEqual(self.calls, ["first"])

    def test_pending_node_refuses_resume(self) -> None:
        with mock.patch.object(graph, "build_graph", self._builder(interrupt=True)):
            graph.run_agent(self.initial, checkpoint=True)
        journal_path = self.run_root / "operation_journal.json"
        journal = json.loads(journal_path.read_text())
        next(iter(journal["operations"].values()))["status"] = "pending"
        journal_path.write_text(json.dumps(journal))
        with mock.patch.object(graph, "build_graph", self._builder(interrupt=False)):
            with self.assertRaises(PendingOperationError):
                graph.run_agent(self.initial, checkpoint=True, resume=True)
        self.assertEqual(self.calls, ["first"])


if __name__ == "__main__":
    unittest.main()
