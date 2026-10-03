"""External commands run as LangGraph tasks: a resume must not re-issue them.

The guarantee under test is the one ``runtime/persistence.py`` used to hand-roll
with its operation journal: a command that already ran is restored from the
checkpoint instead of being executed a second time. The plain path (a node
function called directly, as most tests do) must keep working unchanged.
"""

from __future__ import annotations

import operator
import tempfile
import unittest
from pathlib import Path
from typing import Annotated, TypedDict
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.runtime import checks, effects, stage_services
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


class Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


class CommandEffectTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workdir = Path(temp.name).resolve()
        self.state = {"workdir": str(self.workdir), "run_root": ""}
        self.calls: list[list[str]] = []

        def fake_check(state, toolbox, argv, **kwargs):
            self.calls.append(list(argv))
            return {"returncode": 0, "command": " ".join(argv)}

        self.patches = [
            mock.patch.object(checks, "_run_check", fake_check),
            mock.patch.object(stage_services, "_toolbox_for_stage", lambda state, stage: object()),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _graph(self):
        state = self.state

        def node(_state: Log) -> dict:
            with effects.use_state(state):
                effects.run_command(state, None, ["echo", "before"], stage="validation", rung="r")
                answer = interrupt("pause")
                effects.run_command(state, None, ["echo", "after"], stage="validation", rung="r")
            return {"log": [f"resumed={answer}"]}

        builder = StateGraph(Log)
        builder.add_node("n", node)
        builder.add_edge(START, "n")
        builder.add_edge("n", END)
        return builder.compile(checkpointer=InMemorySaver())

    def test_a_completed_command_is_not_reissued_after_a_resume(self) -> None:
        app = self._graph()
        config = {"configurable": {"thread_id": "effects-1"}}
        app.invoke({"log": []}, config)
        self.assertEqual(self.calls, [["echo", "before"]], "the node must pause before the second command")

        app.invoke(Command(resume="ok"), config)
        self.assertEqual(
            self.calls, [["echo", "before"], ["echo", "after"]],
            "the command recorded before the pause must be restored, not re-run",
        )
        self.assertEqual(app.get_state(config).values["log"], ["resumed=ok"])

    def test_the_plain_path_is_used_outside_a_graph_run(self) -> None:
        self.assertFalse(effects.inside_run())
        result = effects.run_command(self.state, None, ["echo", "direct"], stage="validation", rung="r")
        self.assertEqual(result["command"], "echo direct")
        self.assertEqual(self.calls, [["echo", "direct"]])

    def test_a_published_state_without_a_runnable_context_falls_back(self) -> None:
        """A state outside a graph must degrade, not raise."""
        with effects.use_state(self.state):
            self.assertFalse(effects.inside_run())
            effects.run_command(self.state, None, ["echo", "tasked"], stage="validation", rung="r")
        self.assertEqual(self.calls, [["echo", "tasked"]])

    def test_the_node_wrapper_publishes_the_run_state(self) -> None:
        seen: list[object] = []

        def step(state: dict) -> dict:
            seen.append(effects.current_state())
            return {"errors": []}

        instrument._wrap_node(step)({"workdir": str(self.workdir), "gates": {}, "errors": []})
        self.assertEqual(seen, [{"workdir": str(self.workdir), "gates": {}, "errors": []}])
        self.assertIsNone(effects.current_state(), "the run state must not leak between nodes")


if __name__ == "__main__":
    unittest.main()
