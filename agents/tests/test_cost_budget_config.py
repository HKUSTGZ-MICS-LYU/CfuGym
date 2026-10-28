"""Cost Wall configuration and evidence freshness, without model/toolchain calls."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import get_type_hints
import unittest
from unittest import mock

from langgraph.graph import END, START, StateGraph

from agents.cfu_design_agent import cli, graph
from agents.cfu_design_agent.runtime import cost_budget, dependencies, persistence
from agents.cfu_design_agent.state import CfuDesignState


class CostBudgetCliTest(unittest.TestCase):
    def test_parser_default_and_positive_override(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args([]).max_yosys_cells, 100000)
        self.assertEqual(parser.parse_args(["--max-yosys-cells", "1"]).max_yosys_cells, 1)
        self.assertEqual(parser.parse_args(["--max-yosys-cells=123456"]).max_yosys_cells, 123456)

    def test_invalid_values_fail_before_graph(self):
        for value in ("0", "-1", "1.0", "nan", "True", "", "ten"):
            with self.subTest(value=value), mock.patch.object(cli, "run_agent") as run:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(["--task", "offline", "--max-yosys-cells=" + value])
                run.assert_not_called()

    def _state(self, args):
        with mock.patch.object(cli, "export_graph_visualization", return_value={}), mock.patch.object(
            cli, "enable_progress"
        ), mock.patch.object(cli, "run_agent", return_value={"status": "planned"}) as run:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
        return run.call_args.args[0]

    def test_cli_threads_limit_into_initial_state_for_both_backends(self):
        for backend in ("yosys", "vivado"):
            with self.subTest(backend=backend):
                state = self._state(["--task", "offline", "--dry-run", "--cost-backend", backend,
                                     "--max-yosys-cells", "76543"])
                self.assertEqual(state["max_yosys_cells"], 76543)
                self.assertEqual(state["cost_backend"], backend)
        self.assertEqual(self._state(["--task", "offline", "--dry-run"])["max_yosys_cells"], 100000)

    def test_resume_records_only_explicit_limit_in_both_argument_forms(self):
        self.assertEqual(self._state(["--resume", "prior-run"])["resume_eda_options"], {})
        for extra in (["--max-yosys-cells", "22222"], ["--max-yosys-cells=22222"]):
            with self.subTest(extra=extra):
                state = self._state(["--resume", "prior-run", *extra])
                self.assertEqual(state["resume_eda_options"], {"max_yosys_cells": 22222})

    def test_limit_is_a_native_state_channel(self):
        self.assertIs(get_type_hints(CfuDesignState)["max_yosys_cells"], int)
        self.assertIn("max_yosys_cells", StateGraph(CfuDesignState).channels)


class CostBudgetGraphConfigTest(unittest.TestCase):
    @contextmanager
    def _resume(self, saved=None, explicit=None):
        with tempfile.TemporaryDirectory() as root:
            previous = {"run_id": "prior-run", "workdir": str(Path(root).resolve()), **(saved or {})}
            config = {"configurable": {"thread_id": "prior-run"}}
            persisted = SimpleNamespace(
                saver=object(), run_id="prior-run", config=config,
                journal=SimpleNamespace(_load=lambda: {"operations": {}}),
            )
            app = mock.Mock()
            app.get_state.return_value = SimpleNamespace(values=previous, next=("finish",))
            initial = {"run_id": "prior-run", "workdir": root, "resume_eda_options": explicit or {}}
            with mock.patch.object(persistence, "open_run_persistence", return_value=nullcontext(persisted)), mock.patch.object(
                graph, "build_graph", return_value=app
            ), mock.patch.object(graph, "_invoke_agent_graph", return_value={"status": "planned"}) as invoke:
                yield initial, previous, app, invoke, config

    def test_programmatic_initial_state_gets_default_without_mutating_caller(self):
        initial = {"task": "offline"}
        with mock.patch.object(graph, "build_graph"), mock.patch.object(graph, "_invoke_agent_graph") as invoke:
            graph.run_agent(initial)
        self.assertEqual(invoke.call_args.args[1]["max_yosys_cells"], 100000)
        self.assertNotIn("max_yosys_cells", initial)

    def test_new_checkpoint_input_preserves_explicit_limit(self):
        with self._resume() as (initial, _, app, invoke, _):
            initial["max_yosys_cells"] = 77
            graph.run_agent(initial, checkpoint=True)
        self.assertEqual(invoke.call_args.args[1]["max_yosys_cells"], 77)
        app.get_state.assert_not_called()

    def test_programmatic_invalid_limit_fails_before_graph(self):
        for value in (True, False, 0, -1, 1.0, "100000", None):
            with self.subTest(value=value), mock.patch.object(graph, "build_graph") as build:
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    graph.run_agent({"max_yosys_cells": value})
                build.assert_not_called()

    def test_resume_preserves_recorded_limit_when_cli_default_is_implicit(self):
        with self._resume({"max_yosys_cells": 24680}) as (initial, previous, app, invoke, _):
            initial["max_yosys_cells"] = 100000
            graph.run_agent(initial, checkpoint=True, resume=True)
            self.assertEqual(previous["max_yosys_cells"], 24680)
            app.update_state.assert_not_called()
            self.assertIsNone(invoke.call_args.args[1])

    def test_resume_accepts_matching_explicit_limit(self):
        with self._resume({"max_yosys_cells": 24680}, {"max_yosys_cells": 24680}) as (
            initial, _, app, invoke, _
        ):
            graph.run_agent(initial, checkpoint=True, resume=True)
            app.update_state.assert_not_called()
            invoke.assert_called_once()

    def test_resume_rejects_conflicting_explicit_limit(self):
        for saved in ({"max_yosys_cells": 24680}, {}):
            with self.subTest(saved=saved), self._resume(saved, {"max_yosys_cells": 12345}) as (
                initial, _, app, invoke, _
            ):
                with self.assertRaisesRegex(persistence.PersistenceRiskError, "EDA configuration: max_yosys_cells"):
                    graph.run_agent(initial, checkpoint=True, resume=True)
                app.update_state.assert_not_called()
                invoke.assert_not_called()

    def test_missing_legacy_limit_persists_default(self):
        for explicit in ({}, {"max_yosys_cells": 100000}):
            with self.subTest(explicit=explicit), self._resume(explicit=explicit) as (
                initial, _, app, invoke, config
            ):
                graph.run_agent(initial, checkpoint=True, resume=True)
                app.update_state.assert_called_once_with(config, {"max_yosys_cells": 100000})
                self.assertIsNone(invoke.call_args.args[1])

    def test_invalid_recorded_limit_is_not_treated_as_missing(self):
        for value in (True, 0, -1, 1.0, "100000", None):
            with self.subTest(value=value), self._resume({"max_yosys_cells": value}) as (
                initial, _, app, invoke, _
            ):
                with self.assertRaisesRegex(persistence.PersistenceRiskError, "positive integer"):
                    graph.run_agent(initial, checkpoint=True, resume=True)
                app.update_state.assert_not_called()
                invoke.assert_not_called()

    def test_legacy_migration_preserves_native_pending_node_and_persists_channel(self):
        calls = []

        def builder(*, checkpointer=None, interrupt=False):
            workflow = StateGraph(CfuDesignState)
            workflow.add_node("seed", lambda state: {})

            def finish(state):
                calls.append(state["max_yosys_cells"])
                return {"status": "planned"}

            workflow.add_node("finish", finish)
            workflow.add_edge(START, "seed")
            workflow.add_edge("seed", "finish")
            workflow.add_edge("finish", END)
            return workflow.compile(checkpointer=checkpointer, interrupt_after=["seed"] if interrupt else None)

        with tempfile.TemporaryDirectory() as root:
            initial = {"workdir": root, "run_id": "legacy-config"}
            with persistence.open_run_persistence(Path(root), "legacy-config") as persisted:
                app = builder(checkpointer=persisted.saver, interrupt=True)
                app.invoke(initial, persisted.config)
                self.assertNotIn("max_yosys_cells", app.get_state(persisted.config).values)
                self.assertEqual(app.get_state(persisted.config).next, ("finish",))
            with mock.patch.object(graph, "build_graph", side_effect=builder):
                result = graph.run_agent(initial, checkpoint=True, resume=True)
            self.assertEqual(calls, [100000])
            self.assertEqual(result["max_yosys_cells"], 100000)
            with persistence.open_run_persistence(Path(root), "legacy-config", resume=True) as persisted:
                saved = builder(checkpointer=persisted.saver).get_state(persisted.config)
                self.assertEqual(saved.values["max_yosys_cells"], 100000)


class CostBudgetFingerprintTest(unittest.TestCase):
    def _state(self, root, backend):
        return {"workdir": root, "cost_backend": backend, "flow_version": 2}

    def _assert_cost_only_change(self, before, after):
        changed = {name for name in before if before[name] != after[name]}
        self.assertEqual(changed, {"cost", "validation_pair"})
        gates = {name: {"ok": True} for name in before}
        invalidated = dependencies.invalidate_changed_gates(gates, before, after)
        self.assertEqual({name for name, gate in invalidated.items() if not gate["ok"]}, changed)
        self.assertTrue(invalidated["hw_build"]["ok"])
        self.assertTrue(invalidated["sim"]["ok"])

    def test_missing_limit_matches_explicit_default_for_both_backends(self):
        with tempfile.TemporaryDirectory() as root:
            for backend in ("yosys", "vivado"):
                with self.subTest(backend=backend):
                    state = self._state(root, backend)
                    self.assertEqual(dependencies.input_fingerprints(state), dependencies.input_fingerprints(
                        {**state, "max_yosys_cells": 100000}))

    def test_changed_limit_only_invalidates_cost_and_validation_pair(self):
        with tempfile.TemporaryDirectory() as root:
            for backend in ("yosys", "vivado"):
                with self.subTest(backend=backend):
                    state = self._state(root, backend)
                    before = dependencies.input_fingerprints(state)
                    after = dependencies.input_fingerprints({**state, "max_yosys_cells": 99999})
                    self._assert_cost_only_change(before, after)

    def test_changed_wall_version_only_invalidates_cost_and_validation_pair(self):
        with tempfile.TemporaryDirectory() as root:
            for backend in ("yosys", "vivado"):
                with self.subTest(backend=backend):
                    state = self._state(root, backend)
                    before = dependencies.input_fingerprints(state)
                    with mock.patch.object(cost_budget, "VERSION", cost_budget.VERSION + 1):
                        after = dependencies.input_fingerprints(state)
                    self._assert_cost_only_change(before, after)

    def test_stale_pre_wall_cost_fingerprint_is_not_grandfathered(self):
        with tempfile.TemporaryDirectory() as root:
            state = self._state(root, "yosys")
            old = dependencies.input_fingerprints(state)
            old["cost"] = old["sim"]  # Yosys fingerprint before Cost Wall existed.
            with self.assertRaisesRegex(ValueError, "source or contract changed"):
                dependencies.completed_gate_inputs({**state, "input_fingerprints": old})

    def test_changed_budget_cannot_resume_with_old_cost_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            for backend in ("yosys", "vivado"):
                with self.subTest(backend=backend):
                    state = self._state(root, backend)
                    old = dependencies.input_fingerprints(state)
                    with self.assertRaisesRegex(ValueError, "source or contract changed"):
                        dependencies.completed_gate_inputs({**state, "max_yosys_cells": 200000,
                                                            "input_fingerprints": old})


if __name__ == "__main__":
    unittest.main()
