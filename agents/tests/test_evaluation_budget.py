"""Attempt admission contracts: no LLM, simulator, synthesis, or full ladder."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest import mock

from agents.cfu_design_agent.flow import evaluations, ladder
from agents.cfu_design_agent.runtime import effects
from agents.cfu_design_agent.stages.verify import cost, parallel, soc, software


class EvaluationBudgetTest(unittest.TestCase):
    def setUp(self):
        self.state = {"run_commands": True, "max_evals": 1, "max_iters": 5,
                      "verify_submissions": 1, "iteration": 0}

    def lane(self, rung, ok=True):
        def run(state):
            self.assertEqual(effects.current_state()["evaluation_id"], state["evaluation_id"])
            return {"gates": {rung: {"ok": ok, "evaluation_id": state["evaluation_id"]}}}
        return run

    def ticket(self, state=None):
        current = state or self.state
        return {**current, **evaluations.admission(current)}

    def test_serial_sim_cost_share_one_ticket_even_when_limit_is_full(self):
        initial = self.ticket()
        sim = evaluations.run(initial, "sim", True, self.lane("sim"))
        current = {**initial, **sim}
        result = evaluations.run(current, "cost", True, self.lane("cost"))
        self.assertEqual(evaluations.used(result), 1)
        self.assertEqual(result["evaluation_id"], initial["evaluation_id"])
        self.assertEqual(result["evaluation_attempts"][0]["completed_lanes"], ["cost", "sim"])

    def test_failed_attempt_consumes_budget_and_next_candidate_is_not_executed(self):
        result = evaluations.run(self.state, "sim", True, self.lane("sim", False))
        current = {**self.state, **result, "iteration": 1, "verify_submissions": 2, "evaluation_id": ""}
        callback = mock.Mock()
        result = evaluations.run(current, "sim", True, callback)
        callback.assert_not_called()
        self.assertEqual(result["stopped_at"], "evaluation_budget")
        self.assertEqual(result["termination_reason"], "BUDGET_EXHAUSTED")

    def test_same_source_explicit_repeat_is_new_attempt_but_replay_is_not(self):
        state = {**self.state, "max_evals": 3}
        ticket = self.ticket(state)
        first = evaluations.run(ticket, "sim", True, self.lane("sim"))
        replay = evaluations.run(ticket, "sim", True, self.lane("sim"))
        self.assertEqual(first["evaluation_id"], replay["evaluation_id"])
        repeat = evaluations.run({**ticket, **first}, "sim", True, self.lane("sim"))
        self.assertEqual(evaluations.used(repeat), 2)
        self.assertNotEqual(first["evaluation_id"], repeat["evaluation_id"])

    def test_prepare_preserves_build_verdict_and_checkpoint_ticket(self):
        build = {"gates": {"sw_build": {"ok": True, "next": "simulate"}}}
        result = evaluations.prepare(self.state, build)
        self.assertEqual(evaluations.used(result), 1)
        self.assertTrue(result["gates"]["sw_build"]["ok"])
        full = {**self.state, **result, "evaluation_id": "", "iteration": 1}
        result = evaluations.prepare(full, build)
        self.assertTrue(result["gates"]["sw_build"]["ok"])
        self.assertEqual(result["gates"]["sw_build"]["next"], "review_and_route")

    def test_flags_count_only_enabled_evaluation_groups(self):
        cases = [({"dry_run": True}, False), ({"run_commands": False}, False),
                 ({"skip_sim": True, "skip_cost": True}, False),
                 ({"skip_sim": True, "skip_yosys": True}, False),
                 ({"skip_sim": True, "implementation_design_type": "draft"}, False),
                 ({"skip_sim": True}, True), ({"skip_cost": True}, True),
                 ({"skip_sim": True, "skip_yosys": True, "cost_backend": "vivado"}, True)]
        for flags, charged in cases:
            with self.subTest(flags=flags):
                result = evaluations.prepare({**self.state, **flags},
                    {"gates": {"sw_build": {"ok": False, "skipped": True}}})
                self.assertEqual(evaluations.used(result), int(charged))

    def test_failed_and_draft_routing_stop_without_spending_another_repair(self):
        full = self.ticket()
        extra, target = ladder._ladder_transition(full, "sim", ok=False,
            default="estimate_cost", repair_target="implement")
        self.assertEqual(target, "review_and_route")
        self.assertEqual(extra["stopped_at"], "evaluation_budget")
        self.assertNotIn("repairs_used", extra)
        extra, target = ladder.draft_sim_transition(full)
        self.assertEqual(target, "review_and_route")
        self.assertEqual(extra["stopped_at"], "evaluation_budget")

    def test_all_execution_entrypoints_refuse_new_attempt_before_effects(self):
        full = {**self.ticket(), "evaluation_id": "", "iteration": 1}
        for module, entry, implementation in ((software, "simulate", "_simulate"),
                (cost, "estimate_cost", "_estimate_cost"),
                (parallel, "validate_candidate", "_validate_candidate")):
            with self.subTest(entry=entry), mock.patch.object(module, implementation) as execute, \
                 mock.patch.object(software.stage_services, "_profile_reason", return_value=""):
                result = getattr(module, entry)(full)
                execute.assert_not_called()
                self.assertEqual(result["stopped_at"], "evaluation_budget")
        with mock.patch.object(soc.stage_services, "_profile_reason", return_value=""), \
             mock.patch.object(soc.effects, "run_command") as command:
            self.assertEqual(soc.generate_soc(full)["stopped_at"], "evaluation_budget")
        command.assert_not_called()

    def test_build_wrapper_reserves_ticket_before_executors_are_called(self):
        with mock.patch.object(software, "_build_sw", return_value={
                "gates": {"sw_build": {"ok": True, "next": "simulate"}}}):
            result = software.build_sw(self.state)
        self.assertEqual(evaluations.used(result), 1)
        self.assertTrue(result["evaluation_id"])

    def test_native_pair_replay_has_one_durable_ticket_and_no_repeated_workers(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import Command, interrupt
        from agents.cfu_design_agent.state import CfuDesignState
        calls = []
        ids = []
        def admit(state):
            return evaluations.prepare(state, {"gates": {"sw_build": {"ok": True}}})
        def pair(state):
            with effects.use_state(state):
                update = parallel.validate_candidate(state)
            ids.append(update["evaluation_id"])
            interrupt("after durable pair task")
            return update
        def worker(rung):
            def execute(state, *, transition):
                calls.append((rung, state["evaluation_id"]))
                return {"gates": {rung: {"ok": True}}}
            return execute
        builder = StateGraph(CfuDesignState)
        builder.add_node("admit", admit)
        builder.add_node("pair", pair)
        builder.add_edge(START, "admit")
        builder.add_edge("admit", "pair")
        builder.add_edge("pair", END)
        app = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "unit-eval-budget-replay"}}
        with mock.patch.object(software, "simulate", side_effect=worker("sim")), \
             mock.patch.object(cost, "estimate_cost", side_effect=worker("cost")):
            app.invoke({**self.state, "validation_mode": "serial"}, config)
            pending = app.get_state(config).values
            self.assertEqual(evaluations.used(pending), 1)
            final = app.invoke(Command(resume=True), config)
        self.assertEqual(evaluations.used(final), 1)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual([rung for rung, _ in calls], ["sim", "cost"])
        self.assertEqual({identity for _, identity in calls}, {final["evaluation_id"]})

    def test_pair_after_completed_serial_evaluation_needs_a_new_ticket(self):
        initial = self.ticket()
        sim = evaluations.run(initial, "sim", True, self.lane("sim"))
        serial = evaluations.run({**initial, **sim}, "cost", True, self.lane("cost"))
        full = {**initial, **serial}
        with mock.patch.object(software, "simulate") as simulate, \
             mock.patch.object(cost, "estimate_cost") as estimate:
            result = parallel.validate_candidate(full)
        simulate.assert_not_called()
        estimate.assert_not_called()
        self.assertEqual(result["stopped_at"], "evaluation_budget")
        reservation = evaluations.admission({**full, "max_evals": 2}, "validation_pair")
        self.assertEqual(evaluations.used(reservation), 2)
        self.assertNotEqual(reservation["evaluation_id"], full["evaluation_id"])

    def test_legacy_attempts_are_not_silently_free(self):
        state = {**self.state, "verify_submissions": 3,
                 "validation_results": [{"rung": "sim", "returncode": 1}]}
        self.assertEqual(evaluations.used(state), 3)
        self.assertIsNone(evaluations.admission(state))
        self.assertEqual(evaluations.used({**state, "evaluation_attempts": []}), 0)


if __name__ == "__main__":
    unittest.main()
