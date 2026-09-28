"""Token-usage accounting and stage progress helpers."""

from __future__ import annotations

import types
import unittest
from pathlib import Path

from agents.cfu_design_agent.runtime import llm
from agents.cfu_design_agent.graph import _node_ok, _update_ok
from agents.cfu_design_agent.runtime.progress import NODE_TO_STEP, STAGE_LABELS, StageProgress


class UsageByStageTest(unittest.TestCase):
    """Per-stage input tokens: the "which stage's input is big" view."""

    def setUp(self) -> None:
        llm.reset_usage_log()
        llm.set_stage("")

    def test_by_stage_aggregates_in_first_seen_order(self) -> None:
        llm.record_usage("implement_hw", 100, 10, 110)
        llm.record_usage("design_contract", 5, 1, 6)
        llm.record_usage("implement_hw", 200, 20, 220)
        by_stage = llm.usage_by_stage()
        self.assertEqual(list(by_stage), ["implement_hw", "design_contract"])
        self.assertEqual(
            by_stage["implement_hw"], {"calls": 2, "input": 300, "output": 30, "total": 330}
        )
        self.assertEqual(by_stage["design_contract"]["input"], 5)

    def test_empty_log_yields_no_stages(self) -> None:
        self.assertEqual(llm.usage_by_stage(), {})

    def test_stage_line_and_summary_split_input_output(self) -> None:
        llm.set_stage("implement_hw")
        llm.record_usage("implement_hw", 42, 7, 49)
        progress = StageProgress(list(NODE_TO_STEP.values()))
        progress.end("implement_hw", ok=True, elapsed=1.5, tokens=49)
        self.assertEqual(progress.results["implement_hw"]["input_tokens"], 42)
        self.assertEqual(progress.results["implement_hw"]["output_tokens"], 7)
        stage = progress.summary()["stages"]["implement_hw"]
        self.assertEqual(stage["input_tokens"], 42)
        self.assertEqual(stage["output_tokens"], 7)


class SpecPromptSinkTest(unittest.TestCase):
    """Every LLM call must leave its prompt behind, including the spec call."""

    def test_spec_call_reports_both_prompt_and_answer(self) -> None:
        from agents.cfu_design_agent.spec import normalize_workload_spec

        seen: dict[str, str] = {}
        answers: list[tuple[str, str]] = []

        class _Llm:
            available = True

            def complete(self, system, user, *, fallback):
                return llm.LlmResult("name: t\nunknowns: []", used_llm=True)

        normalize_workload_spec(
            task="accelerate a dot product",
            input_mode="natural_language",
            context={"files": {}},
            workdir=Path("."),
            llm=_Llm(),
            llm_result_sink=lambda label, result: answers.append((label, result.text)),
            llm_prompt_sink=lambda system, user: seen.update(system=system, user=user),
        )
        self.assertTrue(answers, "the spec LLM call must reach the result sink")
        self.assertTrue(seen.get("system"), "the spec LLM call must reach the prompt sink")
        self.assertIn("WorkloadSpec", seen["system"] + seen["user"])


class UsageLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        llm.reset_usage_log()
        llm.set_stage("")

    def test_record_and_totals(self) -> None:
        llm.set_stage("design_contract")
        llm.record_usage("design_contract", 10, 5, 15)
        llm.record_usage("design_contract", 20, 10, 30)
        self.assertEqual(llm.usage_totals(), {"calls": 2, "input": 30, "output": 15, "total": 45})
        self.assertEqual(llm.stage_usage("design_contract")["total"], 45)
        self.assertEqual(llm.stage_usage("other")["calls"], 0)

    def test_extract_chat_usage(self) -> None:
        fake = types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        self.assertEqual(llm._extract_usage(fake, "chat"), (10, 5, 15))

    def test_extract_responses_usage_dict(self) -> None:
        fake = types.SimpleNamespace(usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
        self.assertEqual(llm._extract_usage(fake, "responses"), (7, 3, 10))

    def test_extract_missing_usage(self) -> None:
        fake = types.SimpleNamespace(usage=None)
        self.assertEqual(llm._extract_usage(fake, "chat"), (0, 0, 0))

    def test_llm_result_defaults(self) -> None:
        result = llm.LlmResult("ok", used_llm=True)
        self.assertEqual(result.total_tokens, 0)


class NodeOkTest(unittest.TestCase):
    def test_gate_false_is_fail(self) -> None:
        update = {"gates": {"isa": {"ok": False, "problems": ["x"]}}}
        self.assertFalse(_node_ok(update, "design_contract"))

    def test_gate_true_is_ok(self) -> None:
        update = {"gates": {"isa": {"ok": True}}}
        self.assertTrue(_node_ok(update, "design_contract"))

    def test_no_gate_defaults_ok(self) -> None:
        self.assertTrue(_node_ok({"iteration_history": ["x"]}, "analyze_workload"))

    def test_command_update(self) -> None:
        update = types.SimpleNamespace(update={"gates": {"review": {"ok": False}}})
        self.assertFalse(_node_ok(update, "review_and_route"))


class StageProgressTest(unittest.TestCase):
    ORDER = [
        "load_context", "ingest_input", "route_input", "prepare_benchmark",
        "profile", "hotspot", "ir", "analysis", "isa", "implementation_hw",
        "hw_build", "hw_conformance", "implementation_sw", "sw_build", "sim",
        "cost", "review", "final_report",
    ]

    def test_node_to_step_alignment(self) -> None:
        progress = StageProgress(self.ORDER)
        self.assertEqual(progress._index_of("load_context"), 0)
        self.assertEqual(progress._index_of("instrument_and_profile"), 4)
        self.assertEqual(progress._index_of("check_conformance"), 11)
        self.assertEqual(progress._index_of("review_and_route"), 16)
        self.assertEqual(progress._index_of("final_report"), 17)
        # Unknown node falls back to the end of the list (count), not negative.
        self.assertEqual(progress._index_of("bogus"), len(self.ORDER))

    def test_stage_order_is_the_single_slot_source(self) -> None:
        from agents.cfu_design_agent.graph import STAGE_ORDER

        self.assertEqual(list(STAGE_ORDER), self.ORDER)

    def test_every_node_maps_to_a_valid_slot(self) -> None:
        # Regression: check_conformance mapped to nothing and printed [14/13].
        progress = StageProgress(self.ORDER)
        for name in NODE_TO_STEP:
            self.assertIn(NODE_TO_STEP[name], self.ORDER, name)
            self.assertLess(progress._index_of(name), len(self.ORDER), name)

    def test_every_node_has_a_label(self) -> None:
        self.assertEqual(set(STAGE_LABELS), set(NODE_TO_STEP))
        self.assertIn("check conformance", STAGE_LABELS.values())


class UpdateOkTest(unittest.TestCase):
    def test_conformance_failure_is_fail_even_with_prior_errors(self) -> None:
        # Regression: added_errors subtracted the state's prior error total, so
        # a failing rung printed OK whenever the run had already seen an error
        # (a repaired hw_build before a failing hw_conformance) while the ladder
        # routed back to a repair.
        update = {
            "errors": ["conformance test build failed"],
            "gates": {"hw_conformance": {"ok": False}},
        }
        self.assertFalse(_update_ok(update, "check_conformance"))

    def test_gate_only_failure_is_fail(self) -> None:
        self.assertFalse(_update_ok({"gates": {"hw_conformance": {"ok": False}}}, "check_conformance"))

    def test_clean_node_is_ok(self) -> None:
        update = {"errors": [], "gates": {"hw_conformance": {"ok": True}}}
        self.assertTrue(_update_ok(update, "check_conformance"))

    def test_prepare_node_gate_is_found(self) -> None:
        # Regression: the gate map keyed prepare nodes without the ``_node``
        # suffix of the wrapped function, so their gates were never found.
        update = {"errors": [], "gates": {"prepare_benchmark": {"ok": False}}}
        self.assertFalse(_update_ok(update, "prepare_embench_benchmark_node"))


if __name__ == "__main__":
    unittest.main()
