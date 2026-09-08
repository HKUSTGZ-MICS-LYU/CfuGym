"""Token-usage accounting and stage progress helpers."""

from __future__ import annotations

import types
import unittest

from agents.cfu_design_agent import llm
from agents.cfu_design_agent.graph import _node_ok
from agents.cfu_design_agent.progress import NODE_TO_STEP, StageProgress


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
    def test_node_to_step_alignment(self) -> None:
        order = [v for v in [
            "load_context", "ingest_input", "route_input", "prepare_benchmark",
            "profile", "hotspot", "analysis", "isa", "implementation",
            "validation", "cost", "review", "final_report",
        ]]
        progress = StageProgress(order)
        self.assertEqual(progress._index_of("load_context"), 0)
        self.assertEqual(progress._index_of("instrument_and_profile"), 4)
        self.assertEqual(progress._index_of("review_and_route_python"), 11)
        self.assertEqual(progress._index_of("final_report"), 12)
        # Unknown node falls back to the end of the list (count), not negative.
        self.assertEqual(progress._index_of("bogus"), len(order))


if __name__ == "__main__":
    unittest.main()
