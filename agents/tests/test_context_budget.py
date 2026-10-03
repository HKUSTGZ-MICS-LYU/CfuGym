"""Offline contract tests for conservative context planning and compaction."""

import copy
import unittest

from agents.cfu_design_agent.runtime import context_budget as budget


def batch(ident, content="read output"):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": ident, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": ident, "content": content},
    ]


class ContextBudgetTest(unittest.TestCase):
    def test_unknown_and_model_metadata_override(self):
        self.assertEqual(budget.resolve_window(), budget.WindowSpec(32768, 4096, "unknown-soft", False))
        self.assertEqual(budget.resolve_window(metadata={"model": {"context_length": 64000}}).window_tokens, 64000)
        self.assertEqual(budget.resolve_window({"context_window": 8000, "output_reserve_tokens": 2000},
                                               {"context_length": 64000}).source, "config")
        for value in (True, 0, -1, "64000", 64.0):
            with self.subTest(value=value), self.assertRaises(budget.ContextBudgetError):
                budget.resolve_window(metadata={"context_window": value})
        with self.assertRaisesRegex(budget.ContextBudgetError, "conflicting"):
            budget.resolve_window(metadata={"context_length": 20000, "context_window": 30000})
        self.assertEqual(budget.resolve_window({"context_window": 20000}, {"context_length": 0}).source, "config")
        with self.assertRaisesRegex(budget.ContextBudgetError, "smaller"):
            budget.resolve_window({"context_window": 4000})
        with self.assertRaisesRegex(budget.ContextBudgetError, "conflicting"):
            budget.resolve_window(metadata={"context_window": 20000,
                                            "model": {"context_length": 30000}})

    def test_estimate_includes_tools_response_config_and_utf8(self):
        messages = [{"role": "system", "content": "世界"}, {"role": "user", "content": "hello"}]
        spec = budget.WindowSpec(16000, 4096, "config", True)
        plain = budget.evaluate_budget(messages, window=spec)
        rich = budget.evaluate_budget(messages, window=spec, tools=[{"description": "x" * 800}],
                                      response_config={"response_format": {"type": "json_object"},
                                                       "max_completion_tokens": 5000})
        self.assertGreater(plain.estimated_input_tokens, len('世界'.encode('utf8')))
        self.assertGreater(rich.estimated_input_tokens, plain.estimated_input_tokens + 800)
        self.assertEqual(rich.reserved_output_tokens, 5000)
        with self.assertRaisesRegex(budget.ContextBudgetError, "positive integer"):
            budget.evaluate_budget(messages, response_config={"max_tokens": True})
        with self.assertRaisesRegex(budget.ContextBudgetError, "serialized"):
            budget.evaluate_budget([{"role": "user", "content": object()}])

    def test_threshold_edges_are_inclusive(self):
        messages = [{"role": "user", "content": "abc"}]
        initial = budget.evaluate_budget(messages)
        for percent, level in ((79, "ok"), (80, "warning"), (90, "compact"), (100, "unfit")):
            # Choose a window whose exact threshold matches the fixed total.
            # Multiply request content until divisible by each percent.
            size = percent
            while True:
                messages[0]["content"] = "a" * size
                used = budget.evaluate_budget(messages).total_tokens
                if used * 100 % percent == 0:
                    break
                size += 1
            window = used * 100 // percent
            result = budget.evaluate_budget(messages, window=budget.WindowSpec(window, 4096, "config", True))
            self.assertEqual(result.level, level)

    def test_compacts_atomic_old_batches_without_mutation_and_preserves_evidence(self):
        messages = ([{"role": "system", "content": "policy"},
                     {"role": "user", "content": "initial task"}]
                    + batch("old", "z" * 1300)
                    + batch("failure", "failed build: " + "f" * 350)
                    + batch("edit", "patch applied: " + "e" * 350)
                    + [{"role": "user", "content": "current failure and latest edits must stay"}])
        saved = copy.deepcopy(messages)
        before = budget.evaluate_budget(messages)
        after_old_removed = budget.evaluate_budget(messages[:2] + messages[4:])
        self.assertGreater(before.total_tokens, after_old_removed.total_tokens)
        # Select a window so original >=90%, but removing the old batch <90%.
        window_tokens = max(before.total_tokens, (after_old_removed.total_tokens * 100 // 89) + 1)
        while before.total_tokens * 100 < window_tokens * 90:
            window_tokens -= 1
        result = budget.compact_completed_tool_turns(messages,
            window=budget.WindowSpec(window_tokens, 4096, "config", True))
        self.assertEqual(result.removed_batches, 1)
        self.assertIn("Context compaction", result.messages[2]["content"])
        self.assertEqual(result.messages[:2] + result.messages[3:], messages[:2] + messages[4:])
        self.assertIn(result.after.level, ("ok", "warning"))
        self.assertEqual(messages, saved)

    def test_preserves_current_dynamic_turn_and_explicit_indices(self):
        messages = ([{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}]
                    + batch("old", "z" * 2000)
                    + [{"role": "user", "content": "dynamic reminder"}]
                    + batch("live", "y" * 1000))
        spec = budget.WindowSpec(7100, 4096, "config", True)
        result = budget.compact_completed_tool_turns(messages, window=spec)
        self.assertIn("Context compaction", result.messages[2]["content"])
        self.assertEqual(result.messages[:2] + result.messages[3:], messages[:2] + messages[4:])
        with self.assertRaisesRegex(budget.ContextBudgetError, "cannot fit safely"):
            budget.compact_completed_tool_turns(messages, window=spec, protected_indices=(2,))

    def test_latest_failure_outside_tool_batch_is_protected(self):
        messages = ([{"role": "system", "content": "policy"},
                     {"role": "user", "content": "task"}]
                    + batch("failure", "failed build: " + "f" * 900)
                    + batch("old", "z" * 2000)
                    + batch("latest", "recent observation")
                    + [{"role": "assistant", "content": "Current failure: still failing"},
                       {"role": "user", "content": "continue"}])
        spec = budget.WindowSpec(7300, 4096, "config", True)
        result = budget.compact_completed_tool_turns(messages, window=spec)
        self.assertEqual(result.messages[:4] + result.messages[5:], messages[:4] + messages[6:])
        self.assertIn("Context compaction", result.messages[4]["content"])

    def test_unfit_is_explicit_and_never_splits_calls(self):
        messages = [{"role": "system", "content": "x" * 6000},
                    {"role": "user", "content": "task"}] + batch("old")
        with self.assertRaisesRegex(budget.ContextBudgetError, "cannot fit safely"):
            budget.compact_completed_tool_turns(messages, window=budget.WindowSpec(5000, 4096, "config", True))
        incomplete = [{"role": "assistant", "tool_calls": [
            {"id": "first"}, {"id": "second"}]},
                      {"role": "tool", "tool_call_id": "first", "content": "result"}]
        with self.assertRaises(budget.ContextBudgetError):
            budget.compact_completed_tool_turns(incomplete,
                window=budget.WindowSpec(4500, 4096, "config", True))
        with self.assertRaisesRegex(budget.ContextBudgetError, "orphan"):
            budget.compact_completed_tool_turns([{"role": "tool", "tool_call_id": "first"}])


if __name__ == "__main__":
    unittest.main()
