"""Context compaction must be visible in the operator's CLI log.

Compaction changes what the model can see, so it is not a silent optimization:
the run prints a durable ``[context]`` line when completed tool batches are
dropped, records the counts in the request audit, and reports them on the
per-call ``[llm]`` line. These tests pin that contract offline.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.runtime import context_budget, llm, request_audit
from agents.cfu_design_agent.runtime.llm import OpenAiTextClient

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]

SYSTEM = {"role": "system", "content": "policy"}
USER = {"role": "user", "content": "task"}


def batch(ident: str, content: str = "read output") -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": ident, "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": ident, "content": content},
    ]


def _window(tokens: int) -> context_budget.WindowSpec:
    return context_budget.WindowSpec(tokens, 4096, "config", True)


class CompactionNoticeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name).resolve()
        self.run_root = self.workdir / "agents/generated/runs/notice-run"
        self.run_root.mkdir(parents=True)
        self.audit = self.run_root / request_audit.AUDIT_FILENAME
        self.notices: list[dict] = []
        llm.set_notice_callback(self.notices.append)
        self.addCleanup(llm.set_notice_callback, None)
        # The active stage is process-wide state set by the graph node wrapper;
        # pin it so the notice text is deterministic in a full-suite run.
        llm.set_stage("analyze_workload")
        self.addCleanup(llm.set_stage, "")

    def _events(self) -> list[dict]:
        if not self.audit.exists():
            return []
        return [json.loads(line) for line in self.audit.read_text().splitlines()]

    def _compacting_client_and_messages(self):
        """A transcript whose oldest batch must be dropped to fit ``window``."""
        messages = [SYSTEM, USER] + batch("old", "z" * 1300) + batch("latest", "y" * 300)
        wide = 10 ** 6
        before = context_budget.evaluate_budget(messages, tools=TOOLS, window=_window(wide))
        after = context_budget.evaluate_budget(
            messages[:2] + messages[-2:], tools=TOOLS, window=_window(wide))
        # Smallest window where the full transcript is >=90% but one batch less
        # already fits, so exactly one batch may be removed.
        window_tokens = max(before.total_tokens, (after.total_tokens * 100 // 89) + 1)
        while before.total_tokens * 100 < window_tokens * 90:
            window_tokens -= 1
        client = OpenAiTextClient(config={
            "model": "m", "api_key": "sk-offline", "api_mode": "chat",
            "context_window_tokens": window_tokens, "output_reserve_tokens": 4096,
        })
        return client, messages

    def test_real_compaction_emits_a_notice_and_audits_it(self) -> None:
        client, messages = self._compacting_client_and_messages()

        with request_audit.use_request_audit(self.run_root, self.workdir):
            _, status = client._budget_messages(messages, tools=TOOLS, sdk=object())

        self.assertIn(status.level, {"ok", "warning"})
        self.assertEqual(client._last_compaction["removed_batches"], 1)
        self.assertEqual(client._last_compaction["stage"], "analyze_workload")
        compaction = [notice for notice in self.notices
                      if notice["kind"] == "context.compaction"]
        self.assertEqual(len(compaction), 1)
        self.assertIn("context compacted", compaction[0]["message"])
        self.assertIn("removed 1 completed tool batch,", compaction[0]["message"])
        self.assertIn(f"(level {client._last_compaction['level']}", compaction[0]["message"])
        self.assertIn("stage analyze_workload", compaction[0]["message"])
        audited = [event for event in self._events() if event["event"] == "compaction"]
        self.assertEqual(len(audited), 1)
        self.assertEqual(audited[0]["metadata"]["removed_batches"], 1)

    def test_compaction_is_reported_on_the_result(self) -> None:
        """The stage line can show it: the count travels on ``LlmResult``."""
        compact = context_budget.BudgetStatus(900, 100, 1000, 1000, 1.0, "compact")
        lowered = context_budget.BudgetStatus(100, 100, 200, 1000, 0.2, "ok")
        final = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop",
                                     message=SimpleNamespace(content="DONE", tool_calls=None))],
            usage=None)
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=mock.Mock(return_value=final))))

        with mock.patch.object(context_budget, "evaluate_budget", return_value=compact), \
                mock.patch.object(context_budget, "compact_completed_tool_turns",
                                  return_value=context_budget.CompactionResult(
                                      messages=[SYSTEM, USER], before=compact,
                                      after=lowered, removed_batches=2)), \
                mock.patch("openai.OpenAI", return_value=sdk), \
                request_audit.use_request_audit(self.run_root, self.workdir):
            client = OpenAiTextClient(config={"model": "m", "api_key": "k", "api_mode": "chat"})
            result = client.complete_with_tools(
                "policy", "task", tools=TOOLS, tool_runner=lambda n, a: "x", max_rounds=1)

        self.assertEqual(result.text, "DONE")
        self.assertEqual(result.context_compaction["removed_batches"], 2)
        self.assertEqual(result.context_compaction["after_total_tokens"], 200)
        self.assertTrue(any(notice["kind"] == "context.compaction" for notice in self.notices))

    def test_warning_is_emitted_once_per_climb(self) -> None:
        warning = context_budget.BudgetStatus(850, 100, 950, 1000, 0.95, "warning")
        client = OpenAiTextClient(config={"model": "m", "api_key": "k", "api_mode": "chat"})
        with mock.patch.object(context_budget, "evaluate_budget", return_value=warning), \
                request_audit.use_request_audit(self.run_root, self.workdir):
            client._budget_messages([SYSTEM, USER], sdk=object())
            client._budget_messages([SYSTEM, USER], sdk=object())

        warnings = [notice for notice in self.notices if notice["kind"] == "context.warning"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("95% full", warnings[0]["message"])
        self.assertIn("compaction starts at 90%", warnings[0]["message"])


class CliNoticeRenderingTest(unittest.TestCase):
    def tearDown(self) -> None:
        instrument.disable_progress()
        llm.set_notice_callback(None)

    def test_notice_prints_a_durable_line_including_when_redirected(self) -> None:
        instrument.enable_progress(["analysis"])
        buffer = io.StringIO()
        # A redirected stdout is the case the live progress line deliberately
        # suppresses, so the notice must not depend on it.
        with contextlib.redirect_stdout(buffer):
            llm._emit_notice(
                "context.compaction",
                "context compacted: removed 2 completed tool batches, "
                "est. 121k -> 78k tokens (level ok, stage analyze_workload)")

        printed = buffer.getvalue()
        self.assertIn("[context] context compacted", printed)
        self.assertIn("121k -> 78k", printed)
        self.assertTrue(printed.endswith("\n"))

    def test_notice_is_silent_when_progress_is_disabled(self) -> None:
        instrument.enable_progress(["analysis"])
        instrument.disable_progress()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            llm._emit_notice("context.compaction", "context compacted: removed 1 batch")
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
