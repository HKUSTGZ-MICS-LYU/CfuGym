"""Micro tests for executor-owned submission aliases; no model or file effects."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from agents.cfu_design_agent.runtime import failures, llm_step
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.stage_executor import StageExecutionRequest, StageExecutor


class _RoundClient:
    available = True
    model = "fake-submit-alias"
    _stage_round_client = object()  # Never construct an SDK or access a network.

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def model_round(self, sdk, messages, *, allow_tools, tools, response_format):
        self.seen.append((list(messages), tools))
        if not self.turns:
            raise AssertionError("unexpected model turn")
        text, calls = self.turns.pop(0)
        return SimpleNamespace(
            text=text, reasoning="", reasoning_source="", finish_reason="stop",
            extra_keys=(), input_tokens=1, output_tokens=1, total_tokens=2,
            message=SimpleNamespace(tool_calls=[
                SimpleNamespace(id=f"call_{len(self.seen)}_{i}", function=SimpleNamespace(
                    name=name, arguments=json.dumps(arguments)))
                for i, (name, arguments) in enumerate(calls)
            ]),
        )


class _IsaTools:
    def __init__(self, names=("read_isa", "edit_isa", "check_isa")):
        self.names = names
        self.calls = []
        self.draft = "original"
        self.rolled_back = False

    def schemas(self):
        return [{"type": "function", "function": {"name": name,
                 "parameters": {"type": "object", "properties": {}}}}
                for name in self.names]

    def run(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "edit_isa":
            self.draft = arguments["draft"]
        return "ok"

    def rollback(self):
        self.rolled_back = True


class StageSubmitAliasTest(unittest.TestCase):
    @staticmethod
    def _request(**overrides):
        values = dict(stage="design_contract", label="isa", artifact_stage="isa",
                      tool_calls_relpath="08_isa/isa_tool_calls.yaml",
                      system="system", user="user", fallback="fallback",
                      tools_enabled=True, native_tool_loop=True,
                      submit_tool_name="submit_isa", require_delivery=False)
        values.update(overrides)
        return StageExecutionRequest(**values)

    def _executor(self, toolset=None):
        toolset = toolset if toolset is not None else _IsaTools()
        has_deliverable = mock.Mock(return_value=False)
        executor = StageExecutor(
            complete=mock.Mock(side_effect=AssertionError("plain completion not expected")),
            record_output=mock.Mock(return_value="fake-raw.txt"),
            write_tool_calls=mock.Mock(), has_deliverable=has_deliverable,
            toolset_factory=lambda *_args, **_kwargs: toolset,
        )
        return executor, toolset, has_deliverable

    @staticmethod
    def _names(seen_turn):
        return [schema["function"]["name"] for schema in seen_turn[1]]

    def test_isa_submission_only_runs_independent_gate(self):
        executor, tools, has_deliverable = self._executor()
        gate = mock.Mock(return_value=True)
        client = _RoundClient([("", [("submit_isa", {"delivery": "ISA draft ready"})])])
        result = executor.execute({}, object(), self._request(submit_stage=gate), client=client)
        gate.assert_called_once_with("ISA draft ready")
        has_deliverable.assert_not_called()
        self.assertEqual(tools.calls, [])
        self.assertEqual(result.result.text, "ISA draft ready")
        self.assertIsNone(result.result.accepted_submit)
        self.assertEqual([call["name"] for call in result.tool_calls], ["submit_isa"])
        self.assertTrue(result.tool_calls[0]["ok"])
        self.assertEqual(self._names(client.seen[0]),
                         ["read_isa", "edit_isa", "check_isa", "submit_isa"])
        parameters = client.seen[0][1][-1]["function"]["parameters"]
        self.assertEqual(parameters["required"], ["delivery"])
        self.assertNotIn("design_type", parameters["properties"])

    def test_rejected_gate_can_edit_check_and_resubmit_after_budget(self):
        executor, tools, _ = self._executor()
        judged = []

        def gate(delivery):
            judged.append((delivery, tools.draft))
            return tools.draft == "fixed", "draft is missing operand semantics"

        client = _RoundClient([
            ("", [("read_isa", {})]),
            ("", [("submit_isa", {"delivery": "first draft"})]),
            ("", [("edit_isa", {"draft": "fixed"}), ("check_isa", {})]),
            ("", [("submit_isa", {"delivery": "corrected draft"})]),
        ])
        result = executor.execute({"max_tool_rounds": 1, "max_tool_calls": 1}, object(),
                                  self._request(submit_stage=gate), client=client)
        self.assertEqual(judged, [("first draft", "original"), ("corrected draft", "fixed")])
        self.assertEqual([name for name, _ in tools.calls], ["read_isa", "edit_isa", "check_isa"])
        self.assertEqual([call["ok"] for call in result.tool_calls], [True, False, True, True, True])
        self.assertIn("draft is missing operand semantics", str(client.seen[2][0]))
        self.assertIn("only submit_isa will be available", str(client.seen[2][0]))
        self.assertNotIn("submit_stage", str(client.seen))
        self.assertFalse(tools.rolled_back)

    def test_submit_with_companion_dispatches_only_companion_to_tools(self):
        executor, tools, _ = self._executor()
        drafts = []
        client = _RoundClient([("", [("edit_isa", {"draft": "fixed"}),
                                     ("submit_isa", {"delivery": "ready"})])])
        result = executor.execute({}, object(), self._request(
            submit_stage=lambda _delivery: drafts.append(tools.draft) or True), client=client)
        self.assertEqual(drafts, ["fixed"])
        self.assertEqual(tools.calls, [("edit_isa", {"draft": "fixed"})])
        self.assertEqual([call["name"] for call in result.tool_calls], ["edit_isa", "submit_isa"])

    def test_offered_schema_finalize_and_prose_reminder_use_alias(self):
        executor, _, _ = self._executor()
        client = _RoundClient([
            ("", [("read_isa", {})]), ("prose is not submission", []),
            ("", [("submit_isa", {"delivery": "ready"})]),
        ])
        executor.execute({"max_tool_rounds": 1}, object(),
                         self._request(submit_stage=lambda _: True), client=client)
        self.assertIn("only submit_isa is available", str(client.seen[0][0]))
        self.assertEqual(self._names(client.seen[1]), ["submit_isa"])
        self.assertEqual(self._names(client.seen[2]), ["submit_isa"])
        self.assertIn("Finalize this stage now. Only submit_isa", str(client.seen[1][0]))
        self.assertIn("No stage was submitted. Only submit_isa", str(client.seen[2][0]))
        self.assertNotIn("submit_stage", str(client.seen))

    def test_malformed_alias_submission_skips_companions_and_can_retry(self):
        for malformed, error in (
            ([("submit_isa", {})], "empty submit_isa delivery"),
            ([("submit_isa", {"delivery": "one"}), ("submit_isa", {"delivery": "two"})],
             "exactly one submit_isa call"),
        ):
            with self.subTest(error=error):
                executor, tools, _ = self._executor()
                gate = mock.Mock(return_value=True)
                client = _RoundClient([
                    ("", malformed + [("edit_isa", {"draft": "must not apply"})]),
                    ("", [("submit_isa", {"delivery": "ready"})]),
                ])
                executor.execute({}, object(), self._request(submit_stage=gate), client=client)
                gate.assert_called_once_with("ready")
                self.assertEqual(tools.calls, [])
                self.assertIn(error, str(client.seen[1][0]))
                self.assertIn("fix the malformed submit_isa call first", str(client.seen[1][0]))
                self.assertNotIn("submit_stage", str(client.seen))

    def test_prose_never_passes_and_missing_submit_error_uses_alias(self):
        executor, tools, _ = self._executor()
        gate = mock.Mock(return_value=True)
        client = _RoundClient([("prose", [])] * 3)
        with self.assertRaisesRegex(failures.StageDeliveryFailure, "did not call submit_isa"):
            executor.execute({"max_tool_rounds": 1}, object(),
                             self._request(submit_stage=gate), client=client)
        gate.assert_not_called()
        self.assertTrue(tools.rolled_back)

    def test_original_submit_name_is_not_an_alias_when_not_offered(self):
        executor, tools, _ = self._executor()
        gate = mock.Mock(return_value=True)
        client = _RoundClient([
            ("", [("submit_stage", {"delivery": "not a submission"})]),
            ("", [("submit_isa", {"delivery": "real submission"})]),
        ])
        result = executor.execute({}, object(), self._request(submit_stage=gate), client=client)
        gate.assert_called_once_with("real submission")
        self.assertFalse(result.tool_calls[0]["ok"])
        self.assertIn("tool not offered", result.tool_calls[0]["result"])
        self.assertEqual(tools.calls, [])

    def test_default_submit_name_keeps_implement_declaration_validation(self):
        executor, tools, _ = self._executor()
        gate = mock.Mock(return_value=True)
        request = self._request(stage="implement", submit_stage=gate)
        # Exercise the dataclass default rather than passing the old name explicitly.
        request = StageExecutionRequest(**{key: value for key, value in vars(request).items()
                                           if key != "submit_tool_name"})
        client = _RoundClient([
            ("", [("submit_stage", {"delivery": "missing declaration"})]),
            ("", [("submit_stage", {"delivery": "ready", "design_type": "complete"})]),
        ])
        result = executor.execute({}, object(), request, client=client)
        gate.assert_called_once_with("ready")
        self.assertEqual(tools.calls, [])
        self.assertEqual(result.result.accepted_submit["design_type"], "complete")
        self.assertFalse(result.tool_calls[0]["ok"])
        self.assertIn("design_type must be explicitly", str(client.seen[1][0]))
        self.assertIn("design_type", client.seen[0][1][-1]["function"]["parameters"]["required"])

    def test_schema_name_collision_fails_before_model_or_tools(self):
        for submit_name in ("submit_isa", "submit_stage"):
            with self.subTest(submit_name=submit_name):
                executor, tools, _ = self._executor(_IsaTools(("read_isa", submit_name)))
                gate = mock.Mock(return_value=True)
                client = _RoundClient([])
                with self.assertRaisesRegex(ValueError, f"{submit_name} is reserved"):
                    executor.execute({}, object(), self._request(
                        submit_tool_name=submit_name, submit_stage=gate), client=client)
                self.assertEqual(client.seen, [])
                self.assertEqual(tools.calls, [])
                self.assertTrue(tools.rolled_back)
                gate.assert_not_called()

    def test_invalid_submit_names_fail_closed_and_valid_tokens_are_preserved(self):
        for invalid in ("", " ", " submit_isa", "submit_isa\n", "submit.isa", "submit/isa",
                        "submit isa", "提交", "a" * 65, None, 7, []):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "submit_tool_name.*function token"):
                    self._request(submit_tool_name=invalid)
        for valid in ("submit_isa", "submit-isa", "X", "9", "a" * 64):
            self.assertEqual(self._request(submit_tool_name=valid).submit_tool_name, valid)

    def test_alias_requires_independent_callback(self):
        executor, _, _ = self._executor()
        client = _RoundClient([])
        with self.assertRaisesRegex(ValueError, "independent submit_isa gate"):
            executor.execute({}, object(), self._request(), client=client)
        self.assertEqual(client.seen, [])

    def test_llm_step_forwards_submit_alias_and_callback(self):
        expected = LlmResult("ready", used_llm=True)
        gate = mock.Mock(return_value=True)
        with mock.patch.object(llm_step, "_with_run_memory", side_effect=lambda *a, **k: a[3]), \
             mock.patch.object(llm_step, "StageExecutor") as executor:
            executor.return_value.execute.return_value = SimpleNamespace(
                result=expected, raw_output="fake-raw.txt", tool_calls=())
            result, _, _ = llm_step._run_agent_step(
                {}, object(), stage="design_contract", label="isa", artifact_stage="isa",
                tool_calls_relpath="08_isa/isa_tool_calls.yaml", system="system", user="user",
                fallback="fallback", client=SimpleNamespace(available=True),
                native_tool_loop=True, submit_tool_name="submit_isa", submit_stage=gate,
                require_delivery=False,
            )
        request = executor.return_value.execute.call_args.args[2]
        self.assertIs(result, expected)
        self.assertEqual(request.submit_tool_name, "submit_isa")
        self.assertIs(request.submit_stage, gate)
        self.assertTrue(request.native_tool_loop)
        self.assertFalse(request.require_delivery)


if __name__ == "__main__":
    unittest.main()
