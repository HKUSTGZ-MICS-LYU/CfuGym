"""Unit tests for the stage-local coding executor boundary."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime import llm_step
from agents.cfu_design_agent.runtime.llm import (
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS_PER_ROUND,
    DEFAULT_MAX_TOOL_ROUNDS, LlmResult,
)
from agents.cfu_design_agent.runtime.stage_executor import (
    StageExecutionRequest,
    StageExecutor,
    _RunDirView,
)


class RunDirViewTest(unittest.TestCase):
    """Every tool result the model sees renders this run's root as {{run-dir}}."""

    def test_tool_output_is_virtualized_and_attributes_delegate(self) -> None:
        class Fake:
            rolled_back = False

            def schemas(self):
                return [{"type": "function", "function": {"name": "read_file"}}]

            def run(self, name, arguments):
                return "see agents/generated/runs/r-1/08_isa/cfu_isa.yaml"

            def rollback(self):
                self.rolled_back = True
                return "rolled back"

        inner = Fake()
        view = _RunDirView(inner, "agents/generated/runs/r-1/")
        self.assertEqual(view.schemas(), inner.schemas())
        self.assertEqual(view.run("read_file", {}), "see {{run-dir}}/08_isa/cfu_isa.yaml")
        self.assertEqual(view.rollback(), "rolled back")
        self.assertTrue(inner.rolled_back)
        # Without a run root there is nothing to hide.
        self.assertEqual(_RunDirView(inner, "").run("read_file", {}),
                         "see agents/generated/runs/r-1/08_isa/cfu_isa.yaml")


class _Session:
    def __init__(self) -> None:
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True


class _Toolset:
    latest = None

    def __init__(self, toolbox, observation_index=None, failure_index=None, **kwargs) -> None:
        self.toolbox = toolbox
        self.observation_index = observation_index
        self.failure_index = failure_index
        self.kwargs = kwargs
        self.session = _Session()
        _Toolset.latest = self

    def schemas(self):
        return [{"type": "function", "function": {"name": "read_file"}}]

    def run(self, name, arguments):
        return f"{name}:{arguments.get('path', '')}"


class _ToolClient:
    available = True
    model = "tool-client"

    def __init__(self, result=None, error=None) -> None:
        self.result = result or LlmResult("done", used_llm=True)
        self.error = error
        self.kwargs = None

    def complete_with_tools(self, system, user, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


class _RoundClient:
    available = True
    model = "fake-single-round"
    _stage_round_client = object()  # Never instantiates a network SDK client.

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def _chat_round(self, sdk, messages, *, allow_tools, tools, response_format):
        self.seen.append((list(messages), [item["function"]["name"] for item in tools]))
        if not self.turns:
            raise AssertionError("unexpected model request")
        text, calls = self.turns.pop(0)
        return SimpleNamespace(
            text=text, reasoning="", reasoning_source="", finish_reason="stop",
            extra_keys=(), input_tokens=2, output_tokens=3, total_tokens=5,
            message=SimpleNamespace(content=text, tool_calls=[
                SimpleNamespace(id=f"call_{len(self.seen)}_{i}", function=SimpleNamespace(
                    name=name, arguments=json.dumps(arguments)))
                for i, (name, arguments) in enumerate(calls)
            ]),
        )


class StageExecutorTest(unittest.TestCase):
    def _executor(self):
        calls = {"plain": [], "record": [], "artifacts": [], "live": []}

        def plain(*args, **kwargs):
            calls["plain"].append((args, kwargs))
            return LlmResult("plain", used_llm=True), "plain.txt"

        def record(*args, **kwargs):
            calls["record"].append((args, kwargs))
            return "raw.txt"

        def artifact(*args):
            calls["artifacts"].append(args)

        executor = StageExecutor(
            complete=plain,
            record_output=record,
            write_tool_calls=artifact,
            has_deliverable=lambda text: text == "done",
            on_tool_call=calls["live"].append,
            toolset_factory=_Toolset,
        )
        return executor, calls

    @staticmethod
    def _request(**overrides):
        values = {
            "stage": "implement_sw",
            "label": "software",
            "artifact_stage": "implementation",
            "tool_calls_relpath": "09_implementation/sw_tool_calls.yaml",
            "system": "system",
            "user": "user",
            "fallback": "fallback",
        }
        values.update(overrides)
        return StageExecutionRequest(**values)

    def test_plain_execution_delegates_to_compatible_completion(self) -> None:
        executor, calls = self._executor()
        client = object()
        output = executor.execute({}, object(), self._request(), client=client)
        self.assertEqual(output.result.text, "plain")
        self.assertEqual(output.raw_output, "plain.txt")
        self.assertEqual(output.tool_calls, ())
        args, kwargs = calls["plain"][0]
        self.assertEqual(args[:6], ({}, "implement_sw", "software", "system", "user", "fallback"))
        self.assertIs(kwargs["client"], client)

    def test_tool_execution_forwards_budget_and_records_transcript(self) -> None:
        record = {"round": 1, "name": "read_file", "arguments": {"path": "x"},
                  "result": "body", "ok": True}
        client = _ToolClient(LlmResult("done", used_llm=True, tool_calls=(record,)))
        executor, calls = self._executor()
        output = executor.execute(
            {"max_tool_rounds": 3, "max_tool_calls": 7},
            object(),
            self._request(
                tools_enabled=True,
                require_delivery=True,
                observation_index={"observations": {"x": {}}},
                response_format={"type": "json_object"},
                deliverable_hint="a delivery",
            ),
            client=client,
        )
        self.assertEqual(output.tool_calls, (record,))
        self.assertEqual(client.kwargs["max_rounds"], 3)
        self.assertEqual(client.kwargs["max_total_calls"], 7)
        self.assertEqual(client.kwargs["deliverable_hint"], "a delivery")
        self.assertTrue(client.kwargs["is_deliverable"]("done"))
        self.assertEqual(calls["record"][0][0][1:4], ("implement_sw", "software", client.result))
        self.assertEqual(calls["artifacts"][0][1:3],
                         ("implementation", "09_implementation/sw_tool_calls.yaml"))
        self.assertEqual(calls["artifacts"][0][3], [record])

    def test_tool_execution_uses_shared_defaults_when_state_omits_budgets(self) -> None:
        client = _ToolClient()
        executor, _ = self._executor()
        executor.execute({}, object(), self._request(tools_enabled=True), client=client)
        self.assertEqual(client.kwargs["max_rounds"], DEFAULT_MAX_TOOL_ROUNDS)
        self.assertEqual(client.kwargs["max_total_calls"], DEFAULT_MAX_TOOL_CALLS)

    def test_tool_exception_audits_failure_and_returns_empty_transcript(self) -> None:
        client = _ToolClient(error=RuntimeError("unsupported"))
        executor, calls = self._executor()
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            executor.execute(
                {}, object(), self._request(tools_enabled=True), client=client
            )
        self.assertEqual(len(calls["plain"]), 0)
        self.assertTrue(_Toolset.latest.session.rolled_back)
        failure = calls["artifacts"][0][3][0]
        self.assertEqual(failure["name"], "(tool loop failed)")
        self.assertIn("unsupported", failure["arguments"]["error"])

    def test_degraded_tool_result_fails_closed_and_is_audited(self) -> None:
        result = LlmResult("done", used_llm=True, degraded_reason="no tools")
        client = _ToolClient(result)
        executor, calls = self._executor()
        with self.assertRaisesRegex(RuntimeError, "no tools"):
            executor.execute(
                {}, object(), self._request(tools_enabled=True), client=client
            )
        self.assertEqual(calls["artifacts"][0][3][0]["name"], "(tools unavailable)")
        self.assertTrue(_Toolset.latest.session.rolled_back)

    def test_record_failure_rolls_back_before_propagating(self) -> None:
        executor, _calls = self._executor()
        executor._record_output = mock.Mock(side_effect=OSError("artifact failed"))
        with self.assertRaises(OSError):
            executor.execute(
                {}, object(), self._request(tools_enabled=True),
                client=_ToolClient(LlmResult("done", used_llm=True)),
            )
        self.assertTrue(_Toolset.latest.session.rolled_back)

    def test_native_graph_interleaves_tools_and_gates_explicit_submission(self) -> None:
        client = _RoundClient([
            ("", [("read_file", {"path": "one"})]),
            ("", [("read_file", {"path": "two"})]),
            ("", [("submit_stage", {"delivery": "done"})]),
        ])
        executor, calls = self._executor()
        submitted = []
        saver = InMemorySaver()
        output = executor.execute(
            {"max_tool_rounds": 2, "max_tool_calls": 2}, object(),
            self._request(tools_enabled=True, native_tool_loop=True,
                          submit_stage=lambda delivery: submitted.append(delivery) or True,
                          stage_checkpointer=saver, stage_thread_id="test-stage-native-unique-1"),
            client=client,
        )
        self.assertEqual(submitted, ["done"])
        self.assertEqual(output.result.text, "done")
        self.assertEqual(output.result.tool_rounds, 2)
        self.assertEqual([c["arguments"]["path"] for c in output.tool_calls[:-1]], ["one", "two"])
        self.assertEqual([c["name"] for c in calls["live"]], ["read_file", "read_file", "submit_stage"])
        self.assertTrue(output.tool_calls[-1]["ok"])
        self.assertEqual(client.seen[-1][1], ["submit_stage"])
        self.assertEqual(saver.get_tuple({"configurable": {
            "thread_id": "test-stage-native-unique-1"}}).checkpoint["channel_values"]["total_calls"], 2)

    def test_native_graph_default_accepts_four_calls_in_one_round(self) -> None:
        client = _RoundClient([
            ("", [("read_file", {"path": str(i)})
                   for i in range(DEFAULT_MAX_TOOL_CALLS_PER_ROUND)]),
            ("", [("submit_stage", {"delivery": "done"})]),
        ])
        executor, _ = self._executor()
        output = executor.execute(
            {}, object(),
            self._request(tools_enabled=True, native_tool_loop=True,
                          submit_stage=lambda _delivery: True), client=client,
        )
        self.assertEqual(len(output.tool_calls), DEFAULT_MAX_TOOL_CALLS_PER_ROUND + 1)
        self.assertTrue(all(call["ok"] for call in output.tool_calls))
        self.assertEqual(output.result.tool_rounds, 1)

    def test_native_graph_default_skips_calls_beyond_four(self) -> None:
        client = _RoundClient([
            ("", [("read_file", {"path": str(i)}) for i in range(6)]),
            ("", [("submit_stage", {"delivery": "done"})]),
        ])
        executor, _ = self._executor()
        output = executor.execute(
            {}, object(), self._request(tools_enabled=True, native_tool_loop=True,
                                        submit_stage=lambda _delivery: True), client=client,
        )
        self.assertEqual([c["ok"] for c in output.tool_calls[:6]],
                         [True, True, True, True, False, False])
        self.assertTrue(all(c["result"].startswith("skipped:") for c in output.tool_calls[4:6]))

    def test_native_graph_rejects_unoffered_and_over_budget_calls(self) -> None:
        client = _RoundClient([
            ("", [("edit_hardware", {}), ("read_file", {"path": "one"}),
                   ("read_file", {"path": "two"})]),
            ("", [("submit_stage", {"delivery": "done"})]),
        ])
        executor, calls = self._executor()
        output = executor.execute(
            {"max_tool_rounds": 1, "max_tool_calls": 1}, object(),
            self._request(tools_enabled=True, native_tool_loop=True,
                          submit_stage=lambda _delivery: True), client=client,
        )
        self.assertEqual([c["ok"] for c in output.tool_calls], [False, True, False, True])
        self.assertIn("denied by access policy", output.tool_calls[0]["result"])
        self.assertIn("budget", output.tool_calls[2]["result"])
        self.assertEqual(len(calls["artifacts"]), 1)
        self.assertEqual(client.seen[-1][1], ["submit_stage"])

    def test_native_graph_records_a_tool_usage_error_as_failed(self) -> None:
        # Observed 2026-09-29: every "usage error:" was journaled as ok=True, so
        # the transcript of a stage that spent its whole budget on bad calls
        # looked like a stage that had answered.
        client = _RoundClient([
            ("", [("read_file", {"path": "nope"})]),
            ("", [("submit_stage", {"delivery": "done"})]),
        ])
        executor, _ = self._executor()
        with mock.patch.object(_Toolset, "run",
                               return_value="usage error: no such file: nope"):
            output = executor.execute(
                {}, object(),
                self._request(tools_enabled=True, native_tool_loop=True,
                              submit_stage=lambda _delivery: True), client=client,
            )
        self.assertEqual([c["ok"] for c in output.tool_calls], [False, True])
        self.assertIn("usage error", output.tool_calls[0]["result"])

    def test_a_rejection_can_still_be_fixed_after_the_tool_budget_is_spent(self) -> None:
        # The gate demands an applied change, so a corrective round that could
        # only resubmit would be useless: the tools come back for it.
        client = _RoundClient([
            ("", [("read_file", {"path": "one"})]),
            ("", [("submit_stage", {"delivery": "described, not applied"})]),
            ("", [("read_file", {"path": "two"})]),
            ("", [("submit_stage", {"delivery": "applied"})]),
        ])
        executor, _ = self._executor()
        answers = [True]

        def gate(delivery: str):
            if answers:
                answers.pop()
                return False, "no hardware target was edited"
            return True, ""

        output = executor.execute(
            # One tool round: the first round is the whole exploration budget.
            {"max_tool_rounds": 1, "max_tool_calls": 1}, object(),
            self._request(tools_enabled=True, native_tool_loop=True, submit_stage=gate),
            client=client,
        )
        self.assertEqual(output.result.text, "applied")
        self.assertFalse(output.tool_calls[1]["ok"])
        # The corrective call ran instead of being skipped for budget.
        self.assertTrue(output.tool_calls[2]["ok"], output.tool_calls[2])
        self.assertEqual(output.tool_calls[2]["result"], "read_file:two")

    def test_a_submission_batched_with_other_calls_is_not_fatal(self) -> None:
        # Observed 2026-09-30 (crc32-run-0002): the model asked for one more
        # read in the same turn as `submit_stage`, and the stage died with
        # "submit_stage must be the sole tool call" before the gate ran.
        client = _RoundClient([
            ("", [("read_file", {"path": "one"}), ("submit_stage", {"delivery": "described"})]),
            ("", [("read_file", {"path": "two"})]),
            ("", [("submit_stage", {"delivery": "fixed"})]),
        ])
        executor, _ = self._executor()
        answers = [True]

        def gate(delivery: str):
            if answers:
                answers.pop()
                return False, "no hardware target was edited"
            return True, ""

        output = executor.execute(
            {}, object(),
            self._request(tools_enabled=True, native_tool_loop=True, submit_stage=gate),
            client=client,
        )
        self.assertEqual(output.result.text, "fixed")
        self.assertEqual([c["name"] for c in output.tool_calls],
                         ["read_file", "submit_stage", "read_file", "submit_stage"])
        # The batched read ran, and the gate's reason reached the model.
        self.assertTrue(output.tool_calls[0]["ok"])
        self.assertFalse(output.tool_calls[1]["ok"])
        follow_up = " ".join(str(item.get("content", "")) for item in client.seen[1][0])
        self.assertIn("no hardware target was edited", follow_up)

    def test_two_submissions_in_one_batch_are_refused(self) -> None:
        client = _RoundClient([
            ("", [("submit_stage", {"delivery": "one"}),
                  ("submit_stage", {"delivery": "two"})]),
        ])
        executor, _ = self._executor()
        with self.assertRaisesRegex(RuntimeError, "exactly one submit_stage"):
            executor.execute(
                {}, object(),
                self._request(tools_enabled=True, native_tool_loop=True,
                              submit_stage=lambda _delivery: True), client=client,
            )

    def test_the_last_tool_round_is_announced_while_tools_remain(self) -> None:
        # Observed 2026-09-30: the only budget message told the model to "submit
        # now" after the tools were gone, so a stage that read every round
        # submitted work it had never applied.
        client = _RoundClient([
            ("", [("read_file", {"path": "one"})]),
            ("", [("read_file", {"path": "two"})]),
            ("", [("submit_stage", {"delivery": "done"})]),
        ])
        executor, _ = self._executor()
        executor.execute(
            {"max_tool_rounds": 2}, object(),
            self._request(tools_enabled=True, native_tool_loop=True,
                          submit_stage=lambda _delivery: True), client=client,
        )
        first = " ".join(str(item.get("content", "")) for item in client.seen[0][0])
        second = " ".join(str(item.get("content", "")) for item in client.seen[1][0])
        self.assertNotIn("last tool round", first)
        self.assertIn("This is your last tool round", second)
        self.assertIn("only submit_stage is available", second)

    def test_the_corrective_round_fits_inside_the_turn_ceiling(self) -> None:
        # Observed run-0003: the gate asked for a change, the corrective round
        # was granted, and the run then died on
        # "model-turn hard limit reached without submission" one turn later.
        client = _RoundClient([
            ("", [("read_file", {"path": "one"})]),
            ("", [("submit_stage", {"delivery": "not applied"})]),
            ("", [("read_file", {"path": "two"})]),
            ("", [("read_file", {"path": "three"})]),
            ("", [("submit_stage", {"delivery": "applied"})]),
        ])
        executor, _ = self._executor()
        answers = [True]

        def gate(delivery: str):
            if answers:
                answers.pop()
                return False, "no hardware target was edited"
            return True, ""

        output = executor.execute(
            {"max_tool_rounds": 1}, object(),
            self._request(tools_enabled=True, native_tool_loop=True, submit_stage=gate),
            client=client,
        )
        self.assertEqual(output.result.text, "applied")

    def test_native_graph_rejects_checkpoint_thread_reuse(self) -> None:
        saver = InMemorySaver()
        executor, _ = self._executor()
        request = self._request(tools_enabled=True, native_tool_loop=True,
                                submit_stage=lambda _delivery: True,
                                stage_checkpointer=saver,
                                stage_thread_id="test-stage-native-unique-2")
        executor.execute({}, object(), request, client=_RoundClient([
            ("", [("submit_stage", {"delivery": "done"})])]))
        with self.assertRaisesRegex(ValueError, "resume is not safe"):
            executor.execute({}, object(), request, client=_RoundClient([]))

    def test_native_graph_without_submission_fails_and_rolls_back(self) -> None:
        client = _RoundClient([("just prose", []), ("still prose", []), ("again", [])])
        executor, calls = self._executor()
        with self.assertRaisesRegex(RuntimeError, "did not call submit_stage"):
            executor.execute(
                {"max_tool_rounds": 1}, object(),
                self._request(tools_enabled=True, native_tool_loop=True,
                              submit_stage=lambda _delivery: True), client=client,
            )
        self.assertTrue(_Toolset.latest.session.rolled_back)
        self.assertEqual(calls["artifacts"][-1][3][0]["name"], "(tool loop failed)")

    def test_native_graph_rejected_gate_fails_and_rolls_back(self) -> None:
        # A rejection is correctable once (the reason goes back to the model), so
        # the stage only fails after the retry is rejected too.
        client = _RoundClient([("", [("submit_stage", {"delivery": "done"})]),
                               ("", [("submit_stage", {"delivery": "done again"})])])
        executor, _ = self._executor()
        with self.assertRaisesRegex(RuntimeError, "gate rejected"):
            executor.execute(
                {}, object(), self._request(tools_enabled=True, native_tool_loop=True,
                                             submit_stage=lambda _delivery: False), client=client,
            )
        self.assertTrue(_Toolset.latest.session.rolled_back)
        self.assertEqual(_["artifacts"][-1][3][0]["name"], "submit_stage")
        self.assertFalse(_["artifacts"][-1][3][0]["ok"])

    def test_native_graph_rejection_reason_is_fed_back_and_can_be_fixed(self) -> None:
        # Observed 2026-09-29: a delivery that named no applied edit was rejected
        # with a bare message and the whole run died with it. The model now gets
        # the gate's reason and one chance to fix the work.
        client = _RoundClient([("", [("submit_stage", {"delivery": "nothing applied"})]),
                               ("", [("edit_hardware", {"edits": []})]),
                               ("", [("submit_stage", {"delivery": "applied now"})])])
        executor, _ = self._executor()
        answers = ["no hardware target was edited", True]
        reasons: list[str] = []

        def gate(delivery: str):
            answer = answers.pop(0)
            reasons.append(delivery)
            return (answer, "") if answer is True else (False, answer)

        output = executor.execute(
            {}, object(), self._request(tools_enabled=True, native_tool_loop=True,
                                        submit_stage=gate), client=client,
        )
        self.assertEqual(output.result.text, "applied now")
        self.assertEqual(reasons, ["nothing applied", "applied now"])
        rejected = output.tool_calls[0]
        self.assertFalse(rejected["ok"])
        self.assertIn("no hardware target was edited", rejected["result"])
        # The reason reached the model before it tried again.
        follow_up = " ".join(str(item.get("content", "")) for item in client.seen[1][0])
        self.assertIn("no hardware target was edited", follow_up)
        self.assertTrue(output.tool_calls[-1]["ok"])

    def test_native_graph_requires_external_gate(self) -> None:
        executor, _ = self._executor()
        with self.assertRaisesRegex(ValueError, "independent submit_stage gate"):
            executor.execute({}, object(), self._request(tools_enabled=True,
                native_tool_loop=True), client=_RoundClient([]))

    def test_graph_wrapper_uses_late_bound_complete_patch(self) -> None:
        client = mock.Mock(available=False)
        expected = LlmResult("patched", used_llm=True)
        with mock.patch.object(llm_step, "_complete_llm", return_value=(expected, "patched.txt")) as complete:
            result, raw, calls = llm_step._run_agent_step(
                {"agent_tools": True},
                object(),
                stage="implement_sw",
                label="software",
                artifact_stage="implementation",
                tool_calls_relpath="09_implementation/sw_tool_calls.yaml",
                system="system",
                user="user",
                fallback="fallback",
                client=client,
            )
        self.assertIs(result, expected)
        self.assertEqual(raw, "patched.txt")
        self.assertEqual(calls, [])
        complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
