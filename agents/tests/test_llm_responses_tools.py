"""The tool loop prefers the Responses API and falls back to chat completions.

Offline contract tests: every SDK is a scripted fake, so no provider is called.
They pin the three behaviours that matter for a gateway which refuses function
tools on ``/v1/chat/completions`` (for example gpt-6-luna):

1. the chat-format transcript is translated into native Responses items;
2. a tool round round-trips through ``function_call``/``function_call_output``;
3. ``api_mode=auto`` degrades to chat completions exactly once, while a pinned
   ``responses`` mode surfaces the failure instead of silently changing shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import context_budget, request_audit
from agents.cfu_design_agent.runtime import llm as llm_module
from agents.cfu_design_agent.runtime.llm import (
    OpenAiTextClient,
    _responses_items,
    _responses_round_from_stream,
    _responses_tools,
)

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]


def _usage(input_tokens: int = 10, output_tokens: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )


def _event(kind: str, **fields) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **fields)


def _completed() -> SimpleNamespace:
    return _event("response.completed",
                  response=SimpleNamespace(status="completed", usage=_usage()))


def _call_events(call_id: str = "call_1", name: str = "read_file",
                 arguments: str = '{"path":"a.c"}', *, delta_only: bool = False):
    """One streamed function call, with arguments on ``.done`` or only as deltas."""
    events = [
        _event("response.output_item.added", item=SimpleNamespace(
            type="function_call", id="fc_1", call_id=call_id, name=name, arguments="")),
    ]
    if delta_only:
        for piece in ('{"path"', ':', '"a.c"', '}'):
            events.append(_event("response.function_call_arguments.delta",
                                 item_id="fc_1", delta=piece))
        done_arguments = ""
    else:
        done_arguments = arguments
    events.append(_event("response.output_item.done", item=SimpleNamespace(
        type="function_call", id="fc_1", call_id=call_id, name=name,
        arguments=done_arguments)))
    return events


def _text_events(text: str) -> list[SimpleNamespace]:
    return [_event("response.output_text.delta", delta=text)]


def _chat_tool_call(arguments: str = '{"path": "a"}') -> SimpleNamespace:
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="call-chat", type="function",
            function=SimpleNamespace(name="read_file", arguments=arguments))],
    )
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
                           usage=None)


def _chat_final(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop",
                                 message=SimpleNamespace(content=text, tool_calls=None))],
        usage=None)


class _Endpoint:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("no scripted response left")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeSdk:
    """Scripted stand-in for the OpenAI SDK (chat and Responses side by side)."""

    def __init__(self, responses: list | None = None, chat: list | None = None,
                 *, expose_responses: bool = True):
        self.chat = SimpleNamespace(completions=_Endpoint(chat or []))
        if expose_responses:
            self.responses = _Endpoint(responses or [])


class _Rejected(RuntimeError):
    """A gateway refusal, as the SDK reports a non-retryable HTTP error."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class ResponsesItemsTest(unittest.TestCase):
    def test_transcript_becomes_native_items(self) -> None:
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": '{"path":"a.c"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "1\tint x;"},
        ]
        self.assertEqual(_responses_items(messages), [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "task"},
            {"type": "function_call", "call_id": "call_1", "name": "read_file",
             "arguments": '{"path":"a.c"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "1\tint x;"},
        ])
        # The caller's transcript must not be rewritten in place.
        self.assertIn("tool_calls", messages[2])

    def test_assistant_text_and_calls_both_survive(self) -> None:
        items = _responses_items([{
            "role": "assistant", "content": "checking",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"}}]}])
        self.assertEqual([item.get("type") or item.get("role") for item in items],
                         ["assistant", "function_call"])

    def test_orphan_result_and_incomplete_call_are_rejected(self) -> None:
        with self.assertRaisesRegex(context_budget.ContextBudgetError, "orphan tool result"):
            _responses_items([{"role": "tool", "content": "x"}])
        with self.assertRaisesRegex(context_budget.ContextBudgetError, "incomplete tool call"):
            _responses_items([{"role": "assistant", "tool_calls": [
                {"id": "", "function": {"name": "read_file"}}]}])

    def test_tools_use_the_responses_shape(self) -> None:
        """/responses rejects the chat ``function`` nesting with HTTP 400."""
        converted = _responses_tools(TOOLS)
        self.assertEqual([tool["name"] for tool in converted], ["read_file"])
        self.assertNotIn("function", converted[0])
        self.assertIn("parameters", converted[0])
        # The caller's schemas stay in chat form for the transcript/audit.
        self.assertIn("function", TOOLS[0])


class ResponsesRoundStreamTest(unittest.TestCase):
    def test_text_and_call_are_shaped_like_a_chat_round(self) -> None:
        events = (_text_events("working on it ")
                  + _call_events()
                  + [_completed()])
        round_ = _responses_round_from_stream(iter(events))
        self.assertEqual(round_.text, "working on it ")
        self.assertEqual(round_.input_tokens, 10)
        self.assertEqual(round_.finish_reason, "completed")
        self.assertEqual([(call.id, call.function.name, call.function.arguments)
                          for call in round_.message.tool_calls],
                         [("call_1", "read_file", '{"path":"a.c"}')])

    def test_arguments_are_reassembled_from_deltas_when_done_omits_them(self) -> None:
        round_ = _responses_round_from_stream(
            iter(_call_events(delta_only=True) + [_completed()]))
        self.assertEqual(round_.message.tool_calls[0].function.arguments, '{"path":"a.c"}')

    def test_reasoning_summary_deltas_are_captured(self) -> None:
        events = [
            _event("response.reasoning_summary_text.delta", delta="think "),
            _event("response.reasoning_summary_text.delta", delta="hard"),
            _event("response.output_text.delta", delta="answer"),
            _completed(),
        ]
        round_ = _responses_round_from_stream(iter(events))
        self.assertEqual(round_.reasoning, "think hard")
        self.assertEqual(round_.text, "answer")
        self.assertEqual(round_.reasoning_source, "responses.reasoning_summary.delta")


class ToolLoopEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name).resolve()
        self.run_root = self.workdir / "agents/generated/runs/responses-run"
        self.run_root.mkdir(parents=True)
        self.audit = self.run_root / request_audit.AUDIT_FILENAME
        # The missing-route memo is process-wide; start each case clean.
        llm_module._RESPONSES_ROUTE_MISSING.clear()
        self.addCleanup(llm_module._RESPONSES_ROUTE_MISSING.clear)

    def _client(self, **overrides) -> OpenAiTextClient:
        config = {"model": "m", "api_key": "sk-offline", "stream": True,
                  "base_url": "https://gw.example/v1", "api_mode": "auto", **overrides}
        return OpenAiTextClient(config=config)

    def _events(self) -> list[dict]:
        if not self.audit.exists():
            return []
        return [json.loads(line) for line in self.audit.read_text().splitlines()]

    def _run(self, sdk: _FakeSdk, client: OpenAiTextClient, *, runner=None, rounds: int = 3):
        executed: list[tuple[str, dict]] = []

        def default_runner(name, arguments):
            executed.append((name, arguments))
            return "1\tint x;"

        with mock.patch("openai.OpenAI", return_value=sdk), \
                request_audit.use_request_audit(self.run_root, self.workdir):
            result = client.complete_with_tools(
                "system policy", "user task", tools=TOOLS,
                tool_runner=runner or default_runner, max_rounds=rounds)
        return result, executed

    def test_tool_round_uses_the_responses_endpoint(self) -> None:
        sdk = _FakeSdk(responses=[
            _call_events() + [_completed()],
            _text_events("FINAL ANSWER") + [_completed()],
        ])
        result, executed = self._run(sdk, self._client())

        self.assertEqual(result.text, "FINAL ANSWER")
        self.assertEqual(result.api_mode_used, "responses+tools")
        self.assertEqual(executed, [("read_file", {"path": "a.c"})])
        self.assertEqual(result.tool_rounds, 1)
        # The second request carries the native tool round-trip.
        second = sdk.responses.calls[1]["input"]
        # Tools go out in the Responses shape (flat name/parameters).
        first_tools = sdk.responses.calls[0]["tools"]
        self.assertEqual(first_tools[0]["name"], "read_file")
        self.assertNotIn("function", first_tools[0])
        self.assertIn({"type": "function_call", "call_id": "call_1",
                       "name": "read_file", "arguments": '{"path":"a.c"}'}, second)
        self.assertIn({"type": "function_call_output", "call_id": "call_1",
                       "output": "1\tint x;"}, second)
        # Chat completions were never contacted.
        self.assertEqual(sdk.chat.completions.calls, [])
        endpoints = [event["metadata"]["endpoint"] for event in self._events()
                     if event["event"] == "request"]
        self.assertEqual(endpoints, ["responses", "responses"])
        self.assertNotIn("sk-offline", self.audit.read_text())

    def test_auto_falls_back_to_chat_when_the_endpoint_refuses_tools(self) -> None:
        sdk = _FakeSdk(
            responses=[_Rejected(
                "upstream error: 400 Function tools with reasoning_effort are not "
                "supported for this model in /v1/chat/completions")],
            chat=[_chat_tool_call(), _chat_final("FINAL ANSWER")],
        )
        result, executed = self._run(sdk, self._client())

        self.assertEqual(result.text, "FINAL ANSWER")
        self.assertEqual(result.api_mode_used, "chat+tools")
        self.assertIn("responses unavailable", result.fallback_reason)
        self.assertEqual(executed, [("read_file", {"path": "a"})])
        # Exactly one rejected Responses attempt, then the whole loop stays on chat.
        self.assertEqual(len(sdk.responses.calls), 1)
        self.assertEqual(len(sdk.chat.completions.calls), 2)

    def test_missing_responses_route_also_falls_back(self) -> None:
        sdk = _FakeSdk(chat=[_chat_final("FINAL ANSWER")], expose_responses=False)
        result, _ = self._run(sdk, self._client())
        self.assertEqual(result.text, "FINAL ANSWER")
        self.assertEqual(result.api_mode_used, "chat+tools")
        self.assertIn("responses unavailable", result.fallback_reason)

    def test_pinned_responses_mode_surfaces_the_failure(self) -> None:
        sdk = _FakeSdk(responses=[_Rejected("upstream error: 400 unsupported")])
        client = self._client(api_mode="responses")
        with self.assertRaisesRegex(RuntimeError, "tool-enabled model request failed"):
            self._run(sdk, client)
        self.assertEqual(sdk.chat.completions.calls, [])

    def test_non_streaming_responses_round_round_trips_a_tool(self) -> None:
        def response(items):
            return SimpleNamespace(status="completed", usage=_usage(), output=items)

        sdk = _FakeSdk(responses=[
            response([SimpleNamespace(type="function_call", call_id="call_1",
                                      name="read_file", arguments='{"path":"a.c"}')]),
            response([SimpleNamespace(type="message", content=[
                SimpleNamespace(type="output_text", text="FINAL ANSWER")])]),
        ])
        result, executed = self._run(sdk, self._client(stream=False))

        self.assertEqual(result.text, "FINAL ANSWER")
        self.assertEqual(executed, [("read_file", {"path": "a.c"})])
        self.assertEqual(result.api_mode_used, "responses+tools")
        self.assertFalse(sdk.responses.calls[0].get("stream"))

    def test_pinned_chat_mode_never_probes_responses(self) -> None:
        sdk = _FakeSdk(chat=[_chat_final("FINAL ANSWER")])
        client = self._client(api_mode="chat")
        result, _ = self._run(sdk, client)
        self.assertEqual(result.api_mode_used, "chat+tools")
        self.assertEqual(sdk.responses.calls, [])
        self.assertEqual(result.fallback_reason, "")

    def test_missing_route_is_remembered_for_later_clients(self) -> None:
        """One rejected probe per model, not one per stage."""
        first = _FakeSdk(
            responses=[_Rejected("upstream error: 404 Not Found", status_code=404)],
            chat=[_chat_final("FINAL ANSWER")],
        )
        result, _ = self._run(first, self._client())
        self.assertEqual(result.api_mode_used, "chat+tools")
        self.assertEqual(len(first.responses.calls), 1)

        # A fresh client for the same model must skip the probe entirely.
        second = _FakeSdk(chat=[_chat_final("SECOND")])
        result2, _ = self._run(second, self._client())
        self.assertEqual(result2.text, "SECOND")
        self.assertEqual(second.responses.calls, [])
        # ...and its fallback reason is not reported, because nothing failed.
        self.assertEqual(result2.fallback_reason, "")

    def test_empty_stream_recovers_the_real_endpoint_error(self) -> None:
        """A dropped SSE error event must still surface as the endpoint's error."""
        sdk = _FakeSdk(responses=[[]])  # streaming request yields no events
        client = self._client()
        sdk.responses.script.append(
            _Rejected("upstream error: 404 Not Found", status_code=404))
        with self.assertRaisesRegex(_Rejected, "404"):
            with mock.patch("openai.OpenAI", return_value=sdk), \
                    request_audit.use_request_audit(self.run_root, self.workdir):
                client._responses_round(sdk, [{"role": "user", "content": "x"}],
                                        allow_tools=True, tools=TOOLS)
        # streamed attempt + one non-streaming diagnostic attempt
        self.assertEqual(len(sdk.responses.calls), 2)
        self.assertFalse(sdk.responses.calls[1].get("stream"))


class FallbackNoteTest(unittest.TestCase):
    def test_note_names_the_actual_fallback(self) -> None:
        from agents.cfu_design_agent.runtime.llm_step import _fallback_note

        self.assertEqual(_fallback_note(""), "")
        self.assertEqual(_fallback_note("responses unavailable: RuntimeError: x"),
                         " (endpoint fell back to chat)")
        self.assertEqual(_fallback_note("streaming unavailable: TypeError: y"),
                         " (streaming fell back)")


if __name__ == "__main__":
    unittest.main()
