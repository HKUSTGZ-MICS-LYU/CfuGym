"""Every SDK request attempt must leave an exact, credential-free JSONL record."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.runtime import request_audit
from agents.cfu_design_agent.runtime.llm import OpenAiTextClient


def _response(text: str = "ok", cached: int | None = None):
    details = {"cached_tokens": cached} if cached is not None else {}
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=2, total_tokens=12,
        prompt_tokens_details=details,
    )
    return SimpleNamespace(
        id="resp-1", usage=usage,
        choices=[SimpleNamespace(finish_reason="stop",
                                 message=SimpleNamespace(content=text, tool_calls=None))],
    )


class _ChatCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LlmRequestAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name).resolve()
        self.run_root = self.workdir / "agents/generated/runs/audit-run"
        self.run_root.mkdir(parents=True)
        self.audit = self.run_root / request_audit.AUDIT_FILENAME

    def _events(self) -> list[dict]:
        if not self.audit.exists():
            return []
        return [json.loads(line) for line in self.audit.read_text().splitlines()]

    def _client(self, **overrides) -> OpenAiTextClient:
        config = {"model": "m", "api_key": "sk-super-secret",
                  "api_mode": "chat", "stream": False, **overrides}
        return OpenAiTextClient(config=config)

    def test_plain_chat_records_exact_request_and_usage(self) -> None:
        completions = _ChatCompletions([_response(cached=6)])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with request_audit.use_request_audit(self.run_root, self.workdir):
            result = self._client()._chat(sdk, "system text", "user text")
        events = self._events()
        self.assertEqual([e["event"] for e in events], ["request", "result"])
        self.assertEqual(events[0]["payload"]["messages"],
                         [{"role": "system", "content": "system text"},
                          {"role": "user", "content": "user text"}])
        self.assertEqual(events[0]["payload"]["model"], "m")
        self.assertEqual(events[0]["metadata"]["endpoint"], "chat")
        self.assertNotIn("sk-super-secret", self.audit.read_text())
        self.assertNotIn("api_key", self.audit.read_text())
        self.assertEqual(events[1]["result"]["usage"]["cached_input_tokens"], 6)
        self.assertEqual(result.cached_input_tokens, 6)

    def test_stream_fallback_records_both_attempts(self) -> None:
        completions = _ChatCompletions([
            RuntimeError("stream_options is unsupported"),
            _response(),
        ])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with request_audit.use_request_audit(self.run_root, self.workdir):
            self._client(stream=True)._chat(sdk, "s", "u")
        events = self._events()
        self.assertEqual([e["event"] for e in events],
                         ["request", "error", "request", "result"])
        self.assertTrue(events[0]["payload"].get("stream"))
        self.assertFalse(events[2]["payload"].get("stream"))
        self.assertEqual(events[1]["request_id"], f"req-{events[0]['seq']:08d}")
        self.assertEqual(events[3]["request_id"], f"req-{events[2]['seq']:08d}")
        self.assertEqual(len(completions.calls), 2)

    def test_tool_loop_audits_every_model_round(self) -> None:
        def tool_call_response():
            return SimpleNamespace(
                id="resp-1", usage=None,
                choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
                        name="read_file", arguments='{"path": "a"}'))]))],
            )

        completions = _ChatCompletions([tool_call_response(), _response("done")])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        from unittest import mock

        with mock.patch("openai.OpenAI", return_value=sdk):
            with request_audit.use_request_audit(self.run_root, self.workdir):
                result = self._client().complete_with_tools(
                    "s", "u",
                    tools=[{"type": "function", "function": {"name": "read_file"}}],
                    tool_runner=lambda name, args: "obs",
                    max_rounds=2,
                )
        events = self._events()
        self.assertEqual([e["event"] for e in events], ["request", "result", "request", "result"])
        self.assertIn("tools", events[0]["payload"])
        self.assertEqual(result.text, "done")
        self.assertEqual(len(completions.calls), 2)

    def test_fail_closed_audit_write_blocks_the_sdk_call(self) -> None:
        completions = _ChatCompletions([_response()])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.audit.write_text("{broken json\n")
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaises(request_audit.RequestAuditError):
                self._client()._chat(sdk, "s", "u")
        self.assertEqual(completions.calls, [], "an unrecorded request must never be sent")

    def test_responses_and_auto_chat_fallback_are_separate_attempts(self) -> None:
        from unittest import mock
        class Responses:
            def create(self, **kwargs):
                raise RuntimeError("responses unavailable")
        completions = _ChatCompletions([_response(cached=0)])
        sdk = SimpleNamespace(responses=Responses(), chat=SimpleNamespace(completions=completions))
        with mock.patch("openai.OpenAI", return_value=sdk) as constructor:
            with request_audit.use_request_audit(self.run_root, self.workdir):
                result = self._client(api_mode="auto").complete("s", "u", fallback="")
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        events = self._events()
        requests = [e for e in events if e["event"] == "request"]
        self.assertEqual([e["metadata"]["endpoint"] for e in requests], ["responses", "chat"])
        self.assertEqual(requests[0]["payload"]["max_output_tokens"], 4096)
        self.assertEqual(requests[1]["payload"]["max_completion_tokens"], 4096)
        self.assertEqual(result.cached_input_tokens, 0)
        self.assertEqual(requests[1]["metadata"]["parent_request_id"], requests[0]["request_id"])

    def test_model_metadata_discovery_is_cached_within_run(self) -> None:
        from unittest import mock
        models = SimpleNamespace(retrieve=mock.Mock(return_value={"context_window": 64000}))
        sdk = SimpleNamespace(models=models, chat=SimpleNamespace(completions=_ChatCompletions([_response(), _response()])))
        with request_audit.use_request_audit(self.run_root, self.workdir):
            self._client()._chat(sdk, "s", "u")
        with request_audit.use_request_audit(self.run_root, self.workdir):
            self._client()._chat(sdk, "s", "u")
        models.retrieve.assert_called_once_with("m", timeout=3)
        requests = [e for e in self._events() if e["event"] == "request"]
        self.assertEqual(requests[0]["metadata"]["endpoint"], "models.retrieve")
        self.assertEqual(requests[-1]["metadata"]["context_budget"]["window_source"], "metadata")

    def test_budget_block_is_logged_without_sending(self) -> None:
        from agents.cfu_design_agent.runtime.context_budget import ContextBudgetError
        completions = _ChatCompletions([_response()])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with request_audit.use_request_audit(self.run_root, self.workdir):
            with self.assertRaises(ContextBudgetError):
                self._client(context_window_tokens=5000)._chat(sdk, "s" * 6000, "u")
        self.assertEqual(completions.calls, [])
        self.assertEqual(self._events()[0]["event"], "blocked")
        self.assertFalse(self._events()[0]["metadata"]["sent"])

    def test_effective_body_flattens_extra_body_and_omits_timeout(self) -> None:
        completions = _ChatCompletions([_response()])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with request_audit.use_request_audit(self.run_root, self.workdir):
            self._client(thinking=True, timeout=120)._chat(sdk, "s", "u")
        payload = self._events()[0]["payload"]
        self.assertNotIn("extra_body", payload)
        self.assertNotIn("timeout", payload)
        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_stream_final_usage_preserves_cached_tokens(self) -> None:
        final_usage = SimpleNamespace(
            prompt_tokens=12, completion_tokens=8, total_tokens=20,
            prompt_tokens_details={"cached_tokens": 8},
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
        stream = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None,
                delta=SimpleNamespace(content="ok", reasoning_content="private thought",
                                       tool_calls=None))]),
            SimpleNamespace(usage=final_usage, choices=[]),
        ]
        completions = _ChatCompletions([iter(stream)])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with request_audit.use_request_audit(self.run_root, self.workdir):
            result = self._client(stream=True)._chat(sdk, "s", "u")
        self.assertEqual(result.cached_input_tokens, 8)
        events = self._events()
        self.assertEqual(events[-1]["result"]["cached_input_tokens"], 8)
        self.assertEqual(events[-1]["metadata"]["reasoning_chars"], len("private thought"))
        self.assertEqual(events[-1]["metadata"]["reasoning_source"], "stream.reasoning_content")
        self.assertEqual(events[-1]["metadata"]["reasoning_tokens"], 5)
        self.assertIn("reasoning_content", events[-1]["metadata"]["response_extra_keys"])
        self.assertNotIn("private thought", self.audit.read_text())

    def test_compaction_is_persisted_and_audited(self) -> None:
        def batch(ident, text):
            return [{"role": "assistant", "tool_calls": [{"id": ident, "type": "function",
                     "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]},
                    {"role": "tool", "tool_call_id": ident, "content": text}]
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        messages += batch("old", "a" * 9000) + batch("current", "recent")
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=_ChatCompletions([_response()])))
        client = self._client(context_window_tokens=10000)
        with request_audit.use_request_audit(self.run_root, self.workdir):
            client._chat_round(sdk, messages, allow_tools=False, tools=None)
        events = self._events()
        self.assertEqual(events[0]["event"], "compaction")
        request = next(e for e in events if e["event"] == "request")
        self.assertEqual(client._prepared_messages, request["payload"]["messages"])
        self.assertNotIn("a" * 9000, json.dumps(client._prepared_messages))
        self.assertIn("Context compaction", json.dumps(client._prepared_messages))

    def test_responses_stream_and_nonstream_usage_are_audited(self) -> None:
        response = SimpleNamespace(id="response-id", output_text="ok", status="completed", output=[],
            usage=SimpleNamespace(input_tokens=20, output_tokens=4, total_tokens=24,
                                  input_tokens_details={"cached_tokens": 16}))
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                class Responses:
                    def create(self, **kwargs):
                        if streaming:
                            return iter([SimpleNamespace(type="response.output_text.delta", delta="ok"),
                                         SimpleNamespace(type="response.completed", response=response)])
                        return response
                with request_audit.use_request_audit(self.run_root, self.workdir):
                    result = self._client(stream=streaming)._responses(SimpleNamespace(responses=Responses()), "s", "u")
                self.assertEqual(result.cached_input_tokens, 16)
                self.assertEqual(result.input_tokens, 20)
                self.assertEqual(self._events()[-1]["event"], "result")

    def test_native_stage_model_requests_are_in_same_audit(self) -> None:
        from agents.cfu_design_agent.runtime.stage_executor import StageExecutionRequest, StageExecutor
        submit = SimpleNamespace(id="c-submit", function=SimpleNamespace(
            name="submit_stage", arguments='{"delivery":"done"}'))
        response = _response()
        response.choices[0].message.tool_calls = [submit]
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=_ChatCompletions([response])))
        client = self._client()
        client._stage_round_client = sdk
        class Toolset:
            def __init__(self, *args, **kwargs): pass
            def schemas(self):
                return [{"type": "function", "function": {"name": "read_file"}}]
        executor = StageExecutor(complete=lambda *a, **kw: None,
            record_output=lambda *a, **kw: "raw.txt", write_tool_calls=lambda *a: None,
            has_deliverable=lambda text: True, toolset_factory=Toolset)
        request = StageExecutionRequest(stage="implement", label="stage3", artifact_stage="implementation",
            tool_calls_relpath="calls.yaml", system="s", user="u", fallback="",
            tools_enabled=True, native_tool_loop=True, submit_stage=lambda text: True)
        with request_audit.use_request_audit(self.run_root, self.workdir):
            result = executor.execute({}, object(), request, client=client)
        self.assertEqual(result.result.text, "done")
        self.assertEqual([e["event"] for e in self._events()], ["request", "result"])
        self.assertIn("submit_stage", json.dumps(self._events()[0]["payload"]["tools"]))

    def test_node_wrapper_binds_the_run_audit(self) -> None:
        state = {"run_root": "agents/generated/runs/audit-run",
                 "workdir": str(self.workdir)}

        def node(_state):
            completions = _ChatCompletions([_response()])
            sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
            self._client()._chat(sdk, "s", "u")
            return {}

        instrument._wrap_node(node)(state)
        self.assertEqual(len(self._events()), 2)


if __name__ == "__main__":
    unittest.main()
