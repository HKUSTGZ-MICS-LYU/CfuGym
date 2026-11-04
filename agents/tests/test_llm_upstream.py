"""Offline regressions for status-less gateway SSE failures (abl-1028)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import llm, request_audit


RATE_LIMIT = ("litellm.APIError: Your requests to gpt-6-luna for gpt-6-luna "
              "in eastus have exceeded token rate limit.")
TRUNCATED = ("Response payload is not completed: <TransferEncodingError: 400, "
             "message='Not enough data to satisfy transfer length header.'>")
MESSAGES = [{"role": "user", "content": "task"}]


def error(message=RATE_LIMIT, *, status=None, body=None, headers=None):
    exc = RuntimeError(message)
    exc.status_code = status
    exc.body = body
    exc.response = SimpleNamespace(headers=headers or {})
    return exc


def response():
    return SimpleNamespace(status="completed", output=[], output_text="DONE", usage=None)


def events():
    return [SimpleNamespace(type="response.output_text.delta", delta="DONE"),
            SimpleNamespace(type="response.completed", response=response())]


class Endpoint:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class BrokenStream:
    def __init__(self, exc, prefix=()):
        self.exc = exc
        self.prefix = prefix
        self.closed = False

    def __iter__(self):
        yield from self.prefix
        raise self.exc

    def close(self):
        self.closed = True


class UpstreamTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name).resolve()
        self.root = self.workdir / "agents/generated/runs/upstream-offline"
        self.root.mkdir(parents=True)
        self.audit = self.root / request_audit.AUDIT_FILENAME
        llm._RESPONSES_ROUTE_MISSING.clear()
        self.addCleanup(llm._RESPONSES_ROUTE_MISSING.clear)

    def client(self, **options):
        return llm.OpenAiTextClient(config={
            "api_key": "offline-secret-value", "model": "m", "api_mode": "auto",
            "stream": True, "context_window_tokens": 131072, **options})

    def sdk(self, responses=(), chat=()):
        return SimpleNamespace(responses=Endpoint(responses),
                               chat=SimpleNamespace(completions=Endpoint(chat)))

    def ledger(self):
        return [json.loads(line) for line in self.audit.read_text().splitlines()]

    def run_round(self, sdk, client=None):
        with request_audit.use_request_audit(self.root, self.workdir):
            return (client or self.client()).model_round(
                sdk, MESSAGES, allow_tools=False, tools=None)

    def test_statusless_sse_limit_retries_same_request_not_chat(self):
        broken = BrokenStream(error())
        sdk = self.sdk([broken, iter(events())])
        client = self.client()
        with mock.patch.object(llm.time, "sleep") as sleep:
            result = self.run_round(sdk, client)
        self.assertEqual(result.text, "DONE")
        sleep.assert_called_once_with(60)
        self.assertEqual(sdk.responses.calls[0], sdk.responses.calls[1])
        self.assertEqual(sdk.chat.completions.calls, [])
        self.assertFalse(client._responses_unsupported)
        self.assertTrue(broken.closed)
        records = self.ledger()
        self.assertEqual([r["event"] for r in records],
                         ["request", "error", "request", "result"])
        detail = records[1]["error"]
        self.assertEqual(detail["error_kind"], "rate_limit")
        self.assertIn("exceeded token rate limit", detail["message"])
        self.assertNotIn("status_code", detail, "do not invent an HTTP status")
        self.assertEqual(detail["retry_delay_seconds"], 60)

    def test_request_creation_limit_also_retries_without_chat(self):
        sdk = self.sdk([error(), iter(events())])
        with mock.patch.object(llm.time, "sleep") as sleep:
            result = self.run_round(sdk)
        self.assertEqual(result.text, "DONE")
        sleep.assert_called_once_with(60)
        self.assertEqual(sdk.chat.completions.calls, [])
        self.assertEqual(self.ledger()[1]["error"]["error_kind"], "rate_limit")

    def test_exhausted_sse_retries_preserve_original_limit(self):
        original = error()
        streams = [BrokenStream(original) for _ in range(11)]
        sdk = self.sdk(streams)
        client = self.client()
        with mock.patch.object(llm.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError) as caught:
                self.run_round(sdk, client)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(sdk.responses.calls), 11)
        self.assertEqual(sleep.call_count, 10)
        self.assertEqual(sleep.call_args_list, [mock.call(n) for n in
                         [60, 120, 240, 300, 300, 300, 300, 300, 300, 300]])
        self.assertEqual(sdk.chat.completions.calls, [])
        self.assertFalse(client._responses_unsupported)
        self.assertTrue(all(s.closed for s in streams))
        errors = [r["error"] for r in self.ledger() if r["event"] == "error"]
        self.assertEqual(len(errors), 11)
        self.assertNotIn("retry_delay_seconds", errors[-1])

    def test_nonstream_auto_exhaustion_does_not_change_endpoint(self):
        original = error()
        sdk = self.sdk([original] * 11)
        with mock.patch("openai.OpenAI", return_value=sdk), \
                mock.patch.object(llm.time, "sleep") as sleep, \
                request_audit.use_request_audit(self.root, self.workdir):
            with self.assertRaises(RuntimeError) as caught:
                self.client(stream=False).complete("s", "u", fallback="")
        self.assertIs(caught.exception, original)
        self.assertEqual(len(sdk.responses.calls), 11)
        self.assertEqual(sleep.call_count, 10)
        self.assertEqual(sdk.chat.completions.calls, [])

    def test_truncated_partial_tool_call_is_never_executed(self):
        prefix = [SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(
            type="function_call", id="fc1", call_id="call1", name="read_file",
            arguments='{"path":"a"}'))]
        broken = BrokenStream(error(TRUNCATED), prefix)
        sdk = self.sdk([broken, iter(events())])
        runner = mock.Mock(return_value="unexpected")
        with mock.patch("openai.OpenAI", return_value=sdk), \
                mock.patch.object(llm.time, "sleep") as sleep, \
                request_audit.use_request_audit(self.root, self.workdir):
            result = self.client().complete_with_tools(
                "s", "u", tools=[{"type": "function", "function": {
                    "name": "read_file", "parameters": {"type": "object", "properties": {}}}}],
                tool_runner=runner)
        self.assertEqual(result.text, "DONE")
        runner.assert_not_called()
        sleep.assert_called_once_with(5)
        self.assertTrue(broken.closed)
        self.assertEqual(sdk.responses.calls[0], sdk.responses.calls[1])
        self.assertEqual(sdk.chat.completions.calls, [])
        self.assertEqual(self.ledger()[1]["error"]["error_kind"], "transient")

    def test_chat_stream_truncation_retries_in_chat(self):
        broken = BrokenStream(error(TRUNCATED))
        successful = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            delta=SimpleNamespace(content="DONE", tool_calls=None))], usage=None)
        sdk = self.sdk(chat=[broken, iter([successful])])
        with mock.patch.object(llm.time, "sleep") as sleep:
            result = self.run_round(sdk, self.client(api_mode="chat"))
        self.assertEqual(result.text, "DONE")
        self.assertEqual(len(sdk.chat.completions.calls), 2)
        self.assertEqual(sdk.responses.calls, [])
        sleep.assert_called_once_with(5)
        self.assertTrue(broken.closed)

    def test_plain_complete_stream_is_closed_and_audited(self):
        broken = BrokenStream(error())
        sdk = self.sdk([broken, iter(events())])
        with mock.patch("openai.OpenAI", return_value=sdk), \
                mock.patch.object(llm.time, "sleep") as sleep, \
                request_audit.use_request_audit(self.root, self.workdir):
            result = self.client().complete("s", "u", fallback="")
        self.assertEqual(result.text, "DONE")
        self.assertTrue(broken.closed)
        sleep.assert_called_once_with(60)
        self.assertEqual(sdk.chat.completions.calls, [])
        self.assertEqual(self.ledger()[1]["error"]["error_kind"], "rate_limit")
        self.assertIn("exceeded token rate limit", self.ledger()[1]["error"]["message"])

    def test_nested_response_failed_event_retries_both_entrypoints(self):
        for plain in (False, True):
            with self.subTest(plain=plain):
                failed = SimpleNamespace(type="response.failed", response=SimpleNamespace(
                    error=SimpleNamespace(code="rate_limit_exceeded", message="retry later")))
                sdk = self.sdk([iter([failed]), iter(events())])
                with mock.patch("openai.OpenAI", return_value=sdk), \
                        mock.patch.object(llm.time, "sleep") as sleep, \
                        request_audit.use_request_audit(self.root, self.workdir):
                    client = self.client()
                    result = (client.complete("s", "u", fallback="") if plain else
                              client.model_round(sdk, MESSAGES, allow_tools=False, tools=None))
                self.assertEqual(result.text, "DONE")
                self.assertEqual(len(sdk.responses.calls), 2)
                self.assertEqual(sdk.chat.completions.calls, [])
                self.assertFalse(client._responses_unsupported)
                sleep.assert_called_once_with(60)

    def test_backoff_cancellation_is_not_wrapped_or_fallen_back(self):
        from agents.cfu_design_agent.runtime import processes
        token = processes.CancellationToken()
        broken = BrokenStream(error())
        sdk = self.sdk([broken])
        with mock.patch("openai.OpenAI", return_value=sdk), \
                mock.patch.object(llm.time, "sleep") as sleep, \
                mock.patch.object(token.event, "wait", side_effect=lambda delay: token.cancel()) as wait, \
                processes.use_token(token), \
                request_audit.use_request_audit(self.root, self.workdir):
            with self.assertRaises(processes.ValidationCancelled):
                self.client().complete_with_tools("s", "u", tools=[], tool_runner=mock.Mock())
        wait.assert_called_once_with(60)
        sleep.assert_not_called()
        self.assertEqual(len(sdk.responses.calls), 1)
        self.assertEqual(sdk.chat.completions.calls, [])
        self.assertTrue(broken.closed)

    def test_cancelled_request_is_not_sent(self):
        from agents.cfu_design_agent.runtime import processes
        token = processes.CancellationToken()
        sdk = self.sdk([iter(events())])
        with processes.use_token(token):
            token.cancel()
            with self.assertRaises(processes.ValidationCancelled):
                self.run_round(sdk)
        self.assertEqual(sdk.responses.calls, [])
        self.assertEqual(sdk.chat.completions.calls, [])

    def test_stream_compatibility_fallback_shares_retry_budget(self):
        sdk = self.sdk([error("temporary", status=500) for _ in range(9)] + [
            error("stream_options unsupported"), error("still down", status=503)])
        with mock.patch.object(llm.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "still down"):
                self.run_round(sdk, self.client(api_mode="responses"))
        self.assertEqual(len(sdk.responses.calls), 11)
        self.assertFalse(sdk.responses.calls[-1].get("stream"))

    def test_sdk_body_repr_does_not_leak_basic_cookie_or_quoted_secrets(self):
        import httpx
        from openai import APIError
        body = {"headers": {"Authorization": "Basic dXNlcjpwYXNz",
                            "Cookie": "session=cookie-secret; other=other-secret"},
                "password": "secret with spaces", "message": "invalid request"}
        exc = APIError(f"provider error: {body!r}",
                       request=httpx.Request("POST", "https://offline.invalid"), body=body)
        sdk = self.sdk([exc])
        with self.assertRaises(APIError):
            self.run_round(sdk, self.client(api_mode="responses"))
        text = self.audit.read_text()
        for secret in ("dXNlcjpwYXNz", "cookie-secret", "other-secret", "secret with spaces"):
            self.assertNotIn(secret, text)

    def test_regular_bad_request_is_not_retried(self):
        sdk = self.sdk([error("invalid prompt", status=400)])
        with mock.patch.object(llm.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "invalid prompt"):
                self.run_round(sdk, self.client(api_mode="responses"))
        self.assertEqual(len(sdk.responses.calls), 1)
        sleep.assert_not_called()

    def test_retry_classification_boundaries(self):
        for exc, expected in [
            (error(), True), (error(TRUNCATED), True),
            (error("upstream error: 400"), False),
            (error("invalid prompt", status=400), False),
            (error(RATE_LIMIT, status=401), False),
            (error("insufficient_quota", status=429), False),
            (error("exceeded your current quota", status=429), False),
            (error("rate limit parameter is invalid"), False),
            (error("request rejected", body={"error": {"code": "rate_limit_exceeded"}}), True),
        ]:
            with self.subTest(exc=str(exc)):
                self.assertEqual(llm._is_retryable_request_error(exc), expected)

    def test_retry_after_is_bounded_and_ignores_invalid_values(self):
        for headers, expected in [
            ({"Retry-After": "125"}, 125), ({"retry-after-ms": "125000"}, 125),
            ({"Retry-After": "99999999"}, 300), ({"Retry-After": "nan"}, 60),
            ({"Retry-After": "-1"}, 60), ({"Retry-After": "invalid"}, 60),
            ({"Retry-After": "Thu, 01 Jan 1970 00:02:05 GMT"}, 125),
        ]:
            with self.subTest(headers=headers), mock.patch.object(llm.time, "time", return_value=0):
                self.assertEqual(llm._request_retry_delay(0, error(headers=headers)), expected)

    def test_audit_error_is_informative_bounded_and_redacted(self):
        exc = error("invalid api_key=offline-secret-value Authorization: Bearer hidden-token "
                    "sk-other-secret " + "x" * 4000, status=400, body={
                        "message": "api_key=offline-secret-value", "code": "bad_request",
                        "type": "invalid_request", "headers": {"Authorization": "secret-header"},
                        "request": {"api_key": "do-not-copy"}})
        exc.request_id = "req-provider"
        sdk = self.sdk([exc])
        with self.assertRaises(RuntimeError):
            self.run_round(sdk, self.client(api_mode="responses"))
        detail = self.ledger()[1]["error"]
        self.assertEqual(detail["provider_code"], "bad_request")
        self.assertEqual(detail["provider_request_id"], "req-provider")
        self.assertEqual(detail["status_code"], 400)
        self.assertLessEqual(len(detail["message"]), 2048)
        text = self.audit.read_text()
        for secret in ("offline-secret-value", "hidden-token", "sk-other-secret",
                       "secret-header", "do-not-copy"):
            self.assertNotIn(secret, text)


if __name__ == "__main__":
    unittest.main()
