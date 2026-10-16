"""Named LLM config loading and client config resolution."""

from __future__ import annotations

from pathlib import Path
import unittest

from agents.cfu_design_agent.runtime.llm import OpenAiTextClient
from agents.cfu_design_agent.runtime.llm_config import (
    LlmConfigError,
    effective_config,
    load_config,
)
from agents.tests.common import REPO_ROOT


class LlmConfigTest(unittest.TestCase):
    def test_lists_named_configs(self) -> None:
        import json
        import tempfile

        from agents.cfu_design_agent.runtime.llm_config import list_configs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "agents" / "llm_configs"
            root.mkdir(parents=True)
            (root / "alpha.json").write_text(json.dumps({"name": "alpha"}))
            (root / "beta.json").write_text(json.dumps({"name": "beta"}))
            (root / "ignore.txt").write_text("x")
            names = list_configs(Path(directory))
        self.assertEqual(names, ["alpha", "beta"])

    def test_load_default_resolves_to_env_fallback(self) -> None:
        # effective_config returns an env-only config when default.json is absent.
        config = effective_config(REPO_ROOT, "default")
        self.assertIn(config["api_mode"], {"auto", "responses", "chat"})
        self.assertIsInstance(config["model"], str)
        self.assertIn("api_key", config)
        self.assertEqual(config["name"], "default")

    def test_invalid_mode_rejected(self) -> None:
        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        with self.assertRaises(LlmConfigError):
            resolve_config("bad", {"name": "bad", "api_mode": "nope"})

    def test_missing_config_rejected(self) -> None:
        with self.assertRaises(LlmConfigError):
            load_config(REPO_ROOT, "does_not_exist")

    def test_effective_config_default_missing_uses_env(self) -> None:
        import tempfile

        from agents.cfu_design_agent.runtime.llm_config import effective_config

        with tempfile.TemporaryDirectory() as directory:
            cfg = effective_config(Path(directory), "default")
        self.assertEqual(cfg["name"], "default")
        self.assertIn("api_mode", cfg)
        self.assertIn("api_key", cfg)

    def test_context_options_survive_named_configuration(self) -> None:
        from agents.cfu_design_agent.runtime.llm_config import resolve_config
        from types import SimpleNamespace
        config = resolve_config("explicit", {"context_window_tokens": 64000,
                                              "output_reserve_tokens": 8000})
        client = OpenAiTextClient(config=config)
        window = client._context_window(SimpleNamespace())
        self.assertEqual((window.window_tokens, window.output_reserve_tokens), (64000, 8000))
        self.assertEqual(window.source, "config")
        for invalid in (True, "64000", 0, -1):
            with self.subTest(invalid=invalid), self.assertRaises(LlmConfigError):
                resolve_config("bad", {"context_window_tokens": invalid})
        with self.assertRaises(LlmConfigError):
            resolve_config("bad", {"context_window_tokens": 4096})

    def test_output_reserve_env_overrides_the_config_file(self) -> None:
        """A reasoning model's completion budget is configurable without editing JSON.

        The reserve is both ``max_completion_tokens``/``max_output_tokens`` and the
        context-budget output reservation, so it must stay below the window.
        """
        import os
        from unittest import mock

        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        base = {"context_window_tokens": 922000, "output_reserve_tokens": 8192}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CFU_AGENT_OUTPUT_RESERVE", None)
            self.assertEqual(resolve_config("c", base)["output_reserve_tokens"], 8192)
        with mock.patch.dict(os.environ, {"CFU_AGENT_OUTPUT_RESERVE": "65536"}):
            self.assertEqual(resolve_config("c", base)["output_reserve_tokens"], 65536)
        with mock.patch.dict(os.environ, {"CFU_AGENT_OUTPUT_RESERVE": "bogus"}):
            with self.assertRaisesRegex(LlmConfigError, "positive integer"):
                resolve_config("c", base)
        with mock.patch.dict(os.environ, {"CFU_AGENT_OUTPUT_RESERVE": "9999999"}):
            with self.assertRaisesRegex(LlmConfigError, "smaller than context_window_tokens"):
                resolve_config("c", base)
        # Without a configured window the value still resolves; the soft 32768
        # bound applies later, at window resolution (see test_context_budget).
        with mock.patch.dict(os.environ, {"CFU_AGENT_OUTPUT_RESERVE": "65536"}):
            self.assertEqual(resolve_config("c", {})["output_reserve_tokens"], 65536)

    def test_shipped_configs_keep_a_reasoning_sized_completion_budget(self) -> None:
        """Every shipped config opts into thinking, so none may fall back to 8192."""
        import json

        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        configs = sorted((REPO_ROOT / "agents/llm_configs").glob("*.json"))
        self.assertTrue(configs)
        for path in configs:
            data = json.loads(path.read_text())
            with self.subTest(config=path.name):
                cfg = resolve_config(path.stem, data)
                self.assertGreaterEqual(cfg["output_reserve_tokens"], 32768)
                self.assertLess(cfg["output_reserve_tokens"], cfg["context_window_tokens"])

    def test_run_level_max_output_tokens_wins_over_env_and_config(self) -> None:
        import os
        from unittest import mock

        from agents.cfu_design_agent.runtime import llm_step

        state = {"workdir": str(REPO_ROOT), "llm_config": "gpt-5.6-luna"}
        with mock.patch.dict(os.environ, {"CFU_AGENT_OUTPUT_RESERVE": "40000"}):
            default = llm_step._llm(state)
            self.assertEqual(default.window_config["output_reserve_tokens"], 40000)
            explicit = llm_step._llm({**state, "max_output_tokens": 65536})
            self.assertEqual(explicit.window_config["output_reserve_tokens"], 65536)
            # 0 means "no run-level override", never a zero-token request.
            unset = llm_step._llm({**state, "max_output_tokens": 0})
            self.assertEqual(unset.window_config["output_reserve_tokens"], 40000)

    def test_client_uses_config_fields(self) -> None:
        client = OpenAiTextClient(
            config={
                "model": "my-model",
                "api_key": "sk-test",
                "base_url": "https://example.com/v1",
                "api_mode": "chat",
            }
        )
        self.assertEqual(client.model, "my-model")
        self.assertEqual(client.api_key, "sk-test")
        self.assertEqual(client.base_url, "https://example.com/v1")
        self.assertEqual(client.api_mode, "chat")
        self.assertTrue(client.available)

    def test_json_response_format_is_forwarded_to_both_api_shapes(self) -> None:
        from types import SimpleNamespace

        response_format = {"type": "json_object"}
        response_calls: list[dict] = []

        class Responses:
            def create(self, **kwargs):
                response_calls.append(kwargs)
                return SimpleNamespace(output_text="{}", usage=None)

        class ChatCompletions:
            def create(self, **kwargs):
                response_calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                    usage=None,
                )

        client = OpenAiTextClient(config={"model": "m", "api_key": "k"})
        client._responses(
            SimpleNamespace(responses=Responses()),
            "s",
            "u",
            response_format=response_format,
        )
        client._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions())),
            "s",
            "u",
            response_format=response_format,
        )
        self.assertEqual(response_calls[0]["text"]["format"], response_format)
        self.assertEqual(response_calls[1]["response_format"], response_format)


class ThinkingConfigTest(unittest.TestCase):
    def test_thinking_defaults_off(self) -> None:
        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        cfg = resolve_config("default", {})
        self.assertFalse(cfg["thinking"])
        self.assertIsNone(cfg["reasoning_effort"])

    def test_thinking_and_effort_resolution(self) -> None:
        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        cases = [
            ({"thinking": True}, True, "medium"),
            ({"thinking": True, "reasoning_effort": "high"}, True, "high"),
            ({"reasoning_effort": "low"}, True, "low"),          # effort implies thinking
            ({"thinking": False, "reasoning_effort": "high"}, False, None),
            ({"thinking": "true"}, True, "medium"),
            ({"thinking": "off"}, False, None),
        ]
        for data, expected_thinking, expected_effort in cases:
            cfg = resolve_config("c", data)
            self.assertEqual(cfg["thinking"], expected_thinking, data)
            self.assertEqual(cfg["reasoning_effort"], expected_effort, data)

    def test_invalid_reasoning_effort_rejected(self) -> None:
        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        with self.assertRaises(LlmConfigError):
            resolve_config("c", {"thinking": True, "reasoning_effort": "bogus"})

    def test_thinking_body_must_be_an_object(self) -> None:
        from agents.cfu_design_agent.runtime.llm_config import resolve_config

        with self.assertRaises(LlmConfigError):
            resolve_config("c", {"thinking": True, "thinking_body": "nope"})

    def _chat_kwargs(self, config: dict) -> dict:
        from types import SimpleNamespace

        calls: dict = {}

        class ChatCompletions:
            def create(self, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    usage=None,
                )

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "chat", **config}
        )
        client._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions())), "s", "u"
        )
        return calls

    def test_client_effort_implies_thinking_when_constructed_directly(self) -> None:
        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "reasoning_effort": "low"}
        )
        self.assertTrue(client.thinking)
        self.assertEqual(client.reasoning_effort, "low")

    def test_thinking_off_sends_no_reasoning_parameters(self) -> None:
        kwargs = self._chat_kwargs({})
        self.assertNotIn("extra_body", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)

    def test_thinking_on_sends_gateway_switch_and_effort(self) -> None:
        kwargs = self._chat_kwargs({"thinking": True, "reasoning_effort": "high"})
        self.assertEqual(kwargs["extra_body"]["reasoning_effort"], "high")
        self.assertIn("thinking", kwargs["extra_body"])

    def test_custom_thinking_body_overrides_the_switch_shape(self) -> None:
        kwargs = self._chat_kwargs(
            {"thinking": True, "thinking_body": {"enable_thinking": True}}
        )
        self.assertEqual(kwargs["extra_body"]["enable_thinking"], True)
        self.assertNotIn("thinking", kwargs["extra_body"])

    def test_env_only_config_reads_thinking_env(self) -> None:
        import os
        from unittest import mock

        from agents.cfu_design_agent.runtime.llm_config import env_only_config

        with mock.patch.dict(
            os.environ,
            {"CFU_AGENT_THINKING": "true", "CFU_AGENT_REASONING_EFFORT": "low"},
        ):
            cfg = env_only_config("default")
        self.assertTrue(cfg["thinking"])
        self.assertEqual(cfg["reasoning_effort"], "low")

    def test_chat_result_records_observability(self) -> None:
        from types import SimpleNamespace

        class ChatCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content="ANSWER", reasoning_content="thought"
                            ),
                        )
                    ],
                    usage=None,
                )

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "chat", "thinking": True}
        )
        result = client._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions())), "s", "u"
        )
        self.assertEqual(result.api_mode_used, "chat")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.reasoning_source, "message.reasoning_content")
        self.assertEqual(result.reasoning, "thought")
        self.assertEqual(result.text, "ANSWER")
        self.assertIn("reasoning_content", result.extra_keys)

    def test_responses_api_receives_reasoning_effort(self) -> None:
        from types import SimpleNamespace

        calls: dict = {}

        class Responses:
            def create(self, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(output_text="ok", usage=None)

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "responses",
                    "thinking": True, "reasoning_effort": "low"}
        )
        client._responses(SimpleNamespace(responses=Responses()), "s", "u")
        # Azure/OpenAI hide chain-of-thought; a summary must be requested.
        self.assertEqual(calls["reasoning"], {"effort": "low", "summary": "auto"})

    def test_responses_summary_can_be_disabled_or_tuned(self) -> None:
        from types import SimpleNamespace

        def reasoning_for(config: dict) -> dict:
            calls: dict = {}

            class Responses:
                def create(self, **kwargs):
                    calls.update(kwargs)
                    return SimpleNamespace(output_text="ok", usage=None)

            client = OpenAiTextClient(config={"model": "m", "api_key": "k",
                                              "api_mode": "responses", **config})
            client._responses(SimpleNamespace(responses=Responses()), "s", "u")
            return calls.get("reasoning")

        self.assertEqual(
            reasoning_for({"thinking": True, "reasoning_summary": "detailed"}),
            {"effort": "medium", "summary": "detailed"},
        )
        self.assertEqual(
            reasoning_for({"thinking": True, "reasoning_summary": ""}),
            {"effort": "medium"},
        )
        self.assertIsNone(reasoning_for({"thinking": False}))

    def test_responses_retries_without_summary_when_rejected(self) -> None:
        from types import SimpleNamespace

        attempts: list[dict] = []

        class Responses:
            def create(self, **kwargs):
                attempts.append(dict(kwargs.get("reasoning") or {}))
                if "summary" in (kwargs.get("reasoning") or {}):
                    raise RuntimeError("summary not supported")
                return SimpleNamespace(output_text="ok", usage=None)

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "responses", "thinking": True}
        )
        result = client._responses(SimpleNamespace(responses=Responses()), "s", "u")
        self.assertEqual(result.text, "ok")
        # The first attempt asks for a summary; the successful retry drops it.
        self.assertEqual(attempts[0].get("summary"), "auto")
        self.assertNotIn("summary", attempts[-1])

    def test_chat_reads_reasoning_items_when_content_is_empty(self) -> None:
        from types import SimpleNamespace

        class ChatCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content="ANSWER",
                                reasoning_content="",
                                reasoning_items=[{"type": "reasoning_text", "text": "item thought"}],
                            ),
                        )
                    ],
                    usage=None,
                )

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "chat", "thinking": True}
        )
        result = client._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions())), "s", "u"
        )
        self.assertEqual(result.reasoning, "item thought")
        self.assertEqual(result.reasoning_source, "message.reasoning_items")

    def test_chat_reads_provider_specific_reasoning(self) -> None:
        from types import SimpleNamespace

        # litellm-backed gateways (e.g. GLM/Kimi here) expose the chain of
        # thought under provider_specific_fields.reasoning.
        message = SimpleNamespace(
            content="ANSWER",
            model_extra={"provider_specific_fields": {"reasoning": "provider thought"}},
        )
        from agents.cfu_design_agent.runtime.llm import _extract_chat_reasoning

        text, source = _extract_chat_reasoning(message)
        self.assertEqual(text, "provider thought")
        self.assertEqual(source, "provider_specific_fields.reasoning")

    def test_chat_ignores_a_null_provider_reasoning(self) -> None:
        from types import SimpleNamespace

        from agents.cfu_design_agent.runtime.llm import _extract_chat_reasoning

        # DeepSeek on this gateway reports provider_specific_fields.reasoning: None.
        message = SimpleNamespace(
            content="ANSWER",
            model_extra={"provider_specific_fields": {"reasoning": None, "refusal": None}},
        )
        self.assertEqual(_extract_chat_reasoning(message), ("", ""))

    def test_auto_fallback_to_chat_records_why_responses_was_unavailable(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        class Responses:
            def create(self, **kwargs):
                raise RuntimeError("404: no /responses route for this model")

        class Completions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="ANSWER", tool_calls=None),
                        )
                    ],
                    usage=None,
                )

        fake = SimpleNamespace(responses=Responses(), chat=SimpleNamespace(completions=Completions()))
        client = OpenAiTextClient(config={"model": "m", "api_key": "k", "api_mode": "auto"})
        with mock.patch("openai.OpenAI", return_value=fake):
            result = client.complete("s", "u", fallback="FB")
        self.assertEqual(result.api_mode_used, "chat")
        self.assertEqual(result.text, "ANSWER")
        self.assertIn("responses unavailable", result.fallback_reason)

    def test_responses_reads_the_reasoning_summary(self) -> None:
        from types import SimpleNamespace

        class Responses:
            def create(self, **kwargs):
                return SimpleNamespace(
                    output_text="ok",
                    usage=None,
                    output=[
                        SimpleNamespace(
                            type="reasoning",
                            summary=[SimpleNamespace(type="summary_text", text="summary thought")],
                            content=[],
                        ),
                        SimpleNamespace(type="message"),
                    ],
                )

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "responses", "thinking": True}
        )
        result = client._responses(SimpleNamespace(responses=Responses()), "s", "u")
        self.assertEqual(result.reasoning, "summary thought")
        self.assertEqual(result.reasoning_source, "responses.output.reasoning")


class ReasoningSplitTest(unittest.TestCase):
    def test_split_reasoning_handles_common_formats(self) -> None:
        from agents.cfu_design_agent.runtime.llm import split_reasoning

        lt, gt = chr(60), chr(62)
        cases = [
            (f"{lt}think{gt}draft{lt}/think{gt}\nANSWER", "ANSWER", "draft"),
            (f"{lt}thinking{gt}secret{lt}/thinking{gt}ANSWER", "ANSWER", "secret"),
            (f"PRE\n{lt}reasoning{gt}r1{lt}/reasoning{gt}\nAFTER", "PRE\n\n\nAFTER", "r1"),
            ("```thinking\nsecret\n```\nANSWER", "ANSWER", "secret"),
            ("plain answer", "plain answer", ""),
        ]
        for raw, expected_text, expected_reason in cases:
            text, reason = split_reasoning(raw)
            self.assertEqual(text, expected_text)
            self.assertEqual(reason, expected_reason)

    def test_unclosed_thinking_tag_is_treated_as_reasoning(self) -> None:
        from agents.cfu_design_agent.runtime.llm import split_reasoning

        lt, gt = chr(60), chr(62)
        text, reason = split_reasoning(f"ANSWER?\n{lt}thinking{gt}cut off mid-thought")
        self.assertEqual(text, "ANSWER?")
        self.assertEqual(reason, "cut off mid-thought")

    def test_chat_reasoning_field_is_split_from_the_answer(self) -> None:
        from types import SimpleNamespace

        from agents.cfu_design_agent.runtime.llm import OpenAiTextClient

        lt, gt = chr(60), chr(62)
        seen: dict = {}

        class ChatCompletions:
            def create(self, **kwargs):
                seen.update(kwargs)
                message = SimpleNamespace(
                    content=f"{lt}thinking{gt}inline draft{lt}/thinking{gt}\nFINAL",
                    reasoning_content="separate reasoning",
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)], usage=None
                )

        client = OpenAiTextClient(config={"model": "m", "api_key": "k", "api_mode": "chat"})
        result = client._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions())),
            "s",
            "u",
        )
        self.assertEqual(result.text, "FINAL")
        self.assertIn("separate reasoning", result.reasoning)
        self.assertIn("inline draft", result.reasoning)
        # The configured response_format must still reach the request.
        self.assertNotIn("thinking", result.text)


class StreamingTest(unittest.TestCase):
    """Provider token streaming: accumulation, live progress and fallbacks."""

    def _client(self, **config):
        base = {"model": "m", "api_key": "k", "api_mode": "chat", "thinking": True}
        base.update(config)
        return OpenAiTextClient(config=base)

    @staticmethod
    def _chunk(content=None, reasoning=None, tool_calls=None, usage=None, finish=None):
        from types import SimpleNamespace

        delta = SimpleNamespace(
            content=content, reasoning_content=reasoning, tool_calls=tool_calls
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage
        )

    def test_chat_stream_accumulates_content_reasoning_and_usage(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from agents.cfu_design_agent.runtime import llm

        chunks = [
            self._chunk(reasoning="think-"),
            self._chunk(reasoning="more "),
            self._chunk(content="HEL"),
            self._chunk(content="LO"),
            self._chunk(
                finish="stop",
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=4, total_tokens=14,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
                ),
            ),
        ]

        class Completions:
            def create(self, **kwargs):
                return iter(chunks)

        seen: list[dict] = []
        with mock.patch.object(llm, "_STREAM_EMIT_INTERVAL", 0.0):
            llm.set_stream_callback(seen.append)
            try:
                result = self._client()._chat(
                    SimpleNamespace(chat=SimpleNamespace(completions=Completions())), "s", "u"
                )
            finally:
                llm.set_stream_callback(None)
        self.assertEqual(result.text, "HELLO")
        self.assertEqual(result.reasoning, "think-more")
        self.assertEqual(result.reasoning_source, "stream.reasoning_content")
        self.assertEqual(
            (result.input_tokens, result.output_tokens, result.total_tokens), (10, 4, 14)
        )
        self.assertIn("stream", result.extra_keys)
        self.assertTrue(seen, "the live progress callback must be called")
        self.assertGreater(seen[-1]["chars"], 0)

    def test_stream_reads_sdk_extra_reasoning_and_preserves_exact_tool_transcript(self) -> None:
        from types import SimpleNamespace

        from agents.cfu_design_agent.runtime import llm

        call = SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(
            name="read_file", arguments='{"path":"x"}'))
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=14, total_tokens=24,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=9),
        )
        chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None, delta=
                SimpleNamespace(content=None, tool_calls=None,
                                model_extra={"reasoning_content": "  think"}))]),
            SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None, delta=
                SimpleNamespace(content=None, tool_calls=[call],
                                model_extra={"reasoning_content": "-more \n"}))]),
            SimpleNamespace(usage=usage, choices=[]),
        ]
        round_ = llm._chat_round_from_stream(iter(chunks))
        self.assertEqual(round_.reasoning, "think-more")
        self.assertEqual(round_.reasoning_source, "stream.reasoning_content")
        self.assertEqual(round_.message.reasoning_content, "  think-more \n")
        self.assertEqual(round_.reasoning_tokens, 9)
        self.assertIn("reasoning_content", round_.extra_keys)
        self.assertEqual(llm._assistant_tool_message(round_.message)["reasoning_content"],
                         "  think-more \n")

    def test_chat_stream_accumulates_tool_call_fragments(self) -> None:
        from types import SimpleNamespace

        def fragment(index, call_id, name, arguments):
            return SimpleNamespace(
                index=index,
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )

        chunks = [
            self._chunk(tool_calls=[fragment(0, "call_1", "read_", '{"pa')]),
            self._chunk(tool_calls=[fragment(0, None, "file", 'th": "a"}')]),
            self._chunk(finish="tool_calls"),
        ]

        class Completions:
            def create(self, **kwargs):
                return iter(chunks)

        client = self._client()
        round_ = client._chat_round(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
            [{"role": "user", "content": "x"}],
            allow_tools=True,
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        calls = list(round_.message.tool_calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].id, "call_1")
        self.assertEqual(calls[0].function.name, "read_file")
        self.assertEqual(calls[0].function.arguments, '{"path": "a"}')
        self.assertEqual(round_.finish_reason, "tool_calls")

    def test_chat_stream_falls_back_when_streaming_is_rejected(self) -> None:
        from types import SimpleNamespace

        class Completions:
            def create(self, **kwargs):
                if kwargs.get("stream"):
                    raise RuntimeError("Unsupported parameter: stream_options")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="PLAIN", tool_calls=None),
                        )
                    ],
                    usage=None,
                )

        result = self._client()._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())), "s", "u"
        )
        self.assertEqual(result.text, "PLAIN")
        self.assertIn("streaming unavailable", result.fallback_reason)

    def test_streaming_can_be_disabled(self) -> None:
        from types import SimpleNamespace

        requests: list[dict] = []

        class Completions:
            def create(self, **kwargs):
                requests.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="ANS", tool_calls=None),
                        )
                    ],
                    usage=None,
                )

        client = self._client(stream=False)
        result = client._chat(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())), "s", "u"
        )
        self.assertEqual(result.text, "ANS")
        self.assertNotIn("stream", requests[0])

    def test_responses_stream_accumulates_text_and_summary(self) -> None:
        from types import SimpleNamespace

        events = [
            SimpleNamespace(type="response.reasoning_summary_text.delta", delta="think "),
            SimpleNamespace(type="response.output_text.delta", delta="HE"),
            SimpleNamespace(type="response.output_text.delta", delta="LLO"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed",
                    usage=SimpleNamespace(input_tokens=3, output_tokens=5, total_tokens=8),
                ),
            ),
        ]

        class Responses:
            def create(self, **kwargs):
                assert kwargs.get("stream") is True
                return iter(events)

        client = OpenAiTextClient(
            config={"model": "m", "api_key": "k", "api_mode": "responses", "thinking": True}
        )
        result = client._responses(SimpleNamespace(responses=Responses()), "s", "u")
        self.assertEqual(result.text, "HELLO")
        self.assertEqual(result.reasoning, "think")
        self.assertEqual(result.reasoning_source, "responses.reasoning_summary.delta")
        self.assertEqual((result.input_tokens, result.output_tokens), (3, 5))


if __name__ == "__main__":
    unittest.main()
