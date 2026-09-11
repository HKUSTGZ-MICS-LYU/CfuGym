"""Small OpenAI SDK adapter used by the CFU design graph.

Token usage is recorded into a module-level ledger so that the surrounding
graph can report per-stage and cumulative token consumption. The stage active
during a call is the one set by :func:`set_stage` (the graph node wrapper sets
it before each node runs).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Callable

DEFAULT_MODEL = "gpt-5"
# Reasoning/thinking is OFF unless a config opts in. ``reasoning_effort`` values
# follow the OpenAI spelling; ``thinking_body`` lets a gateway use its own switch
# shape (default: {"thinking": {"type": "enabled"}}).
DEFAULT_REASONING_EFFORT = "medium"
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
DEFAULT_THINKING_BODY: dict[str, Any] = {"thinking": {"type": "enabled"}}

# Usage ledger: each LLM call appends a dict of its token counts, tagged with
# the stage that was active when the call was made.
_USAGE_LOG: list[dict[str, int]] = []
_CURRENT_STAGE = ""


@dataclass
class LlmResult:
    text: str
    used_llm: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Reasoning/thinking text, kept out of ``text`` so artifact parsers never
    # see it. Persisted separately for auditability.
    reasoning: str = ""
    # Observability: what was sent and whether the provider returned reasoning.
    model: str = ""
    api_mode_used: str = ""
    finish_reason: str = ""
    reasoning_source: str = ""
    extra_keys: tuple[str, ...] = ()
    # Tool-calling transcript for a bounded agent loop (see complete_with_tools).
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_rounds: int = 0


# --- Reasoning / thinking handling -----------------------------------------
# Reasoning models sometimes inline their chain of thought in ``content`` (for
# example QwQ/DeepSeek style ``...`` or ``<thinking>`` blocks). Those
# must never reach the YAML/JSON/SEARCH parsers.

_REASONING_TAGS = r"(?:think|thinking|reasoning|thought)"
_REASONING_PAIRED = (
    # <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>, ...
    re.compile(
        rf"<\s*{_REASONING_TAGS}\s*>(.*?)<\s*/\s*{_REASONING_TAGS}\s*>",
        re.IGNORECASE | re.DOTALL,
    ),
    # DeepSeek-R1 / QwQ style: literal  thinking...<｜end▁of▁thinking｜> without angle brackets.
    re.compile(r"<think\s*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(rf"```\s*{_REASONING_TAGS}\s*\n(.*?)```", re.IGNORECASE | re.DOTALL),
    re.compile(rf"~~~\s*{_REASONING_TAGS}\s*\n(.*?)~~~", re.IGNORECASE | re.DOTALL),
)
_REASONING_UNCLOSED = re.compile(
    rf"<\s*{_REASONING_TAGS}\s*>(.*)$", re.IGNORECASE | re.DOTALL
)


def split_reasoning(text: str) -> tuple[str, str]:
    """Return (answer_text, reasoning_text) with inline thinking removed.

    Handles paired ``<thinking>...</thinking>`` tags (also think/reasoning/
    thought), fenced ``thinking`` blocks, and an unclosed opening tag whose
    remainder is treated as reasoning (the common "model was cut off" shape).
    """
    if not text:
        return text, ""
    parts: list[str] = []

    def _capture(match: "re.Match[str]") -> str:
        body = (match.group(1) or "").strip()
        if body:
            parts.append(body)
        return "\n"

    cleaned = text
    for pattern in _REASONING_PAIRED:
        cleaned = pattern.sub(_capture, cleaned)
    unclosed = _REASONING_UNCLOSED.search(cleaned)
    if unclosed:
        body = (unclosed.group(1) or "").strip()
        if body:
            parts.append(body)
        cleaned = cleaned[: unclosed.start()]
    return cleaned.strip("\n"), "\n\n".join(parts).strip()


def clean_llm_text(result: LlmResult) -> LlmResult:
    """Move any inline reasoning out of ``result.text``.

    Idempotent, so the client and the graph can both call it. API-level
    reasoning fields are merged with inline blocks.
    """
    text, inline = split_reasoning(getattr(result, "text", "") or "")
    if not inline:
        return result
    reasoning = "\n\n".join(
        part for part in (getattr(result, "reasoning", "") or "", inline) if part
    ).strip()
    if is_dataclass(result) and not isinstance(result, type):
        return replace(result, text=text, reasoning=reasoning)
    # Duck-typed result (test doubles / other adapters): annotate in place.
    try:
        result.text = text
        result.reasoning = reasoning
        return result
    except Exception:  # pragma: no cover - immutable foreign result
        return LlmResult(
            text,
            used_llm=bool(getattr(result, "used_llm", False)),
            input_tokens=int(getattr(result, "input_tokens", 0) or 0),
            output_tokens=int(getattr(result, "output_tokens", 0) or 0),
            total_tokens=int(getattr(result, "total_tokens", 0) or 0),
            reasoning=reasoning,
        )


def set_stage(stage: str) -> None:
    global _CURRENT_STAGE
    _CURRENT_STAGE = stage or ""


def current_stage() -> str:
    return _CURRENT_STAGE


def reset_usage_log() -> None:
    _USAGE_LOG.clear()


def record_usage(
    stage: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    _USAGE_LOG.append(
        {
            "stage": stage,
            "input": int(input_tokens or 0),
            "output": int(output_tokens or 0),
            "total": int(total_tokens or 0),
        }
    )


def usage_totals() -> dict[str, int]:
    """Cumulative token usage across all recorded calls."""
    return {
        "calls": len(_USAGE_LOG),
        "input": sum(entry["input"] for entry in _USAGE_LOG),
        "output": sum(entry["output"] for entry in _USAGE_LOG),
        "total": sum(entry["total"] for entry in _USAGE_LOG),
    }


def stage_usage(stage: str) -> dict[str, int]:
    matched = [entry for entry in _USAGE_LOG if entry.get("stage") == stage]
    return {
        "calls": len(matched),
        "input": sum(entry["input"] for entry in matched),
        "output": sum(entry["output"] for entry in matched),
        "total": sum(entry["total"] for entry in matched),
    }


def _read(value, *keys):
    """Read ``value`` as a pydantic model or dict, trying ``keys`` in order."""
    if isinstance(value, dict):
        for key in keys:
            if key in value and value[key] is not None:
                return value[key]
        return 0
    for key in keys:
        candidate = getattr(value, key, None)
        if candidate is not None:
            return candidate
    return 0


def _extract_usage(response, kind: str) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    if kind == "responses":
        inp = int(_read(usage, "input_tokens", "input") or 0)
        out = int(_read(usage, "output_tokens", "output") or 0)
    else:
        inp = int(_read(usage, "prompt_tokens") or 0)
        out = int(_read(usage, "completion_tokens") or 0)
    total = int(_read(usage, "total_tokens", "total") or 0)
    if not total:
        total = inp + out
    return inp, out, total


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "summary"):
            text = _scalar_text(value.get(key))
            if text:
                return text
        return ""
    for key in ("text", "content"):
        text = _scalar_text(getattr(value, key, None))
        if text:
            return text
    return ""


def _reasoning_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [_scalar_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return _scalar_text(value)


_CHAT_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")


def _chat_message_keys(message: Any) -> tuple[str, ...]:
    keys: list[str] = []
    for key in _CHAT_REASONING_FIELDS:
        if getattr(message, key, None) is not None:
            keys.append(key)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        keys.extend(str(key) for key in extra.keys())
    return tuple(dict.fromkeys(keys))


def _responses_item_types(response: Any) -> tuple[str, ...]:
    types = [
        str(getattr(item, "type", "") or "")
        for item in getattr(response, "output", None) or []
    ]
    return tuple(dict.fromkeys(item_type for item_type in types if item_type))


def _extract_responses_reasoning(response: Any) -> tuple[str, str]:
    """(reasoning text, source) from an OpenAI Responses API result."""
    parts: list[str] = []
    found = False
    for item in getattr(response, "output", None) or []:
        if str(getattr(item, "type", "") or "") != "reasoning":
            continue
        found = True
        for summary in getattr(item, "summary", None) or []:
            parts.append(_scalar_text(summary))
        for content in getattr(item, "content", None) or []:
            parts.append(_scalar_text(content))
    text = "\n".join(part for part in parts if part).strip()
    return text, ("responses.output.reasoning" if found else "")


def _extract_chat_reasoning(message: Any) -> tuple[str, str]:
    """(reasoning text, source) from an OpenAI-compatible chat message."""
    for key in _CHAT_REASONING_FIELDS:
        text = _reasoning_value(getattr(message, key, None))
        if text:
            return text, f"message.{key}"
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in _CHAT_REASONING_FIELDS:
            text = _reasoning_value(extra.get(key))
            if text:
                return text, f"model_extra.{key}"
    return "", ""


def _reasoning_source(api_source: str, inline_reasoning: str) -> str:
    if api_source:
        return api_source
    return "inline" if inline_reasoning else ""


def _merge_reasoning(api_reasoning: str, inline_reasoning: str) -> str:
    return "\n\n".join(
        part for part in (api_reasoning or "", inline_reasoning or "") if part
    ).strip()


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    """Convert an SDK assistant message with tool_calls back into request form."""
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        calls.append(
            {
                "id": str(getattr(call, "id", "") or ""),
                "type": "function",
                "function": {
                    "name": str(getattr(function, "name", "") or ""),
                    "arguments": str(getattr(function, "arguments", "") or "{}"),
                },
            }
        )
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": calls,
    }


class OpenAiTextClient:
    """OpenAI-compatible text client.

    ``CFU_AGENT_LLM_API`` selects the endpoint style:

    - ``auto`` (default): try the Responses API, then fall back to chat
      completions if the endpoint does not support it.
    - ``responses``: use ``client.responses.create`` only.
    - ``chat``: use ``client.chat.completions.create`` only.
    """

    def __init__(self, model: str | None = None, config: dict | None = None):
        cfg = config or {}
        self.config_name = cfg.get("name")
        self.model = (
            model
            or cfg.get("model")
            or os.environ.get("CFU_AGENT_MODEL")
            or DEFAULT_MODEL
        )
        self.api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        self.temperature = cfg.get("temperature")
        self.timeout = cfg.get("timeout")
        mode = cfg.get("api_mode") or os.environ.get("CFU_AGENT_LLM_API") or "auto"
        self.api_mode = str(mode).lower() if isinstance(mode, str) else "auto"
        # Thinking defaults to off: no reasoning parameter is sent unless the
        # config sets "thinking": true or an explicit effort (mirrors
        # resolve_config, so direct client construction behaves the same).
        effort = cfg.get("reasoning_effort")
        thinking_raw = cfg.get("thinking")
        self.thinking = bool(effort) if thinking_raw is None else bool(thinking_raw)
        self.reasoning_effort = (
            str(effort)
            if self.thinking and effort
            else (DEFAULT_REASONING_EFFORT if self.thinking else None)
        )
        body = cfg.get("thinking_body")
        self.thinking_body = dict(body) if isinstance(body, dict) else None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _chat_reasoning_kwargs(self) -> dict[str, Any]:
        """extra_body/reasoning args for a chat completion, or {} when off."""
        if not self.thinking:
            return {}
        extra_body = (
            dict(self.thinking_body)
            if self.thinking_body is not None
            else dict(DEFAULT_THINKING_BODY)
        )
        if self.reasoning_effort:
            extra_body.setdefault("reasoning_effort", self.reasoning_effort)
        return {"extra_body": extra_body} if extra_body else {}

    def complete(
        self,
        system: str,
        user: str,
        *,
        fallback: str,
        response_format: dict[str, Any] | None = None,
    ) -> LlmResult:
        if not self.available:
            return LlmResult(fallback, used_llm=False)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        try:
            if self.api_mode == "chat":
                return self._chat(client, system, user, response_format=response_format)
            if self.api_mode == "responses":
                return self._responses(client, system, user, response_format=response_format)
        except Exception:
            if self.api_mode != "auto":
                raise
            # auto: try Responses first, then fall back to chat completions.
        try:
            return self._responses(client, system, user, response_format=response_format)
        except Exception as responses_error:
            try:
                return self._chat(client, system, user, response_format=response_format)
            except Exception as chat_error:
                raise RuntimeError(
                    f"LLM call failed in both responses and chat modes. "
                    f"responses: {responses_error}; chat: {chat_error}"
                ) from responses_error

    def complete_with_tools(
        self,
        system: str,
        user: str,
        *,
        tools: list[dict[str, Any]],
        tool_runner: Callable[[str, dict[str, Any]], str],
        max_rounds: int = 6,
        fallback: str = "",
        response_format: dict[str, Any] | None = None,
        on_tool_call: Callable[[dict[str, Any]], None] | None = None,
    ) -> LlmResult:
        """Run a bounded tool-calling loop over chat completions.

        The model may call ``tools`` up to ``max_rounds`` times; ``tool_runner``
        executes each call and the observation is fed back. The final answer
        (after tools are withdrawn) is returned in ``LlmResult.text`` and the
        transcript in ``LlmResult.tool_calls``.

        Implemented on the chat-completions path because that is what
        OpenAI-compatible gateways expose. If the endpoint rejects the tools
        payload, the loop degrades to a plain completion so enabling tools can
        never break a run.
        """
        if not self.available:
            return LlmResult(fallback, used_llm=False)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        transcript: list[dict[str, Any]] = []
        total_in = total_out = total_total = 0
        tool_rounds = 0
        max_tool_rounds = max(0, int(max_rounds))
        mode = "chat+tools" if max_tool_rounds > 0 else "chat"

        for iteration in range(max_tool_rounds + 1):
            allow_tools = iteration < max_tool_rounds
            kwargs: dict[str, Any] = {}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.timeout:
                kwargs["timeout"] = self.timeout
            kwargs.update(self._chat_reasoning_kwargs())
            if allow_tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            elif response_format is not None:
                kwargs["response_format"] = response_format
            try:
                response = client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
            except Exception:
                if allow_tools:
                    # The gateway may not accept tools; degrade to a plain call.
                    return self._chat(client, system, user, response_format=response_format)
                raise

            inp, out, total = _extract_usage(response, "chat")
            total_in += inp
            total_out += out
            total_total += total
            record_usage(current_stage(), inp, out, total)

            choice = response.choices[0]
            message = choice.message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                text, inline = split_reasoning(getattr(message, "content", "") or "")
                api_reasoning, api_source = _extract_chat_reasoning(message)
                return clean_llm_text(
                    LlmResult(
                        text,
                        used_llm=True,
                        input_tokens=total_in,
                        output_tokens=total_out,
                        total_tokens=total_total,
                        reasoning=_merge_reasoning(api_reasoning, inline),
                        model=self.model,
                        api_mode_used=mode,
                        finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                        reasoning_source=_reasoning_source(api_source, inline),
                        extra_keys=_chat_message_keys(message),
                        tool_calls=tuple(transcript),
                        tool_rounds=tool_rounds,
                    )
                )

            tool_rounds += 1
            messages.append(_assistant_tool_message(message))
            for call in tool_calls:
                function = getattr(call, "function", None)
                name = str(getattr(function, "name", "") or "")
                arguments = _parse_tool_arguments(getattr(function, "arguments", ""))
                try:
                    observation = tool_runner(name, arguments)
                    ok = True
                except Exception as exc:  # pragma: no cover - tools report their own errors
                    observation = f"tool error: {type(exc).__name__}: {exc}"
                    ok = False
                entry = {
                    "round": tool_rounds,
                    "name": name,
                    "arguments": arguments,
                    "result": observation,
                    "ok": ok,
                }
                transcript.append(entry)
                if on_tool_call is not None:
                    on_tool_call(entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(getattr(call, "id", "") or ""),
                        "content": observation,
                    }
                )

        return LlmResult(
            fallback,
            used_llm=True,
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total_total,
            model=self.model,
            api_mode_used=mode,
            tool_calls=tuple(transcript),
            tool_rounds=tool_rounds,
        )

    def _responses(
        self,
        client,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LlmResult:
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if response_format is not None:
            kwargs["text"] = {"format": response_format}
        if self.thinking:
            reasoning: dict[str, Any] = {}
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            kwargs["reasoning"] = reasoning or {"effort": DEFAULT_REASONING_EFFORT}
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        inp, out, total = _extract_usage(response, "responses")
        record_usage(current_stage(), inp, out, total)
        text, inline = split_reasoning(getattr(response, "output_text", "") or "")
        api_reasoning, api_source = _extract_responses_reasoning(response)
        result = LlmResult(
            text,
            used_llm=True,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            reasoning=_merge_reasoning(api_reasoning, inline),
            model=self.model,
            api_mode_used="responses",
            finish_reason=str(getattr(response, "status", "") or ""),
            reasoning_source=_reasoning_source(api_source, inline),
            extra_keys=_responses_item_types(response),
        )
        return clean_llm_text(result)

    def _chat(
        self,
        client,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LlmResult:
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if response_format is not None:
            kwargs["response_format"] = response_format
        kwargs.update(self._chat_reasoning_kwargs())
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        inp, out, total = _extract_usage(response, "chat")
        record_usage(current_stage(), inp, out, total)
        choice = response.choices[0]
        message = choice.message
        text, inline = split_reasoning(getattr(message, "content", "") or "")
        api_reasoning, api_source = _extract_chat_reasoning(message)
        result = LlmResult(
            text,
            used_llm=True,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            reasoning=_merge_reasoning(api_reasoning, inline),
            model=self.model,
            api_mode_used="chat",
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            reasoning_source=_reasoning_source(api_source, inline),
            extra_keys=_chat_message_keys(message),
        )
        return clean_llm_text(result)
