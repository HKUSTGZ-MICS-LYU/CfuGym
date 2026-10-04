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
import time
from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

from . import context_budget, request_audit, tools as tools_mod

DEFAULT_MODEL = "gpt-5"
# Reasoning/thinking is OFF unless a config opts in. ``reasoning_effort`` values
# follow the OpenAI spelling; ``thinking_body`` lets a gateway use its own switch
# shape (default: {"thinking": {"type": "enabled"}}).
DEFAULT_REASONING_EFFORT = "medium"
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
# OpenAI/Azure reasoning models hide chain-of-thought and expose a summary only
# when the Responses request asks for one; "auto" lets the provider choose.
DEFAULT_REASONING_SUMMARY = "auto"
DEFAULT_THINKING_BODY: dict[str, Any] = {"thinking": {"type": "enabled"}}
# Bounded tool loop defaults: rounds are model turns that may use tools, calls
# per round caps parallel fan-out, and total calls caps the whole step.
DEFAULT_MAX_TOOL_ROUNDS = 20
DEFAULT_MAX_TOOL_CALLS_PER_ROUND = 4
DEFAULT_MAX_TOOL_CALLS = 96
# Injected before the forced final turn so the model stops exploring and emits
# the deliverable instead of narrating what it would read next. The rung passes
# its own deliverable description (a patch block, an analysis mapping, ...).
DEFAULT_TOOL_DELIVERABLE = (
    "the EDIT_BEGIN/SEARCH/REPLACE, FILE_BEGIN/FILE_END, or "
    "PATCH_BEGIN/PATCH_END block required by the delivery protocol"
)
TOOL_BUDGET_REMINDER = (
    "Tool budget exhausted. Do not call any more tools and do not describe what "
    "you would read next. Reply now with the final deliverable only: "
)


def tool_budget_reminder(deliverable: str = "") -> str:
    return TOOL_BUDGET_REMINDER + (deliverable or DEFAULT_TOOL_DELIVERABLE) + "."


def looks_like_tool_intent(text: str) -> bool:
    """True when a short answer only announces reading a file (no deliverable).

    A degraded tools call, a gateway that drops tool_calls, or a model that
    narrates its plan can leave this as the whole response. It must not be
    mistaken for a finished design.
    """
    body = (text or "").strip()
    if not body or len(body) > 2000:
        return False
    return bool(
        re.search(
            r"\b(?:let me|i'?ll|i will|i need to|i'?m going to|first,? i(?:'?ll| will)?)\b"
            r"[^.\n]{0,100}?"
            r"\b(?:read|examine|inspect|look at|look into|check|open|review|explore|"
            r"verify|confirm|understand|trace|study|consult|grep|search|find|gather|"
            r"determine)\b",
            body,
            re.IGNORECASE,
        )
    )


# Sent when the model announced a read but emitted no tool call, so it can retry
# while its tool budget is still available instead of the rung failing.
TOOL_INTENT_NUDGE = (
    "You announced that you would inspect or verify something, but your reply "
    "contained no tool call and no final deliverable. Do one of these now:\n"
    "- call the tool you intended (one call is enough to start), or\n"
    "- if the context above is already sufficient, emit the final deliverable "
    "immediately.\n"
    "Do not reply with another description of what you plan to read."
)

# Sent when a 200 response carried no content and no tool call. Retried once,
# because that is a transport-level failure rather than an answer.
EMPTY_ANSWER_NUDGE = (
    "Your previous reply was empty: it contained no text and no tool call. "
    "Answer now — call the tool you need, or emit the final deliverable."
)

# Sent when a rung that requires a delivery block got none while tools were
# still available.
DELIVERABLE_NUDGE = (
    "Your reply contained no final deliverable. Emit it now: an "
    "EDIT_BEGIN/SEARCH/REPLACE block, a FILE_BEGIN/FILE_END block, or a "
    "PATCH_BEGIN/PATCH_END diff. If you still need to inspect a file first, call "
    "the tool now; otherwise answer from the context already shown."
)

# Usage ledger: each LLM call appends a dict of its token counts, tagged with
# the stage that was active when the call was made.
_USAGE_LOG: list[dict[str, int]] = []
_CURRENT_STAGE = ""

# Token streaming is on by default: it lets the CLI show live output-token
# progress and, because the read timeout then applies per chunk instead of to
# the whole body, a slow long generation no longer trips the request timeout.
DEFAULT_STREAM = True
# How often a streaming call reports progress to the callback (seconds).
_STREAM_EMIT_INTERVAL = 0.25
# A stream callback receives {"stage", "phase", "chars", "tokens", "elapsed"}.
_STREAM_CALLBACK: "Callable[[dict[str, Any]], None] | None" = None
# Rough live token estimate for the progress line (the exact count arrives with
# the provider's final usage chunk).
_CHARS_PER_TOKEN = 4


def set_stream_callback(callback: "Callable[[dict[str, Any]], None] | None") -> None:
    """Install the process-wide streaming progress sink (None disables it)."""
    global _STREAM_CALLBACK
    _STREAM_CALLBACK = callback


def _emit_stream(
    stage: str, phase: str, chars: int, thinking_chars: int = 0, **extra: Any
) -> None:
    callback = _STREAM_CALLBACK
    if callback is None:
        return
    try:
        callback(
            {
                "stage": stage,
                "phase": phase,
                "chars": int(chars),
                "tokens": int(chars) // _CHARS_PER_TOKEN,
                "thinking_chars": int(thinking_chars),
                "thinking_tokens": int(thinking_chars) // _CHARS_PER_TOKEN,
                **extra,
            }
        )
    except Exception:  # pragma: no cover - progress must never break a call
        pass


@dataclass
class LlmResult:
    text: str
    used_llm: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int | None = None
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
    # Set when a requested tool loop could not run and the call degraded to a
    # plain completion, so the rung knows the model was left without tools.
    degraded_reason: str = ""
    # Set when ``api_mode: auto`` used chat because responses was unavailable
    # (for example the gateway has no /responses route for the model). This
    # explains "no thinking captured" for such a call.
    fallback_reason: str = ""
    # Set when the request itself failed (timeout, gateway error) and the graph
    # degraded to the fallback. This is an environment failure, never a design
    # defect, so a rung can report it without blaming the artifact.
    transport_error: str = ""


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
    cached_input_tokens: int | None = None,
) -> None:
    _USAGE_LOG.append(
        {
            "stage": stage,
            "input": int(input_tokens or 0),
            "output": int(output_tokens or 0),
            "total": int(total_tokens or 0),
            "cached_input": None if cached_input_tokens is None else int(cached_input_tokens),
        }
    )


def usage_totals() -> dict[str, int]:
    """Cumulative token usage across all recorded calls."""
    return {
        "calls": len(_USAGE_LOG),
        "input": sum(entry["input"] for entry in _USAGE_LOG),
        "output": sum(entry["output"] for entry in _USAGE_LOG),
        "total": sum(entry["total"] for entry in _USAGE_LOG),
        "cached_input": sum(entry["cached_input"] or 0 for entry in _USAGE_LOG),
        "cached_usage_reported": sum(entry["cached_input"] is not None for entry in _USAGE_LOG),
    }


def stage_usage(stage: str) -> dict[str, int]:
    matched = [entry for entry in _USAGE_LOG if entry.get("stage") == stage]
    return {
        "calls": len(matched),
        "input": sum(entry["input"] for entry in matched),
        "output": sum(entry["output"] for entry in matched),
        "total": sum(entry["total"] for entry in matched),
        "cached_input": sum(entry["cached_input"] or 0 for entry in matched),
        "cached_usage_reported": sum(entry["cached_input"] is not None for entry in matched),
    }


def usage_by_stage() -> dict[str, dict[str, int]]:
    """Per-stage usage in first-seen stage order.

    This is what makes a big-input stage visible: every request is attributed
    to the node that made it (``set_stage``), so callers can report where the
    input tokens actually went instead of only a run total.
    """
    stages: dict[str, dict[str, int]] = {}
    for entry in _USAGE_LOG:
        name = str(entry.get("stage") or "(unset)")
        bucket = stages.setdefault(name, {"calls": 0, "input": 0, "output": 0, "total": 0,
                                          "cached_input": 0, "cached_usage_reported": 0})
        bucket["calls"] += 1
        bucket["input"] += int(entry.get("input") or 0)
        bucket["output"] += int(entry.get("output") or 0)
        bucket["total"] += int(entry.get("total") or 0)
        bucket["cached_input"] += int(entry.get("cached_input") or 0)
        bucket["cached_usage_reported"] += entry.get("cached_input") is not None
    return stages


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


def _cached_usage(usage: Any, kind: str) -> int | None:
    if usage is None:
        return None
    details = _read(usage, "input_tokens_details" if kind == "responses" else "prompt_tokens_details")
    if not details:
        return None
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return int(cached) if cached is not None and cached != "" else None


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


# Chat-completions reasoning carriers used by OpenAI-compatible gateways.
# ``reasoning_items`` is a list some gateways use when ``reasoning_content`` is
# empty, so both are read.
_CHAT_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking", "reasoning_items")


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
    # litellm-backed gateways may expose the provider reasoning only under
    # provider_specific_fields.reasoning (and set it to None when there is none).
    containers = [getattr(message, "provider_specific_fields", None)]
    if isinstance(extra, dict):
        containers.append(extra.get("provider_specific_fields"))
    for container in containers:
        if isinstance(container, dict):
            text = _reasoning_value(container.get("reasoning"))
            if text:
                return text, "provider_specific_fields.reasoning"
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


@dataclass
class _ChatRound:
    """One chat completion, however it was obtained (streamed or not)."""

    text: str
    reasoning: str
    reasoning_source: str
    finish_reason: str
    extra_keys: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    message: Any
    cached_input_tokens: int | None = None
    fallback_reason: str = ""


def _usage_counts(usage: Any, kind: str) -> tuple[int, int, int]:
    return _extract_usage(SimpleNamespace(usage=usage), kind)


def _chat_round_from_response(response: Any) -> "_ChatRound":
    choice = response.choices[0]
    message = choice.message
    text, inline = split_reasoning(getattr(message, "content", "") or "")
    api_reasoning, api_source = _extract_chat_reasoning(message)
    inp, out, total = _extract_usage(response, "chat")
    return _ChatRound(
        text=text,
        reasoning=_merge_reasoning(api_reasoning, inline),
        reasoning_source=_reasoning_source(api_source, inline),
        finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        extra_keys=_chat_message_keys(message),
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
        message=message,
        cached_input_tokens=_cached_usage(getattr(response, "usage", None), "chat"),
    )


def _chat_round_from_stream(stream: Any) -> "_ChatRound":
    """Accumulate streamed chat deltas into a ``_ChatRound``.

    Content, reasoning and tool-call fragments all arrive incrementally; the
    provider's final usage chunk gives the exact token counts.
    """
    stage = current_stage()
    started = time.monotonic()
    _emit_stream(stage, "start", 0)
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    usage = None
    finish = ""
    last_emit = started
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            finish = str(getattr(choice, "finish_reason", "") or finish)
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            if piece:
                content.append(str(piece))
            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece is None:
                reasoning_piece = getattr(delta, "reasoning", None)
            if reasoning_piece:
                reasoning.append(str(reasoning_piece))
            for call in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(call, "index", 0) or 0)
                slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if getattr(call, "id", None):
                    slot["id"] = str(call.id)
                function = getattr(call, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        slot["name"] += str(function.name)
                    if getattr(function, "arguments", None):
                        slot["arguments"] += str(function.arguments)
        now = time.monotonic()
        if now - last_emit >= _STREAM_EMIT_INTERVAL:
            last_emit = now
            _emit_stream(
                stage,
                "content",
                len("".join(content)),
                thinking_chars=len("".join(reasoning)),
            )
    text = "".join(content)
    # Streamed deltas are fragments of one string: concatenate, do not join.
    thinking = "".join(reasoning).strip()
    tool_calls = [
        SimpleNamespace(
            id=slot["id"] or f"call_{index}",
            type="function",
            function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"] or "{}"),
        )
        for index, slot in sorted(calls.items())
    ]
    message = SimpleNamespace(content=text or None, tool_calls=tool_calls or None)
    clean_text, inline = split_reasoning(text)
    api_source = "stream.reasoning_content" if thinking else ""
    inp, out, total = _usage_counts(usage, "chat")
    return _ChatRound(
        text=clean_text,
        reasoning=_merge_reasoning(thinking, inline),
        reasoning_source=_reasoning_source(api_source, inline),
        finish_reason=finish,
        extra_keys=("stream",) + (("reasoning_content",) if thinking else ()),
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
        message=message,
        cached_input_tokens=_cached_usage(usage, "chat"),
    )


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
        self.window_config = {key: cfg[key] for key in ("context_window_tokens", "output_reserve_tokens")
                              if cfg.get(key) is not None}
        self._window = None
        self._last_budget_metadata: dict[str, Any] = {}
        self._last_request_id: str | None = None
        self._request_attempt = 0
        self._previous_payload: str | None = None
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
        # OpenAI/Azure reasoning models never expose raw chain-of-thought; the
        # Responses API returns only a summary, and only when it is requested.
        summary = cfg.get("reasoning_summary")
        if summary is None:
            summary = DEFAULT_REASONING_SUMMARY if self.thinking else None
        self.reasoning_summary = str(summary) if summary else None
        # Token streaming: on unless the config/caller disables it.
        stream = cfg.get("stream")
        self.stream = DEFAULT_STREAM if stream is None else bool(stream)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _context_window(self, sdk) -> context_budget.WindowSpec:
        if self._window is not None:
            return self._window
        key = (self.base_url or "", self.model)
        metadata = request_audit.model_metadata(key)
        if metadata is None:
            metadata = {}
            if not self.window_config.get("context_window_tokens") and hasattr(sdk, "models"):
                request_id = request_audit.record_request(
                    {"model": self.model}, {"stage": current_stage(), "endpoint": "models.retrieve"})
                try:
                    response = sdk.models.retrieve(self.model, timeout=3)
                    raw = response if isinstance(response, dict) else response.model_dump()
                    # Keep only known capacity fields, never arbitrary gateway
                    # metadata, credentials or transport configuration.
                    metadata = {name: raw[name] for name in (
                        "context_window", "context_window_tokens", "context_length",
                        "max_context_tokens", "max_context_length", "max_model_len",
                        "max_position_embeddings") if name in raw}
                    try:
                        context_budget.resolve_window(self.window_config, metadata)
                    except context_budget.ContextBudgetError:
                        metadata = {}  # malformed discovery is unknown, not authoritative
                    request_audit.record_result(request_id, result={"model_metadata": metadata})
                except request_audit.RequestAuditError:
                    raise
                except Exception as exc:
                    request_audit.record_result(request_id, error={"type": type(exc).__name__})
            request_audit.remember_model_metadata(key, metadata)
        self._window = context_budget.resolve_window(self.window_config, metadata)
        return self._window

    def _budget_messages(self, messages, *, tools=None, response_config=None, sdk=None):
        window = self._context_window(sdk)
        status = context_budget.evaluate_budget(
            messages, tools=tools, response_config=response_config, window=window)
        self._last_budget_metadata = {
            "estimated_input_tokens": status.estimated_input_tokens,
            "reserved_output_tokens": status.reserved_output_tokens,
            "window_tokens": window.window_tokens, "window_source": window.source,
            "capacity_known": window.known, "estimator": "serialized-byte-upper-bound",
            "before_total_tokens": status.total_tokens, "level": status.level,
        }
        if status.level in {"compact", "unfit"}:
            try:
                reduced = context_budget.compact_completed_tool_turns(
                    messages, tools=tools, response_config=response_config, window=window)
            except context_budget.ContextBudgetError:
                request_audit.record_event("blocked", {
                    "stage": current_stage(), "model": self.model, "sent": False,
                    **self._last_budget_metadata})
                raise
            messages = reduced.messages
            status = reduced.after
            self._last_budget_metadata.update(
                estimated_input_tokens=status.estimated_input_tokens,
                after_total_tokens=status.total_tokens, removed_batches=reduced.removed_batches,
                level=status.level)
            request_audit.record_event("compaction", {
                "stage": current_stage(), "model": self.model, **self._last_budget_metadata})
        elif status.level == "warning":
            request_audit.record_event("context_warning", {
                "stage": current_stage(), "model": self.model, **self._last_budget_metadata})
        return messages, status

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

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

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
        except (context_budget.ContextBudgetError, request_audit.RequestAuditError):
            raise
        except Exception as responses_error:
            try:
                result = self._chat(client, system, user, response_format=response_format)
            except Exception as chat_error:
                raise RuntimeError(
                    f"LLM call failed in both responses and chat modes. "
                    f"responses: {responses_error}; chat: {chat_error}"
                ) from responses_error
            # Record why responses was not used, so a run that captured no
            # thinking can be explained after the fact.
            try:
                result.fallback_reason = (
                    f"responses unavailable: {type(responses_error).__name__}: {responses_error}"
                )
            except Exception:  # pragma: no cover - foreign result object
                pass
            return result

    def complete_with_tools(
        self,
        system: str,
        user: str,
        *,
        tools: list[dict[str, Any]],
        tool_runner: Callable[[str, dict[str, Any]], str],
        max_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_calls_per_round: int = DEFAULT_MAX_TOOL_CALLS_PER_ROUND,
        max_total_calls: int = DEFAULT_MAX_TOOL_CALLS,
        fallback: str = "",
        response_format: dict[str, Any] | None = None,
        deliverable_hint: str = "",
        nudge_on_tool_intent: bool = True,
        is_deliverable: Callable[[str], bool] | None = None,
        on_tool_call: Callable[[dict[str, Any]], None] | None = None,
    ) -> LlmResult:
        """Run a bounded tool-calling loop over chat completions.

        The model may call ``tools`` for at most ``max_rounds`` rounds and
        ``max_total_calls`` calls overall (0 disables the total cap). Each round
        executes at most ``max_calls_per_round`` parallel calls; the rest are
        answered with a skip notice so the API still sees a response for every
        call id. The final answer is produced with the tools withdrawn, after an
        explicit reminder to deliver, and is returned in ``LlmResult.text`` with
        the transcript in ``LlmResult.tool_calls``.

        Implemented on the chat-completions path because that is what
        OpenAI-compatible gateways expose. An endpoint that rejects the tools
        payload fails explicitly rather than silently answering without tools.
        """
        if not self.available:
            return LlmResult(fallback, used_llm=False)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        transcript: list[dict[str, Any]] = []
        total_in = total_out = total_total = 0
        total_cached = 0
        cached_reported = False
        tool_rounds = 0
        total_calls = 0
        nudges = 0
        max_nudges = 1 if nudge_on_tool_intent else 0
        empty_retries = 0
        max_tool_rounds = max(0, int(max_rounds))
        round_cap = max(1, int(max_calls_per_round))
        total_cap = max(0, int(max_total_calls))
        mode = "chat+tools" if max_tool_rounds > 0 else "chat"

        for iteration in range(max_tool_rounds + 1):
            budget_exhausted = bool(total_cap) and total_calls >= total_cap
            allow_tools = iteration < max_tool_rounds and not budget_exhausted
            if not allow_tools and tool_rounds:
                # One explicit nudge before the forced final answer.
                messages.append({"role": "user", "content": tool_budget_reminder(deliverable_hint)})
            try:
                round_ = self._chat_round(
                    client,
                    messages,
                    allow_tools=allow_tools,
                    tools=tools,
                    response_format=response_format if not allow_tools else None,
                )
            except Exception as exc:
                if not allow_tools:
                    raise
                raise RuntimeError(
                    f"tool-enabled model request failed: {type(exc).__name__}: {exc}"
                ) from exc

            messages = list(getattr(self, "_prepared_messages", messages))
            inp, out, total = round_.input_tokens, round_.output_tokens, round_.total_tokens
            total_in += inp
            total_out += out
            total_total += total
            if round_.cached_input_tokens is not None:
                cached_reported = True
                total_cached += round_.cached_input_tokens
            record_usage(current_stage(), inp, out, total, round_.cached_input_tokens)

            message = round_.message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            text = round_.text
            if not allow_tools or not tool_calls:
                intent_only = bool(allow_tools and looks_like_tool_intent(text))
                missing_deliverable = bool(
                    allow_tools
                    and is_deliverable is not None
                    and not is_deliverable(text)
                )
                empty_answer = not text.strip()
                if empty_answer and empty_retries < 1:
                    # A 200 response with no content and no tool call is a
                    # transport-level failure, not a design decision (run
                    # 20261003-095209: the analysis loop died on the sixth such
                    # answer after five productive rounds). Re-ask once; the
                    # answer still may not be fabricated, so a second empty
                    # reply raises below.
                    empty_retries += 1
                    messages.append({"role": "user", "content": EMPTY_ANSWER_NUDGE})
                    continue
                if nudges < max_nudges and (intent_only or missing_deliverable):
                    # The model's reply cannot be used: either it announced a read
                    # without calling a tool, or a rung that requires a delivery
                    # block got none. Let it retry once with tools still available
                    # instead of wasting the remaining budget.
                    nudges += 1
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": TOOL_INTENT_NUDGE if intent_only else DELIVERABLE_NUDGE,
                        }
                    )
                    continue
                # Either the model answered, or it kept asking for tools after
                # they were withdrawn. Do not turn an empty answer into a
                # fabricated success when no deliverable was submitted.
                if empty_answer:
                    raise RuntimeError("tool-enabled model returned an empty stage answer")
                return clean_llm_text(
                    LlmResult(
                        text,
                        used_llm=True,
                        input_tokens=total_in,
                        output_tokens=total_out,
                        total_tokens=total_total,
                        cached_input_tokens=total_cached if cached_reported else None,
                        reasoning=round_.reasoning,
                        model=self.model,
                        api_mode_used=mode,
                        finish_reason=round_.finish_reason,
                        reasoning_source=round_.reasoning_source,
                        extra_keys=round_.extra_keys,
                        tool_calls=tuple(transcript),
                        tool_rounds=tool_rounds,
                    )
                )

            tool_rounds += 1
            messages.append(_assistant_tool_message(message))
            for index, call in enumerate(tool_calls):
                function = getattr(call, "function", None)
                name = str(getattr(function, "name", "") or "")
                arguments = _parse_tool_arguments(getattr(function, "arguments", ""))
                over_round = index >= round_cap
                over_total = bool(total_cap) and total_calls >= total_cap
                if over_round or over_total:
                    observation = (
                        "skipped: tool-call budget reached; produce the final "
                        "deliverable instead of reading more"
                    )
                    ok = False
                else:
                    try:
                        observation = tool_runner(name, arguments)
                        # Keep the string runner API, but do not count error text
                        # or a rendered "check failed" as a successful observation.
                        ok = tools_mod.tool_result_ok(observation)
                    except Exception as exc:  # pragma: no cover - tools report their own errors
                        observation = f"tool error: {type(exc).__name__}: {exc}"
                        ok = False
                    total_calls += 1
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

        raise RuntimeError("tool-enabled model exhausted its rounds without a stage answer")

    def _responses_stream(self, stream) -> LlmResult:
        """Consume a Responses event stream into a result.

        Accumulates output text, the reasoning summary, and the final usage.
        """
        stage = current_stage()
        started = time.monotonic()
        _emit_stream(stage, "start", 0)
        text_parts: list[str] = []
        reason_parts: list[str] = []
        usage = None
        response = None
        last_emit = started
        stream_error = ""
        completed = False
        for event in stream:
            etype = str(getattr(event, "type", "") or "")
            if etype in ("error", "response.failed"):
                detail = getattr(event, "message", "") or getattr(event, "error", "")
                stream_error = str(detail or etype)
            if etype == "response.output_text.delta":
                piece = getattr(event, "delta", "") or ""
                if piece:
                    text_parts.append(str(piece))
            elif etype in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                piece = getattr(event, "delta", "") or ""
                if piece:
                    reason_parts.append(str(piece))
            elif etype in ("response.completed", "response.done", "response.incomplete"):
                completed = True
                response = getattr(event, "response", None)
                usage = getattr(response, "usage", None) if response is not None else None
            now = time.monotonic()
            if now - last_emit >= _STREAM_EMIT_INTERVAL:
                last_emit = now
                _emit_stream(
                    stage,
                    "content",
                    len("".join(text_parts)),
                    thinking_chars=len("".join(reason_parts)),
                )
        if stream_error:
            raise RuntimeError(f"responses stream error: {stream_error}")
        if not text_parts and not reason_parts and not completed:
            # Some gateways answer a streaming request with an empty/error SSE
            # stream; raising lets api_mode=auto fall back to chat.
            raise RuntimeError("responses stream produced no content")
        text, inline = split_reasoning("".join(text_parts))
        # Streamed deltas are fragments of one string: concatenate, do not join.
        summary = "".join(reason_parts).strip()
        api_source = "responses.reasoning_summary.delta" if summary else ""
        inp, out, total = _usage_counts(usage, "responses")
        cached = _cached_usage(usage, "responses")
        record_usage(stage, inp, out, total, cached)
        result = LlmResult(
            text,
            used_llm=True,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            cached_input_tokens=cached,
            reasoning=_merge_reasoning(summary, inline),
            model=self.model,
            api_mode_used="responses",
            finish_reason=str(getattr(response, "status", "") or ""),
            reasoning_source=_reasoning_source(api_source, inline),
            extra_keys=("stream",),
        )
        return clean_llm_text(result)

    def _responses(
        self,
        client,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LlmResult:
        kwargs: dict = {"max_output_tokens": self._context_window(client).output_reserve_tokens}
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
            if self.reasoning_summary:
                # Without this the reasoning items come back with an empty
                # summary and no thinking can be captured.
                reasoning["summary"] = self.reasoning_summary
            kwargs["reasoning"] = reasoning or {"effort": DEFAULT_REASONING_EFFORT}
        fallback_reason = ""
        if self.stream:
            try:
                inputs, budget = self._budget_messages(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_config={k: v for k, v in kwargs.items() if k != "timeout"}, sdk=client)
                stream, request_id = self._sdk_request(
                    client, "responses", {"model": self.model, "input": inputs, "stream": True, **kwargs},
                    budget=budget)
            except (context_budget.ContextBudgetError, request_audit.RequestAuditError):
                raise
            except Exception as exc:
                fallback_reason = f"streaming unavailable: {type(exc).__name__}: {exc}"
            else:
                if hasattr(stream, "output") or hasattr(stream, "output_text"):
                    # Some gateways ignore stream=True and return the response.
                    fallback_reason = "streaming ignored by the endpoint"
                    return self._responses_result(stream, fallback_reason)
                try:
                    result = self._responses_stream(stream)
                except Exception as exc:
                    request_audit.record_result(request_id, error={"type": type(exc).__name__})
                    raise
                request_audit.record_result(request_id, result={"status": "completed", "input_tokens": result.input_tokens,
                                                               "cached_input_tokens": result.cached_input_tokens})
                return result
        return self._responses_result(
            self._responses_create(client, system, user, kwargs), fallback_reason
        )

    def _responses_create(self, client, system: str, user: str, kwargs: dict):
        """Non-streaming Responses call, retrying without a rejected summary."""
        inputs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs, budget = self._budget_messages(
            inputs, response_config={k: v for k, v in kwargs.items() if k != "timeout"}, sdk=client)
        try:
            response, _ = self._sdk_request(
                client, "responses", {"model": self.model, "input": inputs, **kwargs}, budget=budget)
            return response
        except (context_budget.ContextBudgetError, request_audit.RequestAuditError):
            raise
        except Exception:
            # A gateway that rejects the summary request must not fail the call.
            reasoning = kwargs.get("reasoning")
            if not isinstance(reasoning, dict) or "summary" not in reasoning:
                raise
            kwargs["reasoning"] = {k: v for k, v in reasoning.items() if k != "summary"}
            response, _ = self._sdk_request(
                client, "responses", {"model": self.model, "input": inputs, **kwargs}, budget=budget)
            return response

    def _responses_result(self, response, fallback_reason: str = "") -> LlmResult:
        inp, out, total = _extract_usage(response, "responses")
        cached = _cached_usage(getattr(response, "usage", None), "responses")
        record_usage(current_stage(), inp, out, total, cached)
        text, inline = split_reasoning(getattr(response, "output_text", "") or "")
        api_reasoning, api_source = _extract_responses_reasoning(response)
        result = LlmResult(
            text,
            used_llm=True,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            cached_input_tokens=cached,
            reasoning=_merge_reasoning(api_reasoning, inline),
            model=self.model,
            api_mode_used="responses",
            finish_reason=str(getattr(response, "status", "") or ""),
            reasoning_source=_reasoning_source(api_source, inline),
            extra_keys=_responses_item_types(response),
            fallback_reason=fallback_reason,
        )
        return clean_llm_text(result)

    @staticmethod
    def _audit_usage(response, endpoint: str) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        inp, out, total = _extract_usage(response, "chat" if endpoint == "chat" else "responses")
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": total,
                "cached_input_tokens": _cached_usage(usage, "chat" if endpoint == "chat" else "responses")}

    def _sdk_request(self, sdk, endpoint: str, body: dict[str, Any], *, budget=None):
        from urllib.parse import urlsplit
        metadata = {"stage": current_stage(), "endpoint": endpoint,
                    "base_url_host": urlsplit(self.base_url or "").hostname or "", "model": self.model}
        self._request_attempt += 1
        metadata.update(attempt=self._request_attempt, parent_request_id=self._last_request_id)
        if budget is not None:
            metadata["context_budget"] = dict(self._last_budget_metadata)
        # SDK-only options do not belong to HTTP JSON. extra_body is merged by
        # the SDK, so audit the actual application body rather than its wrapper.
        payload = {key: value for key, value in body.items() if key not in {"timeout", "extra_body"}}
        payload.update(body.get("extra_body") or {})
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        import hashlib
        metadata["request_body_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
        prior_payload = request_audit.previous_payload(
            (self.base_url or "", self.model, endpoint), serialized) or self._previous_payload
        if prior_payload is not None:
            common = 0
            for left, right in zip(serialized, prior_payload):
                if left != right:
                    break
                common += 1
            metadata["common_prefix_chars"] = common  # diagnostic, NOT provider cached tokens
        self._previous_payload = serialized
        request_id = request_audit.record_request(payload, metadata=metadata)
        self._last_request_id = request_id
        target = sdk.chat.completions.create if endpoint == "chat" else sdk.responses.create
        try:
            response = target(**body)
            # Streaming responses are consumed by their caller. Persist the
            # initiating request immediately; record completion after consumption.
            if body.get("stream") and not hasattr(response, "choices") and not hasattr(response, "output"):
                return response, request_id
            usage = self._audit_usage(response, endpoint)
            request_audit.record_result(request_id, result={"status": "received",
                "response_id": str(getattr(response, "id", "") or ""),
                "usage": usage,
                "input_estimate_error": (usage["input_tokens"] - budget.estimated_input_tokens)
                if budget is not None and usage["input_tokens"] else None})
            return response, request_id
        except Exception as exc:
            request_audit.record_result(request_id, error={"type": type(exc).__name__})
            raise

    def _chat_round(
        self,
        client,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None = None,
    ) -> "_ChatRound":
        """One chat completion, streamed when enabled (falls back if refused)."""
        kwargs: dict[str, Any] = {
            "max_completion_tokens": self._context_window(client).output_reserve_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.timeout:
            kwargs["timeout"] = self.timeout
        kwargs.update(self._chat_reasoning_kwargs())
        if allow_tools and tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        elif response_format is not None:
            kwargs["response_format"] = response_format
        messages, budget = self._budget_messages(
            messages, tools=kwargs.get("tools"), response_config={k: v for k, v in kwargs.items()
            if k not in {"tools", "timeout"}}, sdk=client)
        self._prepared_messages = messages
        if not self.stream:
            response, _ = self._sdk_request(
                client, "chat", {"model": self.model, "messages": messages, **kwargs}, budget=budget)
            return _chat_round_from_response(response)
        stream_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        try:
            stream, request_id = self._sdk_request(
                client, "chat", {"model": self.model, "messages": messages, **stream_kwargs},
                budget=budget)
        except context_budget.ContextBudgetError:
            raise
        except Exception as exc:
            if "stream" not in str(exc).lower():
                # Not a streaming problem (for example the tools payload was
                # rejected): let the caller's retry/degrade logic handle it once
                # instead of doubling the request here.
                raise
            # Streaming or stream_options is unsupported: retry without either,
            # and record why so the log can explain a missing live line.
            response, _ = self._sdk_request(
                client, "chat", {"model": self.model, "messages": messages, **kwargs}, budget=budget)
            round_ = _chat_round_from_response(response)
            round_.fallback_reason = f"streaming unavailable: {type(exc).__name__}: {exc}"
            return round_
        if hasattr(stream, "choices"):
            # Some gateways ignore stream=True and return the whole completion.
            round_ = _chat_round_from_response(stream)
            round_.fallback_reason = "streaming ignored by the endpoint"
            return round_
        try:
            round_ = _chat_round_from_stream(stream)
        except Exception as exc:
            request_audit.record_result(request_id, error={"type": type(exc).__name__})
            raise
        request_audit.record_result(request_id, result={"status": "completed", "input_tokens": round_.input_tokens,
                                                         "output_tokens": round_.output_tokens,
                                                         "cached_input_tokens": round_.cached_input_tokens})
        return round_

    def _chat(
        self,
        client,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LlmResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        round_ = self._chat_round(
            client, messages, allow_tools=False, tools=None, response_format=response_format
        )
        record_usage(current_stage(), round_.input_tokens, round_.output_tokens,
                     round_.total_tokens, round_.cached_input_tokens)
        result = LlmResult(
            round_.text,
            used_llm=True,
            input_tokens=round_.input_tokens,
            output_tokens=round_.output_tokens,
            total_tokens=round_.total_tokens,
            cached_input_tokens=round_.cached_input_tokens,
            reasoning=round_.reasoning,
            model=self.model,
            api_mode_used="chat",
            finish_reason=round_.finish_reason,
            reasoning_source=round_.reasoning_source,
            extra_keys=round_.extra_keys,
            fallback_reason=round_.fallback_reason,
        )
        return clean_llm_text(result)
