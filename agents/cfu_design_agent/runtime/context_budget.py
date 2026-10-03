"""Conservative, offline context-window planning for chat/tool requests.

The caller retains ownership of the live transcript.  In particular compaction
returns new message dictionaries and never mutates the input or calls a model.
All estimates are *planning bounds*, not provider tokenizer measurements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

UNKNOWN_WINDOW_TOKENS = 32_768
DEFAULT_OUTPUT_RESERVE = 4_096
# JSON framing alone is not all a provider counts (message delimiters, roles,
# tool-call framing). Include a fixed per-message/request allowance as well.
_REQUEST_OVERHEAD = 256
_MESSAGE_OVERHEAD = 32
_WINDOW_FIELDS = (
    "context_window", "context_window_tokens", "context_length", "max_context_tokens",
    "max_context_length", "max_model_len", "max_position_embeddings",
)
_RESERVE_FIELDS = ("output_reserve_tokens", "context_output_reserve_tokens")
_RESPONSE_LIMIT_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")
_FAILURE = re.compile(r"\b(?:failure|failed|error|exception|traceback|rejected|diagnosis)\b", re.I)
_EDIT = re.compile(r"\b(?:edit(?:ed|ing)?|patch(?:ed)?|replace(?:d|ment)?|diff|write(?:n)?|modif(?:y|ied))\b", re.I)


class ContextBudgetError(ValueError):
    """Invalid metadata/transcript, or protected content cannot fit safely."""


@dataclass(frozen=True)
class WindowSpec:
    window_tokens: int
    output_reserve_tokens: int
    source: str
    known: bool

    def __post_init__(self) -> None:
        _positive(self.window_tokens, "window_tokens")
        _positive(self.output_reserve_tokens, "output_reserve_tokens")
        if self.output_reserve_tokens >= self.window_tokens:
            raise ContextBudgetError("output_reserve_tokens must be smaller than window_tokens")


@dataclass(frozen=True)
class BudgetStatus:
    estimated_input_tokens: int
    reserved_output_tokens: int
    total_tokens: int
    window_tokens: int
    utilization: float
    level: str  # ok (<80%), warning (>=80%), compact (>=90%), unfit (>=100%)


@dataclass(frozen=True)
class CompactionResult:
    messages: list[dict[str, Any]]
    before: BudgetStatus
    after: BudgetStatus
    removed_batches: int


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContextBudgetError(f"{name} must be a positive integer")
    return value


def _optional_number(data: Mapping[str, Any], keys: Sequence[str], owner: str) -> int | None:
    values = [(key, _positive(data[key], f"{owner}.{key}")) for key in keys if data.get(key) is not None]
    if len({value for _, value in values}) > 1:
        raise ContextBudgetError(f"conflicting {owner} fields: {', '.join(key for key, _ in values)}")
    return values[0][1] if values else None


def resolve_window(
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> WindowSpec:
    """Resolve optional config overrides and ``GET /models/{model}`` metadata.

    Accept flat metadata or its ``model`` object. A valid explicit config takes
    precedence and does not inspect irrelevant (possibly malformed) metadata.
    Missing fields use a conservative *soft* 32768-token window, never an
    inferred provider capability. No endpoint is contacted by this helper.
    """
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise ContextBudgetError("config must be an object")
    configured = _optional_number(config, _WINDOW_FIELDS, "config")
    reserve = _optional_number(config, _RESERVE_FIELDS, "config") or DEFAULT_OUTPUT_RESERVE
    if configured is not None:
        return WindowSpec(configured, reserve, "config", True)
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, Mapping):
        raise ContextBudgetError("model metadata must be an object")
    nested = metadata.get("model")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ContextBudgetError("metadata.model must be an object")
        reported_outer = _optional_number(metadata, _WINDOW_FIELDS, "metadata")
        reported_nested = _optional_number(nested, _WINDOW_FIELDS, "metadata.model")
        if reported_outer is not None and reported_nested is not None and reported_outer != reported_nested:
            raise ContextBudgetError("conflicting metadata and metadata.model window fields")
        reported = reported_nested or reported_outer
    else:
        reported = _optional_number(metadata, _WINDOW_FIELDS, "metadata")
    return WindowSpec(reported or UNKNOWN_WINDOW_TOKENS, reserve,
                      "metadata" if reported else "unknown-soft", bool(reported))


def _serial_size(value: Any) -> int:
    try:
        # ensure_ascii=True counts escaped non-ASCII at least as generously as
        # its UTF-8 representation; separators match normal compact wire JSON.
        return len(json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextBudgetError(f"request cannot be serialized as JSON: {exc}") from exc


def evaluate_budget(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    response_config: Mapping[str, Any] | None = None,
    window: WindowSpec | None = None,
) -> BudgetStatus:
    """Bound request tokens by serialized UTF-8 bytes plus framing and output.

    ``response_config`` contains request fields such as response_format,
    reasoning, max_tokens; limits there may *increase* the output reservation,
    never silently lower the configured reserve. Include the exact tools offered
    for this turn (including submit_stage if offered).
    """
    window = window or resolve_window()
    if not isinstance(messages, (list, tuple)) or any(not isinstance(m, Mapping) for m in messages):
        raise ContextBudgetError("messages must be a sequence of objects")
    if tools is not None and (not isinstance(tools, (list, tuple)) or any(not isinstance(t, Mapping) for t in tools)):
        raise ContextBudgetError("tools must be a sequence of objects")
    if response_config is not None and not isinstance(response_config, Mapping):
        raise ContextBudgetError("response_config must be an object")
    response_config = response_config or {}
    requested = _optional_number(response_config, _RESPONSE_LIMIT_FIELDS, "response_config")
    reserved = max(window.output_reserve_tokens, requested or 0)
    input_tokens = (_serial_size({"messages": messages, "tools": tools or [],
                                  "response_config": response_config})
                    + _REQUEST_OVERHEAD + len(messages) * _MESSAGE_OVERHEAD)
    total = input_tokens + reserved
    level = ("unfit" if total * 100 >= window.window_tokens * 100 else
             "compact" if total * 100 >= window.window_tokens * 90 else
             "warning" if total * 100 >= window.window_tokens * 80 else "ok")
    return BudgetStatus(input_tokens, reserved, total, window.window_tokens,
                        total / window.window_tokens, level)


def _tool_batches(messages: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Find only completed assistant+all-results batches; reject orphan results."""
    batches: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.get("role") == "tool":
            raise ContextBudgetError(f"orphan tool result at message {i}")
        calls = message.get("tool_calls")
        if not calls:
            i += 1
            continue
        if message.get("role") != "assistant" or not isinstance(calls, list):
            raise ContextBudgetError(f"invalid assistant tool batch at message {i}")
        ids = []
        for call in calls:
            if not isinstance(call, Mapping) or not isinstance(call.get("id"), str) or not call["id"]:
                raise ContextBudgetError(f"invalid tool call id at message {i}")
            ids.append(call["id"])
        if len(set(ids)) != len(ids):
            raise ContextBudgetError(f"duplicate tool call ids at message {i}")
        j = i + 1
        seen: set[str] = set()
        while j < len(messages) and messages[j].get("role") == "tool":
            call_id = messages[j].get("tool_call_id")
            if call_id not in ids or call_id in seen:
                raise ContextBudgetError(f"unmatched or duplicate tool result at message {j}")
            seen.add(call_id)
            j += 1
        if seen != set(ids):
            # An unfinished call at the tail is live context, not a candidate.
            if j != len(messages):
                raise ContextBudgetError(f"incomplete tool batch at message {i}")
        else:
            batches.append((i, j))
        i = j
    return batches


def _batch_summary(messages: Sequence[Mapping[str, Any]], start: int, end: int) -> str:
    """Bounded provenance-only memo, never a fresh-file or passing-gate claim."""
    calls = messages[start].get("tool_calls") or []
    results = {m.get("tool_call_id"): m.get("content", "") for m in messages[start + 1:end]}
    pieces: list[str] = []
    for call in calls[:4]:
        function = call.get("function") or {}
        name = str(function.get("name") or "tool")[:60]
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (TypeError, ValueError):
            args = {}
        # Only named selectors may leave the batch; never echo old SEARCH or
        # replacement code as though it were the live target after an edit.
        selectors = []
        if isinstance(args, Mapping):
            for key in ("path", "id", "start_line", "end_line"):
                if key in args:
                    value = str(args[key]).replace("\n", " ")[:160]
                    selectors.append(f"{key}={value}")
        result = str(results.get(call.get("id"), ""))
        # For checks retain their one-line historical outcome and log pointer;
        # for reads retain selector/extent but no arbitrary source instructions.
        evidence = []
        if name != "read_file":
            for row in result.splitlines():
                line = row.strip()
                if line and (not evidence or "full log:" in line.lower() or "failure:" in line.lower()):
                    evidence.append(line[:180])
                if len(evidence) >= 3:
                    break
        detail = "; ".join(evidence) if evidence else f"{len(result)} historical result characters omitted"
        pieces.append(f"{name}({', '.join(selectors)}): {detail}")
    if len(calls) > 4:
        pieces.append(f"and {len(calls) - 4} additional tool results omitted")
    return " | ".join(pieces)[:650]


def _with_compaction_memo(transcript, removed, summaries):
    candidate = [dict(m) for i, m in enumerate(transcript) if i not in removed]
    memo = {
        "role": "assistant",
        "content": (
            "[Context compaction: completed historical tool batches were replaced by this memo. "
            "These are past observations, not current file bytes or authoritative gate results; "
            "re-read exact current content before editing.]\n"
            + "\n".join(summaries[-6:])
            + (f"\n{len(summaries) - 6} earlier batches also omitted." if len(summaries) > 6 else "")
        ),
    }
    # Place the memo at the oldest removed batch, preserving the relative order
    # of every retained user/system and full tool exchange.
    insert_at = sum(i < min(removed) and i not in removed for i in range(len(transcript)))
    candidate.insert(insert_at, memo)
    return candidate


def compact_completed_tool_turns(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    response_config: Mapping[str, Any] | None = None,
    window: WindowSpec | None = None,
    protected_indices: Sequence[int] = (),
) -> CompactionResult:
    """Replace oldest *whole* completed batches with honest memos below 90%.

    Protect every system/developer/user turn, the most recent completed tool
    batch, any explicitly protected indices, and batches carrying the latest
    failure or latest edit evidence. A user message does NOT blanket-protect all
    later batches: in a normal tool loop the initial prompt is the latest user,
    and doing so makes compaction impossible. Original messages are never
    modified. Raises rather than truncating any protected turn when a fitting
    request cannot be produced.
    """
    window = window or resolve_window()
    if not isinstance(messages, (list, tuple)) or any(not isinstance(m, Mapping) for m in messages):
        raise ContextBudgetError("messages must be a sequence of objects")
    transcript = list(messages)
    batches = _tool_batches(transcript)
    if not isinstance(protected_indices, (list, tuple)):
        raise ContextBudgetError("protected_indices must be a sequence of positions")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(transcript) for i in protected_indices):
        raise ContextBudgetError("protected_indices must refer to existing message positions")
    protected = set(protected_indices)
    protected.update(i for i, m in enumerate(transcript)
                     if m.get("role") in {"system", "developer", "user"})
    if batches:
        protected.update(range(*batches[-1]))
    for pattern in (_FAILURE, _EDIT):
        matching_batches = [batch for batch in batches
                            if pattern.search(_json_text(transcript[batch[0]:batch[1]]))]
        if matching_batches:
            protected.update(range(*matching_batches[-1]))
        matching = [i for i, message in enumerate(transcript)
                    if pattern.search(_json_text(message))]
        if matching:
            protected.add(matching[-1])
    # An explicitly protected result or current failure/last edit protects its
    # entire assistant batch, never just one half of a tool-call exchange.
    for start, end in batches:
        if any(i in protected for i in range(start, end)):
            protected.update(range(start, end))
    before = evaluate_budget(transcript, tools=tools, response_config=response_config, window=window)
    if before.level in {"ok", "warning"}:
        return CompactionResult([dict(m) for m in transcript], before, before, 0)
    removed: set[int] = set()
    after = before
    count = 0
    summaries: list[str] = []
    for start, end in batches:
        if any(i in protected for i in range(start, end)):
            continue
        removed.update(range(start, end))
        count += 1
        summaries.append(_batch_summary(transcript, start, end))
        candidate = _with_compaction_memo(transcript, removed, summaries)
        after = evaluate_budget(candidate, tools=tools, response_config=response_config, window=window)
        if after.level in {"ok", "warning"}:
            return CompactionResult(candidate, before, after, count)
    raise ContextBudgetError(
        f"context cannot fit safely below 90%: {after.total_tokens} / "
        f"{window.window_tokens} estimated tokens after {count} completed batches removed"
    )


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ContextBudgetError(f"cannot inspect tool batch: {exc}") from exc
