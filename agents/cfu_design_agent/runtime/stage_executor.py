"""Stage-local model execution for the CFU design agent.

The domain graph owns prompts, state transitions, edits, builds and acceptance
criteria.  This module owns only the bounded interaction used to obtain one
stage deliverable.  Keeping that boundary explicit lets implementation stages
grow into richer coding loops without allowing the model to decide whether a
CFU contract, simulation or QoR gate passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypedDict

from .agent_tools import ReadOnlyToolset
from . import tools as tools_mod
from .llm import (
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS_PER_ROUND,
    DEFAULT_MAX_TOOL_ROUNDS,
    LlmResult, _parse_tool_arguments, clean_llm_text,
    current_stage, record_usage,
)


class Toolset(Protocol):
    """Small capability interface consumed by :class:`StageExecutor`."""

    def schemas(self) -> list[dict[str, Any]]: ...

    def run(self, name: str, arguments: dict[str, Any]) -> str: ...


class _RunDirView:
    """Delegate a toolset while rendering this run's root as ``{{run-dir}}``.

    Every tool result the model sees goes through one of the two dispatch paths
    below, so virtualizing here keeps list/search/log/failure output consistent
    with the prompt instead of leaking a run id the model would then paste back.
    """

    def __init__(self, toolset: Any, run_root: str):
        self._toolset = toolset
        self._run_root = str(run_root or "").strip("/")

    def schemas(self) -> list[dict[str, Any]]:
        return self._toolset.schemas()

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        return tools_mod.virtualize_run_paths(self._toolset.run(name, arguments), self._run_root)

    def __getattr__(self, item: str) -> Any:  # rollback/commit/session and friends
        return getattr(self._toolset, item)


@dataclass(frozen=True)
class StageExecutionRequest:
    """Everything needed to obtain one stage deliverable.

    ``tools_enabled`` is decided by the control plane after applying CLI and
    provider availability. The executor does not broaden stage permissions;
    the injected toolset is a stage capability object whose file access still
    routes through the supplied ``RepoToolbox``.
    """

    stage: str
    label: str
    artifact_stage: str
    tool_calls_relpath: str
    system: str
    user: str
    fallback: str
    tools_enabled: bool = False
    response_format: dict[str, Any] | None = None
    deliverable_hint: str = ""
    require_delivery: bool = False
    observation_index: dict[str, Any] | None = None
    failure_index: dict[str, Any] | None = None
    toolset_kwargs: dict[str, Any] | None = None
    # Opt in stage by stage; the default retains the existing client-owned loop.
    native_tool_loop: bool = False
    # An independent, graph-owned final gate. No submission capability by default.
    # The gate returns True/False or ``(accepted, reason)``; the reason is fed
    # back to the model on a rejection instead of ending the run silently.
    submit_stage: Callable[[str], bool | tuple[bool, str]] | None = None
    # Experimental checkpoints audit state; never replay side-effecting tools.
    # A fresh unique thread id is mandatory and resumption is refused. This is
    # not a production-ready resumable Stage 3 implementation.
    stage_checkpointer: Any = None
    stage_thread_id: str = ""


@dataclass(frozen=True)
class StageExecutionResult:
    """Model result plus the audit references produced by one execution."""

    result: LlmResult
    raw_output: str
    tool_calls: tuple[dict[str, Any], ...] = ()


CompleteCallback = Callable[..., tuple[LlmResult, str]]
RecordCallback = Callable[..., str]
ToolArtifactCallback = Callable[[Any, str, str, list[dict[str, Any]]], None]
ToolsetFactory = Callable[..., Toolset]


class StageExecutor:
    """Run a plain completion or a bounded, policy-checked tool interaction.

    Dependencies are injected deliberately.  In particular, the graph passes
    its existing completion and artifact functions on every invocation, which
    preserves their public patch points while moving execution mechanics out of
    the domain workflow module.
    """

    def __init__(
        self,
        *,
        complete: CompleteCallback,
        record_output: RecordCallback,
        write_tool_calls: ToolArtifactCallback,
        has_deliverable: Callable[[str], bool],
        on_tool_call: Callable[[dict[str, Any]], None] | None = None,
        toolset_factory: ToolsetFactory = ReadOnlyToolset,
    ) -> None:
        self._complete = complete
        self._record_output = record_output
        self._write_tool_calls = write_tool_calls
        self._has_deliverable = has_deliverable
        self._on_tool_call = on_tool_call
        self._toolset_factory = toolset_factory

    def execute(
        self,
        state: Any,
        toolbox: Any,
        request: StageExecutionRequest,
        *,
        client: Any,
    ) -> StageExecutionResult:
        """Obtain a final deliverable without applying or validating it."""
        if not request.tools_enabled:
            result, raw_output = self._plain(state, request, client)
            return StageExecutionResult(result=result, raw_output=raw_output)

        if request.native_tool_loop and request.submit_stage is None:
            raise ValueError("native tool loop requires an independent submit_stage gate")
        toolset = _RunDirView(
            self._toolset_factory(
                toolbox,
                observation_index=request.observation_index,
                failure_index=request.failure_index,
                **(request.toolset_kwargs or {}),
            ),
            str(state.get("run_root", "") or ""),
        )
        native_trace: list[dict[str, Any]] = []
        def report_native_call(entry: dict[str, Any]) -> None:
            native_trace.append(entry)
            if self._on_tool_call is not None:
                self._on_tool_call(entry)

        try:
            if request.native_tool_loop:
                result = _native_stage_loop(
                    client, request, state, toolset, self._has_deliverable,
                    report_native_call,
                )
            else:
                result = client.complete_with_tools(
                    request.system,
                    request.user,
                    tools=toolset.schemas(),
                    tool_runner=toolset.run,
                    max_rounds=int(state.get("max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS) or DEFAULT_MAX_TOOL_ROUNDS),
                    max_total_calls=int(
                        state.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS) or 0
                    ),
                    fallback=request.fallback,
                    response_format=request.response_format,
                    deliverable_hint=request.deliverable_hint,
                    is_deliverable=(
                        self._has_deliverable if request.require_delivery else None
                    ),
                    on_tool_call=self._on_tool_call,
                )
            result = clean_llm_text(result)
        except Exception as exc:
            # A failed tool loop cannot turn into an apparently accepted stage.
            rollback = getattr(toolset, "rollback", None)
            session = getattr(toolset, "session", None)
            if callable(rollback):
                rollback()
            elif session is not None and hasattr(session, "rollback"):
                session.rollback()
            self._write_tool_calls(
                state, request.artifact_stage, request.tool_calls_relpath,
                native_trace + [{"round": 0, "name": "(tool loop failed)",
                                 "arguments": {"error": str(exc)}, "result": "",
                                 "ok": False}],
            )
            raise

        try:
            raw_output = self._record_output(
                state,
                request.stage,
                request.label,
                result,
                client=client,
                system=request.system,
                user=request.user,
            )
            tool_calls = list(getattr(result, "tool_calls", ()) or ())
            degraded = str(getattr(result, "degraded_reason", "") or "")
            if degraded:
                self._write_tool_calls(
                    state, request.artifact_stage, request.tool_calls_relpath,
                    tool_calls + [{"round": 0, "name": "(tools unavailable)",
                                   "arguments": {"error": degraded}, "result": "",
                                   "ok": False}],
                )
                raise RuntimeError(f"stage tools unavailable: {degraded}")
            if tool_calls:
                self._write_tool_calls(
                    state,
                    request.artifact_stage,
                    request.tool_calls_relpath,
                    tool_calls,
                )
        except Exception:
            rollback = getattr(toolset, "rollback", None)
            session = getattr(toolset, "session", None)
            if callable(rollback):
                rollback()
            elif session is not None and hasattr(session, "rollback"):
                session.rollback()
            raise
        return StageExecutionResult(
            result=result,
            raw_output=raw_output,
            tool_calls=tuple(tool_calls),
        )

    def _plain(
        self, state: Any, request: StageExecutionRequest, client: Any
    ) -> tuple[LlmResult, str]:
        return self._complete(
            state,
            request.stage,
            request.label,
            request.system,
            request.user,
            request.fallback,
            response_format=request.response_format,
            client=client,
        )


class _LoopState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    round_data: dict[str, Any]
    calls: list[dict[str, str]]
    transcript: list[dict[str, Any]]
    tool_rounds: int
    total_calls: int
    model_turns: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    cached_usage_reported: bool
    delivered: bool
    delivery: str
    empty_turns: int
    # Rejected submissions so far. A gate rejection is correctable once; the
    # bound keeps a model that cannot satisfy the gate from looping forever.
    rejections: int


DEFAULT_SUBMIT_RETRIES = 1


def _submit_verdict(
    request: StageExecutionRequest,
    delivery: str,
    has_deliverable: Callable[[str], bool],
) -> tuple[bool, str]:
    """Run the independent stage gate, tolerating a bare ``bool`` verdict.

    The reason is what makes a rejection actionable: without it the model only
    learns that "something" was wrong, and a stage that never edited a file
    cannot be distinguished from one that edited the wrong thing.
    """
    if request.require_delivery and not has_deliverable(delivery):
        return False, "the delivery does not contain this stage's required answer"
    verdict = request.submit_stage(delivery)
    if isinstance(verdict, tuple):
        accepted = bool(verdict[0]) if verdict else False
        reason = str(verdict[1]) if len(verdict) > 1 else ""
        return accepted, reason
    return bool(verdict), "" if verdict else "the stage gate rejected the delivery without a reason"


def _native_stage_loop(
    client: Any,
    request: StageExecutionRequest,
    state: Any,
    toolset: Toolset,
    has_deliverable: Callable[[str], bool],
    on_tool_call: Callable[[dict[str, Any]], None] | None,
) -> LlmResult:
    """One model request per LangGraph decide node; only an explicit submit can pass.

    Submission is not a repository tool and is never forwarded to ``toolset.run``.
    The injected gate owns final validation, not this execution plane or the model.
    """
    from langgraph.graph import END, START, StateGraph

    if not getattr(client, "available", False) or not callable(getattr(client, "_chat_round", None)):
        raise RuntimeError("native loop needs an available single-round chat client")
    schemas = toolset.schemas()
    permitted = {str(item.get("function", {}).get("name", "")) for item in schemas}
    if "submit_stage" in permitted:
        raise ValueError("submit_stage is reserved for the independent stage gate")
    if not schemas or not all(name for name in permitted):
        raise ValueError("native stage tool schemas must have explicit names")
    offered = schemas + [{"type": "function", "function": {
        "name": "submit_stage",
        "description": (
            "Finish this stage once the work is already applied: edits go through the "
            "editing tools, and this argument is a short summary of the change, the "
            "evidence you ran, and anything left unknown. It is not a patch channel: "
            "file content pasted here is not written anywhere, and a delivery that "
            "reports no applied change to a required target is rejected."
        ),
        "parameters": {"type": "object", "properties": {
            "delivery": {"type": "string", "description": "Summary of the applied change (not a patch)."}},
            "required": ["delivery"]}}}]
    if request.stage_checkpointer is not None and not request.stage_thread_id:
        raise ValueError("checkpointed native loop requires a unique stage_thread_id")
    if request.stage_checkpointer is not None and request.stage_checkpointer.get_tuple(
        {"configurable": {"thread_id": request.stage_thread_id}}
    ) is not None:
        raise ValueError("stage_thread_id already has a checkpoint; resume is not safe for side-effecting tools")
    round_limit = max(0, int(state.get("max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS) or DEFAULT_MAX_TOOL_ROUNDS))
    total_limit = max(0, int(state.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS) or 0))
    # Never broaden the existing per-model-round fanout limit.
    per_round = DEFAULT_MAX_TOOL_CALLS_PER_ROUND
    submit_retries = DEFAULT_SUBMIT_RETRIES
    sdk = getattr(client, "_stage_round_client", None)
    if sdk is None:
        from openai import OpenAI
        sdk = OpenAI(api_key=client.api_key, base_url=client.base_url, max_retries=0)

    def decide(s: _LoopState) -> dict[str, Any]:
        # A rejection grants a corrective round (tools plus a resubmission), so
        # the turn ceiling has to leave room for it: observed run-0003 died on
        # "model-turn hard limit reached without submission" one turn after the
        # gate asked for a change it was still allowed to make.
        if s["model_turns"] >= round_limit + 3 + 2 * submit_retries:
            raise RuntimeError("native stage failed: model-turn hard limit reached without submission")
        # A rejected submission grants one corrective round with the tools
        # available again. The gate asked for an applied change, and a turn that
        # can only resubmit cannot produce one.
        correction = s["rejections"] > 0
        allow_tools = correction or (
            s["tool_rounds"] < round_limit
            and (total_limit == 0 or s["total_calls"] < total_limit)
        )
        # Submission remains available after exploration budget is exhausted.
        tools = offered if allow_tools else offered[-1:]
        messages = list(s["messages"])
        if not allow_tools:
            messages.append({"role": "user", "content": (
                "Tool exploration budget exhausted. Submit the final stage deliverable "
                "with submit_stage now; otherwise this stage fails."
            )})
        elif not correction and s["tool_rounds"] == round_limit - 1:
            # Warn while the tools are still there. Observed 2026-09-30: a stage
            # that reads for every round and is then told only "submit now"
            # submits work it never applied, which the gate must reject.
            messages.append({"role": "user", "content": (
                "This is your last tool round. Make every remaining change with the "
                "offered tools now; after this round only submit_stage is available."
            )})
        round_ = client._chat_round(
            sdk, messages, allow_tools=True, tools=tools,
            response_format=None,
        )
        messages = list(getattr(client, "_prepared_messages", messages))
        cached = getattr(round_, "cached_input_tokens", None)
        record_usage(current_stage(), round_.input_tokens, round_.output_tokens,
                     round_.total_tokens, cached)
        calls = list(getattr(round_.message, "tool_calls", None) or [])
        return {
            "messages": messages, "round_data": {
                "text": round_.text, "reasoning": round_.reasoning,
                "finish_reason": round_.finish_reason,
                "reasoning_source": round_.reasoning_source,
                "extra_keys": tuple(round_.extra_keys),
            },
            "calls": [{"id": str(getattr(call, "id", "") or ""),
                       "name": str(getattr(getattr(call, "function", None), "name", "") or ""),
                       "arguments": str(getattr(getattr(call, "function", None), "arguments", "") or "")}
                      for call in calls],
            "model_turns": s["model_turns"] + 1,
            "input_tokens": s["input_tokens"] + round_.input_tokens,
            "output_tokens": s["output_tokens"] + round_.output_tokens,
            "total_tokens": s["total_tokens"] + round_.total_tokens,
            "cached_input_tokens": s["cached_input_tokens"] + (cached or 0),
            "cached_usage_reported": s["cached_usage_reported"] or cached is not None,
        }

    def route_decide(s: _LoopState) -> str:
        if any(call["name"] == "submit_stage" for call in s["calls"]):
            return "final_deliver"
        if s["calls"]:
            return "guarded_tools"
        return "observe"

    def execute_batch(
        s: _LoopState, calls: list[dict[str, str]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Run one model batch through the guards.

        Returns ``(transcript entries, tool messages, total calls)``. Shared by
        the ordinary tool turn and by a final turn that also asks for other
        tools, so both apply exactly the same permission and budget rules.
        """
        entries: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        used = s["total_calls"]
        round_no = s["tool_rounds"] + 1
        correction = s["rejections"] > 0
        for index, call in enumerate(calls):
            name = call["name"]
            arguments = _parse_tool_arguments(call["arguments"])
            if name not in permitted:
                observation = "denied by access policy: tool not offered for this stage"
                ok = False
            elif index >= per_round or (
                not correction
                and (s["tool_rounds"] >= round_limit or (total_limit and used >= total_limit))
            ):
                observation = "skipped: tool-call budget reached; submit the stage instead"
                ok = False
            else:
                try:
                    observation = toolset.run(name, arguments)
                    # `usage error:` is a fixable argument problem, not an
                    # answer: the toolset renders it distinctly for exactly this
                    # check, so a failed call must not be journaled as ok.
                    ok = not observation.startswith((
                        "error:", "tool error:", "denied by", "skipped:", "usage error:",
                    ))
                except Exception as exc:
                    observation = f"tool error: {type(exc).__name__}: {exc}"
                    ok = False
                used += 1
            entry = {"round": round_no, "name": name, "arguments": arguments,
                     "result": observation, "ok": ok}
            entries.append(entry)
            if on_tool_call is not None:
                on_tool_call(entry)
            tool_messages.append({"role": "tool", "tool_call_id": call["id"],
                                  "content": observation})
        return entries, tool_messages, used

    @staticmethod
    def assistant_message(s: _LoopState, calls: list[dict[str, str]]) -> dict[str, Any]:
        return {"role": "assistant", "content": s["round_data"]["text"],
                "tool_calls": [{"id": call["id"], "type": "function",
                                "function": {"name": call["name"],
                                             "arguments": call["arguments"]}}
                               for call in calls]}

    def guarded_tools(s: _LoopState) -> dict[str, Any]:
        entries, tool_messages, used = execute_batch(s, s["calls"])
        messages = list(s["messages"]) + [assistant_message(s, s["calls"])] + tool_messages
        return {"messages": messages, "transcript": list(s["transcript"]) + entries,
                "total_calls": used, "tool_rounds": s["tool_rounds"] + 1}

    def observe(s: _LoopState) -> dict[str, Any]:
        if s["calls"]:
            return {}
        # A prose answer is not submission. Give exactly one corrective re-ask;
        # never mistake the forced reminder or a final paragraph for acceptance.
        if s["empty_turns"] >= 2:
            raise RuntimeError("native stage failed: model did not call submit_stage")
        messages = list(s["messages"])
        messages.append({"role": "assistant", "content": s["round_data"]["text"]})
        messages.append({"role": "user", "content":
                         "No stage was submitted. Call an authorized tool, or call "
                         "submit_stage with the final delivery. Prose alone cannot pass."})
        return {"messages": messages, "empty_turns": s["empty_turns"] + 1}

    def final_deliver(s: _LoopState) -> dict[str, Any]:
        # A model may put the submission in the same turn as other calls
        # (`edit_hardware` + `submit_stage` means "apply this, then hand over").
        # Observed 2026-09-30: that batching ended the whole run with
        # "submit_stage must be the sole tool call", before the gate even ran.
        # Run the other calls through the same guards first, then judge the
        # submission; the verdict stays the gate's alone.
        submits = [call for call in s["calls"] if call["name"] == "submit_stage"]
        companions = [call for call in s["calls"] if call["name"] != "submit_stage"]
        if len(submits) != 1:
            raise RuntimeError(
                "native stage failed: the final batch needs exactly one submit_stage call"
            )
        submit = submits[0]
        args = _parse_tool_arguments(submit["arguments"])
        delivery = args.get("delivery")
        if not isinstance(delivery, str) or not delivery.strip():
            raise RuntimeError("native stage failed: empty submit_stage delivery")
        entries, tool_messages, used = execute_batch(s, companions)
        accepted, reason = _submit_verdict(request, delivery, has_deliverable)
        entry = {"round": s["tool_rounds"] + 1, "name": "submit_stage",
                 "arguments": {"delivery": delivery},
                 "result": ("accepted by independent stage gate" if accepted
                            else f"stage gate rejected delivery: {reason or 'no reason given'}"),
                 "ok": accepted}
        if on_tool_call is not None:
            on_tool_call(entry)
        transcript = list(s["transcript"]) + entries + [entry]
        # Every tool call of the batch needs exactly one response: the
        # companions' observations, then the gate's verdict for the submission.
        tool_messages = tool_messages + [{"role": "tool", "tool_call_id": submit["id"],
                                          "content": entry["result"]}]
        if not accepted:
            if s["rejections"] >= submit_retries:
                raise RuntimeError(
                    "native stage failed: independent submit_stage gate rejected "
                    f"delivery ({reason or 'no reason given'})"
                )
            # A rejected delivery is a correctable mistake, not a dead end: hand
            # the gate's reason back as a tool observation and let the model fix
            # the work and submit once more. The verdict itself is unchanged.
            messages = (list(s["messages"]) + [assistant_message(s, s["calls"])]
                        + tool_messages)
            messages.append({"role": "user", "content": (
                "That submission did not pass the stage gate. Fix the work itself "
                "(apply the changes with the editing tools, then re-check) and submit "
                "again; describing the change is not enough."
            )})
            return {"messages": messages, "calls": [], "transcript": transcript,
                    "total_calls": used, "rejections": s["rejections"] + 1}
        return {"delivery": delivery, "delivered": True, "transcript": transcript,
                "total_calls": used}

    graph = StateGraph(_LoopState)
    graph.add_node("llm_decide", decide)
    graph.add_node("guarded_tools", guarded_tools)
    graph.add_node("observe", observe)
    graph.add_node("final_deliver", final_deliver)
    graph.add_edge(START, "llm_decide")
    graph.add_conditional_edges("llm_decide", route_decide, {
        "guarded_tools": "guarded_tools", "observe": "observe", "final_deliver": "final_deliver",
    })
    graph.add_edge("guarded_tools", "observe")
    graph.add_edge("observe", "llm_decide")
    graph.add_conditional_edges("final_deliver", lambda s: "end" if s["delivered"] else "retry",
                                {"end": END, "retry": "llm_decide"})
    result = graph.compile(checkpointer=request.stage_checkpointer).invoke({
        "messages": [{"role": "system", "content": request.system},
                     {"role": "user", "content": request.user}],
        "round_data": {}, "calls": [], "transcript": [], "tool_rounds": 0,
        "total_calls": 0, "model_turns": 0, "input_tokens": 0,
        "output_tokens": 0, "total_tokens": 0, "cached_input_tokens": 0,
        "cached_usage_reported": False, "delivered": False, "delivery": "",
        "empty_turns": 0, "rejections": 0,
    }, config={"recursion_limit": max(25, (round_limit + 3) * 4),
               "configurable": {"thread_id": request.stage_thread_id}}
    )
    if not result["delivered"]:
        raise RuntimeError("native stage failed: no accepted submit_stage delivery")
    last = result["round_data"]
    return clean_llm_text(LlmResult(
        result["delivery"], used_llm=True,
        input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
        total_tokens=result["total_tokens"],
        cached_input_tokens=(result["cached_input_tokens"] if result["cached_usage_reported"] else None),
        reasoning=last["reasoning"],
        model=client.model, api_mode_used="chat+langgraph-tools",
        finish_reason=last["finish_reason"], reasoning_source=last["reasoning_source"],
        extra_keys=last["extra_keys"], tool_calls=tuple(result["transcript"]),
        tool_rounds=result["tool_rounds"],
    ))
