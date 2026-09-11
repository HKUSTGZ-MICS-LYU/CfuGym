"""LangGraph workflow for CFU design iterations.

A pure-Python sequential runner is provided as ``run_agent_python`` so the
framework stays usable and testable even when the ``langgraph`` dependency is
not installed. ``build_graph`` and ``run_agent`` remain available for the
LangGraph-backed path.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Callable

from .agent_tools import ReadOnlyToolset
from .artifacts import RunArtifacts
from .benchmarks import (
    prepare_c_project_profile as prepare_c_project_profile_workspace,
    prepare_generated_benchmark,
    profile_commands,
    profile_result_from_commands,
)
from .embench import (
    benchmark_source_files,
    build_command as embench_build_command,
    cfu_source_targets,
    config_sources,
    config_workspace as embench_config_workspace,
    elf_path as embench_elf_path,
    hotspot_from_profile,
    parse_profile as parse_embench_profile,
    prepare_embench_benchmark,
    workspace_rel as embench_workspace,
)
from .embench_instrument import instrument_sources
from .isa import (
    IsaLintError,
    intrinsic_symbols,
    render_cfu_intrinsics,
    render_isa_markdown,
    validate_isa_text,
)
from .llm import (
    LlmResult,
    OpenAiTextClient,
    clean_llm_text,
    reset_usage_log,
    set_stage,
    usage_totals,
)
from .llm_config import effective_config, LlmConfigError
from .policy import PolicyError, load_policy, normalize_repo_path
from .progress import StageProgress
from .prompts import (
    SYSTEM_PROMPT,
    contract_prompt,
    hardware_prompt,
    software_prompt,
    workload_prompt,
)
from .spec import normalize_workload_spec, validate_workload_spec
from .state import CfuDesignState, append_history
from .tools import (
    DEFAULT_CONTEXT_FILES,
    DESIGN_SOURCE_FILES,
    DESIGN_WORKSPACE_SUBDIR,
    SOC_WORKSPACE_SUBDIR,
    RepoToolbox,
    design_workspace_paths,
    extract_edit_blocks,
    extract_file_blocks,
    extract_patch,
    filter_output,
    infer_command_id,
)

try:  # pragma: no cover - optional dependency
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, RetryPolicy
except ImportError:  # pragma: no cover
    InMemorySaver = None  # type: ignore
    StateGraph = None  # type: ignore
    START = None  # type: ignore
    END = None  # type: ignore
    Command = None  # type: ignore
    RetryPolicy = None  # type: ignore
    _LANGRAPH_AVAILABLE = False
else:
    _LANGRAPH_AVAILABLE = True

# The node wrapper is a no-op unless a progress reporter is enabled. Tests and
# programmatic callers leave it disabled to avoid noisy stdout.
_PROGRESS: StageProgress | None = None


def enable_progress(stages: list[str] | None = None) -> None:
    global _PROGRESS
    reset_usage_log()
    _PROGRESS = StageProgress(list(stages) if stages else list(STAGE_ORDER))


def disable_progress() -> None:
    global _PROGRESS
    _PROGRESS = None


def progress() -> StageProgress | None:
    return _PROGRESS


# Map a node's return value to the gate key that reports its success. Nodes
# whose gate is not present in the map are considered OK unless they add errors.
_NODE_TO_GATE: dict[str, str] = {
    "load_context": "load_context",
    "ingest_input": "workload_spec",
    "prepare_c_project_profile": "prepare_benchmark",
    "create_benchmark_project": "prepare_benchmark",
    "prepare_embench_benchmark": "prepare_benchmark",
    "instrument_and_profile": "profile",
    "profile_embench": "profile",
    "extract_hotspot": "hotspot",
    "design_contract": "isa",
    "implement_hw": "implementation_hw",
    "generate_soc": "hw_build",
    "implement_sw": "implementation_sw",
    "build_sw": "sw_build",
    "simulate": "sim",
    "estimate_cost": "cost",
    "review_and_route": "review",
    "final_report": "final_report",
}


def _node_update(update: Any) -> Any:
    """Return the dict payload of a node update.

    LangGraph nodes may return a plain dict or a ``Command`` whose ``update``
    field holds the dict payload. ``dict`` instances have an ``update`` method,
    so the attribute-based check must come second.
    """
    if isinstance(update, dict):
        return update
    candidate = getattr(update, "update", None)
    if isinstance(candidate, dict):
        return candidate
    return None


def _node_ok(update: Any, name: str) -> bool:
    effective = _node_update(update)
    if not isinstance(effective, dict):
        return True
    gate = effective.get("gates", {}).get(_NODE_TO_GATE.get(name, name))
    if isinstance(gate, dict) and "ok" in gate:
        return bool(gate.get("ok"))
    return True


def _wrap_node(fn):
    @functools.wraps(fn)
    def wrapper(state):
        name = fn.__name__
        set_stage(name)
        if _PROGRESS is None:
            return fn(state)
        prior_errors = len((state.get("errors") or []))
        prior_total = _PROGRESS._running_total()
        start = time.monotonic()
        _PROGRESS.begin(name)
        try:
            update = fn(state)
        except Exception as exc:
            _PROGRESS.end(
                name,
                ok=False,
                elapsed=time.monotonic() - start,
                tokens=_PROGRESS._running_total() - prior_total,
            )
            raise
        effective = _node_update(update)
        added_errors = 0
        if isinstance(effective, dict):
            added_errors = len(effective.get("errors") or []) - prior_errors
        _PROGRESS.end(
            name,
            ok=_node_ok(update, name) and added_errors <= 0,
            elapsed=time.monotonic() - start,
            tokens=_PROGRESS._running_total() - prior_total,
        )
        return update

    return wrapper


def _wrapped(fn):
    return _wrap_node(fn) if _PROGRESS is not None else fn


def _identity_map(*names: str) -> dict[str, str]:
    """Conditional-edge path map whose router returns node names directly."""
    return {name: name for name in names}


def build_graph(*, checkpointer: Any = None, debug: bool = False):  # type: ignore[misc]
    if not _LANGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed; use run_agent_python()")
    graph = StateGraph(CfuDesignState)
    graph.add_node("load_context", _wrap_node(load_context), retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node("ingest_input", _wrap_node(ingest_input))
    graph.add_node("route_input", _wrap_node(route_input))
    graph.add_node("prepare_c_project_profile", _wrap_node(prepare_c_project_profile_node))
    graph.add_node("create_benchmark_project", _wrap_node(create_benchmark_project_node))
    graph.add_node("prepare_embench_benchmark", _wrap_node(prepare_embench_benchmark_node))
    graph.add_node("profile_embench", _wrap_node(profile_embench))
    graph.add_node("instrument_and_profile", _wrap_node(instrument_and_profile))
    graph.add_node("extract_hotspot", _wrap_node(extract_hotspot))
    graph.add_node("analyze_workload", _wrap_node(analyze_workload))
    graph.add_node("design_contract", _wrap_node(design_contract))
    graph.add_node("implement_hw", _wrap_node(implement_hw))
    graph.add_node("generate_soc", _wrap_node(generate_soc))
    graph.add_node("implement_sw", _wrap_node(implement_sw))
    graph.add_node("build_sw", _wrap_node(build_sw))
    graph.add_node("simulate", _wrap_node(simulate))
    graph.add_node("estimate_cost", _wrap_node(estimate_cost))
    graph.add_node("review_and_route", _wrap_node(review_and_route))
    graph.add_node("final_report", _wrap_node(final_report))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "ingest_input")
    graph.add_edge("ingest_input", "route_input")
    graph.add_conditional_edges(
        "route_input",
        _route_input_target,
        _identity_map("prepare_c_project_profile", "create_benchmark_project", "prepare_embench_benchmark"),
    )
    graph.add_conditional_edges(
        "prepare_c_project_profile",
        _route_after_prepare,
        _identity_map("instrument_and_profile", "review_and_route"),
    )
    graph.add_conditional_edges(
        "create_benchmark_project",
        _route_after_prepare,
        _identity_map("instrument_and_profile", "review_and_route"),
    )
    graph.add_conditional_edges(
        "prepare_embench_benchmark",
        _route_after_prepare,
        _identity_map("profile_embench", "review_and_route"),
    )
    graph.add_edge("profile_embench", "extract_hotspot")
    graph.add_edge("instrument_and_profile", "extract_hotspot")
    graph.add_edge("extract_hotspot", "analyze_workload")
    graph.add_edge("analyze_workload", "design_contract")
    # Validation ladder: each artifact is checked by the tool that owns it and
    # either repairs its own rung or stops the run immediately.
    graph.add_conditional_edges(
        "design_contract",
        _route_after_isa,
        _identity_map("design_contract", "implement_hw", "review_and_route"),
    )
    graph.add_edge("implement_hw", "generate_soc")
    graph.add_conditional_edges(
        "generate_soc",
        _route_after_hw_build,
        _identity_map("implement_sw", "implement_hw", "review_and_route"),
    )
    graph.add_edge("implement_sw", "build_sw")
    graph.add_conditional_edges(
        "build_sw",
        _route_after_sw_build,
        _identity_map("simulate", "implement_sw", "review_and_route"),
    )
    graph.add_conditional_edges(
        "simulate",
        _route_after_sim,
        _identity_map("estimate_cost", "implement_sw", "implement_hw", "review_and_route"),
    )
    graph.add_conditional_edges(
        "estimate_cost",
        _route_after_cost,
        _identity_map("review_and_route", "implement_hw"),
    )
    graph.add_edge("review_and_route", "final_report")
    graph.add_edge("final_report", END)
    return graph.compile(checkpointer=checkpointer, debug=debug, name="cfu_design_agent")


# State keys annotated with operator.add in CfuDesignState. Nodes return only
# their own increment for these keys; LangGraph appends automatically, and the
# pure-Python runner merges them the same way in _merge_state_update.
APPEND_STATE_KEYS = (
    "errors",
    "iteration_history",
    "validation_results",
    "cost_results",
    "graph_events",
    "llm_raw_outputs",
    "tool_calls",
)


def _merge_state_update(state: CfuDesignState, update: Any) -> None:
    """Apply a node's partial update with the same semantics as LangGraph."""
    if not isinstance(update, dict):
        return
    for key, value in update.items():
        if key in APPEND_STATE_KEYS and isinstance(value, list):
            state[key] = [*(state.get(key) or []), *value]
        else:
            state[key] = value


def _finalize(state: CfuDesignState) -> CfuDesignState:
    """Attach run-wide token usage and (when enabled) the progress summary."""
    state["token_usage"] = usage_totals()
    if _PROGRESS is not None:
        state["progress_summary"] = _PROGRESS.summary()
    return state


def run_agent(
    initial_state: CfuDesignState,
    *,
    thread_id: str = "",
    checkpoint: bool = False,
    stream: bool = False,
) -> CfuDesignState:
    reset_usage_log()
    if not _LANGRAPH_AVAILABLE:
        return run_agent_python(initial_state)
    app = build_graph(checkpointer=InMemorySaver() if checkpoint else None)
    config = graph_config(thread_id) if checkpoint or thread_id else None
    if not stream:
        return _finalize(app.invoke(initial_state, config=config))
    final_state: CfuDesignState = {}
    snapshots = 0
    for snapshot in app.stream(initial_state, config=config, stream_mode="values"):
        if isinstance(snapshot, dict):
            snapshots += 1
            final_state = snapshot
    if snapshots:
        final_state = dict(final_state)
        final_state["graph_events"] = [f"streamed {snapshots} state snapshots"]
    return _finalize(final_state)


STAGE_ORDER = (
    "load_context",
    "ingest_input",
    "route_input",
    "prepare_benchmark",
    "profile",
    "hotspot",
    "analysis",
    "isa",
    "implementation",
    "validation",
    "cost",
    "review",
    "final_report",
)


def run_agent_python(initial_state: CfuDesignState) -> CfuDesignState:
    reset_usage_log()
    state = dict(initial_state)
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = RunArtifacts.create(
        workdir,
        policy,
        out_dir=state.get("out_dir", "agents/generated"),
        run_id=state.get("run_id", ""),
    )
    # Seed run identity so every node reuses this run's artifact directory.
    state["run_id"] = artifacts.run_id
    state["run_root"] = artifacts.root_rel

    # Node returns are partial updates; merge them like LangGraph state does,
    # including operator.add semantics for the append-annotated keys. The walk
    # follows the same static edges and conditional routers as build_graph().
    nodes = python_nodes()
    node: str | None = "load_context"
    guard = 0
    while node is not None:
        guard += 1
        if guard > 200:  # pragma: no cover - defensive: routers must terminate
            raise RuntimeError(f"python runner did not terminate at node {node!r}")
        fn = nodes[node]
        _merge_state_update(state, _wrapped(fn)(state))
        router = PY_ROUTERS.get(node)
        node = router(state) if router is not None else PY_SUCCESSORS.get(node)

    state["run_id"] = artifacts.run_id
    state["run_root"] = artifacts.root_rel
    return _finalize(state)


def python_nodes() -> dict[str, Any]:
    """Node name -> callable, mirroring the LangGraph node table."""
    return {
        "load_context": load_context,
        "ingest_input": ingest_input,
        "route_input": route_input,
        "prepare_c_project_profile": prepare_c_project_profile_node,
        "create_benchmark_project": create_benchmark_project_node,
        "prepare_embench_benchmark": prepare_embench_benchmark_node,
        "profile_embench": profile_embench,
        "instrument_and_profile": instrument_and_profile,
        "extract_hotspot": extract_hotspot,
        "analyze_workload": analyze_workload,
        "design_contract": design_contract,
        "implement_hw": implement_hw,
        "generate_soc": generate_soc,
        "implement_sw": implement_sw,
        "build_sw": build_sw,
        "simulate": simulate,
        "estimate_cost": estimate_cost,
        "review_and_route": review_and_route,
        "final_report": final_report,
    }


def export_graph_visualization(out_dir: str | Path) -> dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not _LANGRAPH_AVAILABLE:
        ascii_text = fallback_ascii_graph()
        mermaid_text = ""
        root.joinpath("cfu_design_agent.mmd").write_text(mermaid_text)
        root.joinpath("cfu_design_agent.txt").write_text(ascii_text)
        return {
            "mermaid": str(root.joinpath("cfu_design_agent.mmd").resolve()),
            "ascii": str(root.joinpath("cfu_design_agent.txt").resolve()),
        }
    app = build_graph()
    drawable = app.get_graph()
    mermaid_path = root / "cfu_design_agent.mmd"
    ascii_path = root / "cfu_design_agent.txt"
    mermaid_path.write_text(drawable.draw_mermaid())
    try:
        ascii_text = drawable.draw_ascii()
    except ImportError:
        ascii_text = fallback_ascii_graph()
    ascii_path.write_text(ascii_text)
    return {
        "mermaid": str(mermaid_path.resolve()),
        "ascii": str(ascii_path.resolve()),
    }


def fallback_ascii_graph() -> str:
    return "\n".join(
        [
            "START",
            "  -> load_context",
            "  -> ingest_input",
            "  -> route_input",
            "       | goto prepare_c_project_profile/create_benchmark_project/prepare_embench_benchmark",
            "  -> instrument_and_profile",
            "  -> extract_hotspot",
            "  -> analyze_workload",
            "  -> design_contract (ISA spec)",
            "  -> implement_hw (AgentCfu.scala/Fiber)",
            "  -> generate_soc (HW gate)",
            "       | fail -> implement_hw / stop",
            "  -> implement_sw (kernel + intrinsics)",
            "  -> build_sw (compile gate)",
            "       | fail -> implement_sw / stop",
            "  -> simulate (sim gate)",
            "       | fail -> implement_sw, then implement_hw / stop",
            "  -> estimate_cost (yosys)",
            "  -> review_and_route",
            "  -> final_report",
            "  -> END",
            "",
        ]
    )


def graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": (thread_id or "cfu-design-agent")[:255]}}


def _policy(state: CfuDesignState):
    return load_policy(
        Path(state.get("workdir", ".")).resolve(),
        visibility=str(state.get("visibility", "") or ""),
    )


def _artifacts(state: CfuDesignState, policy=None) -> RunArtifacts:
    return RunArtifacts.create(
        Path(state.get("workdir", ".")).resolve(),
        policy or _policy(state),
        out_dir=state.get("out_dir", "agents/generated"),
        run_id=state.get("run_id", ""),
    )


def _llm(state: CfuDesignState) -> OpenAiTextClient:
    workdir = Path(state.get("workdir", ".")).resolve()
    config_name = state.get("llm_config") or os.environ.get("CFU_AGENT_LLM_CONFIG") or "default"
    try:
        config = effective_config(workdir, config_name)
    except LlmConfigError:
        # Fall back to the environment-based client so a bad config name
        # does not silently disable the LLM path.
        config = effective_config(workdir, "default")
    return OpenAiTextClient(state.get("model"), config=config)


def _llm_request_info(client) -> dict[str, Any]:
    return {
        "config": getattr(client, "config_name", "") or "",
        "model": getattr(client, "model", "") or "",
        "base_url": getattr(client, "base_url", "") or "",
        "api_mode": getattr(client, "api_mode", "") or "",
        "thinking": bool(getattr(client, "thinking", False)),
        "reasoning_effort": getattr(client, "reasoning_effort", None),
        "thinking_body": getattr(client, "thinking_body", None),
        "available": bool(getattr(client, "available", False)),
    }


def _record_llm_output(
    state: CfuDesignState,
    stage: str,
    label: str,
    result: LlmResult,
    *,
    client=None,
) -> str:
    """Persist the exact returned text without exposing the prompt/context."""
    stage_info = {
        "ingest_input": ("ingest_input", "02_workload"),
        "analyze_workload": ("analysis", "07_analysis"),
        "design_contract": ("isa", "08_isa"),
        "implement_hw": ("implementation", "09_implementation"),
        "implement_sw": ("implementation", "09_implementation"),
    }
    artifact_stage, stage_dir = stage_info.get(stage, (None, None))
    if artifact_stage is None or stage_dir is None:
        raise ValueError(f"LLM raw output stage is not configured: {stage}")

    artifacts = _artifacts_for_stage(state, artifact_stage)
    prefix = f"{stage_dir}/llm_raw/{label}_"
    # Count only the answer files so a matching reasoning sidecar never shifts
    # the numbering of later attempts.
    existing = [
        ref for ref in artifacts.manifest.get("artifacts", [])
        if ref.get("kind") == "llm_raw_output"
        and str(ref.get("path", "")).startswith(f"{artifacts.root_rel}/{prefix}")
    ]
    index = len(existing)
    relative = f"{prefix}{index:03d}.txt"
    reasoning = str(getattr(result, "reasoning", "") or "").strip()
    metadata = {
        "llm_stage": stage,
        "llm_label": label,
        "used_llm": bool(getattr(result, "used_llm", False)),
        "input_tokens": int(getattr(result, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(result, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(result, "total_tokens", 0) or 0),
        # Observability: prove whether the provider returned reasoning at all.
        "model": str(getattr(result, "model", "") or ""),
        "api_mode_used": str(getattr(result, "api_mode_used", "") or ""),
        "finish_reason": str(getattr(result, "finish_reason", "") or ""),
        "reasoning_source": str(getattr(result, "reasoning_source", "") or ""),
        "reasoning_chars": len(reasoning),
        "response_extra_keys": list(getattr(result, "extra_keys", ()) or ()),
    }
    info_client = client
    if info_client is None:
        try:
            info_client = _llm(state)
        except Exception:  # pragma: no cover - _llm already falls back to default
            info_client = None
    if info_client is not None:
        # Record once, run-level, so a run says what it actually asked for.
        artifacts.set_manifest_metadata(artifact_stage, "llm_request", _llm_request_info(info_client))
    ref = artifacts.write_text(
        artifact_stage,
        relative,
        result.text,
        kind="llm_raw_output",
        metadata=metadata,
    )
    if reasoning:
        # Thinking is kept for audit but never parsed as the answer.
        artifacts.write_text(
            artifact_stage,
            f"{prefix}{index:03d}.reasoning.txt",
            reasoning,
            kind="llm_reasoning_output",
            metadata={**metadata, "kind": "reasoning"},
        )
    return str(ref["path"])


def _complete_llm(
    state: CfuDesignState,
    stage: str,
    label: str,
    system: str,
    user: str,
    fallback: str,
    response_format: dict[str, Any] | None = None,
    client=None,
) -> tuple[LlmResult, str]:
    client = client or _llm(state)
    result = clean_llm_text(
        client.complete(
            system,
            user,
            fallback=fallback,
            **({"response_format": response_format} if response_format is not None else {}),
        )
    )
    return result, _record_llm_output(state, stage, label, result, client=client)


def load_context(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = _artifacts(state, policy)
    toolbox = _toolbox_for_stage(state, "load_context")

    context_sources = ["agents/specs/cfu_isa_template.yaml"]
    if state.get("benchmark_name"):
        # The workload source is what the LLM analyzes and patches.
        context_sources.extend(benchmark_source_files(workdir, str(state["benchmark_name"])))
    if state.get("kernel_file"):
        context_sources.append(state["kernel_file"])
    if state.get("workload_spec_path"):
        context_sources.append(state["workload_spec_path"])
    files: dict[str, str] = {}
    denied: list[str] = []
    # A context file the selected visibility tier hides is expected, not a policy
    # misconfiguration, so it must not fail the load_context gate.
    withheld: list[str] = []
    seen: set[str] = set()
    for path in [*DEFAULT_CONTEXT_FILES, *context_sources]:
        if path in seen:
            continue
        seen.add(path)
        blocked = policy.visibility_block(path)
        if blocked:
            withheld.append(f"{path}: requires visibility level {blocked!r}")
            continue
        artifact_path = f"01_context/excerpt/{path.replace('/', '__')}.txt"
        try:
            text = toolbox.read_text(path, limit=20000)
        except PolicyError as exc:
            denied.append(f"{path}: {exc}")
            continue
        if not text:
            continue
        files[path] = text
        artifacts.write_text("load_context", artifact_path, text, kind="context_excerpt", inputs=[path])

    # Seed the isolated hardware design workspace from the tracked scaffolds.
    # The agent may only write these two files; the repo copies stay read-only.
    design_toolbox = RepoToolbox(
        workdir,
        dry_run=state.get("dry_run", False),
        policy=policy,
        stage="implementation",
        run_root=artifacts.root_rel,
        apply_patch=state.get("apply_patch", False),
    )
    design_files = design_toolbox.seed_design_workspace()

    manifest = {
        "visibility": policy.visibility,
        "visible_files": sorted(files),
        "withheld_files": withheld,
        "denied_files": denied,
        "git_status": toolbox.git_status(),
        "design_workspace": {
            "root": f"{artifacts.root_rel}/{DESIGN_WORKSPACE_SUBDIR}",
            "files": design_files,
            "seeded_from": list(DESIGN_SOURCE_FILES),
        },
    }
    artifacts.write_yaml("load_context", "01_context/manifest.yaml", manifest, status="produced")
    artifacts.set_manifest_metadata(
        "load_context",
        "visibility",
        {"level": policy.visibility, "withheld": withheld},
    )
    return {
        "context": {
            "files": files,
            "git_status": manifest["git_status"],
            "visibility": policy.visibility,
            "withheld": withheld,
        },
        "visibility": policy.visibility,
        "run_id": artifacts.run_id,
        "run_root": artifacts.root_rel,
        "design_workspace": manifest["design_workspace"],
        "design_files": design_files,
        "status": "initialized",
        "stage_jobs": "ingest_input",
        "gates": {**state.get("gates", {}), "load_context": {"ok": not denied, "denied": denied}},
        "iteration_history": append_history(
            state,
            f"Loaded repo context ({policy.visibility} visibility) and seeded the isolated CFU design workspace.",
        ),
    }


def ingest_input(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = _artifacts(state, policy)
    llm_raw_outputs: list[str] = []

    def record_spec_output(_label: str, result: LlmResult) -> None:
        llm_raw_outputs.append(_record_llm_output(state, "ingest_input", "spec", result))

    spec, spec_text = normalize_workload_spec(
        task=state.get("task", ""),
        input_mode=state.get("input_mode", "natural_language"),
        context=state.get("context", {}),
        workdir=workdir,
        workload_spec_path=state.get("workload_spec_path", ""),
        project_root=state.get("project_root", ""),
        kernel_file=state.get("kernel_file", ""),
        kernel_function=state.get("kernel_function", ""),
        build_cmd=state.get("build_cmd", ""),
        run_cmd=state.get("run_cmd", ""),
        test_cmd=state.get("test_cmd", ""),
        benchmark_name=state.get("benchmark_name", ""),
        llm=_llm(state),
        llm_result_sink=record_spec_output,
    )
    problems = validate_workload_spec(spec)
    artifacts.write_yaml("ingest_input", "02_workload/workload_spec.yaml", dict(spec), status="produced")
    artifacts.write_yaml(
        "ingest_input",
        "02_workload/validation.yaml",
        {"status": "valid" if not problems else "invalid", "problems": problems},
    )
    return {
        "workload_spec": spec,
        "spec_text": spec_text,
        "spec_unknowns": spec.get("unknowns", []),
        "llm_raw_outputs": llm_raw_outputs,
        "stage_jobs": "route_input",
        "status": "blocked" if problems else "initialized",
        "gates": {**state.get("gates", {}), "workload_spec": {"ok": not problems, "problems": problems}},
        "iteration_history": append_history(state, f"Normalized input into WorkloadSpec `{spec.get('name', 'cfu_workload')}`."),
    }


def route_input(state: CfuDesignState) -> CfuDesignState:
    kind, _ = _route_kind(state)
    _artifacts_for_stage(state, "route_input").write_yaml(
        "route_input",
        "03_route/route.yaml",
        {"input_kind": kind, "description": "input classification result"},
    )
    return {
        "input_kind": kind,
        "stage_jobs": "prepare_benchmark",
        "iteration_history": append_history(state, f"Selected input route `{kind}`."),
    }


def _route_input_target(state: CfuDesignState) -> str:
    """Conditional edge: which benchmark-preparation node to run."""
    return _route_kind(state)[1]


def _route_kind(state: CfuDesignState) -> tuple[str, str]:
    kind = state.get("input_kind") or state.get("input_mode", "natural_language")
    if kind not in ("c_project", "natural_language", "embench"):
        kind = "natural_language"
    if kind == "embench":
        return kind, "prepare_embench_benchmark"
    return kind, ("prepare_c_project_profile" if kind == "c_project" else "create_benchmark_project")


PREPARE_GATE = "prepare_benchmark"


def _prepare_failure_update(state: CfuDesignState, node: str, exc: Exception) -> dict[str, Any]:
    """Turn a benchmark-preparation policy/input error into a clean failure.

    A denied read (for example a C project outside the allowlist) must stop the
    run with ``status=failed`` instead of raising out of the graph.
    """
    message = f"{node} failed: {exc}"
    try:
        _artifacts_for_stage(state, "prepare_benchmark").write_yaml(
            "prepare_benchmark",
            "04_benchmark/manifest.yaml",
            {"kind": "failed", "node": node, "error": str(exc)},
            status="failed",
        )
    except (PolicyError, OSError, ValueError):
        # Reporting the failure must never mask the original error.
        pass
    return {
        "benchmark_project": {"kind": "failed", "notes": [message]},
        "stage_jobs": "review",
        "stopped_at": PREPARE_GATE,
        "errors": [message],
        "gates": {
            **state.get("gates", {}),
            PREPARE_GATE: {
                "ok": False,
                "node": node,
                "next": "review_and_route",
                "errors": [str(exc)],
            },
        },
        "iteration_history": append_history(state, message),
    }


def _prepare_gates(state: CfuDesignState, node: str, next_node: str) -> dict:
    return {
        **state.get("gates", {}),
        PREPARE_GATE: {"ok": True, "node": node, "next": next_node},
    }


def _route_after_prepare(state: CfuDesignState) -> str:
    gate = (state.get("gates", {}) or {}).get(PREPARE_GATE, {}) or {}
    return str(gate.get("next") or "review_and_route")


def prepare_c_project_profile_node(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = _artifacts(state, policy)
    workspace = f"{artifacts.root_rel}/workspace"
    spec = dict(state.get("workload_spec", {}))
    spec["project_root"] = state.get("project_root", spec.get("project_root", ""))
    spec["kernel_file"] = state.get("kernel_file", spec.get("kernel_file", ""))
    try:
        project = prepare_c_project_profile_workspace(
            _toolbox_for_stage(state, "prepare_benchmark"),
            spec,
        )
    except (PolicyError, ValueError, OSError) as exc:
        return _prepare_failure_update(state, "prepare_c_project_profile", exc)
    artifacts.write_yaml(
        "prepare_benchmark",
        "04_benchmark/manifest.yaml",
        dict(project),
        status="planned" if state.get("dry_run") else "produced",
    )
    return {
        "benchmark_project": project,
        "stage_jobs": "profile",
        "gates": _prepare_gates(state, "prepare_c_project_profile", "instrument_and_profile"),
        "iteration_history": append_history(state, f"Prepared C-project profiling workspace `{project.get('workspace', '')}`."),
    }


def create_benchmark_project_node(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = _artifacts(state, policy)
    try:
        project = prepare_generated_benchmark(
            _toolbox_for_stage(state, "prepare_benchmark"), state.get("workload_spec", {})
        )
    except (PolicyError, ValueError, OSError) as exc:
        return _prepare_failure_update(state, "create_benchmark_project", exc)
    artifacts.write_yaml(
        "prepare_benchmark",
        "04_benchmark/manifest.yaml",
        dict(project),
        status="planned" if state.get("dry_run") else "produced",
    )
    return {
        "benchmark_project": project,
        "stage_jobs": "profile",
        "gates": _prepare_gates(state, "create_benchmark_project", "instrument_and_profile"),
        "iteration_history": append_history(state, f"Prepared generated benchmark workspace `{project.get('workspace', '')}`."),
    }

def prepare_embench_benchmark_node(state: CfuDesignState) -> CfuDesignState:
    """Copy an embench-iot benchmark into the run workspace and plan its build."""
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = _artifacts(state, policy)
    spec = dict(state.get("workload_spec", {}))
    toolbox = _toolbox_for_stage(state, "prepare_benchmark")
    try:
        project = prepare_embench_benchmark(toolbox, spec)
    except (PolicyError, ValueError, OSError) as exc:
        return _prepare_failure_update(state, "prepare_embench_benchmark", exc)
    artifacts.write_yaml(
        "prepare_benchmark",
        "04_benchmark/embench.yaml",
        dict(project),
        status="planned" if state.get("dry_run") else "produced",
    )
    return {
        "benchmark_project": project,
        "stage_jobs": "profile",
        "gates": _prepare_gates(state, "prepare_embench_benchmark", "profile_embench"),
        "iteration_history": append_history(
            state, f"Prepared embench benchmark `{project.get('benchmark', '')}`."
        ),
    }


def profile_embench(state: CfuDesignState) -> CfuDesignState:
    """Build and simulate the reference/profile(/cfu) configurations.

    reference : plain build, must pass Embench verify, gives baseline cycles
    profile   : reference + observation points, gives the per-function table
    cfu       : only when a design patch was applied; must still pass
    """
    spec = state.get("workload_spec", {})
    project = state.get("benchmark_project", {})
    run_root = state.get("run_root", "")
    name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
    reason = _profile_reason(state)
    execute = bool(state.get("run_commands", False)) and not state.get("dry_run", False) and not state.get("skip_sim", False)

    # The cfu configuration is built and simulated by the validation ladder
    # (build_sw/simulate), never during profiling.
    configs = ["reference", "profile"]

    results: list[dict] = []
    profiles: dict[str, dict] = {}
    errors: list[str] = []
    observation: dict = {}

    if not execute:
        planned = [
            toolbox_skipped(embench_build_command(run_root, name, config), state, reason)
            for config in configs
        ]
        profile = profile_result_from_commands(planned, dry_run=True, planned_reason=reason)
        _write_embench_profile_artifact(state, name, {}, planned, reason)
        return {
            "profile_result": profile,
            "embench_profiles": {},
            "observation_report": {},
            "stage_jobs": "hotspot",
            "gates": {**state.get("gates", {}), "profile": {"ok": False, "skipped": True, "reason": reason}},
            "iteration_history": append_history(state, f"Skipped embench profiling ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "profile")

    if "profile" in configs:
        sources = {
            rel: toolbox.read_text(rel, limit=2_000_000)
            for rel in config_sources(toolbox.workdir, run_root, name, "profile")
        }
        instrumented, report = instrument_sources(sources, state.get("observation_plan"))
        observation = report.to_dict()
        for rel, text in instrumented.items():
            toolbox.write_text(rel, text)

    for config in configs:
        build = toolbox.run_command(
            embench_build_command(run_root, name, config),
            command_id="embench_build",
            timeout=1800,
            mutate=True,
            execute=True,
        )
        results.append(build)
        if build.get("returncode") not in (0, None):
            errors.append(f"embench {config} build failed")
            continue
        elf = embench_elf_path(run_root, name, config)
        sim_commands = agent_cfu_sim_commands(state, state.get("isa_spec", {}) or {}, elf=elf)
        for sim_command in sim_commands:
            sim = toolbox.run_command(
                sim_command, command_id="soc_sim", timeout=3600, mutate=False, execute=True
            )
            results.append(sim)
            text = "\n".join([sim.get("stdout_tail", ""), sim.get("stderr_tail", "")])
            profiles[config] = parse_embench_profile(text)
            if sim.get("returncode") not in (0, None):
                errors.append(f"embench {config} simulation failed (Embench verify did not pass)")

    reference = profiles.get("reference", {})
    instrumented = embench_instrumented_profile(profiles)
    hot = hotspot_from_profile(instrumented, exclude=(name,)) or name
    candidates = [region for region in instrumented if region != name]
    ok = bool(reference) and not errors
    profile = {
        "status": "profiled" if ok else "failed",
        "command_results": results,
        "regions": [
            {"name": region, "cycles": data.get("cycles", 0), "source": "embench"}
            for region, data in instrumented.items()
        ],
        "hot_spot": hot,
        "summary": summarize_embench_profiles(profiles, name),
        "configs": sorted(profiles),
    }
    _write_embench_profile_artifact(state, name, profiles, results, "", observation)
    return {
        "profile_result": profile,
        "embench_profiles": profiles,
        "observation_report": observation,
        "stage_jobs": "hotspot",
        "errors": errors,
        "gates": {**state.get("gates", {}), "profile": {"ok": ok, "skipped": False, "configs": sorted(profiles)}},
        "iteration_history": append_history(
            state,
            "Profiled embench " + name + " (" + ", ".join(sorted(profiles)) + ")",
        ),
    }


def check_feedback(state: CfuDesignState, rung: str, *, limit: int = 2) -> str:
    """Tail of the failing commands of one validation rung.

    Feeding this back is what turns the ladder into a repair loop instead of a
    one-shot generation: compiler and simulator errors name the exact lines.
    ``sbt`` reports its errors on stdout, so both streams are included.
    """
    chunks: list[str] = []
    for result in (state.get("validation_results") or []):
        if result.get("rung") != rung:
            continue
        if result.get("returncode") in (0, None):
            continue
        command = str(result.get("command", ""))[:200]
        stdout = str(result.get("stdout_tail", ""))[-2500:]
        stderr = str(result.get("stderr_tail", ""))[-1500:]
        chunks.append(f"$ {command}\n{stdout}\n{stderr}".strip())
    return "\n\n".join(chunks[-limit:])


def cost_feedback(state: CfuDesignState, *, limit: int = 2) -> str:
    """Tail of the failing Yosys cost commands.

    ``estimate_cost`` records its commands in ``cost_results`` instead of
    ``validation_results``, so the ladder needs a separate reader to feed a
    cost failure into the hardware repair prompt.
    """
    chunks: list[str] = []
    for result in (state.get("cost_results") or []):
        if result.get("returncode") in (0, None):
            continue
        command = str(result.get("command", ""))[:200]
        stdout = str(result.get("stdout_tail", ""))[-2500:]
        stderr = str(result.get("stderr_tail", ""))[-1500:]
        chunks.append(f"$ {command}\n{stdout}\n{stderr}".strip())
    return "\n\n".join(chunks[-limit:])


def labeled_feedback(label: str, text: str) -> str:
    """Prefix a feedback block with the rung that produced it."""
    text = (text or "").strip()
    return f"### {label}\n{text}" if text else ""


def hardware_repair_feedback(state: CfuDesignState) -> str:
    """Every authoritative error source for a hardware repair, clearly labeled.

    A hardware repair can be triggered by the SoC build, by a repeated
    simulation failure that escalated from software, or by a failed Yosys cost
    run. Each source is labeled so the model can attribute the exact error
    instead of guessing, and empty sources are omitted.
    """
    parts = [
        labeled_feedback("hw_build (SoC generation) errors", check_feedback(state, "hw_build")),
        labeled_feedback("sim (simulation) errors", check_feedback(state, "sim")),
        labeled_feedback("cost (Yosys) errors", cost_feedback(state)),
        labeled_feedback(
            "implementation_hw rejection", implementation_feedback(state, "implementation_hw")
        ),
    ]
    return "\n\n".join(part for part in parts if part)


def implementation_feedback(state: CfuDesignState, rung: str, *, limit: int = 3) -> str:
    """Why the previous attempt at an implementation rung was rejected.

    A build rung reports tool errors through ``validation_results``; an
    implementation rung that produced no usable patch never reaches a tool, so
    its rejection reason is carried on the gate itself. Both feed the next
    prompt.
    """
    gate = (state.get("gates", {}) or {}).get(rung, {}) or {}
    errors = [str(item) for item in (gate.get("errors") or []) if str(item).strip()]
    return "\n".join(errors[-limit:])


def embench_instrumented_profile(profiles: dict[str, dict]) -> dict:
    """The config that carries observation points, falling back to reference."""
    return profiles.get("profile", {}) or profiles.get("reference", {})


def summarize_embench_profiles(profiles: dict[str, dict], name: str) -> str:
    parts: list[str] = []
    for config, data in sorted(profiles.items()):
        cycles = data.get(name, {}).get("cycles", 0)
        parts.append(f"{config}={cycles}")
    return ", ".join(parts) if parts else "no profile lines parsed"


def _write_embench_profile_artifact(
    state: CfuDesignState,
    name: str,
    profiles: dict[str, dict],
    results: list[dict],
    reason: str,
    observation: dict | None = None,
) -> None:
    artifacts = _artifacts_for_stage(state, "profile")
    artifacts.write_yaml(
        "profile",
        "05_profile/embench.yaml",
        {
            "benchmark": name,
            "reason": reason,
            "profiles": profiles,
            "observation": observation or {},
            "commands": [dict(item) for item in results],
        },
        status="planned" if reason else "produced",
    )


def instrument_and_profile(state: CfuDesignState) -> CfuDesignState:
    project = state.get("benchmark_project", {})
    commands = profile_commands(project)
    reason = _profile_reason(state)
    execute = state.get("run_commands", False) and not state.get("dry_run", False) and not state.get("skip_sim", False)
    errors: list[str] = []
    if not execute:
        if not commands:
            commands = [["make", "TARGET=vexii_soc", "MARCH=rv32imc_zicsr_zifencei", "compile"]]
        results = [
            toolbox_skipped(command, state, reason)
            for command in commands
        ]
        profile = profile_result_from_commands(results, dry_run=True, planned_reason=reason)
    else:
        toolbox = _toolbox_for_stage(state, "profile")
        results = []
        for command in commands:
            try:
                command_id = infer_command_id(command, toolbox.policy)
            except PolicyError:
                errors.append(f"profile command is not allowlisted and was skipped: {' '.join(command)}")
                continue
            if not toolbox.command_allowed(command, command_id=command_id):
                errors.append(f"profile command is not allowlisted and was skipped: {' '.join(command)}")
                continue
            result = toolbox.run_command(command, command_id=command_id, timeout=3600, mutate=True, execute=True)
            results.append(result)
            if result.get("returncode") not in (0, None):
                break
        profile = profile_result_from_commands(results, dry_run=False)
    _write_profile_artifact(state, profile, reason)
    return {
        "profile_result": profile,
        "stage_jobs": "hotspot",
        "gates": {
            **state.get("gates", {}),
            "profile": {"ok": _profile_ok(profile) and not errors, "skipped": not execute, "reason": reason},
        },
        "errors": errors,
        "iteration_history": append_history(state, "Profiled the prepared workload."),
    }


def extract_hotspot(state: CfuDesignState) -> CfuDesignState:
    profile = state.get("profile_result", {})
    spec = dict(state.get("workload_spec", {}))
    project = state.get("benchmark_project", {})
    candidates = project.get("candidate_hotspots", [])
    preferred_kernel = spec.get("kernel_function", "")
    if preferred_kernel == "main" and candidates:
        preferred_kernel = candidates[0]
    hot = (
        profile.get("hot_spot")
        or preferred_kernel
        or (candidates[0] if candidates else "")
        or spec.get("operation", "scalar_kernel")
    )
    source = "profile" if profile.get("hot_spot") else "static"
    if profile.get("regions"):
        candidates = [region.get("name", "") for region in profile["regions"]] or candidates
    if not spec.get("hot_spot"):
        spec["hot_spot"] = hot
    spec_text = state.get("spec_text", "")
    if "hot_spot:" not in spec_text:
        spec_text = spec_text.rstrip() + f"\nhot_spot: {hot}\n"
    _write_hotspot_artifact(state, hot, source, candidates)
    return {
        "workload_spec": spec,
        "spec_text": spec_text,
        "hot_spot": hot,
        "stage_jobs": "analysis",
        "gates": {**state.get("gates", {}), "hotspot": {"ok": bool(hot), "source": source}},
        "iteration_history": append_history(state, f"Selected hot spot `{hot}` for CFU analysis."),
    }


def analyze_workload(state: CfuDesignState) -> CfuDesignState:
    task = state["task"]
    fallback = {
        "operation": state.get("workload_spec", {}).get("operation", "unknown"),
        "pattern": state.get("workload_spec", {}).get("pattern", "unknown"),
        "hot_spot": state.get("hot_spot", ""),
        "element_type": state.get("workload_spec", {}).get("element_type", ""),
        "unknowns": state.get("workload_spec", {}).get("unknowns", []),
        "roofline": deterministic_roofline_analysis(state.get("workload_spec", {})),
    }
    result, raw_output = _complete_llm(
        state,
        "analyze_workload",
        "analysis",
        SYSTEM_PROMPT,
        workload_prompt(
            task,
            state.get("workload_spec", {}),
            state.get("context", {}),
            profile=state.get("embench_profiles"),
        ),
        _yaml_str(fallback),
    )
    analysis = parse_yaml_mapping(result.text) or fallback
    note = "Analyzed workload with LLM." if result.used_llm else "Analyzed workload with deterministic fallback."
    _write_analysis_artifact(state, analysis)
    return {
        "workload_analysis": _yaml_str(analysis),
        "llm_raw_outputs": [raw_output],
        "stage_jobs": "isa",
        "iteration_history": append_history(state, note),
    }


def isa_repair_feedback(state: CfuDesignState) -> str:
    """Linter problems from the previous ISA contract attempt, if any."""
    problems = [str(item) for item in (state.get("isa_spec_errors") or []) if str(item).strip()]
    return "\n".join(f"- {problem}" for problem in problems)


def design_contract(state: CfuDesignState) -> CfuDesignState:
    spec = state.get("workload_spec", {})
    fallback = _isa_from_template(state.get("isa_spec_text", ""), spec, state.get("workload_analysis", ""))
    result, raw_output = _complete_llm(
        state,
        "design_contract",
        "isa",
        SYSTEM_PROMPT,
        contract_prompt(
            state["task"],
            spec,
            state.get("workload_analysis", ""),
            state.get("context", {}),
            profile=state.get("embench_profiles"),
            feedback=isa_repair_feedback(state),
        ),
        json.dumps(fallback, ensure_ascii=True, indent=2),
        response_format={"type": "json_object"},
    )
    try:
        isa_spec, problems = validate_isa_text(result.text)
    except IsaLintError:
        isa_spec, problems = fallback, _isa_problem_note(result.text, fallback)
    if not problems and not _has_isaable_commands(isa_spec):
        isa_spec, problems = fallback, ["no usable commands after ISA extraction"]
    _write_isa_artifact(state, isa_spec, problems)
    ok = not problems
    extra, next_node = _ladder_transition(
        state,
        "isa",
        ok=ok,
        default="implement_hw",
        repair_target="design_contract",
    )
    update = {
        "isa_spec": isa_spec,
        "isa_spec_text": _yaml_str(isa_spec),
        "isa_spec_errors": problems,
        "design_contract": result.text,
        "llm_raw_outputs": [raw_output],
        "stage_jobs": "implementation",
        "gates": {
            **state.get("gates", {}),
            "isa": {"ok": ok, "problems": problems, "next": next_node},
        },
        "iteration_history": append_history(state, "Produced CFU ISA/integration contract."),
    }
    return _apply_transition(update, extra)


def _apply_agent_output(
    state: CfuDesignState,
    text: str,
    *,
    toolbox: RepoToolbox,
) -> dict[str, Any]:
    """Apply an LLM response's edits, FILE_BEGIN blocks, or patch.

    Shared by the hardware and software rungs so all three delivery protocols
    honour the same write policy and the same dry-run semantics. Targeted
    SEARCH/REPLACE edits are preferred: they are the cheapest to emit and they
    expose the actual change instead of forcing a full-file rewrite.

    The prompt shows, and the protocol edits, the *live run-workspace* content:
    repairs therefore compose on top of earlier attempts instead of being reset
    against the tracked original (which the prompt does not show).
    """
    edit_blocks = extract_edit_blocks(text)
    file_blocks = extract_file_blocks(text)
    patch_text = extract_patch(text)
    summary = text
    errors: list[str] = []
    apply_result: dict | None = None
    applied_files: list[str] = []
    dry_run = bool(state.get("dry_run", False))
    # ``--apply-patch`` is the explicit opt-in for writing LLM-authored content
    # into the run workspace. Without it, every delivery protocol stays a
    # proposal even when external commands are allowed to run.
    apply_patch = bool(state.get("apply_patch", False))
    allow_apply = apply_patch and not dry_run

    if edit_blocks and allow_apply:
        try:
            apply_result = toolbox.apply_edit_blocks(edit_blocks)
            summary += "\n\nEdit result:\n" + str(apply_result)
            if apply_result.get("returncode") not in (0, None):
                reason = str(apply_result.get("stderr_tail", "")).strip()[:400]
                errors.append(f"targeted edits did not apply: {reason}")
        except (PolicyError, ValueError) as exc:
            errors.append(f"targeted edits failed: {exc}")
    elif file_blocks and allow_apply:
        for target, content in file_blocks:
            try:
                resolved = toolbox.resolve_target(target)
                if not toolbox.is_write_allowed(resolved):
                    raise ValueError(f"path is not writable: {target}")
                toolbox.write_text(resolved, content)
                applied_files.append(resolved)
                summary += f"\n\nWrote {resolved} ({len(content)} bytes)"
            except (PolicyError, ValueError) as exc:
                errors.append(f"file block rejected ({target}): {exc}")
        apply_result = {
            "returncode": 0 if applied_files and not errors else 1,
            "parsed": {"touched_paths": applied_files},
        }
    elif patch_text and allow_apply:
        try:
            apply_result = toolbox.apply_patch_text(patch_text)
            summary += "\n\nPatch result:\n" + str(apply_result)
            if apply_result.get("returncode") not in (0, None):
                errors.append(f"patch did not apply: {apply_result.get('stderr_tail', '').strip()[:400]}")
        except (PolicyError, ValueError) as exc:
            errors.append(f"patch failed: {exc}")
    elif not dry_run:
        if not apply_patch and (edit_blocks or file_blocks or patch_text):
            summary += (
                "\n\nPatch application is disabled (--apply-patch not set); "
                "this rung remains a design proposal and was not applied."
            )
            errors.append(
                "patch application is disabled: pass --apply-patch to let the agent "
                "modify the run workspace"
            )
        else:
            summary += "\n\nNo edit, file block, or patch was produced; this rung remains a design proposal."

    applied = bool(apply_result) and apply_result.get("returncode") == 0 and not apply_result.get("skipped")
    files: list[str] = []
    if applied:
        parsed = (apply_result or {}).get("parsed", {}) or {}
        files = list(parsed.get("touched_paths", []) or []) or list(parsed.get("redirected_paths", []) or [])
    proposed = bool(edit_blocks or file_blocks or patch_text)
    gate_ok = proposed if dry_run else applied
    return {
        "patch": patch_text,
        "summary": summary,
        "errors": errors,
        "applied": applied,
        "files": files,
        "apply_result": apply_result,
        "gate_ok": gate_ok,
        "proposed": proposed,
    }


def _hardware_context(state: CfuDesignState, toolbox: RepoToolbox) -> dict:
    """Repo context whose hardware targets carry the live workspace content.

    The repair prompt must show the same bytes the edit is applied to. The
    snapshotted ``context`` captured at load_context holds the tracked scaffold,
    which stops matching once an earlier attempt has edited the overlay, so the
    run-workspace copy is substituted here.
    """
    context = dict(state.get("context", {}) or {})
    files = dict(context.get("files", {}) or {})
    run_root = state.get("run_root", "")
    overlays = design_workspace_paths(run_root) if run_root else {}
    for tracked in DESIGN_SOURCE_FILES:
        overlay = overlays.get(tracked)
        if not overlay:
            continue
        try:
            text = toolbox.read_text(overlay, limit=1_000_000)
        except PolicyError:
            continue
        if text:
            files[tracked] = text
    context["files"] = files
    return context


def _log_tool_call(record: dict[str, Any]) -> None:
    """Print each tool call while a run is in progress (CLI log)."""
    if _PROGRESS is None:
        return
    arguments = record.get("arguments") or {}
    rendered = ", ".join(f"{key}={value!r}" for key, value in list(arguments.items())[:4])
    status = "ok" if record.get("ok") else "ERROR"
    print(
        f"    [tool] {record.get('name', '?')}({rendered}) {status} "
        f"-> {len(str(record.get('result', '')))} chars",
        flush=True,
    )


def _tool_call_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": record.get("round"),
        "tool": record.get("name"),
        "arguments": record.get("arguments"),
        "ok": record.get("ok"),
        "result_chars": len(str(record.get("result", ""))),
    }


def _write_tool_calls_artifact(state: CfuDesignState, label: str, tool_calls: list[dict]) -> None:
    artifacts = _artifacts_for_stage(state, "implementation")
    artifacts.write_yaml(
        "implementation",
        f"09_implementation/{label}_tool_calls.yaml",
        {
            "rounds": max((int(call.get("round", 0) or 0) for call in tool_calls), default=0),
            "call_count": len(tool_calls),
            "calls": [
                {**_tool_call_summary(call), "result_excerpt": str(call.get("result", ""))[:2000]}
                for call in tool_calls
            ],
        },
        status="produced",
    )


def implement_hw(state: CfuDesignState) -> CfuDesignState:
    """Hardware rung: design AgentCfu.scala / AgentCfuFiber.scala only."""
    toolbox = _toolbox_for_stage(state, "implementation")
    feedback = hardware_repair_feedback(state)
    system = SYSTEM_PROMPT
    client = _llm(state)
    tools_enabled = bool(state.get("agent_tools", True)) and getattr(client, "available", False)
    user = hardware_prompt(
        state["task"],
        state.get("workload_spec", {}),
        state.get("workload_analysis", ""),
        state.get("design_contract", ""),
        _hardware_context(state, toolbox),
        profile=state.get("embench_profiles"),
        feedback=feedback,
        agent_tools=tools_enabled,
    )
    fallback = deterministic_hardware_plan(state["task"], state.get("design_contract", ""))
    tool_calls: list[dict] = []
    if tools_enabled:
        toolset = ReadOnlyToolset(toolbox)
        try:
            result = clean_llm_text(
                client.complete_with_tools(
                    system,
                    user,
                    tools=toolset.schemas(),
                    tool_runner=toolset.run,
                    max_rounds=int(state.get("max_tool_rounds", 6) or 6),
                    fallback=fallback,
                    on_tool_call=_log_tool_call,
                )
            )
        except Exception as exc:
            # Tool calling must never break the rung: degrade to a plain call.
            result, raw_output = _complete_llm(
                state, "implement_hw", "hardware", system, user, fallback, client=client
            )
            result = clean_llm_text(result)
            _write_tool_calls_artifact(
                state, "hw", [{"round": 0, "name": "(tool loop failed)", "arguments": {"error": str(exc)},
                               "result": "", "ok": False}]
            )
        else:
            raw_output = _record_llm_output(
                state, "implement_hw", "hardware", result, client=client
            )
            tool_calls = list(getattr(result, "tool_calls", ()) or ())
            if tool_calls:
                _write_tool_calls_artifact(state, "hw", tool_calls)
    else:
        result, raw_output = _complete_llm(
            state, "implement_hw", "hardware", system, user, fallback, client=client
        )
    outcome = _apply_agent_output(
        state,
        result.text,
        toolbox=toolbox,
    )
    _write_hardware_artifact(state, result.text, outcome, feedback)
    return {
        "hardware_plan": result.text,
        "hardware_patch": outcome["patch"],
        "implementation_hw_summary": outcome["summary"],
        "changed_design_files": outcome["files"],
        "llm_raw_outputs": [raw_output],
        "tool_calls": [_tool_call_summary(call) for call in tool_calls],
        "stage_jobs": "validation",
        "errors": outcome["errors"],
        "gates": {
            **state.get("gates", {}),
            "implementation_hw": {
                "ok": outcome["gate_ok"],
                "patch": bool(outcome["patch"]),
                "applied": outcome["applied"],
                "files": outcome["files"],
                "repair": bool(feedback),
                "tool_calls": len(tool_calls),
                "repair_diagnosis": ("REPAIR_DIAGNOSIS" in result.text) if feedback else None,
                "errors": outcome["errors"],
            },
        },
        "iteration_history": append_history(
            state, "Completed the hardware rung (AgentCfu.scala/AgentCfuFiber.scala)."
        ),
    }


def implement_sw(state: CfuDesignState) -> CfuDesignState:
    """Software rung: the accelerated kernel that calls the new intrinsics."""
    toolbox = _toolbox_for_stage(state, "implementation")
    isa_spec = state.get("isa_spec", {}) or {}
    targets = _software_targets(state, toolbox)
    if not targets:
        # No software surface exists (for example a C-project workload without a
        # resolvable kernel): fail the rung instead of spending an LLM call.
        note = "no software target exists for this workload; the software rung cannot run"
        outcome = {"patch": "", "files": [], "applied": False, "proposed": False, "errors": [note]}
        _write_software_artifact(state, "", outcome, [note])
        return {
            "software_plan": "",
            "software_patch": "",
            "implementation_sw_summary": note,
            "llm_raw_outputs": [],
            "changed_sw_files": [],
            "stage_jobs": "validation",
            "errors": [note],
            "gates": {
                **state.get("gates", {}),
                "implementation_sw": {
                    "ok": False,
                    "patch": False,
                    "applied": False,
                    "files": [],
                    "intrinsics_header": "",
                    "errors": [note],
                },
            },
            "iteration_history": append_history(state, note),
        }
    errors: list[str] = []
    intrinsics_path = _write_intrinsics_header(state, toolbox)
    feedback = "\n\n".join(
        part
        for part in (
            labeled_feedback("sw_build (compile) errors", check_feedback(state, "sw_build")),
            labeled_feedback("sim (simulation) errors", check_feedback(state, "sim")),
            labeled_feedback(
                "implementation_sw rejection", implementation_feedback(state, "implementation_sw")
            ),
        )
        if part
    )
    result, raw_output = _complete_llm(
        state,
        "implement_sw",
        "software",
        SYSTEM_PROMPT,
        software_prompt(
            state["task"],
            state.get("workload_spec", {}),
            state.get("workload_analysis", ""),
            state.get("design_contract", ""),
            state.get("context", {}),
            profile=state.get("embench_profiles"),
            intrinsics=render_cfu_intrinsics(isa_spec),
            targets=targets,
            feedback=feedback,
        ),
        deterministic_software_plan(state["task"], state.get("design_contract", "")),
    )
    outcome = _apply_agent_output(
        state,
        result.text,
        toolbox=toolbox,
    )
    errors.extend(outcome["errors"])
    if outcome["applied"]:
        uses_cfu, cfu_problem = _software_exercises_cfu(state, toolbox, targets)
        if not uses_cfu:
            errors.append(cfu_problem)
    summary = outcome["summary"]
    if intrinsics_path:
        summary += f"\n\nWrote {intrinsics_path} from the CFU ISA spec."
    _write_software_artifact(state, result.text, outcome, errors, feedback)
    return {
        "software_plan": result.text,
        "software_patch": outcome["patch"],
        "implementation_sw_summary": summary,
        "changed_sw_files": outcome["files"],
        "llm_raw_outputs": [raw_output],
        "stage_jobs": "validation",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            "implementation_sw": {
                "ok": outcome["gate_ok"] and not errors,
                "patch": bool(outcome["patch"]),
                "applied": outcome["applied"],
                "files": outcome["files"],
                "intrinsics_header": intrinsics_path,
                "repair": bool(feedback),
                "repair_diagnosis": ("REPAIR_DIAGNOSIS" in result.text) if feedback else None,
                "errors": errors,
            },
        },
        "iteration_history": append_history(
            state, "Completed the software rung (accelerated kernel with CFU intrinsics)."
        ),
    }


def _software_targets(state: CfuDesignState, toolbox: RepoToolbox) -> list[tuple[str, str]]:
    """[(writable path, current text)] the software rung may rewrite."""
    spec = state.get("workload_spec", {})
    project = state.get("benchmark_project", {})
    run_root = state.get("run_root", "")
    workdir = Path(state.get("workdir", ".")).resolve()
    kind = str(project.get("kind", ""))

    if kind == "embench_benchmark":
        name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
        return cfu_source_targets(workdir, run_root, name) if name else []
    if kind == "generated_benchmark":
        source = str(project.get("source_path", ""))
        return [(source, toolbox.read_text(source, limit=1_000_000))] if source else []
    if kind == "instrumented_copy":
        source = str(project.get("source_path", ""))
        return [(source, toolbox.read_text(source, limit=1_000_000))] if source else []
    return []


# Helpers that only enable/fence the CFU do not exercise the datapath, so a
# kernel that calls them and then runs the scalar loop is not an acceleration.
CFU_NON_DATAPATH_HELPERS = ("cfu_enable", "cfu_memory_fence")


def _software_exercises_cfu(
    state: CfuDesignState,
    toolbox: RepoToolbox,
    targets: list[tuple[str, str]],
) -> tuple[bool, str]:
    """True when the applied software actually calls a CFU datapath intrinsic.

    ``cfu_enable()`` / ``cfu_memory_fence()`` alone, or a scalar fallback inside
    the ``#ifdef USE_AGENT_CFU`` branch, is not acceleration. The check runs on
    the applied workspace bytes so a Protocol A/B/C change is all covered.
    """
    expected = intrinsic_symbols(state.get("isa_spec", {}) or {})
    sources: list[tuple[str, str]] = []
    for path, _ in targets:
        try:
            text = toolbox.read_text(path, limit=2_000_000)
        except PolicyError:
            continue
        if text:
            # Strip comments so a commented-out call cannot satisfy the check.
            stripped = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
            stripped = re.sub(r"//[^\n]*", " ", stripped)
            sources.append((path, stripped))

    for _, text in sources:
        for symbol in expected:
            if re.search(rf"\b{re.escape(symbol)}\s*\(", text):
                return True, ""
    # Accept a differently named CFU intrinsic the model may have emitted, as
    # long as it is not just the enable/fence helpers.
    for _, text in sources:
        for match in re.finditer(r"\b(cfu_\w+)\s*\(", text):
            if match.group(1) not in CFU_NON_DATAPATH_HELPERS:
                return True, ""

    wanted = ", ".join(expected) or "a generated cfu_<command>() intrinsic"
    return False, (
        f"the CFU build does not call any CFU datapath intrinsic (expected {wanted}); "
        "an #ifdef USE_AGENT_CFU branch that only calls cfu_enable() and then runs the "
        "scalar loop is not acceleration"
    )


def _write_intrinsics_header(state: CfuDesignState, toolbox: RepoToolbox) -> str:
    """Write the generated CFU intrinsics beside the software it is called from."""
    if state.get("dry_run", False) or not state.get("apply_patch", False):
        # Without --apply-patch the software rung cannot apply a kernel either,
        # so do not leave a half-generated workspace behind.
        return ""
    project = state.get("benchmark_project", {})
    run_root = state.get("run_root", "")
    spec = state.get("workload_spec", {})
    text = render_cfu_intrinsics(state.get("isa_spec", {}) or {})
    kind = str(project.get("kind", ""))
    if kind == "embench_benchmark":
        name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
        if not name:
            return ""
        path = f"{embench_config_workspace(run_root, name, 'cfu')}/include/cfu_intrinsics.h"
    else:
        workspace = str(project.get("workspace", ""))
        if not workspace:
            return ""
        path = f"{workspace}/cfu_intrinsics.h"
    toolbox.write_text(path, text)
    return path


def _run_check(
    state: CfuDesignState,
    toolbox: RepoToolbox,
    argv: list[str],
    *,
    rung: str,
    command_id: str = "",
    timeout: int = 3600,
    mutate: bool = False,
) -> dict:
    """Run one ladder command, turning hard failures into a failed rung result.

    A hanging simulator (``TimeoutExpired``), a missing toolchain (``OSError``),
    or a command that the access policy does not allow must be a failing rung
    the ladder can repair, not an exception that kills the run.
    """
    if not command_id:
        # Resolve the declared command id here so an unallowlisted user command
        # (for example a custom C-project run_cmd) becomes a repairable failure
        # instead of a PolicyError escaping the graph.
        try:
            command_id = infer_command_id(argv, toolbox.policy)
        except PolicyError as exc:
            return {
                "command": " ".join(argv),
                "cwd": str(toolbox.workdir),
                "returncode": -2,
                "stdout_tail": "",
                "stderr_tail": f"command is not allowlisted for stage {toolbox.stage}: {exc}",
                "rung": rung,
                "parsed": {},
            }
    try:
        result = toolbox.run_command(
            argv,
            command_id=command_id or None,
            timeout=timeout,
            mutate=mutate,
            execute=True,
        )
    except PolicyError as exc:  # e.g. a denied path inside an allowed command
        result = {
            "command": " ".join(argv),
            "cwd": str(toolbox.workdir),
            "returncode": -2,
            "stdout_tail": "",
            "stderr_tail": f"command rejected by the access policy: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        result = {
            "command": " ".join(argv),
            "cwd": str(toolbox.workdir),
            "returncode": -9,
            "stdout_tail": filter_output(stdout)[-2000:] if isinstance(stdout, str) else "",
            "stderr_tail": f"command timed out after {timeout}s",
            "timed_out": True,
        }
    except OSError as exc:
        result = {
            "command": " ".join(argv),
            "cwd": str(toolbox.workdir),
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": f"command could not start: {exc}",
        }
    result["rung"] = rung
    return result


def _missing_design_files(state: CfuDesignState) -> list[str]:
    """Overlay hardware files that are absent or empty (agent-CFU runs only)."""
    run_root = state.get("run_root", "")
    if not run_root or not is_agent_cfu_run(state):
        return []
    workdir = Path(state.get("workdir", ".")).resolve()
    missing: list[str] = []
    for overlay in design_workspace_paths(run_root).values():
        path = workdir / overlay
        if not path.is_file() or not path.read_text(errors="replace").strip():
            missing.append(overlay)
    return missing


def _soc_rtl_path(state: CfuDesignState) -> Path | None:
    run_root = state.get("run_root", "")
    if not run_root:
        return None
    return Path(state.get("workdir", ".")).resolve() / run_root / SOC_WORKSPACE_SUBDIR / "MiCoSoc.v"


def _legacy_ladder_commands(
    state: CfuDesignState,
    toolbox: RepoToolbox,
) -> dict[str, list[list[str]]]:
    """Split the non-embench command set into generate/build/sim rungs."""
    spec = state.get("workload_spec", {})
    isa_spec = state.get("isa_spec", {}) or {}
    overlay = agent_cfu_overlay_commands(state, isa_spec)
    if overlay:
        return {
            "generate": overlay,
            "build": [],
            "sim": agent_cfu_sim_commands(state, isa_spec),
        }
    commands, _errors = validation_commands_for_spec(spec, state.get("task", ""), toolbox)
    generate = [c for c in commands if "MiCoSocGen" in " ".join(c)]
    sim = [c for c in commands if "MiCoSocSim" in " ".join(c)]
    build = [c for c in commands if c not in generate and c not in sim]
    return {"generate": generate, "build": build, "sim": sim}


def _software_build_commands(state: CfuDesignState, toolbox: RepoToolbox) -> list[list[str]]:
    project = state.get("benchmark_project", {})
    spec = state.get("workload_spec", {})
    run_root = state.get("run_root", "")
    if str(project.get("kind", "")) == "embench_benchmark":
        name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
        return [embench_build_command(run_root, name, "cfu")] if name else []
    commands: list[list[str]] = []
    for key in ("build_cmd", "test_cmd"):
        raw = str(project.get(key, "")).strip()
        if raw:
            commands.append(shlex.split(raw))
    if commands:
        return commands
    return _legacy_ladder_commands(state, toolbox)["build"]


def _software_sim_commands(state: CfuDesignState, toolbox: RepoToolbox) -> list[list[str]]:
    project = state.get("benchmark_project", {})
    spec = state.get("workload_spec", {})
    run_root = state.get("run_root", "")
    isa_spec = state.get("isa_spec", {}) or {}
    if str(project.get("kind", "")) == "embench_benchmark":
        name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
        if not name:
            return []
        return agent_cfu_sim_commands(state, isa_spec, elf=embench_elf_path(run_root, name, "cfu"))
    overlay_sim = agent_cfu_sim_commands(state, isa_spec)
    if overlay_sim:
        return overlay_sim
    raw = str(project.get("run_cmd", "")).strip()
    if raw:
        return [shlex.split(raw)]
    return _legacy_ladder_commands(state, toolbox)["sim"]


def generate_soc(state: CfuDesignState) -> CfuDesignState:
    """Rung 2: generate the SoC from the overlaid agent design.

    This is the hardware gate: Scala compilation, SpinalHDL elaboration and
    Verilog generation all happen here, so a broken design stops the run before
    any software is written.
    """
    rung = "hw_build"
    reason = _validation_reason(state)
    if reason:
        result = _skipped_result("sbt MiCoSocGen (agent overlay)", state, reason)
        result["rung"] = rung
        _write_ladder_artifact(state, rung, [result], "planned", reason)
        return {
            "validation_results": [result],
            "stage_jobs": "validation",
            "gates": {**state.get("gates", {}), rung: {"ok": False, "skipped": True, "reason": reason}},
            "iteration_history": append_history(state, f"Skipped SoC generation ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "validation")
    isa_spec = state.get("isa_spec", {}) or {}
    commands = agent_cfu_overlay_commands(state, isa_spec) or _legacy_ladder_commands(state, toolbox)["generate"]
    results: list[dict] = []
    impl_gate = _rung_gate(state, "implementation_hw")
    if impl_gate and not impl_gate.get("ok", False):
        # The hardware rung produced no applied design, so the workspace still
        # holds the seeded scaffold. Compiling that would "pass" while validating
        # nothing, so fail the rung here and let the ladder repair implement_hw.
        errors: list[str] = [
            "the hardware rung did not apply a design change; "
            "refusing to generate the SoC from the scaffold"
        ]
    else:
        errors = _missing_design_files(state)
        if errors:
            errors = [f"hardware design file is missing or empty: {', '.join(errors)}"]
        elif not commands:
            errors.append("no SoC generation command is available for this run")
        else:
            for command in commands:
                result = _run_check(
                    state,
                    toolbox,
                    command,
                    rung=rung,
                )
                results.append(result)
                if result.get("returncode") not in (0, None):
                    errors.append(f"SoC generation failed: {' '.join(command)[:200]}")
                    break

    rtl = _soc_rtl_path(state)
    if not errors and rtl is not None and not rtl.is_file():
        errors.append(f"SoC generation produced no RTL at {state.get('run_root')}/{SOC_WORKSPACE_SUBDIR}/MiCoSoc.v")

    ok = not errors and bool(results) and all(r.get("returncode") == 0 for r in results)
    _write_ladder_artifact(state, rung, results, "passed" if ok else "failed", "")
    extra, next_node = _ladder_transition(
        state, rung, ok=ok, default="implement_sw", repair_target="implement_hw"
    )
    update = {
        "validation_results": results,
        "stage_jobs": "validation",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            rung: {"ok": ok, "skipped": False, "commands": len(results), "next": next_node},
        },
        "iteration_history": append_history(state, "Generated the SoC from the agent design."),
    }
    return _apply_transition(update, extra)


def build_sw(state: CfuDesignState) -> CfuDesignState:
    """Rung 4: compile the accelerated software against the new intrinsics."""
    rung = "sw_build"
    reason = _validation_reason(state)
    if reason:
        result = _skipped_result("make (accelerated software compile)", state, reason)
        result["rung"] = rung
        _write_ladder_artifact(state, rung, [result], "planned", reason)
        return {
            "validation_results": [result],
            "stage_jobs": "validation",
            "gates": {**state.get("gates", {}), rung: {"ok": False, "skipped": True, "reason": reason}},
            "iteration_history": append_history(state, f"Skipped the software build ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "validation")
    commands = _software_build_commands(state, toolbox)
    results: list[dict] = []
    errors: list[str] = []
    impl_gate = _rung_gate(state, "implementation_sw")
    if impl_gate and not impl_gate.get("ok", False):
        # The software rung produced no applied change, so a compile would test
        # the tracked original rather than the accelerated kernel. Fail here and
        # let the ladder repair implement_sw instead of compiling stale code.
        errors.append(
            "the software rung did not apply a change; refusing to compile the unmodified source"
        )
    else:
        if not commands:
            errors.append("no software build command is available for this workload")
        for command in commands:
            result = _run_check(
                state,
                toolbox,
                command,
                rung=rung,
                timeout=1800,
                mutate=True,
            )
            results.append(result)
            if result.get("returncode") not in (0, None):
                errors.append(f"software build failed: {' '.join(command)[:200]}")
                break

    elf = _software_elf(state)
    if not errors and elf and not (Path(state.get("workdir", ".")).resolve() / elf).is_file():
        errors.append(f"software build produced no ELF at {elf}")

    ok = not errors and bool(results) and all(r.get("returncode") == 0 for r in results)
    _write_ladder_artifact(state, rung, results, "passed" if ok else "failed", "")
    extra, next_node = _ladder_transition(
        state, rung, ok=ok, default="simulate", repair_target="implement_sw"
    )
    update = {
        "validation_results": results,
        "stage_jobs": "validation",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            rung: {"ok": ok, "skipped": False, "commands": len(results), "elf": elf, "next": next_node},
        },
        "iteration_history": append_history(state, "Compiled the accelerated software."),
    }
    return _apply_transition(update, extra)


def _software_elf(state: CfuDesignState) -> str:
    project = state.get("benchmark_project", {})
    spec = state.get("workload_spec", {})
    run_root = state.get("run_root", "")
    if str(project.get("kind", "")) == "embench_benchmark":
        name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
        return embench_elf_path(run_root, name, "cfu") if name else ""
    return str(project.get("elf_path", "") or "")


def simulate(state: CfuDesignState) -> CfuDesignState:
    """Rung 5: run the ELF on the generated SoC and check the result."""
    rung = "sim"
    reason = _validation_reason(state)
    if reason:
        result = _skipped_result("sbt MiCoSocSim", state, reason)
        result["rung"] = rung
        _write_ladder_artifact(state, rung, [result], "planned", reason)
        return {
            "validation_results": [result],
            "stage_jobs": "cost",
            "gates": {**state.get("gates", {}), rung: {"ok": False, "skipped": True, "reason": reason}},
            "iteration_history": append_history(state, f"Skipped simulation ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "validation")
    project = state.get("benchmark_project", {})
    spec = state.get("workload_spec", {})
    run_root = state.get("run_root", "")
    name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
    is_embench = str(project.get("kind", "")) == "embench_benchmark"
    commands = _software_sim_commands(state, toolbox)
    results: list[dict] = []
    errors: list[str] = []
    sim_text = ""

    if not commands:
        errors.append("no simulation command is available for this workload")
    for command in commands:
        result = _run_check(
            state,
            toolbox,
            command,
            rung=rung,
        )
        results.append(result)
        sim_text = "\n".join([result.get("stdout_tail", ""), result.get("stderr_tail", "")])
        if result.get("returncode") not in (0, None):
            errors.append(f"simulation failed: {' '.join(command)[:200]}")
            break

    reference = state.get("embench_profiles", {}).get("reference", {})
    ref_cycles = int(reference.get(name, {}).get("cycles", 0) or 0) if is_embench else 0
    cfu_cycles = 0
    speedup_x100 = 0
    verification = str(project.get("verification", "none"))

    if is_embench:
        cfu_profile = parse_embench_profile(sim_text)
        cfu_cycles = int(cfu_profile.get(name, {}).get("cycles", 0) or 0)
        speedup_x100 = (ref_cycles * 100 // cfu_cycles) if cfu_cycles else 0
        if not cfu_profile:
            errors.append("simulation produced no CFU profile; the accelerated kernel did not run")
        if verification != "bench_verify":
            errors.append("benchmark provides no verification oracle; correctness cannot be claimed")
        if not state.get("changed_design_files"):
            # The cfu build is then identical to the reference configuration, so
            # any speedup would come from the compiler, not the new CFU.
            errors.append("no design change was applied; the cfu build equals the reference configuration")
        if ref_cycles <= 0:
            errors.append(
                "no reference cycle count is available; acceleration cannot be verified"
            )
        elif cfu_cycles > 0 and speedup_x100 < MIN_EMBENCH_SPEEDUP_X100:
            errors.append(
                f"the CFU build is not meaningfully faster than the reference "
                f"({cfu_cycles} vs {ref_cycles} cycles, {speedup_x100 / 100:.2f}x); "
                "reverting to scalar inside the CFU build is not acceleration"
            )
        artifacts = _artifacts_for_stage(state, "validation")
        artifacts.write_yaml(
            "validation",
            "10_validation/embench.yaml",
            {
                "benchmark": name,
                "verification": verification,
                "reference_cycles": ref_cycles,
                "cfu_cycles": cfu_cycles,
                "speedup_x100": speedup_x100,
                "min_speedup_x100": MIN_EMBENCH_SPEEDUP_X100,
                "status": "passed" if not errors else "failed",
            },
        )
    elif not state.get("changed_design_files"):
        # Without an applied design change the ELF exercises the scaffold, so
        # there is nothing to validate.
        errors.append("no design change was applied; the simulation runs the scaffold CFU")

    ok = not errors and bool(results) and all(r.get("returncode") == 0 for r in results)
    _write_ladder_artifact(state, rung, results, "passed" if ok else "failed", "")
    extra, next_node = _ladder_transition(
        state, rung, ok=ok, default="estimate_cost", repair_target=_sim_repair_target
    )
    update = {
        "validation_results": results,
        "stage_jobs": "cost",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            rung: {
                "ok": ok,
                "skipped": False,
                "reference_cycles": ref_cycles,
                "cfu_cycles": cfu_cycles,
                "speedup_x100": speedup_x100,
                "next": next_node,
            },
        },
        "iteration_history": append_history(state, "Ran the SoC simulation."),
    }
    return _apply_transition(update, extra)


def _rtl_ready(workdir: Path, rtl: str) -> bool:
    """True when a generated Verilog file/directory exists and is non-empty."""
    path = workdir / rtl
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(
            item.is_file() and item.suffix in {".v", ".sv"} and item.stat().st_size > 0
            for item in path.rglob("*")
        )
    return False


def estimate_cost(state: CfuDesignState) -> CfuDesignState:
    if state.get("skip_yosys", False):
        result = [_skipped_result("python3 skills/cfu-designer/scripts/yosys_cost_report.py", state, "--skip-yosys")]
        _write_cost_artifact(state, result, "planned", "--skip-yosys")
        return {
            "cost_results": result,
            "stage_jobs": "review",
            "gates": {
                **state.get("gates", {}),
                "cost": {"ok": False, "skipped": True, "reason": "--skip-yosys"},
            },
            "iteration_history": append_history(state, "Skipped Yosys cost estimation by request."),
        }

    toolbox = _toolbox_for_stage(state, "cost")
    agent_cfu = is_agent_cfu_run(state)
    spec = state.get("workload_spec", {}) or {}
    task = state.get("task", "")
    case = cost_case_for_spec(spec, task, agent_cfu=agent_cfu)
    execute = bool(state.get("run_commands", False))
    workdir = Path(state.get("workdir", ".")).resolve()

    # Command side effects must stay inside the stage's declared write roots, so
    # the report goes into the run directory when one exists.
    run_root = state.get("run_root", "")
    if run_root:
        out_dir = f"{run_root}/11_cost/yosys/{case}"
        generator_commands = agent_cfu_cost_commands(state, spec) if agent_cfu else []
        rtl = (
            f"{run_root}/11_cost/cfu_rtl"
            if generator_commands
            else f"{run_root}/{SOC_WORKSPACE_SUBDIR}/MiCoSoc.v"
        )
    else:
        out_dir = f"{base_artifact_root(state)}/yosys/{case}"
        generator_commands = []
        rtl = "MiCoSoc.v"

    yosys_command = [
        "python3",
        "skills/cfu-designer/scripts/yosys_cost_report.py",
        rtl,
        "--top",
        cost_top_for_spec(spec, task, agent_cfu=agent_cfu),
        "--out-dir",
        out_dir,
        "--flatten",
    ]

    results: list[dict] = []
    errors: list[str] = []
    if not execute:
        for argv in [*generator_commands, yosys_command]:
            results.append(toolbox.run_command(argv, execute=False))
        ok = False
    else:
        # Agent runs synthesize the standalone CFU first: GenAgentCfu elaborates
        # only the overlaid AgentCfu class, so Yosys never sees the CPU/SoC.
        for argv in generator_commands:
            result = _run_check(state, toolbox, argv, rung="cost", timeout=1800, mutate=True)
            results.append(result)
            if result.get("returncode") not in (0, None):
                errors.append(f"CFU generation failed: {' '.join(argv)[:200]}")
                break
        if not errors and generator_commands and not _rtl_ready(workdir, rtl):
            errors.append(f"CFU generation produced no Verilog under {rtl}")
        if not errors:
            result = _run_check(state, toolbox, yosys_command, rung="cost", timeout=1800, mutate=True)
            results.append(result)
            if result.get("returncode") not in (0, None):
                errors.append(f"Yosys cost estimation failed: {' '.join(yosys_command)[:200]}")
        ok = not errors and bool(results) and all(item.get("returncode") == 0 for item in results)

    reason = "; ".join(errors)[:400] if errors else ""
    _write_cost_artifact(
        state, results, "passed" if ok else ("planned" if not execute else "failed"), reason
    )
    extra, next_node = _ladder_transition(
        state,
        "cost",
        ok=ok,
        default="review_and_route",
        repair_target="implement_hw",
        skipped=not execute,
    )
    update = {
        "cost_results": results,
        "stage_jobs": "review",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            "cost": {
                "ok": ok,
                "skipped": not execute,
                "reason": reason or ("" if execute else "execution-disabled"),
                "rtl": rtl,
                "next": next_node,
            },
        },
        "iteration_history": append_history(state, "Ran Yosys cost estimation."),
    }
    return _apply_transition(update, extra)


def review_and_route(state: CfuDesignState) -> CfuDesignState:
    """Terminal decision node.

    Retry routing lives in the per-rung routers, so review only reports the
    honest outcome of the ladder: complete, planned, failed, or blocked.
    """
    decision, status = _review_decision(state)
    _report_review_decision(state, decision, status)
    stopped = state.get("stopped_at", "")
    note = f"Reviewed the run: {status}."
    if stopped:
        note = f"Reviewed the run: {status} (stopped at {stopped})."
    return {
        "status": status,
        "stage_jobs": "final_report",
        "gates": {**state.get("gates", {}), "review": {"decision": decision, "status": status}},
        "iteration_history": append_history(state, note),
    }


def final_report(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = _policy(state)
    artifacts = _artifacts(state, policy)
    state = {**state, "llm_request": artifacts.manifest.get("llm_request") or {}}
    report = render_report(state)
    # The review node already decided the outcome (planned/complete/failed/
    # blocked). Recomputing it here would overwrite an honest "planned" with
    # "blocked" whenever execution was simply skipped.
    status = str(state.get("status", "") or "")
    if status not in {"planned", "complete", "failed", "blocked"}:
        status = "planned" if state.get("dry_run", False) else ("complete" if _is_complete(state) else "blocked")
    if status == "complete" and not _is_complete(state):
        # Never claim completion without every required gate passing.
        status = "blocked"
    artifacts.write_text("review", "report.md", report, kind="markdown", status=status)
    artifacts.set_status("review", status)
    return {
        "final_report_path": str(artifacts.path("report.md")),
        "run_root": artifacts.root_rel,
        "status": status,
        "stage_jobs": "final_report",
        "iteration_history": append_history(state, "Wrote final report."),
    }


def deterministic_roofline_analysis(spec: dict) -> dict[str, Any]:
    """Return conservative Roofline fields without inventing workload ceilings."""
    unknowns = list(spec.get("unknowns", []) or [])
    unknowns.extend(
        item
        for item in (
            "hot-loop operation count",
            "external byte count and memory boundary",
            "compute peak",
            "sustained bandwidth peak",
            "frequency/cache behavior for end-to-end performance",
        )
        if item not in unknowns
    )
    return {
        "hot_function": spec.get("kernel_function") or spec.get("hot_spot") or "unknown",
        "hot_loop": "unknown",
        "trip_count": "unknown",
        "memory_boundary": "unknown",
        "operations": "unknown",
        "operation_convention": "unknown",
        "bytes": "unknown",
        "arithmetic_intensity": {"value": "unknown", "unit": "ops/byte", "kind": "unknown"},
        "compute_peak": {"value": "unknown", "basis": "unknown", "unit": "ops/cycle"},
        "bandwidth_peak": {"value": "unknown", "basis": "unknown", "unit": "bytes/cycle"},
        "ridge_point": "unknown",
        "predicted_ceiling": "unknown",
        "bottleneck": "unknown",
        "gap_to_ceiling": "unknown",
        "design_action": "Measure operation count, external bytes, and both ceilings before selecting CFU resources.",
        "verification_metric": "Recompute achieved ops/cycle and external bytes/cycle from scalar/CFU cycles and traffic evidence.",
        "unknowns": unknowns,
    }


def deterministic_analysis(task: str, spec: dict) -> str:
    pattern = spec.get("pattern") or "unknown"
    operation = spec.get("operation") or "unknown"
    roofline = deterministic_roofline_analysis(spec)
    return (
        f"Operation: {operation}.\n"
        f"Pattern: {pattern}.\n"
        f"Selected hot spot: {spec.get('hot_spot', '') or 'unknown'}.\n"
        f"Known element type: {spec.get('element_type', '') or 'unknown'}.\n"
        f"Unknowns: {', '.join(spec.get('unknowns', [])) or 'none'}.\n"
        f"Roofline: {roofline}.\n"
        "Likely CFU shape: extend AgentCfu's placeholder compute/config function ids; keep the Vector RegFile and CfuLsu for memory-backed vectors or multi-operation reuse, and keep the CfuBus response contract."
    )


def deterministic_contract(task: str, spec: dict, analysis: str) -> str:
    operation = spec.get("operation", "")
    if operation in {"weighted_average", "dot_product"}:
        op_line = "For weighted average/dot product, use CFU-owned vector registers and `maclen` slices only if the Roofline analysis shows resident-vector reuse is worthwhile; otherwise use a direct or streamed datapath, widen locally, and accumulate into the required scalar result."
    elif operation == "elementwise_add":
        op_line = "For element-wise add, use CFU RF slots only when reuse justifies them; otherwise process direct packed operands, and store the packed result through the shared LSU only when the output must escape."
    else:
        op_line = "Select the smallest direct or TileLink CFU contract after confirming the hot-loop operation, Roofline bottleneck, and data lifetime; unused vector dimensions remain unset."
    return (
        "Proposed contract:\n"
        "- Use custom0 function IDs documented beside the C intrinsics.\n"
        "- Keep reset/config/load/compute/read commands explicit.\n"
        f"- {op_line}\n"
        "- Keep the SoC integration fixed: the run overlays your AgentCfu/AgentCfuFiber copies through the `--mico-agent-cfu` build, so no SoC parameter or CLI flag changes are needed or permitted.\n"
        "- Choose the AgentCfu memory mode through its parameter (`withTilelink`/`withLoad`/`withStore`), which the run passes from the ISA datapath.\n"
        "- Validate with scalar C reference, SoC generation, and a Yosys cost report from the run's generated RTL."
    )


def deterministic_hardware_plan(task: str, contract: str) -> str:
    return (
        "No LLM hardware patch was generated. Hardware plan:\n"
        "1. Implement the datapath in the run workspace copy of AgentCfu.scala (compute/config function ids, Vector RegFile use, CfuLsu memory access); touch AgentCfuFiber.scala only if the fiber contract must change.\n"
        "2. Keep the public contract and the single CPU CFU owner rule; do not edit any other repository source.\n"
        "3. Choose the AgentCfu memory mode through its parameter (`withTilelink`/`withLoad`/`withStore`), which the run passes from the ISA datapath.\n"
        "4. The run generates MiCoSoc.v from the overlay into workspace/soc; a generation error is reported here before any software is written."
    )


def deterministic_software_plan(task: str, contract: str) -> str:
    return (
        "No LLM software patch was generated. Software plan:\n"
        "1. Include the generated cfu_intrinsics.h and call cfu_enable() once before the first CFU instruction.\n"
        "2. Rewrite the hot kernel to use the CFU intrinsics; keep a scalar fallback path so the reference build stays correct.\n"
        "3. The kernel must produce exactly the same observable result as the scalar path.\n"
        "4. Iterate on the intrinsic encoding and data layout based on the compile and simulation errors."
    )


def _toolbox_for_stage(state: CfuDesignState, stage: str) -> RepoToolbox:
    policy = _policy(state)
    return RepoToolbox(
        Path(state.get("workdir", ".")).resolve(),
        dry_run=state.get("dry_run", False),
        policy=policy,
        stage=stage,
        run_root=state.get("run_root", ""),
        embench_benchmark=str(state.get("benchmark_name", "") or ""),
        apply_patch=state.get("apply_patch", False),
    )


def _artifacts_for_stage(state: CfuDesignState, stage: str) -> RunArtifacts:
    return RunArtifacts.create(
        Path(state.get("workdir", ".")).resolve(),
        _policy(state),
        out_dir=state.get("out_dir", "agents/generated"),
        run_id=state.get("run_id", ""),
    )


def _write_profile_artifact(state: CfuDesignState, profile: dict, reason: str) -> None:
    artifacts = _artifacts_for_stage(state, "profile")
    artifacts.write_yaml("profile", "05_profile/result.yaml", dict(profile), status=reason or "produced")


def _write_hotspot_artifact(state: CfuDesignState, hot: str, source: str, candidates: list) -> None:
    artifacts = _artifacts_for_stage(state, "hotspot")
    artifacts.write_yaml(
        "hotspot",
        "06_hotspot/hotspot.yaml",
        {"hot_spot": hot, "source": source, "candidates": candidates},
    )


def _write_analysis_artifact(state: CfuDesignState, analysis: dict) -> None:
    artifacts = _artifacts_for_stage(state, "analysis")
    artifacts.write_yaml("analysis", "07_analysis/workload_analysis.yaml", dict(analysis))


def _write_isa_artifact(state: CfuDesignState, isa_spec: dict, problems: list[str]) -> None:
    artifacts = _artifacts_for_stage(state, "isa")
    artifacts.write_yaml("isa", "08_isa/cfu_isa.yaml", dict(isa_spec), status="invalid" if problems else "produced")
    artifacts.write_text("isa", "08_isa/cfu_isa.md", render_isa_markdown(isa_spec), kind="markdown")
    artifacts.write_yaml("isa", "08_isa/validation.yaml", {"status": "invalid" if problems else "valid", "problems": problems})


def _write_hardware_artifact(
    state: CfuDesignState, plan: str, outcome: dict, feedback: str = ""
) -> None:
    artifacts = _artifacts_for_stage(state, "implementation")
    artifacts.write_text("implementation", "09_implementation/hw_plan.md", plan, kind="markdown")
    if feedback:
        artifacts.write_text(
            "implementation",
            "09_implementation/hw_repair_feedback.txt",
            feedback,
            kind="text",
            status="repair",
        )
    if outcome.get("patch"):
        artifacts.write_text("implementation", "09_implementation/hw_patch.diff", outcome["patch"], kind="text")
    artifacts.write_yaml(
        "implementation",
        "09_implementation/hw_changed_files.yaml",
        {
            "design_files": list(outcome.get("files") or []),
            "patch_applied": bool(outcome.get("applied")),
            "proposed": bool(outcome.get("proposed")),
            "repair": bool(feedback),
            # Whether this repair actually reasoned about the injected error log.
            "repair_diagnosis": ("REPAIR_DIAGNOSIS" in plan) if feedback else None,
            "errors": outcome.get("errors") or [],
        },
    )


def _write_software_artifact(
    state: CfuDesignState, plan: str, outcome: dict, errors: list[str], feedback: str = ""
) -> None:
    artifacts = _artifacts_for_stage(state, "implementation")
    artifacts.write_text("implementation", "09_implementation/sw_plan.md", plan, kind="markdown")
    if feedback:
        artifacts.write_text(
            "implementation",
            "09_implementation/sw_repair_feedback.txt",
            feedback,
            kind="text",
            status="repair",
        )
    if outcome.get("patch"):
        artifacts.write_text("implementation", "09_implementation/sw_patch.diff", outcome["patch"], kind="text")
    artifacts.write_yaml(
        "implementation",
        "09_implementation/sw_changed_files.yaml",
        {
            "software_files": list(outcome.get("files") or []),
            "patch_applied": bool(outcome.get("applied")),
            "proposed": bool(outcome.get("proposed")),
            "repair": bool(feedback),
            # Whether this repair actually reasoned about the injected error log.
            "repair_diagnosis": ("REPAIR_DIAGNOSIS" in plan) if feedback else None,
            "errors": errors,
        },
    )


def _write_ladder_artifact(state: CfuDesignState, rung: str, results: list, status: str, reason: str) -> None:
    artifacts = _artifacts_for_stage(state, "validation")
    artifacts.write_yaml(
        "validation",
        f"10_validation/{rung}.yaml",
        {"rung": rung, "status": status, "reason": reason, "commands": [dict(item) for item in results]},
    )


def _write_cost_artifact(state: CfuDesignState, results: list, status: str, reason: str) -> None:
    artifacts = _artifacts_for_stage(state, "cost")
    artifacts.write_yaml(
        "cost",
        "11_cost/result.yaml",
        {"status": status, "reason": reason, "commands": [dict(item) for item in results]},
    )


def _report_review_decision(state: CfuDesignState, decision: str, status: str) -> None:
    artifacts = _artifacts_for_stage(state, "review")
    artifacts.write_yaml(
        "review",
        "12_review/decision.yaml",
        {"decision": decision, "status": status, "gates": state.get("gates", {})},
    )


def _profile_reason(state: CfuDesignState) -> str:
    if state.get("dry_run", False):
        return "dry-run"
    if state.get("skip_sim", False):
        return "--skip-sim"
    if not state.get("run_commands", False):
        return "execution-disabled"
    return ""


def _validation_reason(state: CfuDesignState) -> str:
    if state.get("dry_run", False):
        return "dry-run"
    if state.get("skip_sim", False):
        return "--skip-sim"
    if not state.get("run_commands", False):
        return "execution-disabled"
    return ""


def _skipped_result(command: str, state: CfuDesignState, reason: str) -> dict:
    return {
        "command": command,
        "cwd": state.get("workdir", "."),
        "returncode": None,
        "skipped": True,
        "reason": reason,
    }


def toolbox_skipped(command: list[str], state: CfuDesignState, reason: str) -> dict:
    return {
        "command": " ".join(command),
        "cwd": state.get("workdir", "."),
        "returncode": None,
        "skipped": True,
        "reason": reason,
    }


def _profile_ok(profile: dict) -> bool:
    return profile.get("status") == "profiled"


def is_agent_cfu_run(state: CfuDesignState) -> bool:
    """True when this run targets the AgentCfu scaffold rather than a legacy CFU."""
    return bool(state.get("agent_cfu", True))


LADDER_RUNGS = ("hw_build", "sw_build", "sim", "cost")

# A CFU design must show a measurable cycle improvement over the scalar
# reference; 105 means at least 1.05x. Without a floor, a kernel that reverts to
# scalar inside the CFU build (or calls an intrinsic but ignores it) would be
# reported as a successful "acceleration" for a run that is effectively scalar.
MIN_EMBENCH_SPEEDUP_X100 = 105
REQUIRED_GATES = (
    "workload_spec",
    "isa",
    "implementation_hw",
    "hw_build",
    "implementation_sw",
    "sw_build",
    "sim",
    "cost",
)


def max_repairs(state: CfuDesignState) -> int:
    """Repair rounds the global ``--max-iters`` budget still allows.

    ``--max-iters 1`` (the "iteration not allowed" case) yields zero repairs, so
    the first failing rung stops the run.
    """
    return max(0, int(state.get("max_iters", 2) or 2) - 1)


def _repairs_left(state: CfuDesignState) -> bool:
    return int(state.get("repairs_used", 0) or 0) < max_repairs(state)


def _ladder_transition(
    state: CfuDesignState,
    rung: str,
    *,
    ok: bool,
    default: str,
    repair_target: str | Any,
    skipped: bool = False,
) -> tuple[dict, str]:
    """Decide a rung's successor and the bookkeeping that goes with it.

    Conditional edges cannot update state, so the failing node records its own
    repair/stop decision (``gates.<rung>.next``) and the router only reads it.
    """
    if ok or skipped:
        return {}, default
    target = repair_target(state) if callable(repair_target) else repair_target
    if _repairs_left(state):
        note = (
            f"{rung} failed; repairing {target} "
            f"(repair {int(state.get('repairs_used', 0) or 0) + 1}/{max_repairs(state)})."
        )
        attempts = dict(state.get("attempts", {}) or {})
        attempts[rung] = int(attempts.get(rung, 0)) + 1
        return {
            "attempts": attempts,
            "repairs_used": int(state.get("repairs_used", 0) or 0) + 1,
            "iteration": int(state.get("iteration", 0) or 0) + 1,
            "last_repair_target": target,
            "stage_jobs": target,
            "errors": [note],
            "iteration_history": append_history(state, note),
        }, target
    note = (
        f"{rung} failed and no repair budget remains "
        f"(--max-iters {state.get('max_iters', 2)}); stopping the run."
    )
    return {
        "stopped_at": rung,
        "stage_jobs": "review",
        "errors": [note],
        "iteration_history": append_history(state, note),
    }, "review_and_route"


def _apply_transition(update: dict, extra: dict) -> dict:
    """Fold a ``_ladder_transition`` payload into a node's state update."""
    extra = dict(extra)
    notes = extra.pop("errors", []) or []
    if notes:
        update["errors"] = [*(update.get("errors") or []), *notes]
    update.update(extra)
    return update


def _rung_gate(state: CfuDesignState, rung: str) -> dict:
    return (state.get("gates", {}) or {}).get(rung, {}) or {}


def _rung_next(state: CfuDesignState, rung: str, default: str) -> str:
    return str(_rung_gate(state, rung).get("next") or default)


def _route_after_isa(state: CfuDesignState) -> str:
    return _rung_next(state, "isa", "implement_hw")


def _route_after_hw_build(state: CfuDesignState) -> str:
    return _rung_next(state, "hw_build", "implement_sw")


def _route_after_sw_build(state: CfuDesignState) -> str:
    return _rung_next(state, "sw_build", "simulate")


def _route_after_sim(state: CfuDesignState) -> str:
    return _rung_next(state, "sim", "estimate_cost")


def _route_after_cost(state: CfuDesignState) -> str:
    return _rung_next(state, "cost", "review_and_route")


def _sim_repair_target(state: CfuDesignState) -> str:
    """Simulation failures repair the software first, then the hardware.

    Escalation must key off *simulation* repairs only. Reading the global
    ``last_repair_target`` would let an earlier ``sw_build`` repair make the very
    first simulation failure jump straight to the hardware, skipping the
    software retry the ladder promises.
    """
    sim_repairs = int((state.get("attempts", {}) or {}).get("sim", 0) or 0)
    return "implement_hw" if sim_repairs > 0 else "implement_sw"


# Static edges of the pure-Python runner. Nodes whose successor depends on a
# gate are listed in PY_ROUTERS instead; ``None`` ends the walk.
PY_SUCCESSORS: dict[str, str | None] = {
    "load_context": "ingest_input",
    "ingest_input": "route_input",
    "prepare_c_project_profile": "instrument_and_profile",
    "create_benchmark_project": "instrument_and_profile",
    "prepare_embench_benchmark": "profile_embench",
    "profile_embench": "extract_hotspot",
    "instrument_and_profile": "extract_hotspot",
    "extract_hotspot": "analyze_workload",
    "analyze_workload": "design_contract",
    "implement_hw": "generate_soc",
    "implement_sw": "build_sw",
    "review_and_route": "final_report",
    "final_report": None,
}

PY_ROUTERS: dict[str, Any] = {
    "route_input": _route_input_target,
    "prepare_c_project_profile": _route_after_prepare,
    "create_benchmark_project": _route_after_prepare,
    "prepare_embench_benchmark": _route_after_prepare,
    "design_contract": _route_after_isa,
    "generate_soc": _route_after_hw_build,
    "build_sw": _route_after_sw_build,
    "simulate": _route_after_sim,
    "estimate_cost": _route_after_cost,
}


def _review_decision(state: CfuDesignState) -> tuple[str, str]:
    gates = state.get("gates", {})
    if isinstance(gates.get(PREPARE_GATE), dict) and gates[PREPARE_GATE].get("ok") is False:
        # Benchmark preparation could not read/copy the workload (for example a
        # C project outside the access policy): the run cannot proceed honestly.
        return "final_report", "failed"
    if state.get("dry_run", False):
        # Dry-run is a planning pass: never claims a measured or complete result.
        return "final_report", "planned"
    if not gates.get("workload_spec", {}).get("ok", False):
        return "final_report", "blocked"
    if not gates.get("isa", {}).get("ok", False):
        return "final_report", "blocked"
    present = [rung for rung in LADDER_RUNGS if isinstance(gates.get(rung), dict)]
    executed = [rung for rung in present if not gates[rung].get("skipped")]
    skipped = [rung for rung in present if gates[rung].get("skipped")]
    # A terminal rung failure always reports failed, even when a later rung was
    # never reached because the ladder stopped early.
    if any(not _rung_gate(state, rung).get("ok", False) for rung in executed):
        return "final_report", "failed"
    if not executed:
        return "final_report", "planned"
    if skipped:
        return "final_report", "planned"
    if not all(gates.get(key, {}).get("ok", False) for key in REQUIRED_GATES):
        return "final_report", "blocked"
    return "final_report", "complete"


def _is_complete(state: CfuDesignState) -> bool:
    gates = state.get("gates", {})
    return all(gates.get(key, {}).get("ok", False) for key in REQUIRED_GATES)


def _yaml_str(value: Any) -> str:
    import yaml as _yaml
    return _yaml.safe_dump(value, sort_keys=False) if isinstance(value, dict) else str(value)


def parse_yaml_mapping(text: str) -> dict | None:
    import yaml as _yaml
    try:
        data = _yaml.safe_load(text)
    except _yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _isa_from_template(template: str, spec: dict, analysis: str) -> dict:
    try:
        template_data = parse_yaml_mapping(template) or {}
    except Exception:
        template_data = {}
    operation = spec.get("operation", "unknown")
    analysis_data = parse_yaml_mapping(analysis) or {}
    roofline = analysis_data.get("roofline")
    if not isinstance(roofline, dict):
        roofline = deterministic_roofline_analysis(spec)
    commands = [
        {"function_id": i, "name": f"op_{i}", "operand": "", "result": "", "semantics": f"{operation} step {i}", "response": "status"}
        for i in range(1, 4)
    ]
    return {
        "spec_version": "1",
        "identity": {
            "name": f"{operation}_cfu",
            "status": "draft",
            "opcode": "custom0 (0x0B)",
            "function_id_width": 3,
            "reference_workload": spec.get("name", operation),
        },
        "commands": commands,
        "operations_contract": {"pattern": spec.get("pattern", "")},
        "roofline": roofline,
        "state": template_data.get("state", {}),
        "memory": {"operator_owned": False, "load": True, "store": False},
        # Keep fallback datapath metadata conservative. These are independent
        # optional knobs; do not invent a vector RF or MAC slice for a scalar CFU.
        "datapath": {
            "vlen": 0,
            "xlen": 0,
            "maclen": 0,
            "accWidth": 0,
            "regDepth": 0,
        },
        "software": {"intrinsics_header": "cfu_intrinsics.h", "scalar_fallback": True},
        "integration": {
            "hardware_component": "AgentCfu",
            "hardware_fiber": "AgentCfuFiber",
            "bus_style": "CfuBus + optional TileLink",
            "soc_cli_flags": ["--mico-agent-cfu"],
            "single_cpu_bus_owner": True,
        },
        "validation": {
            "status": "draft",
            "scalar_reference": True,
            "soc_simulation": False,
            "cycle_profile": True,
            "yosys_cost": True,
            "requirements": [analysis] if analysis and isinstance(analysis, str) else ["Scalar reference must match the CFU output."],
        },
        "assumptions": {},
        "unknowns": spec.get("unknowns", []),
    }


def _has_isaable_commands(isa_spec: dict) -> bool:
    commands = isa_spec.get("commands")
    return isinstance(commands, list) and bool(commands) and all(isinstance(item, dict) for item in commands)


def _isa_problem_note(text: str, fallback: dict) -> list[str]:
    return [f"LLM ISA output did not persist a valid contract; fell back to a template spec."]


def base_artifact_root(state: CfuDesignState) -> str:
    root = state.get("out_dir", "agents/generated").strip("/")
    if root == "agents/generated" or root.startswith("agents/generated/"):
        return root
    return "agents/generated"


def inference_command_id(args: list[str], policy) -> str:
    return infer_command_id(args, policy)


def agent_cfu_overlay_commands(state: CfuDesignState, spec: dict) -> list[list[str]]:
    """Build the SoC generation command that overlays the run workspace design.

    The tracked AgentCfu*.scala scaffolds are excluded and the workspace copies
    are compiled in their place, so an agent-designed CFU reaches the generated
    hardware without any repo file changing.
    """
    if not is_agent_cfu_run(state):
        return []
    settings = agent_cfu_overlay_settings(state)
    if not settings:
        return []
    # MiCoSocGen writes MiCoSoc.v/soc.h into the process working directory, so
    # the forked run is pointed at the run workspace to keep the repo clean.
    set_out_dir = (
        "set run / baseDirectory := (Compile/baseDirectory).value / "
        f'"{state.get("run_root", "")}/{SOC_WORKSPACE_SUBDIR}"'
    )
    flags = agent_cfu_gen_flags(spec)
    # sbt takes each command as one argv element, so the subcommand and its
    # flags are joined into a single element.
    run_main = " ".join(["runMain vexiiriscv.soc.mico.MiCoSocGen", *flags])
    return [["sbt", *settings, set_out_dir, run_main]]


def agent_cfu_overlay_settings(state: CfuDesignState) -> list[str]:
    """sbt settings that compile the run workspace CFU over the tracked scaffold."""
    run_root = state.get("run_root", "")
    if not run_root or not is_agent_cfu_run(state):
        return []
    design_dir = f"{run_root}/{DESIGN_WORKSPACE_SUBDIR}"
    # Add the workspace design dir and drop the tracked scaffolds from the
    # resolved source list. AgentCfuSourceFilter lives in project/ and keeps the
    # logic metacharacter-free, so the command policy check stays meaningful.
    add_overlay = (
        "set Compile/unmanagedSourceDirectories += "
        f'(Compile/baseDirectory).value / "{design_dir}"'
    )
    swap_sources = (
        "set Compile/unmanagedSources := "
        "AgentCfuSourceFilter.sources((Compile/unmanagedSourceDirectories).value)"
    )
    return [add_overlay, swap_sources]


def agent_cfu_cost_commands(state: CfuDesignState, spec: dict) -> list[list[str]]:
    """sbt command that elaborates the standalone CFU Verilog for the cost rung.

    Unlike the SoC build this does not need the CPU: GenAgentCfu elaborates only
    the overlaid AgentCfu class, so Yosys can synthesize the CFU in isolation.
    """
    run_root = state.get("run_root", "")
    if not run_root or not is_agent_cfu_run(state):
        return []
    settings = agent_cfu_overlay_settings(state)
    if not settings:
        return []
    out_dir = f"{run_root}/11_cost/cfu_rtl"
    flags = agent_cfu_datapath_flags(spec)
    run_main = " ".join(
        ["runMain vexiiriscv.soc.mico.GenAgentCfu", "--out-dir", out_dir, *flags]
    )
    return [["sbt", *settings, run_main]]


def agent_cfu_sim_commands(
    state: CfuDesignState,
    spec: dict,
    elf: str = "",
) -> list[list[str]]:
    """Simulation command for the overlaid agent design, when an ELF is known."""
    elf = str(elf or state.get("benchmark_project", {}).get("elf_path", "") or "")
    if not elf:
        return []
    run_root = state.get("run_root", "")
    if not run_root:
        return []
    design_dir = f"{run_root}/{DESIGN_WORKSPACE_SUBDIR}"
    add_overlay = (
        "set Compile/unmanagedSourceDirectories += "
        f'(Compile/baseDirectory).value / "{design_dir}"'
    )
    swap_sources = (
        "set Compile/unmanagedSources := "
        "AgentCfuSourceFilter.sources((Compile/unmanagedSourceDirectories).value)"
    )
    flags = agent_cfu_gen_flags(spec)
    run_main = " ".join(
        ["runMain vexiiriscv.soc.mico.MiCoSocSim", "--load-elf", elf, *flags]
    )
    return [["sbt", add_overlay, swap_sources, run_main]]


def agent_cfu_datapath_flags(spec: dict) -> list[str]:
    """CFU datapath flags (vlen/bus width/reg depth) from the ISA spec."""
    datapath = spec.get("datapath", {}) if isinstance(spec.get("datapath"), dict) else {}
    flags: list[str] = []
    vlen = datapath.get("vlen")
    xlen = datapath.get("xlen")
    reg_depth = datapath.get("regDepth")
    if isinstance(vlen, int) and vlen > 0:
        flags += ["--agent-cfu-vlen", str(vlen)]
    if isinstance(xlen, int) and xlen > 0:
        flags += ["--agent-cfu-bus-width", str(xlen)]
    if isinstance(reg_depth, int) and reg_depth > 0:
        flags += ["--agent-cfu-reg-depth", str(reg_depth)]
    return flags


def agent_cfu_gen_flags(spec: dict) -> list[str]:
    """CLI flags for an agent-CFU SoC build, derived from the ISA datapath."""
    return [
        "--with-rvc",
        "--with-rvm",
        "--with-rdtime",
        "--mico-agent-cfu",
        *agent_cfu_datapath_flags(spec),
    ]


def validation_commands_for_spec(spec: dict, task: str, toolbox: RepoToolbox | None = None) -> tuple[list[list[str]], list[str]]:
    project_commands, errors = project_validation_commands(spec, toolbox)
    if project_commands:
        return project_commands, errors

    operation = str(spec.get("operation", "")).lower()
    lowered = f"{task.lower()} {operation}"
    common = "--with-rvc --with-rvm --with-rdtime"
    if "simd8" in lowered or ("sum" in lowered and "weighted" not in lowered):
        flag = "--mico-simd8-sum-cfu"
        elf = "sw/tests/simd8_sum_cfu_test.elf"
    elif "vadd" in lowered or ("element" in lowered and "add" in lowered):
        flag = "--mico-i8-vadd-cfu"
        elf = "sw/tests/i8_vadd_cfu_test.elf"
    else:
        flag = "--mico-u8-wavg-cfu"
        elf = "sw/tests/u8_wavg_cfu_test.elf"
    return [
        ["sbt", f"runMain vexiiriscv.soc.mico.MiCoSocGen {common} {flag}"],
        ["sbt", f"runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf {elf} {common} {flag}"],
    ], errors


def project_validation_commands(spec: dict, toolbox: RepoToolbox | None) -> tuple[list[list[str]], list[str]]:
    commands: list[list[str]] = []
    errors: list[str] = []
    if spec.get("source_kind") != "c_project":
        return commands, errors
    for key in ("build_cmd", "test_cmd", "run_cmd"):
        raw = str(spec.get(key, "")).strip()
        if not raw:
            continue
        try:
            args = shlex.split(raw)
        except ValueError as exc:
            errors.append(f"{key} could not be parsed: {exc}")
            continue
        if toolbox is not None and not toolbox.command_allowed(args):
            errors.append(f"{key} is not allowlisted and was skipped: {raw}")
            continue
        commands.append(args)
    return commands, errors


def cost_top_for_spec(spec: dict, task: str, *, agent_cfu: bool = False) -> str:
    if agent_cfu:
        return "AgentCfu"
    operation = str(spec.get("operation", "")).lower()
    lowered = f"{task.lower()} {operation}"
    if "simd8" in lowered or ("sum" in lowered and "weighted" not in lowered):
        return "Simd8SumCfu"
    if "vadd" in lowered or ("element" in lowered and "add" in lowered):
        return "I8VAddCfu"
    return "U8WeightedAvgCfu"


def cost_case_for_spec(spec: dict, task: str, *, agent_cfu: bool = False) -> str:
    return cost_top_for_spec(spec, task, agent_cfu=agent_cfu).replace("Cfu", "_cfu").lower()


def validate_render(text: dict) -> str:
    import yaml as _yaml
    return _yaml.safe_dump(text, sort_keys=False)


def render_ladder(state: CfuDesignState) -> str:
    """Per-rung status of the validation ladder."""
    gates = state.get("gates", {}) or {}
    attempts = state.get("attempts", {}) or {}
    lines: list[str] = []
    rungs = ("isa", "implementation_hw", "hw_build", "implementation_sw", "sw_build", "sim", "cost")
    for rung in rungs:
        gate = gates.get(rung)
        if not isinstance(gate, dict):
            lines.append(f"- {rung}: not reached")
            continue
        if gate.get("skipped"):
            status = f"skipped ({gate.get('reason', '')})"
        else:
            status = "pass" if gate.get("ok") else "FAIL"
        repairs = attempts.get(rung, 0)
        suffix = f", repairs={repairs}" if repairs else ""
        lines.append(f"- {rung}: {status}{suffix}")
    stopped = state.get("stopped_at", "")
    if stopped:
        lines.append(f"- stopped at: {stopped}")
    lines.append(
        f"- repairs used: {state.get('repairs_used', 0)}/{max_repairs(state)} "
        f"(iteration={state.get('iteration', 0)})"
    )
    return "\n".join(lines)


def render_report(state: CfuDesignState) -> str:
    lines: list[str] = [
        "# CFU Design Agent Report",
        f"Task: {state.get('task', '')}",
        f"Status: {state.get('status', '')}",
        f"Run root: {state.get('run_root', '')}",
        "",
    ]
    llm_request = state.get("llm_request") or {}
    if llm_request:
        lines.extend([
            "## LLM Request",
            f"- config: {llm_request.get('config', '')}",
            f"- model: {llm_request.get('model', '')}",
            f"- base_url: {llm_request.get('base_url', '')}",
            f"- api_mode: {llm_request.get('api_mode', '')}",
            f"- thinking: {llm_request.get('thinking')}",
            f"- reasoning_effort: {llm_request.get('reasoning_effort')}",
            "",
        ])
    context = state.get("context") or {}
    visibility = str(state.get("visibility") or context.get("visibility") or "")
    if visibility:
        lines.extend([
            "## Visibility",
            f"- level: {visibility}",
            "- scope: read-only file visibility for the agent (writes and command "
            "allowlists are unaffected)",
        ])
        withheld = list(context.get("withheld") or [])
        if withheld:
            lines.append("- withheld from the agent at this level:")
            lines.extend(f"  - {item}" for item in withheld)
        lines.append("")
    lines.append("## Gates")
    for key, value in (state.get("gates") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Validation Ladder")
    lines.append(render_ladder(state))
    lines.append("")
    lines.append("## Normalized WorkloadSpec")
    lines.append("```yaml")
    lines.append(state.get("spec_text", "").strip())
    lines.append("```")
    lines.append("")
    lines.append("## Benchmark Project")
    lines.append(format_mapping(state.get("benchmark_project", {})))
    lines.append("")
    lines.append("## Profile Result")
    lines.append(format_mapping(state.get("profile_result", {})))
    lines.append("")
    lines.append("## Workload Analysis")
    lines.append(str(state.get("workload_analysis", "")))
    lines.append("")
    lines.append("## CFU ISA Spec")
    if state.get("isa_spec"):
        import yaml as _yaml
        lines.append("```yaml")
        lines.append(_yaml.safe_dump(state["isa_spec"], sort_keys=False).strip())
        lines.append("```")
    else:
        lines.append("No ISA spec was produced.")
    if state.get("isa_spec_errors"):
        lines.append("### ISA errors")
        lines.append("\n".join(f"- {err}" for err in state["isa_spec_errors"]))
    lines.append("")
    lines.append("## Hardware Design")
    lines.append(str(state.get("implementation_hw_summary", "")))
    lines.append("")
    lines.append("## Software Design")
    lines.append(str(state.get("implementation_sw_summary", "")))
    lines.append("")
    lines.append("## Hardware Build (SoC generation)")
    lines.append(format_rung_results(state, "hw_build"))
    lines.append("")
    lines.append("## Software Build (compile)")
    lines.append(format_rung_results(state, "sw_build"))
    lines.append("")
    lines.append("## Simulation")
    lines.append(format_rung_results(state, "sim"))
    lines.append("")
    lines.append("## Hardware Cost")
    lines.append(format_results(state.get("cost_results", [])))
    lines.append("")
    lines.append("## Tool Calls")
    tool_calls = state.get("tool_calls") or []
    if tool_calls:
        for call in tool_calls:
            lines.append(
                f"- round {call.get('round')}: `{call.get('tool')}`"
                f"({call.get('arguments')}) ok={call.get('ok')}"
                f" result_chars={call.get('result_chars')}"
            )
    else:
        lines.append("No agent tool calls were recorded for this run.")
    lines.append("")
    lines.append("## LLM Raw Outputs")
    raw_outputs = state.get("llm_raw_outputs", [])
    if raw_outputs:
        lines.extend(f"- `{path}`" for path in raw_outputs)
    else:
        lines.append("No LLM calls were recorded for this run.")
    lines.append("")
    lines.append("## Token Usage")
    usage = usage_totals()
    lines.append(f"- llm_calls: {usage['calls']}")
    lines.append(f"- input tokens: {usage['input']}")
    lines.append(f"- output tokens: {usage['output']}")
    lines.append(f"- total tokens: {usage['total']}")
    lines.append("")
    lines.append("## Iteration History")
    lines.append("\n".join(f"- {line}" for line in state.get("iteration_history", [])))
    lines.append("")
    lines.append("## Errors")
    lines.append("\n".join(f"- {err}" for err in state.get("errors", [])) or "None")
    return "\n".join(lines) + "\n"


def format_rung_results(state: CfuDesignState, rung: str) -> str:
    results = [r for r in (state.get("validation_results") or []) if r.get("rung") == rung]
    return format_results(results)


def format_results(results: list[dict]) -> str:
    if not results:
        return "No results."
    lines: list[str] = []
    for result in results:
        lines.append(f"- `{result.get('command', '')}` rc={result.get('returncode')}")
        if result.get("skipped"):
            lines.append(f"  skipped: {result.get('reason', '')}")
        parsed = result.get("parsed")
        if parsed:
            lines.append(f"  parsed: {parsed}")
        stdout_tail = result.get("stdout_tail", "").strip()
        if stdout_tail:
            lines.append("  stdout tail:")
            lines.append("  ```text")
            lines.append(stdout_tail[-1200:])
            lines.append("  ```")
    return "\n".join(lines)


def format_mapping(value: dict) -> str:
    if not value:
        return "No data."
    lines = []
    for key, item in value.items():
        lines.append(f"- {key}: {item}")
    return "\n".join(lines)
