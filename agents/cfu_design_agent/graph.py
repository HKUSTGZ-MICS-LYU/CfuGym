"""LangGraph workflow for CFU design iterations.

A pure-Python sequential runner is provided as ``run_agent_python`` so the
framework stays usable and testable even when the ``langgraph`` dependency is
not installed. ``build_graph`` and ``run_agent`` remain available for the
LangGraph-backed path.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
import shlex
import time
from typing import Any, Literal

from .artifacts import RunArtifacts
from .benchmarks import (
    prepare_c_project_profile as prepare_c_project_profile_workspace,
    prepare_generated_benchmark,
    profile_commands,
    profile_result_from_commands,
)
from .isa import IsaLintError, render_isa_markdown, validate_isa_text
from .llm import OpenAiTextClient, reset_usage_log, usage_totals
from .llm_config import effective_config, LlmConfigError
from .policy import PolicyError, load_policy, normalize_repo_path
from .progress import StageProgress
from .prompts import SYSTEM_PROMPT, contract_prompt, implementation_prompt, workload_prompt
from .spec import normalize_workload_spec, validate_workload_spec
from .state import CfuDesignState, append_history
from .tools import DEFAULT_CONTEXT_FILES, RepoToolbox, extract_patch, infer_command_id

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
    "instrument_and_profile": "profile",
    "extract_hotspot": "hotspot",
    "design_contract": "isa",
    "implement_hw_sw": "implementation",
    "validate": "validation",
    "estimate_cost": "cost",
    "review_and_route": "review",
    "review_and_route_python": "review",
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
        if _PROGRESS is None:
            return fn(state)
        name = fn.__name__
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


def build_graph(*, checkpointer: Any = None, debug: bool = False):  # type: ignore[misc]
    if not _LANGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed; use run_agent_python()")
    graph = StateGraph(CfuDesignState)
    graph.add_node("load_context", _wrap_node(load_context), retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node("ingest_input", _wrap_node(ingest_input))
    graph.add_node(
        "route_input",
        _wrap_node(route_input),
        destinations=("prepare_c_project_profile", "create_benchmark_project"),
    )
    graph.add_node("prepare_c_project_profile", _wrap_node(prepare_c_project_profile_node))
    graph.add_node("create_benchmark_project", _wrap_node(create_benchmark_project_node))
    graph.add_node("instrument_and_profile", _wrap_node(instrument_and_profile))
    graph.add_node("extract_hotspot", _wrap_node(extract_hotspot))
    graph.add_node("analyze_workload", _wrap_node(analyze_workload))
    graph.add_node("design_contract", _wrap_node(design_contract))
    graph.add_node("implement_hw_sw", _wrap_node(implement_hw_sw))
    graph.add_node("validate", _wrap_node(validate))
    graph.add_node("estimate_cost", _wrap_node(estimate_cost))
    graph.add_node("review_and_route", _wrap_node(review_and_route), destinations=("implement_hw_sw", "final_report"))
    graph.add_node("final_report", _wrap_node(final_report))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "ingest_input")
    graph.add_edge("ingest_input", "route_input")
    graph.add_edge("prepare_c_project_profile", "instrument_and_profile")
    graph.add_edge("create_benchmark_project", "instrument_and_profile")
    graph.add_edge("instrument_and_profile", "extract_hotspot")
    graph.add_edge("extract_hotspot", "analyze_workload")
    graph.add_edge("analyze_workload", "design_contract")
    graph.add_edge("design_contract", "implement_hw_sw")
    graph.add_edge("implement_hw_sw", "validate")
    graph.add_edge("validate", "estimate_cost")
    graph.add_edge("estimate_cost", "review_and_route")
    graph.add_edge("final_report", END)
    return graph.compile(checkpointer=checkpointer, debug=debug, name="cfu_design_agent")


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
    policy = load_policy(workdir)
    artifacts = RunArtifacts.create(
        workdir,
        policy,
        out_dir=state.get("out_dir", "agents/generated"),
        run_id=state.get("run_id", ""),
    )
    # Seed run identity so every node reuses this run's artifact directory.
    state["run_id"] = artifacts.run_id
    state["run_root"] = artifacts.root_rel

    # Node returns are partial updates; merge them like LangGraph state does.
    for node in (load_context, ingest_input, route_input_python):
        state.update(_wrapped(node)(state))
    prepare = prepare_c_project_profile_node if state.get("input_kind") == "c_project" else create_benchmark_project_node
    for node in (
        prepare,
        instrument_and_profile,
        extract_hotspot,
        analyze_workload,
        design_contract,
        implement_hw_sw,
        validate,
        estimate_cost,
        review_and_route_python,
        final_report,
    ):
        state.update(_wrapped(node)(state))

    state["run_id"] = artifacts.run_id
    state["run_root"] = artifacts.root_rel
    return _finalize(state)


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
            "       | goto prepare_c_project_profile/create_benchmark_project",
            "  -> instrument_and_profile",
            "  -> extract_hotspot",
            "  -> analyze_workload",
            "  -> design_contract (ISA spec)",
            "  -> implement_hw_sw",
            "  -> validate",
            "  -> estimate_cost",
            "  -> review_and_route",
            "       | goto implement_hw_sw/final_report",
            "  -> END",
            "",
        ]
    )


def graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": (thread_id or "cfu-design-agent")[:255]}}


def _policy(state: CfuDesignState):
    return load_policy(Path(state.get("workdir", ".")).resolve())


def _toolbox(state: CfuDesignState, policy=None, artifacts: RunArtifacts | None = None) -> RepoToolbox:
    policy = policy or _policy(state)
    return RepoToolbox(
        Path(state.get("workdir", ".")).resolve(),
        dry_run=state.get("dry_run", False),
        policy=policy,
        stage=stage_name(state),
        run_root=state.get("run_root", ""),
        apply_patch=state.get("apply_patch", False),
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


def stage_name(state: CfuDesignState) -> str:
    return "load_context"


def load_context(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = load_policy(workdir)
    artifacts = _artifacts(state, policy)
    toolbox = _toolbox_for_stage(state, "load_context")

    context_sources = ["agents/specs/cfu_isa_template.yaml"]
    if state.get("kernel_file"):
        context_sources.append(state["kernel_file"])
    if state.get("workload_spec_path"):
        context_sources.append(state["workload_spec_path"])
    files: dict[str, str] = {}
    denied: list[str] = []
    for path in [*DEFAULT_CONTEXT_FILES, *context_sources]:
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

    manifest = {
        "visible_files": sorted(files),
        "denied_files": denied,
        "git_status": toolbox.git_status(),
    }
    artifacts.write_yaml("load_context", "01_context/manifest.yaml", manifest, status="produced")
    return {
        "context": {"files": files, "git_status": manifest["git_status"]},
        "run_id": artifacts.run_id,
        "run_root": artifacts.root_rel,
        "status": "initialized",
        "stage_jobs": "ingest_input",
        "gates": {**state.get("gates", {}), "load_context": {"ok": not denied, "denied": denied}},
        "iteration_history": append_history(state, "Loaded repo context and CFU design references."),
    }


def ingest_input(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = load_policy(workdir)
    artifacts = _artifacts(state, policy)
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
        llm=_llm(state),
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
        "stage_jobs": "route_input",
        "status": "blocked" if problems else "initialized",
        "gates": {**state.get("gates", {}), "workload_spec": {"ok": not problems, "problems": problems}},
        "iteration_history": append_history(state, f"Normalized input into WorkloadSpec `{spec.get('name', 'cfu_workload')}`."),
    }


def route_input(
    state: CfuDesignState,
) -> Command[Literal["prepare_c_project_profile", "create_benchmark_project"]]:
    kind, goal = _route_kind(state)
    _artifacts_for_stage(state, "route_input").write_yaml(
        "route_input",
        "03_route/route.yaml",
        {"input_kind": kind, "description": "input classification result"},
    )
    return Command(
        update={
            "input_kind": kind,
            "stage_jobs": "prepare_benchmark",
            "iteration_history": append_history(state, f"Selected input route `{kind}`."),
        },
        goto=goal,
    )


def route_input_python(state: CfuDesignState) -> CfuDesignState:
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


def _route_kind(state: CfuDesignState) -> tuple[str, str]:
    kind = state.get("input_kind") or state.get("input_mode", "natural_language")
    if kind not in ("c_project", "natural_language"):
        kind = "natural_language"
    return kind, ("prepare_c_project_profile" if kind == "c_project" else "create_benchmark_project")


def prepare_c_project_profile_node(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = load_policy(workdir)
    artifacts = _artifacts(state, policy)
    workspace = f"{artifacts.root_rel}/workspace"
    spec = dict(state.get("workload_spec", {}))
    spec["project_root"] = state.get("project_root", spec.get("project_root", ""))
    spec["kernel_file"] = state.get("kernel_file", spec.get("kernel_file", ""))
    project = prepare_c_project_profile_workspace(
        _toolbox_for_stage(state, "prepare_benchmark"),
        spec,
    )
    artifacts.write_yaml(
        "prepare_benchmark",
        "04_benchmark/manifest.yaml",
        dict(project),
        status="planned" if state.get("dry_run") else "produced",
    )
    return {
        "benchmark_project": project,
        "stage_jobs": "profile",
        "iteration_history": append_history(state, f"Prepared C-project profiling workspace `{project.get('workspace', '')}`."),
    }


def create_benchmark_project_node(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = load_policy(workdir)
    artifacts = _artifacts(state, policy)
    project = prepare_generated_benchmark(_toolbox_for_stage(state, "prepare_benchmark"), state.get("workload_spec", {}))
    artifacts.write_yaml(
        "prepare_benchmark",
        "04_benchmark/manifest.yaml",
        dict(project),
        status="planned" if state.get("dry_run") else "produced",
    )
    return {
        "benchmark_project": project,
        "stage_jobs": "profile",
        "iteration_history": append_history(state, f"Prepared generated benchmark workspace `{project.get('workspace', '')}`."),
    }


def instrument_and_profile(state: CfuDesignState) -> CfuDesignState:
    project = state.get("benchmark_project", {})
    commands = profile_commands(project)
    reason = _profile_reason(state)
    execute = state.get("run_commands", False) and not state.get("dry_run", False) and not state.get("skip_sim", False)
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
        errors: list[str] = []
        for command in commands:
            command_id = infer_command_id(command, toolbox.policy)
            if not toolbox.command_allowed(command, command_id=command_id):
                errors.append(f"profile command is not allowlisted and was skipped: {' '.join(command)}")
                continue
            result = toolbox.run_command(command, command_id=command_id, timeout=3600, mutate=True, execute=True)
            results.append(result)
            if result.get("returncode") not in (0, None):
                break
        profile = profile_result_from_commands(results, dry_run=False)
        errors = errors
    _write_profile_artifact(state, profile, reason)
    return {
        "profile_result": profile,
        "stage_jobs": "hotspot",
        "gates": {**state.get("gates", {}), "profile": {"ok": _profile_ok(profile)}},
        "errors": state.get("errors", []),
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
    }
    result = _llm(state).complete(
        SYSTEM_PROMPT,
        workload_prompt(task, state.get("workload_spec", {}), state.get("context", {})),
        fallback=_yaml_str(fallback),
    )
    analysis = parse_yaml_mapping(result.text) or fallback
    note = "Analyzed workload with LLM." if result.used_llm else "Analyzed workload with deterministic fallback."
    _write_analysis_artifact(state, analysis)
    return {
        "workload_analysis": _yaml_str(analysis),
        "stage_jobs": "isa",
        "iteration_history": append_history(state, note),
    }


def design_contract(state: CfuDesignState) -> CfuDesignState:
    spec = state.get("workload_spec", {})
    fallback = _isa_from_template(state.get("isa_spec_text", ""), spec, state.get("workload_analysis", ""))
    result = _llm(state).complete(
        SYSTEM_PROMPT,
        contract_prompt(
            state["task"],
            spec,
            state.get("workload_analysis", ""),
            state.get("context", {}),
        ),
        fallback=_yaml_str(fallback),
    )
    try:
        isa_spec, problems = validate_isa_text(result.text)
    except IsaLintError:
        isa_spec, problems = fallback, _isa_problem_note(result.text, fallback)
    if not problems and not _has_isaable_commands(isa_spec):
        isa_spec, problems = fallback, ["no usable commands after ISA extraction"]
    _write_isa_artifact(state, isa_spec, problems)
    return {
        "isa_spec": isa_spec,
        "isa_spec_text": _yaml_str(isa_spec),
        "isa_spec_errors": problems,
        "design_contract": result.text,
        "stage_jobs": "implementation",
        "gates": {**state.get("gates", {}), "isa": {"ok": not problems, "problems": problems}},
        "iteration_history": append_history(state, "Produced CFU ISA/integration contract."),
    }


def implement_hw_sw(state: CfuDesignState) -> CfuDesignState:
    result = _llm(state).complete(
        SYSTEM_PROMPT,
        implementation_prompt(
            state["task"],
            state.get("workload_spec", {}),
            state.get("workload_analysis", ""),
            state.get("design_contract", ""),
            state.get("context", {}),
        ),
        fallback=deterministic_implementation_plan(state["task"], state.get("design_contract", "")),
    )
    patch_text = extract_patch(result.text)
    summary = result.text
    errors: list[str] = list(state.get("errors", []))
    apply_result = None
    if patch_text:
        try:
            apply_result = _toolbox_for_stage(state, "implementation").apply_patch_text(patch_text)
            summary += "\n\nPatch result:\n" + str(apply_result)
        except PolicyError as exc:
            errors = [*errors, f"patch failed: {exc}"]
    elif not state.get("dry_run", False):
        summary += "\n\nNo patch block was produced; implementation remains a design proposal."
    _write_implementation_artifact(state, result.text, patch_text, apply_result, errors)
    return {
        "implementation_plan": result.text,
        "patch_text": patch_text,
        "implementation_summary": summary,
        "stage_jobs": "validation",
        "errors": errors,
        "gates": {**state.get("gates", {}), "implementation": {"ok": bool(patch_text) or state.get("dry_run"), "patch": bool(patch_text)}},
        "iteration_history": append_history(state, "Completed implementation/proposal node."),
    }


def validate(state: CfuDesignState) -> CfuDesignState:
    with_context = state.get("workload_spec", {})
    if state.get("skip_sim", False) or state.get("dry_run", False) or not state.get("run_commands", False):
        reason = _validation_reason(state)
        results = [_skipped_result("sbt MiCoSocGen/MiCoSocSim", state, reason)]
        profile_status = "planned" if reason else "skipped"
        _write_validation_artifact(state, results, profile_status, reason)
        return {
            "validation_results": results,
            "stage_jobs": "cost",
            "gates": {**state.get("gates", {}), "validation": {"ok": False, "reason": reason}},
            "iteration_history": append_history(state, f"Skipped validation commands ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "validation")
    commands, command_errors = validation_commands_for_spec(with_context, state.get("task", ""), toolbox)
    results = []
    errors: list[str] = list(command_errors)
    for command in commands:
        command_id = infer_command_id(command, toolbox.policy)
        if not toolbox.command_allowed(command, command_id=command_id):
            errors.append(f"validation command is not allowlisted and was skipped: {' '.join(command)}")
            continue
        results.append(toolbox.run_command(command, command_id=command_id, timeout=3600, mutate=False, execute=True))
        if results[-1]["returncode"] not in (0, None):
            break
    ok = all(r.get("returncode") == 0 for r in results) and not errors
    _write_validation_artifact(state, results, "passed" if ok else "failed", "")
    return {
        "validation_results": results,
        "stage_jobs": "cost",
        "errors": errors,
        "gates": {**state.get("gates", {}), "validation": {"ok": ok, "results": len(results)}},
        "iteration_history": append_history(state, "Ran generation/simulation validation."),
    }


def estimate_cost(state: CfuDesignState) -> CfuDesignState:
    if state.get("skip_yosys", False):
        result = [_skipped_result("python3 skills/cfu-designer/scripts/yosys_cost_report.py", state, "--skip-yosys")]
        _write_cost_artifact(state, result, "planned", "--skip-yosys")
        return {
            "cost_results": result,
            "stage_jobs": "review",
            "gates": {**state.get("gates", {}), "cost": {"ok": False, "reason": "--skip-yosys"}},
            "iteration_history": append_history(state, "Skipped Yosys cost estimation by request."),
        }

    toolbox = _toolbox_for_stage(state, "cost")
    out_dir = f"{base_artifact_root(state)}/yosys/{cost_case_for_spec(state.get('workload_spec', {}), state.get('task', ''))}"
    command = [
        "python3",
        "skills/cfu-designer/scripts/yosys_cost_report.py",
        "MiCoSoc.v",
        "--top",
        cost_top_for_spec(state.get("workload_spec", {}), state.get("task", "")),
        "--out-dir",
        out_dir,
        "--flatten",
    ]
    if not state.get("run_commands", False):
        result = [toolbox.run_command(command, command_id="yosys_cost", execute=False)]
    else:
        result = [toolbox.run_command(command, command_id="yosys_cost", timeout=1800, mutate=True, execute=True)]
    ok = all(item.get("returncode") == 0 for item in result)
    _write_cost_artifact(state, result, "passed" if ok else "failed", "")
    return {
        "cost_results": result,
        "stage_jobs": "review",
        "gates": {**state.get("gates", {}), "cost": {"ok": ok}},
        "iteration_history": append_history(state, "Ran Yosys cost estimation."),
    }


def review_and_route(state: CfuDesignState) -> Command[Literal["implement_hw_sw", "final_report"]]:
    decision, status = _review_decision(state)
    _report_review_decision(state, decision, status)
    return Command(
        update={
            "iteration": state.get("iteration", 0) + 1,
            "status": status,
            "stage_jobs": "final_report",
            "iteration_history": append_history(state, f"Reviewed iteration {state.get('iteration', 0) + 1}: {status}."),
        },
        goto="implement_hw_sw" if status == "needs_iteration" else "final_report",
    )


def review_and_route_python(state: CfuDesignState) -> CfuDesignState:
    decision, status = _review_decision(state)
    _report_review_decision(state, decision, status)
    return {
        "iteration": state.get("iteration", 0) + 1,
        "status": status,
        "stage_jobs": "review",
        "gates": {**state.get("gates", {}), "review": {"decision": decision, "status": status}},
        "iteration_history": append_history(state, f"Reviewed iteration {state.get('iteration', 0) + 1}: {status}."),
    }


def final_report(state: CfuDesignState) -> CfuDesignState:
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = load_policy(workdir)
    artifacts = _artifacts(state, policy)
    report = render_report(state)
    artifacts.write_text("review", "report.md", report, kind="markdown", status="complete")
    complete = _is_complete(state)
    if state.get("dry_run", False):
        status = "planned"
    else:
        status = "complete" if complete else "blocked"
    return {
        "final_report_path": str(artifacts.path("report.md")),
        "run_root": artifacts.root_rel,
        "status": status,
        "stage_jobs": "final_report",
        "iteration_history": append_history(state, "Wrote final report."),
    }


def deterministic_analysis(task: str, spec: dict) -> str:
    pattern = spec.get("pattern") or "unknown"
    operation = spec.get("operation") or "unknown"
    return (
        f"Operation: {operation}.\n"
        f"Pattern: {pattern}.\n"
        f"Selected hot spot: {spec.get('hot_spot', '') or 'unknown'}.\n"
        f"Known element type: {spec.get('element_type', '') or 'unknown'}.\n"
        f"Unknowns: {', '.join(spec.get('unknowns', [])) or 'none'}.\n"
        "Likely CFU shape: use DirectCfuSpec for register-sized packed work; use TilelinkCfuSpec plus CfuLsu and optional VPU-style RF for memory-backed vectors or multi-operation reuse."
    )


def deterministic_contract(task: str, spec: dict, analysis: str) -> str:
    operation = spec.get("operation", "")
    if operation in {"weighted_average", "dot_product"}:
        op_line = "For weighted average/dot product, load input vectors into RF slots, multiply int8 lanes in `maclen` slices, widen locally, and accumulate into a 32-bit scalar."
    elif operation == "elementwise_add":
        op_line = "For element-wise add, load source vectors into CFU RF slots, add packed int8 lanes, and store the packed result through the shared LSU when the output must escape."
    else:
        op_line = "Select the smallest direct or TileLink CFU contract after confirming the hot-loop operation and data lifetime."
    return (
        "Proposed contract:\n"
        "- Use custom0 function IDs documented beside the C intrinsics.\n"
        "- Keep reset/config/load/compute/read commands explicit.\n"
        f"- {op_line}\n"
        "- Integrate through `TilelinkCfuSpec` when memory is owned by the CFU; set `vexii.withCfu = true` and align `lsuMemDataWidthMin` with the CFU bus width.\n"
        "- Validate with scalar C reference, SoC sim profile line, and Yosys cost report from generated RTL."
    )


def deterministic_implementation_plan(task: str, contract: str) -> str:
    return (
        "No LLM patch was generated. Implementation plan:\n"
        "1. Add or update the Scala CFU component using CfuBus plus DirectCfuSpec/TilelinkCfuSpec.\n"
        "2. Add SoC parameters and CLI flags in MiCoSocParam, then instantiate through the shared CFU container.\n"
        "3. Add C intrinsics and a scalar-vs-CFU test under sw/tests.\n"
        "4. Run MiCoSocGen, build the ELF, run MiCoSocSim, parse cycle profile lines, then run Yosys cost on generated RTL.\n"
        "5. Iterate on vector width, bus width, RF backend, burst mode, and pipeline knobs based on correctness, speed, and cells."
    )


def _toolbox_for_stage(state: CfuDesignState, stage: str) -> RepoToolbox:
    policy = load_policy(Path(state.get("workdir", ".")).resolve())
    return RepoToolbox(
        Path(state.get("workdir", ".")).resolve(),
        dry_run=state.get("dry_run", False),
        policy=policy,
        stage=stage,
        run_root=state.get("run_root", ""),
        apply_patch=state.get("apply_patch", False),
    )


def _artifacts_for_stage(state: CfuDesignState, stage: str) -> RunArtifacts:
    return RunArtifacts.create(
        Path(state.get("workdir", ".")).resolve(),
        load_policy(Path(state.get("workdir", ".")).resolve()),
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


def _write_implementation_artifact(state: CfuDesignState, plan: str, patch: str, apply_result: dict, errors: list[str]) -> None:
    artifacts = _artifacts_for_stage(state, "implementation")
    artifacts.write_text("implementation", "09_implementation/plan.md", plan, kind="markdown")
    if patch:
        artifacts.write_text("implementation", "09_implementation/patch.diff", patch, kind="text")
    changed = patch and apply_result.get("parsed", {}).get("touched_paths", []) or []
    artifacts.write_yaml(
        "implementation",
        "09_implementation/changed_files.yaml",
        {"touched_paths": changed, "patch_applied": bool(apply_result and apply_result.get("skipped") is not True), "errors": errors},
    )


def _write_validation_artifact(state: CfuDesignState, results: list, status: str, reason: str) -> None:
    artifacts = _artifacts_for_stage(state, "validation")
    artifacts.write_yaml(
        "validation",
        "10_validation/result.yaml",
        {"status": status, "reason": reason, "commands": [dict(item) for item in results]},
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


def _review_decision(state: CfuDesignState) -> tuple[str, str]:
    gates = state.get("gates", {})
    if state.get("dry_run", False):
        # Dry-run is a planning pass: never claims a measured or complete result.
        return "final_report", "planned"
    workload_ok = gates.get("workload_spec", {}).get("ok", False)
    isa_ok = gates.get("isa", {}).get("ok", False)
    validation = gates.get("validation", {})
    cost = gates.get("cost", {})
    iteration = state.get("iteration", 0) + 1
    max_iters = state.get("max_iters", 2)
    if not workload_ok:
        return "final_report", "blocked"
    if not isa_ok:
        return "final_report", "blocked"
    if not validation.get("ok", False) or not cost.get("ok", False):
        if iteration < max_iters:
            return "implement_hw_sw", "needs_iteration"
        return "final_report", "failed"
    return "final_report", "complete"


def _is_complete(state: CfuDesignState) -> bool:
    gates = state.get("gates", {})
    keys = ("workload_spec", "isa", "validation", "cost")
    return all(gates.get(key, {}).get("ok", False) for key in keys)


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
        "state": template_data.get("state", {}),
        "memory": {"operator_owned": False, "load": True, "store": False},
        "datapath": {
            "vlen": spec.get("vector_bytes", 0) * 8,
            "xlen": 32,
            "maclen": 32,
            "accWidth": 32,
            "regDepth": 2,
        },
        "software": {"intrinsics_header": "", "scalar_fallback": True},
        "integration": {"bus_style": "TilelinkCfuSpec", "single_cpu_bus_owner": True},
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


def cost_top_for_spec(spec: dict, task: str) -> str:
    operation = str(spec.get("operation", "")).lower()
    lowered = f"{task.lower()} {operation}"
    if "simd8" in lowered or ("sum" in lowered and "weighted" not in lowered):
        return "Simd8SumCfu"
    if "vadd" in lowered or ("element" in lowered and "add" in lowered):
        return "I8VAddCfu"
    return "U8WeightedAvgCfu"


def cost_case_for_spec(spec: dict, task: str) -> str:
    return cost_top_for_spec(spec, task).replace("Cfu", "_cfu").lower()


def validate_render(text: dict) -> str:
    import yaml as _yaml
    return _yaml.safe_dump(text, sort_keys=False)


def render_report(state: CfuDesignState) -> str:
    lines: list[str] = [
        "# CFU Design Agent Report",
        f"Task: {state.get('task', '')}",
        f"Status: {state.get('status', '')}",
        f"Run root: {state.get('run_root', '')}",
        "",
        "## Gates",
    ]
    for key, value in (state.get("gates") or {}).items():
        lines.append(f"- {key}: {value}")
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
    lines.append("## Implementation")
    lines.append(str(state.get("implementation_summary", "")))
    lines.append("")
    lines.append("## Validation")
    lines.append(format_results(state.get("validation_results", [])))
    lines.append("")
    lines.append("## Hardware Cost")
    lines.append(format_results(state.get("cost_results", [])))
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
