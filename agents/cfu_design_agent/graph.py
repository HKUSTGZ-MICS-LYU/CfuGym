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
from .embench import (
    build_command as embench_build_command,
    config_sources,
    config_workspace as embench_config_workspace,
    elf_path as embench_elf_path,
    hotspot_from_profile,
    parse_profile as parse_embench_profile,
    prepare_embench_benchmark,
    workspace_rel as embench_workspace,
)
from .embench_instrument import instrument_sources
from .isa import IsaLintError, render_cfu_intrinsics, render_isa_markdown, validate_isa_text
from .llm import OpenAiTextClient, reset_usage_log, set_stage, usage_totals
from .llm_config import effective_config, LlmConfigError
from .policy import PolicyError, load_policy, normalize_repo_path
from .progress import StageProgress
from .prompts import SYSTEM_PROMPT, contract_prompt, implementation_prompt, workload_prompt
from .spec import normalize_workload_spec, validate_workload_spec
from .state import CfuDesignState, append_history
from .tools import (
    DEFAULT_CONTEXT_FILES,
    DESIGN_SOURCE_FILES,
    DESIGN_WORKSPACE_SUBDIR,
    SOC_WORKSPACE_SUBDIR,
    RepoToolbox,
    design_workspace_paths,
    extract_patch,
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
    "instrument_and_profile": "profile",
    "profile_embench": "profile",
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


def build_graph(*, checkpointer: Any = None, debug: bool = False):  # type: ignore[misc]
    if not _LANGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed; use run_agent_python()")
    graph = StateGraph(CfuDesignState)
    graph.add_node("load_context", _wrap_node(load_context), retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node("ingest_input", _wrap_node(ingest_input))
    graph.add_node(
        "route_input",
        _wrap_node(route_input),
        destinations=("prepare_c_project_profile", "create_benchmark_project", "prepare_embench_benchmark"),
    )
    graph.add_node("prepare_c_project_profile", _wrap_node(prepare_c_project_profile_node))
    graph.add_node("create_benchmark_project", _wrap_node(create_benchmark_project_node))
    graph.add_node("prepare_embench_benchmark", _wrap_node(prepare_embench_benchmark_node))
    graph.add_node("profile_embench", _wrap_node(profile_embench))
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
    graph.add_edge("prepare_embench_benchmark", "profile_embench")
    graph.add_edge("profile_embench", "extract_hotspot")
    graph.add_edge("instrument_and_profile", "extract_hotspot")
    graph.add_edge("extract_hotspot", "analyze_workload")
    graph.add_edge("analyze_workload", "design_contract")
    graph.add_edge("design_contract", "implement_hw_sw")
    graph.add_edge("implement_hw_sw", "validate")
    graph.add_edge("validate", "estimate_cost")
    graph.add_edge("estimate_cost", "review_and_route")
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

    # Node returns are partial updates; merge them like LangGraph state does,
    # including operator.add semantics for the append-annotated keys.
    for node in (load_context, ingest_input, route_input_python):
        _merge_state_update(state, _wrapped(node)(state))
    if state.get("input_kind") == "embench":
        prepare = prepare_embench_benchmark_node
        profile_node = profile_embench
    elif state.get("input_kind") == "c_project":
        prepare = prepare_c_project_profile_node
        profile_node = instrument_and_profile
    else:
        prepare = create_benchmark_project_node
        profile_node = instrument_and_profile
    for node in (
        prepare,
        profile_node,
        extract_hotspot,
        analyze_workload,
        design_contract,
        implement_hw_sw,
        validate,
        estimate_cost,
        review_and_route_python,
        final_report,
    ):
        _merge_state_update(state, _wrapped(node)(state))

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
        "visible_files": sorted(files),
        "denied_files": denied,
        "git_status": toolbox.git_status(),
        "design_workspace": {
            "root": f"{artifacts.root_rel}/{DESIGN_WORKSPACE_SUBDIR}",
            "files": design_files,
            "seeded_from": list(DESIGN_SOURCE_FILES),
        },
    }
    artifacts.write_yaml("load_context", "01_context/manifest.yaml", manifest, status="produced")
    return {
        "context": {"files": files, "git_status": manifest["git_status"]},
        "run_id": artifacts.run_id,
        "run_root": artifacts.root_rel,
        "design_workspace": manifest["design_workspace"],
        "design_files": design_files,
        "status": "initialized",
        "stage_jobs": "ingest_input",
        "gates": {**state.get("gates", {}), "load_context": {"ok": not denied, "denied": denied}},
        "iteration_history": append_history(state, "Loaded repo context and seeded the isolated CFU design workspace."),
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
        benchmark_name=state.get("benchmark_name", ""),
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
) -> Command[Literal["prepare_c_project_profile", "create_benchmark_project", "prepare_embench_benchmark"]]:
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
    if kind not in ("c_project", "natural_language", "embench"):
        kind = "natural_language"
    if kind == "embench":
        return kind, "prepare_embench_benchmark"
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

def prepare_embench_benchmark_node(state: CfuDesignState) -> CfuDesignState:
    """Copy an embench-iot benchmark into the run workspace and plan its build."""
    workdir = Path(state.get("workdir", ".")).resolve()
    policy = load_policy(workdir)
    artifacts = _artifacts(state, policy)
    spec = dict(state.get("workload_spec", {}))
    toolbox = _toolbox_for_stage(state, "prepare_benchmark")
    project = prepare_embench_benchmark(toolbox, spec)
    artifacts.write_yaml(
        "prepare_benchmark",
        "04_benchmark/embench.yaml",
        dict(project),
        status="planned" if state.get("dry_run") else "produced",
    )
    return {
        "benchmark_project": project,
        "stage_jobs": "profile",
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

    configs = ["reference", "profile"]
    if state.get("changed_design_files"):
        configs.append("cfu")

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
    _write_embench_profile_artifact(state, name, profiles, results, "")
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
) -> None:
    artifacts = _artifacts_for_stage(state, "profile")
    artifacts.write_yaml(
        "profile",
        "05_profile/embench.yaml",
        {
            "benchmark": name,
            "reason": reason,
            "profiles": profiles,
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
            command_id = infer_command_id(command, toolbox.policy)
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
            "profile": {"ok": _profile_ok(profile), "skipped": not execute, "reason": reason},
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
    errors: list[str] = []
    apply_result = None
    if patch_text:
        toolbox = _toolbox_for_stage(state, "implementation")
        if state.get("iteration", 0) > 0:
            # Every iteration's patch is written against the tracked scaffold
            # (that is what the prompt shows), so reset the workspace copy
            # before applying it. Otherwise the patch is rejected against the
            # previous iteration's edits.
            toolbox.seed_design_workspace()
            summary += "\n\nRe-seeded the design workspace from the tracked scaffold."
        try:
            apply_result = toolbox.apply_patch_text(patch_text)
            summary += "\n\nPatch result:\n" + str(apply_result)
            if apply_result.get("returncode") not in (0, None):
                errors.append(f"design patch did not apply: {apply_result.get('stderr_tail', '').strip()[:400]}")
        except (PolicyError, ValueError) as exc:
            errors.append(f"patch failed: {exc}")
    elif not state.get("dry_run", False):
        summary += "\n\nNo patch block was produced; implementation remains a design proposal."

    applied = bool(apply_result) and apply_result.get("returncode") == 0 and not apply_result.get("skipped")
    if state.get("dry_run", False):
        # A planning pass delivers a proposal; it never claims a change.
        gate_ok = bool(patch_text)
    else:
        gate_ok = applied
    changed_design = (
        list((apply_result or {}).get("parsed", {}).get("redirected_paths", []) or []) if applied else []
    )
    _write_implementation_artifact(state, result.text, patch_text, apply_result, errors, changed_design)
    return {
        "implementation_plan": result.text,
        "patch_text": patch_text,
        "implementation_summary": summary,
        "changed_design_files": changed_design,
        "stage_jobs": "validation",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            "implementation": {
                "ok": gate_ok,
                "patch": bool(patch_text),
                "applied": applied,
            },
        },
        "iteration_history": append_history(state, "Completed implementation/proposal node."),
    }


def validate_embench(state: CfuDesignState) -> CfuDesignState:
    """Build and simulate the CFU configuration, then compare with reference."""
    spec = state.get("workload_spec", {})
    project = state.get("benchmark_project", {})
    run_root = state.get("run_root", "")
    name = str(project.get("benchmark", "") or spec.get("benchmark_name", ""))
    reason = _validation_reason(state)
    if reason:
        results = [_skipped_result("embench reference/cfu build+sim", state, reason)]
        _write_validation_artifact(state, results, "planned", reason)
        return {
            "validation_results": results,
            "stage_jobs": "cost",
            "gates": {**state.get("gates", {}), "validation": {"ok": False, "skipped": True, "reason": reason}},
            "iteration_history": append_history(state, f"Skipped embench validation ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "validation")
    results: list[dict] = []
    errors: list[str] = []
    isa_spec = state.get("isa_spec", {}) or {}

    # Intrinsics header for the agent design; the patched kernel includes it.
    cfu_ws = embench_config_workspace(run_root, name, "cfu")
    toolbox.write_text(f"{cfu_ws}/include/cfu_intrinsics.h", render_cfu_intrinsics(isa_spec))

    for command in agent_cfu_overlay_commands(state, isa_spec):
        results.append(toolbox.run_command(command, command_id="soc_generate_agent", timeout=3600, mutate=False, execute=True))

    build = toolbox.run_command(
        embench_build_command(run_root, name, "cfu"),
        command_id="embench_build",
        timeout=1800,
        mutate=True,
        execute=True,
    )
    results.append(build)
    cfu_profile: dict = {}
    if build.get("returncode") == 0:
        elf = embench_elf_path(run_root, name, "cfu")
        for sim_command in agent_cfu_sim_commands(state, isa_spec, elf=elf):
            sim = toolbox.run_command(sim_command, command_id="soc_sim", timeout=3600, mutate=False, execute=True)
            results.append(sim)
            text = "\n".join([sim.get("stdout_tail", ""), sim.get("stderr_tail", "")])
            cfu_profile = parse_embench_profile(text)
            if sim.get("returncode") not in (0, None):
                errors.append("embench cfu simulation failed: Embench verify_benchmark did not pass")
    else:
        errors.append("embench cfu build failed")

    reference = state.get("embench_profiles", {}).get("reference", {})
    ref_cycles = int(reference.get(name, {}).get("cycles", 0) or 0)
    cfu_cycles = int(cfu_profile.get(name, {}).get("cycles", 0) or 0)
    speedup_x100 = (ref_cycles * 100 // cfu_cycles) if cfu_cycles else 0
    ok = bool(cfu_profile) and not errors
    verification = str(project.get("verification", "none"))
    if verification != "bench_verify":
        ok = False
        errors.append("benchmark provides no verification oracle; correctness cannot be claimed")

    _write_validation_artifact(state, results, "passed" if ok else "failed", "")
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
            "status": "passed" if ok else "failed",
        },
    )
    return {
        "validation_results": results,
        "stage_jobs": "cost",
        "errors": errors,
        "gates": {
            **state.get("gates", {}),
            "validation": {
                "ok": ok,
                "skipped": False,
                "reference_cycles": ref_cycles,
                "cfu_cycles": cfu_cycles,
                "speedup_x100": speedup_x100,
            },
        },
        "iteration_history": append_history(state, f"Validated embench {name} against Embench verify."),
    }

def validate(state: CfuDesignState) -> CfuDesignState:
    if state.get("input_kind") == "embench":
        return validate_embench(state)
    with_context = state.get("workload_spec", {})
    if state.get("skip_sim", False) or state.get("dry_run", False) or not state.get("run_commands", False):
        reason = _validation_reason(state)
        results = [_skipped_result("sbt MiCoSocGen/MiCoSocSim", state, reason)]
        profile_status = "planned" if reason else "skipped"
        _write_validation_artifact(state, results, profile_status, reason)
        return {
            "validation_results": results,
            "stage_jobs": "cost",
            "gates": {
                **state.get("gates", {}),
                "validation": {"ok": False, "skipped": True, "reason": reason},
            },
            "iteration_history": append_history(state, f"Skipped validation commands ({reason})."),
        }

    toolbox = _toolbox_for_stage(state, "validation")
    spec_for_commands = with_context.get("isa_spec", {}) or state.get("isa_spec", {})
    overlay_commands = agent_cfu_overlay_commands(state, spec_for_commands)
    if overlay_commands:
        # An agent CFU run validates the overlaid workspace design. The legacy
        # VPU/BitNet flags would select a different CFU owner, so they are not
        # mixed in here.
        commands = [*overlay_commands, *agent_cfu_sim_commands(state, spec_for_commands)]
        command_errors = []
    else:
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
        "gates": {
            **state.get("gates", {}),
            "validation": {"ok": ok, "skipped": False, "results": len(results)},
        },
        "iteration_history": append_history(state, "Ran generation/simulation validation."),
    }


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
    case = cost_case_for_spec(state.get("workload_spec", {}), state.get("task", ""), agent_cfu=agent_cfu)
    # Command side effects must stay inside the stage's declared write roots,
    # so the report goes into the run directory when one exists.
    run_root = state.get("run_root", "")
    if run_root:
        out_dir = f"{run_root}/11_cost/yosys/{case}"
        rtl = f"{run_root}/{SOC_WORKSPACE_SUBDIR}/MiCoSoc.v"
    else:
        out_dir = f"{base_artifact_root(state)}/yosys/{case}"
        rtl = "MiCoSoc.v"
    command = [
        "python3",
        "skills/cfu-designer/scripts/yosys_cost_report.py",
        rtl,
        "--top",
        cost_top_for_spec(
            state.get("workload_spec", {}),
            state.get("task", ""),
            agent_cfu=agent_cfu,
        ),
        "--out-dir",
        out_dir,
        "--flatten",
    ]
    execute = bool(state.get("run_commands", False))
    if not execute:
        result = [toolbox.run_command(command, command_id="yosys_cost", execute=False)]
    else:
        result = [toolbox.run_command(command, command_id="yosys_cost", timeout=1800, mutate=True, execute=True)]
    ok = execute and all(item.get("returncode") == 0 for item in result)
    _write_cost_artifact(state, result, "passed" if ok else ("planned" if not execute else "failed"), "")
    return {
        "cost_results": result,
        "stage_jobs": "review",
        "gates": {
            **state.get("gates", {}),
            "cost": {"ok": ok, "skipped": not execute, "reason": "" if execute else "execution-disabled"},
        },
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
        "Likely CFU shape: extend AgentCfu's placeholder compute/config function ids; keep the Vector RegFile and CfuLsu for memory-backed vectors or multi-operation reuse, and keep the CfuBus response contract."
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
        "- Keep the SoC integration fixed: the run overlays your AgentCfu/AgentCfuFiber copies through the `--mico-agent-cfu` build, so no SoC parameter or CLI flag changes are needed or permitted.\n"
        "- Choose the AgentCfu memory mode through its parameter (`withTilelink`/`withLoad`/`withStore`), which the run passes from the ISA datapath.\n"
        "- Validate with scalar C reference, SoC generation, and a Yosys cost report from the run's generated RTL."
    )


def deterministic_implementation_plan(task: str, contract: str) -> str:
    return (
        "No LLM patch was generated. Implementation plan:\n"
        "1. Implement the datapath in the run workspace copy of AgentCfu.scala (compute/config function ids, Vector RegFile use, CfuLsu memory access); touch AgentCfuFiber.scala only if the fiber contract must change.\n"
        "2. Keep the public contract and the single CPU CFU owner rule; do not edit any other repository source.\n"
        "3. Add C intrinsics and a scalar-vs-CFU test under sw/tests.\n"
        "4. The run generates MiCoSoc.v from the overlay into workspace/soc, then costs it with Yosys at top AgentCfu.\n"
        "5. Iterate on datapath width, RF depth, and memory mode based on correctness and cells."
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


def _write_implementation_artifact(
    state: CfuDesignState,
    plan: str,
    patch: str,
    apply_result: dict,
    errors: list[str],
    changed_design: list[str] | None = None,
) -> None:
    artifacts = _artifacts_for_stage(state, "implementation")
    artifacts.write_text("implementation", "09_implementation/plan.md", plan, kind="markdown")
    if patch:
        artifacts.write_text("implementation", "09_implementation/patch.diff", patch, kind="text")
    parsed = (apply_result or {}).get("parsed", {}) or {}
    changed = patch and parsed.get("touched_paths", []) or []
    artifacts.write_yaml(
        "implementation",
        "09_implementation/changed_files.yaml",
        {
            "touched_paths": changed,
            "design_files": list(changed_design or []),
            "patch_applied": bool(apply_result and apply_result.get("skipped") is not True),
            "errors": errors,
        },
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


def is_agent_cfu_run(state: CfuDesignState) -> bool:
    """True when this run targets the AgentCfu scaffold rather than a legacy CFU."""
    return bool(state.get("agent_cfu", True))


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
    if validation.get("skipped") or cost.get("skipped"):
        # Nothing was executed (execution disabled or skipped by request), so
        # this is an honest planning pass rather than a failure. It can never be
        # reported as complete.
        return "final_report", "planned"
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
    # MiCoSocGen writes MiCoSoc.v/soc.h into the process working directory, so
    # the forked run is pointed at the run workspace to keep the repo clean.
    set_out_dir = (
        "set run / baseDirectory := (Compile/baseDirectory).value / "
        f'"{run_root}/{SOC_WORKSPACE_SUBDIR}"'
    )
    flags = agent_cfu_gen_flags(spec)
    # sbt takes each command as one argv element, so the subcommand and its
    # flags are joined into a single element.
    run_main = " ".join(["runMain vexiiriscv.soc.mico.MiCoSocGen", *flags])
    return [[
        "sbt",
        add_overlay,
        swap_sources,
        set_out_dir,
        run_main,
    ]]


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


def agent_cfu_gen_flags(spec: dict) -> list[str]:
    """CLI flags for an agent-CFU SoC build, derived from the ISA datapath."""
    datapath = spec.get("datapath", {}) if isinstance(spec.get("datapath"), dict) else {}
    flags = ["--with-rvc", "--with-rvm", "--with-rdtime", "--mico-agent-cfu"]
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
