"""LangGraph workflow for run-scoped CFU design and validation.

This module is the composition root: it builds the graph from the flow
specification, exposes the public entry points, and re-exports the node
functions that ``python_nodes()`` resolves from this module's globals.
Stage logic lives in ``stages/``, flow control in ``flow/``, and execution
services in ``runtime/``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pathlib import Path
from typing import Any

from .flow import instrument, policies, wiring
from .flow.instrument import STAGE_ORDER, disable_progress, enable_progress, progress
from .flow.ladder import _route_after_analysis, _route_after_cost, _route_after_hw_build, _route_after_hw_conformance, _route_after_implement, _route_after_isa, _route_after_prepare, _route_after_profile, _route_after_review, _route_after_sim, _route_after_sw_build, _route_after_validation, _route_input_target
from .flow.wiring import NODE_NAMES, ROUTER_NAMES, ROUTE_TARGETS, SUCCESSORS as FLOW_SUCCESSORS
from .runtime import dependencies, failures, llm, persistence, tools
from .runtime.llm import LlmResult
from .runtime.persistence import PersistenceRiskError
from .runtime.policy import PolicyError
from .runtime.tools import RepoToolbox
from .stages.analysis import analyze_workload
from .stages.context import create_benchmark_project_node, ingest_input, load_context, prepare_c_project_profile_node, prepare_embench_benchmark_node, route_input
from .stages.contract import design_contract, extend_isa
from .stages.implement import implement
from .stages.profile import collect_ir_evidence, extract_hotspot, instrument_and_profile, profile_embench
from .stages.review import final_report, review_and_route
from .stages.verify.conformance import check_conformance
from .stages.verify.cost import estimate_cost
from .stages.verify.parallel import validate_candidate
from .stages.verify.soc import generate_soc
from .stages.verify.software import build_sw, simulate
from .state import CfuDesignState


def build_graph(*, checkpointer: Any = None, debug: bool = False):
    graph = StateGraph(CfuDesignState)
    wiring.register_langgraph(
        graph, python_nodes(), {name: globals()[symbol] for name, symbol in ROUTER_NAMES.items()},
        wrap=instrument._wrap_node, start=START, end=END, node_options=policies.node_options(),
    )
    return graph.compile(checkpointer=checkpointer, debug=debug, name="cfu_design_agent")


def _finalize(state: CfuDesignState) -> CfuDesignState:
    """Attach run-wide token usage and (when enabled) the progress summary."""
    state["token_usage"] = llm.usage_totals()
    # Per-stage breakdown (input/output/calls): this is what the CLI and the
    # report use to show WHICH stage consumed the input tokens.
    state["token_usage_by_stage"] = llm.usage_by_stage()
    reporter = progress()
    if reporter is not None:
        state["progress_summary"] = reporter.summary()
    return state


def run_agent(
    initial_state: CfuDesignState,
    *,
    thread_id: str = "",
    checkpoint: bool = False,
    stream: bool = False,
    resume: bool = False,
) -> CfuDesignState:
    """Run one LangGraph thread; opt-in persistence refuses unsafe replay.

    A checkpoint is a snapshot of graph state, not a transaction for workspace
    writes or external commands. Resume checks the run manifest and operation
    journal and refuses uncertain effects rather than reapplying them blindly.
    """
    llm.reset_usage_log()
    if resume and not checkpoint:
        raise ValueError("resume requires persistent checkpointing")
    run_id = str(initial_state.get("run_id", "") or "")
    if checkpoint:
        if not run_id:
            raise ValueError("checkpoint requires an explicit run_id")
        if thread_id and thread_id != run_id:
            raise ValueError("checkpoint thread_id must equal run_id")
        with persistence.open_run_persistence(
            Path(initial_state.get("workdir", ".")), run_id,
            out_dir=str(initial_state.get("out_dir", "agents/generated")), resume=resume,
        ) as persisted:
            app = build_graph(checkpointer=persisted.saver)
            if resume:
                snapshot = app.get_state(persisted.config)
                if not snapshot or not snapshot.values or not snapshot.next:
                    raise PersistenceRiskError("no unfinished checkpoint is available for this run")
                previous = snapshot.values
                # Historical defaults interpret missing checkpoint fields; never
                # retarget an old run when the default for new runs changes.
                defaults = {"cost_backend": "yosys", "fpga_part": "xc7a200tsbg484-1", "target_frequency_mhz": 100.0,
                            "eda_timeout": 3600, "vivado_threads": 4, "validation_mode": "serial"}
                for key, value in (initial_state.get("resume_eda_options") or {}).items():
                    if value != previous.get(key, defaults.get(key)):
                        raise PersistenceRiskError(f"resume cannot change recorded EDA configuration: {key}")
                if previous.get("run_id") != persisted.run_id:
                    raise PersistenceRiskError("checkpoint run_id differs from run manifest")
                if str(previous.get("workdir", "")) != str(Path(initial_state.get("workdir", ".")).resolve()):
                    raise PersistenceRiskError("checkpoint workdir differs from requested workdir")
                # A node can commit its external-effect journal before LangGraph
                # durably records the node boundary. Refuse to replay that node:
                # committed effects are not proof of a committed graph checkpoint.
                unfinished = set(snapshot.next)
                operations = persisted.journal._load()["operations"]
                if any(entry.get("status") == "committed" and entry.get("node") in unfinished
                       for entry in operations.values()):
                    raise PersistenceRiskError(
                        "unfinished node has committed external effects; checkpoint may lag its journal; "
                        "manual reconciliation is required before resume"
                    )
                try:
                    dependencies.completed_gate_inputs(previous, resume=True)
                except ValueError as exc:
                    raise PersistenceRiskError(str(exc)) from exc
                graph_input = None
                max_iters = int(previous.get("max_iters", 2) or 2)
            else:
                graph_input = dict(initial_state, run_id=persisted.run_id)
                max_iters = int(initial_state.get("max_iters", 2) or 2)
            config = {**graph_config(persisted.run_id, max_iters=max_iters),
                      "configurable": persisted.config["configurable"]}
            token = instrument._ACTIVE_PERSISTENCE.set(persisted)
            tool_token = tools.set_tool_persistence(persisted)
            try:
                return _invoke_agent_graph(app, graph_input, config, stream=stream)
            finally:
                tools.reset_tool_persistence(tool_token)
                instrument._ACTIVE_PERSISTENCE.reset(token)
    if thread_id:
        raise ValueError("thread_id requires checkpointing")
    app = build_graph()
    max_iters = int(initial_state.get("max_iters", 2) or 2)
    return _invoke_agent_graph(app, initial_state, {"recursion_limit": recursion_limit(max_iters)}, stream=stream)


def _invoke_agent_graph(app: Any, graph_input: CfuDesignState | None, config: dict[str, Any], *, stream: bool) -> CfuDesignState:
    final_state: CfuDesignState = dict(graph_input or {})
    snapshots = 0
    try:
        if not stream:
            final = app.invoke(graph_input, config=config, **({"durability": "sync"} if app.checkpointer else {}))
            return _finalize(final)
        for snapshot in app.stream(graph_input, config=config, stream_mode="values",
                                   **({"durability": "sync"} if app.checkpointer else {})):
            if isinstance(snapshot, dict):
                snapshots += 1
                final_state = snapshot
    except Exception as exc:
        observed = getattr(exc, "partial_state", None) or final_state
        source = getattr(exc, "partial_state_source", "") or ("stream_snapshot" if snapshots else "graph_input")
        if app.checkpointer:
            try:
                snapshot = app.get_state(config)
                if snapshot and isinstance(snapshot.values, dict) and snapshot.values:
                    observed = snapshot.values
                    source = "native_checkpoint"
            except Exception as recovery_exc:
                # State recovery must never replace the original error/traceback.
                exc.add_note(f"checkpoint state recovery failed: {type(recovery_exc).__name__}: {recovery_exc}")
        failures.attach_partial_state(exc, _finalize(dict(observed)), source=source)
        raise
    if snapshots:
        final_state = dict(final_state)
        final_state["graph_events"] = [f"streamed {snapshots} state snapshots"]
    return _finalize(final_state)


def python_nodes() -> dict[str, Any]:
    """Resolve shared flow node names against this module's patchable globals."""
    aliases = {
        "prepare_c_project_profile": "prepare_c_project_profile_node",
        "create_benchmark_project": "create_benchmark_project_node",
        "prepare_embench_benchmark": "prepare_embench_benchmark_node",
    }
    return {name: globals()[aliases.get(name, name)] for name in NODE_NAMES}


def export_graph_visualization(out_dir: str | Path) -> dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
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
    """Render the shared wiring without maintaining a second hand-drawn graph."""
    lines = ["START -> load_context"]
    for name in NODE_NAMES:
        if name in ROUTE_TARGETS:
            lines.append(f"{name} -> {' | '.join(ROUTE_TARGETS[name])}")
        else:
            lines.append(f"{name} -> {FLOW_SUCCESSORS[name] or 'END'}")
    return "\n".join(lines) + "\n"


def recursion_limit(max_iters: int = 2) -> int:
    """Graph step budget.

    The ladder is a cycle (repairs and --loop optimization rounds), so the step
    limit has to grow with --max-iters instead of using LangGraph's default of 25,
    which a long --loop run would exceed.
    """
    return 60 + 12 * max(1, int(max_iters or 2))


def graph_config(thread_id: str, *, max_iters: int = 2) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": (thread_id or "cfu-design-agent")[:255]},
        "recursion_limit": recursion_limit(max_iters),
    }
