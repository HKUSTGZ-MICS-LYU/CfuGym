"""Workflow specification for the LangGraph runner.

Node implementations and routing decisions stay in ``graph.py`` so existing
module-level patch targets continue to work. This module owns the graph shape.
"""

from __future__ import annotations

from typing import Any, Callable


# (node, static successor, router function name, allowed conditional targets).
# Exactly one of a static successor or a router is used; final_report terminates.
_NODE_LAYOUT: tuple[tuple[str, str | None, str | None, tuple[str, ...]], ...] = (
    ("load_context", "ingest_input", None, ()),
    ("ingest_input", "route_input", None, ()),
    ("route_input", None, "_route_input_target",
     ("prepare_c_project_profile", "create_benchmark_project", "prepare_embench_benchmark")),
    ("prepare_c_project_profile", None, "_route_after_prepare",
     ("instrument_and_profile", "review_and_route")),
    ("create_benchmark_project", None, "_route_after_prepare",
     ("instrument_and_profile", "review_and_route")),
    ("prepare_embench_benchmark", None, "_route_after_prepare",
     ("profile_embench", "review_and_route")),
    ("profile_embench", "extract_hotspot", None, ()),
    ("instrument_and_profile", "extract_hotspot", None, ()),
    ("extract_hotspot", "collect_ir_evidence", None, ()),
    ("collect_ir_evidence", "analyze_workload", None, ()),
    # A workload analysis that is not a parseable mapping is an answer error:
    # repair it from the answer pool or stop, but never hand the empty mapping
    # to every later stage.
    ("analyze_workload", None, "_route_after_analysis",
     ("analyze_workload", "design_contract", "review_and_route")),
    # One implementation stage; every repair below routes back into it.
    ("design_contract", None, "_route_after_isa",
     ("design_contract", "implement", "review_and_route")),
    ("extend_isa", None, "_route_after_isa",
     ("extend_isa", "implement", "review_and_route")),
    ("implement", "generate_soc", None, ()),
    ("generate_soc", None, "_route_after_hw_build",
     ("check_conformance", "implement", "review_and_route")),
    ("check_conformance", None, "_route_after_hw_conformance",
     ("build_sw", "implement", "review_and_route")),
    ("build_sw", None, "_route_after_sw_build",
     ("simulate", "implement", "review_and_route")),
    ("simulate", None, "_route_after_sim",
     ("estimate_cost", "implement", "review_and_route")),
    ("estimate_cost", None, "_route_after_cost",
     ("review_and_route", "implement")),
    ("review_and_route", None, "_route_after_review",
     ("final_report", "implement", "extend_isa")),
    ("final_report", None, None, ()),
)

NODE_NAMES = tuple(name for name, _, _, _ in _NODE_LAYOUT)
SUCCESSORS: dict[str, str | None] = {
    name: next_node for name, next_node, router, _ in _NODE_LAYOUT if router is None
}
ROUTER_NAMES: dict[str, str] = {
    name: router for name, _, router, _ in _NODE_LAYOUT if router is not None
}
ROUTE_TARGETS: dict[str, tuple[str, ...]] = {
    name: targets for name, _, router, targets in _NODE_LAYOUT if router is not None
}


def _validate_layout() -> None:
    nodes = set(NODE_NAMES)
    if len(nodes) != len(NODE_NAMES):
        raise ValueError("duplicate flow node name")
    for name, next_node, router, targets in _NODE_LAYOUT:
        if (router is not None) != bool(targets):
            raise ValueError(f"flow node {name!r} has inconsistent routing")
        if router is not None and next_node is not None:
            raise ValueError(f"flow node {name!r} has both static and conditional edges")
        if next_node is not None and next_node not in nodes:
            raise ValueError(f"unknown flow successor {next_node!r}")
        if set(targets) - nodes:
            raise ValueError(f"flow node {name!r} has unknown targets")
    if SUCCESSORS.get("final_report", "missing") is not None:
        raise ValueError("final_report must terminate the workflow")


_validate_layout()


def register_langgraph(
    graph: Any,
    nodes: dict[str, Callable[..., Any]],
    routers: dict[str, Callable[..., Any]],
    *,
    wrap: Callable[[Callable[..., Any]], Callable[..., Any]],
    start: Any,
    end: Any,
    node_options: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Register the workflow with LangGraph.

    ``node_options`` carries the declarative per-node LangGraph policies
    (``retry_policy``, ``error_handler``, ...) declared in ``flow/policies.py``;
    they are applied here so a node body never has to encode them.
    """
    if set(nodes) != set(NODE_NAMES):
        raise ValueError("flow node registry differs from workflow specification")
    if set(routers) != set(ROUTER_NAMES):
        raise ValueError("flow router registry differs from workflow specification")
    options = dict(node_options or {})
    unknown = set(options) - set(NODE_NAMES)
    if unknown:
        raise ValueError(f"node options for unknown flow nodes: {sorted(unknown)}")
    for name in NODE_NAMES:
        graph.add_node(name, wrap(nodes[name]), **options.get(name, {}))
    graph.add_edge(start, "load_context")
    for source, destination in SUCCESSORS.items():
        graph.add_edge(source, end if destination is None else destination)
    for source, targets in ROUTE_TARGETS.items():
        graph.add_conditional_edges(source, routers[source], {target: target for target in targets})