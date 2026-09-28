"""Single workflow specification shared by LangGraph and the Python runner.

Node implementations and routing decisions stay in ``graph.py`` so existing
module-level patch targets continue to work. This module owns the graph shape.
"""

from __future__ import annotations

from typing import Any, Callable


# (node, static successor, router function name, allowed conditional targets).
# Exactly one of a static successor or a router is used; final_report ends the walk.
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
    ("extract_hotspot", "analyze_workload", None, ()),
    ("analyze_workload", "design_contract", None, ()),
    ("design_contract", None, "_route_after_isa",
     ("design_contract", "implement_hw", "review_and_route")),
    ("extend_isa", None, "_route_after_isa",
     ("extend_isa", "implement_hw", "review_and_route")),
    ("implement_hw", "generate_soc", None, ()),
    ("generate_soc", None, "_route_after_hw_build",
     ("check_conformance", "build_sw", "implement_hw", "review_and_route")),
    ("check_conformance", None, "_route_after_hw_conformance",
     ("implement_sw", "build_sw", "check_conformance", "implement_hw", "review_and_route")),
    ("implement_sw", "build_sw", None, ()),
    ("build_sw", None, "_route_after_sw_build",
     ("simulate", "implement_sw", "review_and_route")),
    ("simulate", None, "_route_after_sim",
     ("estimate_cost", "implement_sw", "implement_hw", "review_and_route")),
    ("estimate_cost", None, "_route_after_cost",
     ("review_and_route", "implement_hw")),
    ("review_and_route", None, "_route_after_review",
     ("final_report", "implement_hw", "implement_sw", "extend_isa")),
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
    retry_policy: Any,
) -> None:
    """Register the same workflow traversed by the fallback runner."""
    if set(nodes) != set(NODE_NAMES):
        raise ValueError("flow node registry differs from workflow specification")
    if set(routers) != set(ROUTER_NAMES):
        raise ValueError("flow router registry differs from workflow specification")
    for name in NODE_NAMES:
        kwargs = {"retry_policy": retry_policy} if name == "load_context" else {}
        graph.add_node(name, wrap(nodes[name]), **kwargs)
    graph.add_edge(start, "load_context")
    for source, destination in SUCCESSORS.items():
        graph.add_edge(source, end if destination is None else destination)
    for source, targets in ROUTE_TARGETS.items():
        graph.add_conditional_edges(source, routers[source], {target: target for target in targets})


def walk_python(
    state: dict[str, Any],
    nodes: dict[str, Callable[[dict[str, Any]], Any]],
    routers: dict[str, Callable[[dict[str, Any]], str]],
    *,
    invoke: Callable[[Callable[..., Any], dict[str, Any]], Any],
    merge: Callable[[dict[str, Any], Any], None],
    limit: int,
) -> None:
    """Traverse the shared edge specification; caller owns node execution."""
    node: str | None = "load_context"
    guard = 0
    while node is not None:
        guard += 1
        if guard > limit:
            raise RuntimeError(f"python runner did not terminate at node {node!r}")
        merge(state, invoke(nodes[node], state))
        router = routers.get(node)
        node = router(state) if router is not None else SUCCESSORS.get(node)
