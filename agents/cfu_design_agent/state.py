"""Typed state shared across the CFU design LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict


AgentMode = Literal["dry-run", "autonomous"]
InputKind = Literal["natural_language", "c_project", "embench"]
Visibility = Literal["basic", "standard", "extended"]


class CommandResult(TypedDict, total=False):
    command: str
    cwd: str
    returncode: int
    stdout_tail: str
    stderr_tail: str
    parsed: dict[str, Any]
    skipped: bool
    reason: str
    # Validation-ladder rung that produced this result: hw_build/sw_build/sim.
    rung: str
    timed_out: bool


class RepoContext(TypedDict, total=False):
    files: dict[str, str]
    git_status: str
    cfu_entrypoints: list[str]
    software_tests: list[str]
    # Selected visibility tier and the context files it withheld from the agent.
    visibility: str
    withheld: list[str]


class WorkloadSpec(TypedDict, total=False):
    spec_version: str
    name: str
    source_kind: Literal["natural_language", "formatted_spec", "c_project", "embench"]
    benchmark_suite: str
    benchmark_name: str
    benchmark_scale: int
    warmup_heat: int
    observation_plan: dict[str, Any]
    description: str
    project_root: str
    kernel_file: str
    kernel_function: str
    build_cmd: str
    run_cmd: str
    test_cmd: str
    operation: str
    pattern: str
    element_type: str
    signedness: str
    output_type: str
    vector_bytes: int
    memory_access: str
    alignment_bytes: int
    cfu_style: str
    hot_spot: str
    hot_loop: dict[str, Any]
    validation: dict[str, Any]
    unknowns: list[str]


class BenchmarkProject(TypedDict, total=False):
    kind: Literal["generated_benchmark", "instrumented_copy", "embench_benchmark", "planned"]
    suite: str
    benchmark: str
    configs: list[str]
    elf_paths: dict[str, str]
    verification: str
    source_files: list[str]
    copied_files: list[str]
    workspace: str
    source_path: str
    header_path: str
    build_cmd: str
    run_cmd: str
    test_cmd: str
    elf_path: str
    candidate_hotspots: list[str]
    notes: list[str]


class ProfileRegion(TypedDict, total=False):
    name: str
    cycles: int
    source: str


class ProfileResult(TypedDict, total=False):
    status: Literal["planned", "profiled", "skipped", "failed"]
    command_results: list[CommandResult]
    regions: list[ProfileRegion]
    hot_spot: str
    hot_spot_point: dict[str, Any]
    summary: str


class CfuDesignState(TypedDict, total=False):
    task: str
    input_kind: InputKind
    input_mode: Literal["natural_language", "formatted_spec", "c_project", "embench"]
    workload_spec_path: str
    benchmark_name: str
    observation_plan: dict[str, Any]
    project_root: str
    kernel_file: str
    kernel_function: str
    build_cmd: str
    run_cmd: str
    test_cmd: str
    workdir: str
    out_dir: str
    model: str
    llm_config: str
    mode: AgentMode
    dry_run: bool
    apply_patch: bool
    run_commands: bool
    agent_tools: bool
    max_tool_rounds: int
    # Cumulative cap on tool calls per agent step (0 = unlimited).
    max_tool_calls: int
    # Reference/profile measurement cache: on | off | refresh.
    profile_cache: str
    # Stream provider output tokens (live progress + per-chunk read timeout).
    stream_llm: bool
    # Failed-simulation repair routing: auto (LLM decides sw/hw) | fixed.
    repair_routing: str
    # Whether the software rung must run after the hardware build:
    # auto (the agent may reuse a kernel that already drives the CFU) | always.
    software_change: str
    # --loop: a passing run may spend leftover budget on optimization rounds.
    loop_optimize: bool
    # explore reconsiders a premature stop; conservative retains one-shot routing.
    loop_policy: str
    loop_patience: int
    isa_revision: int
    isa_extension_base: dict[str, Any]
    isa_added_symbols: list[str]
    best_candidate: dict[str, Any]
    delivered_round: int
    # The optimization round currently being executed ({} outside --loop).
    optimization_plan: dict[str, Any]
    # Index of the round whose measurements the sim/cost gates carry.
    optimization_round: int
    # One measured result per round, for the ledger and the report.
    optimization_rounds: list[dict[str, Any]]
    # Read-only file visibility tier: basic | standard | extended.
    visibility: Visibility
    skip_sim: bool
    skip_yosys: bool
    max_iters: int
    iteration: int
    run_id: str
    run_root: str
    agent_cfu: bool
    design_workspace: dict[str, Any]
    design_files: list[str]
    changed_design_files: list[str]
    stage_jobs: str
    context: RepoContext
    workload_spec: WorkloadSpec
    spec_text: str
    spec_unknowns: list[str]
    benchmark_project: BenchmarkProject
    embench_profiles: dict[str, Any]
    observation_report: dict[str, Any]
    profile_result: ProfileResult
    hot_spot: str
    workload_analysis: str
    isa_spec: dict[str, Any]
    isa_spec_text: str
    isa_spec_errors: list[str]
    design_contract: str
    hardware_plan: str
    software_plan: str
    hardware_patch: str
    software_patch: str
    implementation_hw_summary: str
    implementation_sw_summary: str
    changed_sw_files: list[str]
    validation_results: Annotated[list[CommandResult], operator.add]
    cost_results: Annotated[list[CommandResult], operator.add]
    iteration_history: Annotated[list[str], operator.add]
    # Validation-ladder bookkeeping. ``attempts`` counts repairs per rung (the
    # simulation escalation reads ``attempts["sim"]``), ``repairs_used`` is the
    # global repair budget consumed, and ``last_repair_target`` records the most
    # recent repair for reporting.
    attempts: dict[str, int]
    repairs_used: int
    last_repair_target: str
    stopped_at: str
    final_report_path: str
    status: Literal["initialized", "needs_iteration", "failed", "complete", "blocked", "planned"]
    errors: Annotated[list[str], operator.add]
    thread_id: str
    graph_events: Annotated[list[str], operator.add]
    gates: dict[str, dict[str, Any]]
    token_usage: dict[str, int]
    # Run-relative paths of the exact text returned by each LLM call.
    llm_raw_outputs: Annotated[list[str], operator.add]
    # One summary entry per agent tool call (read_file/list_dir/search_text).
    tool_calls: Annotated[list[dict], operator.add]
    # (response_hash, error_hash) per software attempt; two identical consecutive
    # entries mean the repair prompt produced the same answer again.
    sw_attempts: Annotated[list[dict], operator.add]
    # The same ledger for the hardware rung: a byte-identical stuck retry must
    # switch delivery strategy instead of burning the whole repair budget.
    hw_attempts: Annotated[list[dict], operator.add]
    # Rejected conformance-oracle attempts (problems_hash): a rejection that
    # recurs means the repair is oscillating between two validation errors.
    conformance_attempts: Annotated[list[dict], operator.add]
    progress_summary: dict[str, Any]


def append_history(state: CfuDesignState, line: str) -> list[str]:
    return [line]
