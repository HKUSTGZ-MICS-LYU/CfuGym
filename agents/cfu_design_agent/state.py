"""Typed state shared across the CFU design LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict


AgentMode = Literal["dry-run", "autonomous"]
InputKind = Literal["natural_language", "c_project"]


class CommandResult(TypedDict, total=False):
    command: str
    cwd: str
    returncode: int
    stdout_tail: str
    stderr_tail: str
    parsed: dict[str, Any]
    skipped: bool
    reason: str


class RepoContext(TypedDict, total=False):
    files: dict[str, str]
    git_status: str
    cfu_entrypoints: list[str]
    software_tests: list[str]


class WorkloadSpec(TypedDict, total=False):
    name: str
    source_kind: Literal["natural_language", "formatted_spec", "c_project"]
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
    validation: dict[str, Any]
    unknowns: list[str]


class BenchmarkProject(TypedDict, total=False):
    kind: Literal["generated_benchmark", "instrumented_copy", "planned"]
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
    summary: str


class CfuDesignState(TypedDict, total=False):
    task: str
    input_kind: InputKind
    input_mode: Literal["natural_language", "formatted_spec", "c_project"]
    workload_spec_path: str
    project_root: str
    kernel_file: str
    kernel_function: str
    build_cmd: str
    run_cmd: str
    test_cmd: str
    workdir: str
    out_dir: str
    model: str
    mode: AgentMode
    dry_run: bool
    skip_sim: bool
    skip_yosys: bool
    max_iters: int
    iteration: int
    context: RepoContext
    workload_spec: WorkloadSpec
    spec_text: str
    spec_unknowns: list[str]
    benchmark_project: BenchmarkProject
    profile_result: ProfileResult
    hot_spot: str
    workload_analysis: str
    design_contract: str
    implementation_plan: str
    patch_text: str
    implementation_summary: str
    validation_results: Annotated[list[CommandResult], operator.add]
    cost_results: Annotated[list[CommandResult], operator.add]
    iteration_history: Annotated[list[str], operator.add]
    final_report_path: str
    status: Literal["initialized", "needs_iteration", "passed", "failed", "complete"]
    errors: Annotated[list[str], operator.add]
    thread_id: str
    graph_events: Annotated[list[str], operator.add]


def append_history(state: CfuDesignState, line: str) -> list[str]:
    return [line]
