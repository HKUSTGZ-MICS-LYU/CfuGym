"""Domain sub-schemas: the shapes the evidence group refers to.

They live here rather than beside the run state so the state groups can import
only what they annotate.
"""

from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict

from ..runtime.observations import FailureObservation

AgentMode = Literal["dry-run", "autonomous"]
InputKind = Literal["natural_language", "c_project", "embench"]
Visibility = Literal["basic", "standard", "extended"]


class CommandResult(TypedDict, total=False):
    command: str
    cwd: str
    returncode: int
    stdout_tail: str
    stderr_tail: str
    # Run-scoped full log; large output is never stored directly in graph state.
    log_path: str
    full_log_path: str
    command_id: str
    stdout_truncated: bool
    stderr_truncated: bool
    parsed: dict[str, Any]
    skipped: bool
    reason: str
    # Validation-ladder rung that produced this result: hw_build/sw_build/sim.
    rung: str
    timed_out: bool
    failure_observation: FailureObservation


class RepoContext(TypedDict, total=False):
    files: dict[str, str]
    cfu_entrypoints: list[str]
    software_tests: list[str]
    # Selected visibility tier and the context files it withheld from the agent.
    visibility: str
    withheld: list[str]
    # Where the agent works: the repository root and this run's artifact root.
    # Prompts report both so every path resolves without guessing.
    workdir: str
    run_root: str


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
