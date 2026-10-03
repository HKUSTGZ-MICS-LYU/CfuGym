"""Evidence: What the run has observed or produced: the domain artifacts."""

from __future__ import annotations

from typing import Any, TypedDict
from .schemas import BenchmarkProject, ProfileResult, RepoContext, WorkloadSpec


class Evidence(TypedDict, total=False):
    """What the run has observed or produced: the domain artifacts."""

    ir_result: dict[str, Any]
    design_workspace: dict[str, Any]
    design_files: list[str]
    changed_design_files: list[str]
    stage_jobs: str
    context: RepoContext
    # Bounded, source-fingerprinted knowledge shared by all nodes in this run.
    flow_memory: dict[str, Any]
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
    conformance_delivery: dict[str, Any]
    repair_scope: str
