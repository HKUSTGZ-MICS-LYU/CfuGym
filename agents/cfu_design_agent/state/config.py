"""RunConfig: The run's inputs: what to do, how far, with which permissions."""

from __future__ import annotations

from typing import Any, Literal, TypedDict
from .schemas import AgentMode, InputKind, Visibility


class RunConfig(TypedDict, total=False):
    """The run's inputs: what to do, how far, with which permissions."""

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
    max_tool_calls: int
    profile_cache: str
    ir_features: str
    stream_llm: bool
    max_output_tokens: int
    loop_optimize: bool
    loop_policy: str
    loop_patience: int
    max_evals: int
    visibility: Visibility
    skip_sim: bool
    skip_yosys: bool
    # Ablation switches (all default True; the --no-* CLI flags set False).
    llvm_ir: bool
    require_conformance: bool
    isa_contract: bool
    max_iters: int
    run_id: str
    run_root: str
    agent_cfu: bool
    thread_id: str
