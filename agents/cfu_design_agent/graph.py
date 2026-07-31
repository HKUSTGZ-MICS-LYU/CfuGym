"""LangGraph workflow for CFU design iterations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shlex
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from .benchmarks import (
    prepare_c_project_profile as prepare_c_project_profile_workspace,
    prepare_generated_benchmark,
    profile_commands,
    profile_result_from_commands,
)
from .llm import OpenAiTextClient
from .prompts import SYSTEM_PROMPT, contract_prompt, implementation_prompt, workload_prompt
from .spec import normalize_workload_spec
from .state import CfuDesignState, append_history
from .tools import RepoToolbox, extract_patch


def build_graph(*, checkpointer: Any = None, debug: bool = False):
    graph = StateGraph(CfuDesignState)
    graph.add_node("load_context", load_context, retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node("ingest_input", ingest_input)
    graph.add_node(
        "route_input",
        route_input,
        destinations=("prepare_c_project_profile", "create_benchmark_project"),
    )
    graph.add_node("prepare_c_project_profile", prepare_c_project_profile)
    graph.add_node("create_benchmark_project", create_benchmark_project)
    graph.add_node("instrument_and_profile", instrument_and_profile)
    graph.add_node("extract_hotspot", extract_hotspot)
    graph.add_node("analyze_workload", analyze_workload)
    graph.add_node("design_contract", design_contract)
    graph.add_node("implement_hw_sw", implement_hw_sw)
    graph.add_node("validate", validate)
    graph.add_node("estimate_cost", estimate_cost)
    graph.add_node(
        "review_and_route",
        review_and_route,
        destinations=("implement_hw_sw", "final_report"),
    )
    graph.add_node("final_report", final_report)

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


def run_agent(
    initial_state: CfuDesignState,
    *,
    thread_id: str = "",
    checkpoint: bool = False,
    stream: bool = False,
) -> CfuDesignState:
    app = build_graph(checkpointer=InMemorySaver() if checkpoint else None)
    config = graph_config(thread_id) if checkpoint or thread_id else None
    if not stream:
        return app.invoke(initial_state, config=config)

    final_state: CfuDesignState = {}
    snapshots = 0
    for snapshot in app.stream(initial_state, config=config, stream_mode="values"):
        if isinstance(snapshot, dict):
            snapshots += 1
            final_state = snapshot
    if snapshots:
        final_state = dict(final_state)
        final_state["graph_events"] = [f"streamed {snapshots} state snapshots"]
    return final_state


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
    return "\n".join(
        [
            "START",
            "  -> load_context",
            "  -> ingest_input",
            "  -> route_input",
            "       | Command goto prepare_c_project_profile/create_benchmark_project",
            "  -> instrument_and_profile",
            "  -> extract_hotspot",
            "  -> analyze_workload",
            "  -> design_contract",
            "  -> implement_hw_sw",
            "  -> validate",
            "  -> estimate_cost",
            "  -> review_and_route",
            "       | Command goto implement_hw_sw/final_report",
            "  -> END",
            "",
        ]
    )


def graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": (thread_id or "cfu-design-agent")[:255]}}


def _toolbox(state: CfuDesignState) -> RepoToolbox:
    return RepoToolbox(Path(state.get("workdir", ".")).resolve(), dry_run=state.get("dry_run", False))


def _llm(state: CfuDesignState) -> OpenAiTextClient:
    return OpenAiTextClient(state.get("model"))


def load_context(state: CfuDesignState) -> CfuDesignState:
    toolbox = _toolbox(state)
    extra_files = []
    if state.get("kernel_file"):
        extra_files.append(state["kernel_file"])
    if state.get("workload_spec_path"):
        extra_files.append(state["workload_spec_path"])
    context = toolbox.discover_context(extra_files)
    return {
        "context": context,
        "iteration": state.get("iteration", 0),
        "status": "initialized",
        "iteration_history": append_history(state, "Loaded repo context and CFU design references."),
    }


def ingest_input(state: CfuDesignState) -> CfuDesignState:
    spec, spec_text = normalize_workload_spec(
        task=state.get("task", ""),
        input_mode=state.get("input_mode", "natural_language"),
        context=state.get("context", {}),
        workdir=Path(state.get("workdir", ".")).resolve(),
        workload_spec_path=state.get("workload_spec_path", ""),
        project_root=state.get("project_root", ""),
        kernel_file=state.get("kernel_file", ""),
        kernel_function=state.get("kernel_function", ""),
        build_cmd=state.get("build_cmd", ""),
        run_cmd=state.get("run_cmd", ""),
        test_cmd=state.get("test_cmd", ""),
        llm=_llm(state),
    )
    return {
        "workload_spec": spec,
        "spec_text": spec_text,
        "spec_unknowns": spec.get("unknowns", []),
        "iteration_history": append_history(state, f"Normalized input into WorkloadSpec `{spec.get('name', 'cfu_workload')}`."),
    }


def route_input(
    state: CfuDesignState,
) -> Command[Literal["prepare_c_project_profile", "create_benchmark_project"]]:
    kind = state.get("input_kind") or state.get("input_mode", "natural_language")
    if kind not in ("c_project", "natural_language"):
        kind = "natural_language"
    goto = "prepare_c_project_profile" if kind == "c_project" else "create_benchmark_project"
    return Command(
        update={
            "input_kind": kind,
            "iteration_history": append_history(state, f"Selected input route `{kind}`."),
        },
        goto=goto,
    )


def prepare_c_project_profile(state: CfuDesignState) -> CfuDesignState:
    project = prepare_c_project_profile_workspace(_toolbox(state), state.get("workload_spec", {}))
    return {
        "benchmark_project": project,
        "iteration_history": append_history(state, f"Prepared C-project profiling workspace `{project.get('workspace', '')}`."),
    }


def create_benchmark_project(state: CfuDesignState) -> CfuDesignState:
    project = prepare_generated_benchmark(_toolbox(state), state.get("workload_spec", {}))
    return {
        "benchmark_project": project,
        "iteration_history": append_history(state, f"Prepared generated benchmark workspace `{project.get('workspace', '')}`."),
    }


def instrument_and_profile(state: CfuDesignState) -> CfuDesignState:
    project = state.get("benchmark_project", {})
    commands = profile_commands(project)
    if state.get("dry_run", False) or state.get("skip_sim", False):
        reason = "dry-run" if state.get("dry_run", False) else "--skip-sim"
        results = [
            {
                "command": " ".join(command),
                "cwd": state.get("workdir", "."),
                "returncode": 0,
                "skipped": True,
                "reason": reason,
            }
            for command in commands
        ]
        if not results:
            results = [{
                "command": "profile benchmark/project",
                "cwd": state.get("workdir", "."),
                "returncode": 0,
                "skipped": True,
                "reason": reason,
            }]
        profile = profile_result_from_commands(results, dry_run=True, planned_reason=reason)
        return {
            "profile_result": profile,
            "iteration_history": append_history(state, "Planned profiling commands without executing them."),
        }

    toolbox = _toolbox(state)
    results = []
    errors: list[str] = []
    for command in commands:
        if not toolbox.command_allowed(command):
            errors.append(f"profile command is not allowlisted and was skipped: {' '.join(command)}")
            continue
        result = toolbox.run_command(command, timeout=3600, mutate=True)
        results.append(result)
        if result.get("returncode", 0) != 0:
            break
    profile = profile_result_from_commands(results, dry_run=False)
    return {
        "profile_result": profile,
        "errors": errors,
        "iteration_history": append_history(state, "Ran profiling commands for prepared workload."),
    }


def extract_hotspot(state: CfuDesignState) -> CfuDesignState:
    profile = state.get("profile_result", {})
    spec = dict(state.get("workload_spec", {}))
    project = state.get("benchmark_project", {})
    candidates = project.get("candidate_hotspots", [])
    preferred_kernel = spec.get("kernel_function", "")
    if preferred_kernel == "main" and candidates:
        preferred_kernel = candidates[0]
    hot = profile.get("hot_spot") or preferred_kernel or (candidates[0] if candidates else "") or spec.get("operation", "scalar_kernel")
    if not spec.get("hot_spot"):
        spec["hot_spot"] = hot
    spec_text = state.get("spec_text", "")
    if "hot_spot:" not in spec_text:
        spec_text = spec_text.rstrip() + f"\nhot_spot: {hot}\n"
    return {
        "workload_spec": spec,
        "spec_text": spec_text,
        "hot_spot": hot,
        "iteration_history": append_history(state, f"Selected hot spot `{hot}` for CFU analysis."),
    }


def analyze_workload(state: CfuDesignState) -> CfuDesignState:
    task = state["task"]
    fallback = deterministic_analysis(task, state.get("workload_spec", {}))
    result = _llm(state).complete(
        SYSTEM_PROMPT,
        workload_prompt(task, state.get("workload_spec", {}), state.get("context", {})),
        fallback=fallback,
    )
    note = "Analyzed workload with LLM." if result.used_llm else "Analyzed workload with deterministic fallback."
    return {
        "workload_analysis": result.text,
        "iteration_history": append_history(state, note),
    }


def design_contract(state: CfuDesignState) -> CfuDesignState:
    task = state["task"]
    fallback = deterministic_contract(task, state.get("workload_spec", {}), state.get("workload_analysis", ""))
    result = _llm(state).complete(
        SYSTEM_PROMPT,
        contract_prompt(task, state.get("workload_spec", {}), state.get("workload_analysis", ""), state.get("context", {})),
        fallback=fallback,
    )
    return {
        "design_contract": result.text,
        "iteration_history": append_history(state, "Produced CFU ISA/integration contract."),
    }


def implement_hw_sw(state: CfuDesignState) -> CfuDesignState:
    task = state["task"]
    fallback = deterministic_implementation_plan(task, state.get("design_contract", ""))
    result = _llm(state).complete(
        SYSTEM_PROMPT,
        implementation_prompt(
            task,
            state.get("workload_spec", {}),
            state.get("workload_analysis", ""),
            state.get("design_contract", ""),
            state.get("context", {}),
        ),
        fallback=fallback,
    )
    patch_text = extract_patch(result.text)
    summary = result.text
    errors: list[str] = []

    if patch_text:
        try:
            apply_result = _toolbox(state).apply_patch_text(patch_text)
            summary += "\n\nPatch result:\n" + str(apply_result)
        except Exception as exc:
            errors = [*errors, f"patch failed: {exc}"]
    elif not state.get("dry_run", False):
        summary += "\n\nNo patch block was produced; implementation remains a design proposal."

    return {
        "implementation_plan": result.text,
        "patch_text": patch_text,
        "implementation_summary": summary,
        "errors": errors,
        "iteration_history": append_history(state, "Completed implementation/proposal node."),
    }


def validate(state: CfuDesignState) -> CfuDesignState:
    if state.get("skip_sim", False) or state.get("dry_run", False):
        result = {
            "command": "sbt MiCoSocGen/MiCoSocSim",
            "cwd": state.get("workdir", "."),
            "returncode": 0,
            "skipped": True,
            "reason": "--skip-sim" if state.get("skip_sim", False) else "dry-run",
        }
        return {
            "validation_results": [result],
            "iteration_history": append_history(state, "Skipped validation commands."),
        }

    toolbox = _toolbox(state)
    commands, command_errors = validation_commands_for_spec(state.get("workload_spec", {}), state.get("task", ""), toolbox)
    results = []
    errors: list[str] = []
    if command_errors:
        errors = [*errors, *command_errors]
    for command in commands:
        results.append(toolbox.run_command(command, timeout=3600, mutate=False))
        if results[-1]["returncode"] != 0:
            break
    return {
        "validation_results": results,
        "errors": errors,
        "iteration_history": append_history(state, "Ran generation/simulation validation."),
    }


def estimate_cost(state: CfuDesignState) -> CfuDesignState:
    if state.get("skip_yosys", False):
        result = {
            "command": "python3 skills/cfu-designer/scripts/yosys_cost_report.py",
            "cwd": state.get("workdir", "."),
            "returncode": 0,
            "skipped": True,
            "reason": "--skip-yosys",
        }
        return {
            "cost_results": [result],
            "iteration_history": append_history(state, "Skipped Yosys cost estimation by request."),
        }

    toolbox = _toolbox(state)
    out_dir = f"{artifact_root(state)}/yosys/{cost_case_for_spec(state.get('workload_spec', {}), state.get('task', ''))}"
    result = toolbox.run_command(
        [
            "python3",
            "skills/cfu-designer/scripts/yosys_cost_report.py",
            "MiCoSoc.v",
            "--top",
            cost_top_for_spec(state.get("workload_spec", {}), state.get("task", "")),
            "--out-dir",
            out_dir,
            "--flatten",
        ],
        timeout=1800,
        mutate=True,
    )
    return {
        "cost_results": [result],
        "iteration_history": append_history(state, "Ran Yosys cost estimation."),
    }


def review_and_route(state: CfuDesignState) -> Command[Literal["implement_hw_sw", "final_report"]]:
    failed_validation = any(r.get("returncode", 0) != 0 for r in state.get("validation_results", []))
    failed_cost = any(r.get("returncode", 0) != 0 for r in state.get("cost_results", []))
    iteration = state.get("iteration", 0) + 1
    max_iters = state.get("max_iters", 2)
    status = "needs_iteration" if (failed_validation or failed_cost) and iteration < max_iters else "passed"
    if failed_validation or failed_cost:
        status = "failed" if iteration >= max_iters else "needs_iteration"
    goto = "implement_hw_sw" if status == "needs_iteration" else "final_report"
    return Command(
        update={
            "iteration": iteration,
            "status": status,
            "iteration_history": append_history(state, f"Reviewed iteration {iteration}: {status}."),
        },
        goto=goto,
    )


def final_report(state: CfuDesignState) -> CfuDesignState:
    toolbox = _toolbox(state)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    rel_path = f"{artifact_root(state)}/runs/{timestamp}/report.md"
    report = render_report(state)
    write_note = toolbox.write_text(rel_path, report)
    return {
        "final_report_path": str(toolbox.resolve(rel_path)),
        "status": "complete",
        "iteration_history": append_history(state, f"Wrote final report: {write_note}."),
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


def artifact_root(state: CfuDesignState) -> str:
    root = state.get("out_dir", "agents/generated").strip("/")
    if root == "agents/generated" or root.startswith("agents/generated/"):
        return root
    return "agents/generated"


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


def render_report(state: CfuDesignState) -> str:
    return "\n\n".join(
        [
            "# CFU Design Agent Report",
            f"Task: {state.get('task', '')}",
            f"Status: {state.get('status', '')}",
            "## Normalized WorkloadSpec\n```yaml\n" + state.get("spec_text", "").strip() + "\n```",
            "## Benchmark Project\n" + format_mapping(state.get("benchmark_project", {})),
            "## Profile Result\n" + format_mapping(state.get("profile_result", {})),
            "## Workload Analysis\n" + state.get("workload_analysis", ""),
            "## Design Contract\n" + state.get("design_contract", ""),
            "## Implementation\n" + state.get("implementation_summary", ""),
            "## Validation\n" + format_results(state.get("validation_results", [])),
            "## Hardware Cost\n" + format_results(state.get("cost_results", [])),
            "## Iteration History\n" + "\n".join(f"- {line}" for line in state.get("iteration_history", [])),
            "## Errors\n" + ("\n".join(f"- {err}" for err in state.get("errors", [])) or "None"),
        ]
    ) + "\n"


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
