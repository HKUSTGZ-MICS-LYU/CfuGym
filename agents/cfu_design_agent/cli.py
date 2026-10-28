"""CLI for the CFU design LangGraph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from .graph import enable_progress, export_graph_visualization, run_agent
from .runtime.llm import (
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_ROUNDS, OpenAiTextClient,
)
from .runtime.llm_config import LlmConfigError, effective_config, list_configs
from .flow import optimize as flow_optimize
from .runtime import cost_budget, provenance, ppa, vivado
from .stages.review import report as report_mod
from .state import CfuDesignState


def _stream_default() -> bool | None:
    """CFU_AGENT_STREAM_LLM as a tri-state default (None keeps the config value)."""
    raw = os.environ.get("CFU_AGENT_STREAM_LLM", "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return None


def _flag_env(name: str) -> bool:
    """Optional boolean default for a store_true flag."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


def _max_yosys_cells(value: str) -> int:
    try:
        return cost_budget.max_cells({"max_yosys_cells": int(value)})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CfuGym CFU design agent.")
    parser.add_argument("--task", default="", help="Natural-language workload or CFU design request.")
    parser.add_argument("--project-root", default="", help="C project root for source-backed workload ingestion.")
    parser.add_argument("--kernel-file", default="", help="C source file containing the hot kernel.")
    parser.add_argument("--kernel-function", default="", help="Hot kernel function name.")
    parser.add_argument("--build-cmd", default="", help="Project build command for the workload.")
    parser.add_argument("--run-cmd", default="", help="Project run/benchmark command for the workload.")
    parser.add_argument("--test-cmd", default="", help="Project correctness test command for the workload.")
    parser.add_argument("--workdir", default=".", help="Repo root. Defaults to the current directory.")
    parser.add_argument("--out-dir", default="agents/generated", help="Agent artifact directory.")
    parser.add_argument(
        "--model",
        default="",
        help="Explicit model override. If empty, the selected LLM config's `model` wins, "
        "then CFU_AGENT_MODEL, then the built-in default.",
    )
    parser.add_argument(
        "--llm-config",
        default=os.environ.get("CFU_AGENT_LLM_CONFIG", "default"),
        help="Named LLM config under agents/llm_configs/<name>.json (default: default).",
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Only run an API connectivity check against the selected LLM config, then exit.",
    )
    parser.add_argument(
        "--embench",
        default="",
        help="Run an embench-iot benchmark as the workload, e.g. --embench crc32.",
    )
    parser.add_argument("--run-id", default="", help="Explicit run id; defaults to a timestamped unique id.")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=2,
        help="Maximum candidate submissions to the verification ladder, including the first (default: 2).",
    )
    parser.add_argument(
        "--max-preverify-repairs",
        type=int,
        default=2,
        help="Separate bound for implementation reroutes before a candidate reaches verification (default: 2).",
    )
    parser.add_argument(
        "--max-answer-repairs",
        type=int,
        default=2,
        help="Separate repair budget for malformed analysis/oracle/ISA answers; any candidate rerun still needs a --max-iters slot (default: 2).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Force read-only: never mutate source or run external commands "
                        "(the opt-out; applying patches and running allowlisted commands is the default).")
    parser.add_argument(
        "--apply-patch",
        dest="apply_patch",
        action="store_true",
        default=True,
        help="Deprecated no-op: an LLM patch may be applied to CFU/software allowlisted "
        "paths by default (use --no-apply-patch to disable).",
    )
    parser.add_argument(
        "--no-apply-patch",
        dest="apply_patch",
        action="store_false",
        help="Record LLM deliverables as proposals only; never write them into the workspace.",
    )
    parser.add_argument(
        "--run-commands",
        dest="run_commands",
        action="store_true",
        default=True,
        help="Deprecated no-op: allowlisted build/simulator/Yosys commands execute by "
        "default (use --no-run-commands to disable).",
    )
    parser.add_argument(
        "--no-run-commands",
        dest="run_commands",
        action="store_false",
        help="Never execute allowlisted build/simulator/Yosys commands.",
    )
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Deprecated no-op: mutation and command execution are the default "
                        "(use --dry-run or the --no-* effect flags to restrict).")
    parser.add_argument(
        "--no-llvm-ir",
        action="store_true",
        help="Ablation: remove LLVM IR content from the agent (forces --ir-features off and "
        "drops the inspect_observation ir/features views; asm/summary evidence stays).",
    )
    parser.add_argument(
        "--no-conformance",
        action="store_true",
        help="Ablation: drop the conformance requirement (no oracle delivery; the hw_conformance "
        "rung is recorded as disabled instead of validated).",
    )
    parser.add_argument(
        "--no-isa-contract",
        action="store_true",
        help="Ablation: skip the ISA contract stage; the agent writes cfu_intrinsics.h itself "
        "from the intrinsic template with free encodings. Implies --no-conformance.",
    )
    parser.add_argument("--skip-sim", action="store_true", help="Skip SBT generation/simulation.")
    parser.add_argument("--skip-yosys", action="store_true", help="Skip Yosys cost estimation (Yosys backend only).")
    parser.add_argument("--skip-cost", action="store_true", help="Skip the selected EDA cost backend.")
    parser.add_argument("--cost-backend", choices=("yosys", "vivado"), default="yosys",
                        help="Cost backend: Yosys proxy (default) or real Vivado FPGA implementation.")
    parser.add_argument("--max-yosys-cells", type=_max_yosys_cells,
                        default=cost_budget.DEFAULT_MAX_CELLS,
                        help=f"Positive Yosys cell limit for either cost backend (default: {cost_budget.DEFAULT_MAX_CELLS}).")
    parser.add_argument("--fpga-part", default=vivado.DEFAULT_PART,
                        help=f"Vivado part (default: {vivado.DEFAULT_PART}, Artix UltraScale+ AU25P).")
    parser.add_argument("--target-frequency-mhz", type=float, default=100.0)
    parser.add_argument("--eda-timeout", type=int, default=3600, help="EDA command timeout in seconds.")
    parser.add_argument("--vivado-threads", type=int, default=4)
    parser.add_argument("--validation-mode", choices=("parallel", "serial"), default="parallel",
                        help="Run simulation and EDA concurrently with fail-fast cancellation (default).")
    parser.add_argument(
        "--agent-tools",
        dest="agent_tools",
        action="store_true",
        default=True,
        help="Let the agent call read-only tools (read_file/list_dir/search_text) before answering (default: on).",
    )
    parser.add_argument(
        "--no-agent-tools",
        dest="agent_tools",
        action="store_false",
        help="Disable the agent tool loop (one-shot prompting only).",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=DEFAULT_MAX_TOOL_ROUNDS,
        help=f"Maximum tool-calling rounds per agent step (default: {DEFAULT_MAX_TOOL_ROUNDS}).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
        help=(f"Maximum tool calls per agent step, across all rounds "
              f"(default: {DEFAULT_MAX_TOOL_CALLS}; 0 disables the cap)."),
    )
    parser.add_argument(
        "--profile-cache",
        choices=("on", "off", "refresh"),
        default=os.environ.get("CFU_AGENT_PROFILE_CACHE", "on"),
        help="Reuse the cross-run reference/profile measurement when its inputs "
        "match: on (default), off, or refresh to recompute and overwrite.",
    )
    parser.add_argument(
        "--ir-features",
        choices=("off", "auto", "require", "approx"),
        default=os.environ.get("CFU_AGENT_IR_FEATURES", "auto"),
        help="LLVM IR evidence from the uninstrumented Embench reference sources: "
        "auto (default; prefers the exact reference target and falls back to the "
        "labelled approximate rv32imc export when the installed clang rejects the "
        "GCC-only ISA), off, require (strict; a mismatch fails the run), or approx "
        "(force the approximate export). IR never replaces measured cycles or "
        "selects loops.",
    )
    parser.add_argument(
        "--loop",
        dest="loop_optimize",
        action="store_true",
        default=_flag_env("CFU_AGENT_LOOP"),
        help="Keep optimizing after a fully passing run: while --max-iters verification-submission slots "
        "remain, review explores hardware, software or append-only ISA rounds. "
        "Premature stops are reconsidered once; patience and total budget bound exploration.",
    )
    parser.add_argument("--loop-policy", choices=("explore", "conservative"), default="explore",
                        help="Explore reconsiders a stop once and allows append-only ISA extensions (default).")
    parser.add_argument("--loop-patience", type=int, default=3,
                        help="Stop after this many consecutive non-improving measured optimization rounds (default: 3).")
    parser.add_argument("--max-evals", type=int, default=0,
                        help="Maximum measured evaluation reports including the baseline (0: bounded only by --max-iters).")
    parser.add_argument(
        "--stream-llm",
        dest="stream_llm",
        action="store_true",
        default=_stream_default(),
        help="Stream model output tokens: prints live output-token progress and avoids a "
        "whole-response timeout on a slow generation (default: on unless the config or "
        "CFU_AGENT_STREAM_LLM disables it).",
    )
    parser.add_argument(
        "--no-stream-llm",
        dest="stream_llm",
        action="store_false",
        help="Disable provider token streaming (single non-streaming completion).",
    )
    parser.add_argument(
        "--max-output-tokens",
        dest="max_output_tokens",
        type=int,
        default=0,
        help="Completion budget for every request (max_completion_tokens / max_output_tokens). "
        "0 keeps the config's output_reserve_tokens, or CFU_AGENT_OUTPUT_RESERVE when set. "
        "Reasoning models share this budget with their thinking tokens, so a small value can "
        "return an empty answer; it is also the context-budget output reservation and must "
        "stay below the configured context_window_tokens.",
    )
    parser.add_argument(
        "--visibility",
        choices=("basic", "standard", "extended"),
        default=os.environ.get("CFU_AGENT_VISIBILITY", "extended"),
        help="Read-only file visibility tier, shared by every stage. basic: existing "
        "integration paths; standard: adds skills and the full VexiiRiscv source tree "
        "(except denied Vpu/BitNet example CFUs); extended: adds ext/SpinalHDL. "
        "Vpu/BitNet examples stay denied at every tier. Default: extended.",
    )
    parser.add_argument("--export-graph", default="", help="Directory for Mermaid/ASCII graph visualization.")
    parser.add_argument("--checkpoint", action="store_true", help="Persist LangGraph checkpoints within this run (requires an explicit --run-id to resume later).")
    parser.add_argument("--thread-id", default="", help="Deprecated; when checkpointing the thread is always the run id.")
    parser.add_argument("--resume", default="", metavar="RUN_ID", help="Resume a checkpointed run after validating its run artifacts.")
    parser.add_argument("--stream", action="store_true", help="Run through LangGraph streaming state snapshots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()

    if args.test_api:
        return run_api_test(workdir, args)

    if args.resume and args.run_id and args.resume != args.run_id:
        raise SystemExit("--resume and --run-id must name the same run")
    if args.thread_id and args.thread_id != (args.resume or args.run_id):
        raise SystemExit("--thread-id must equal --run-id (or --resume)")
    if args.checkpoint and not (args.run_id or args.resume):
        raise SystemExit("--checkpoint requires --run-id for safe recovery")
    import math
    import re
    if args.skip_yosys and args.cost_backend != "yosys":
        raise SystemExit("--skip-yosys requires --cost-backend yosys; use --skip-cost for Vivado")
    if not math.isfinite(args.target_frequency_mhz) or not 0 < args.target_frequency_mhz <= 1000:
        raise SystemExit("--target-frequency-mhz must be finite and between 0 and 1000")
    if args.eda_timeout <= 0 or not 1 <= args.vivado_threads <= 16:
        raise SystemExit("--eda-timeout must be positive; --vivado-threads must be between 1 and 16")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.fpga_part):
        raise SystemExit("--fpga-part must be a Vivado part name, not a path or Tcl expression")
    if args.max_evals < 0:
        raise SystemExit("--max-evals must be zero or positive")
    if args.max_iters < 1:
        raise SystemExit("--max-iters must be at least 1")
    if args.max_preverify_repairs < 0:
        raise SystemExit("--max-preverify-repairs must be zero or positive")
    if args.no_llvm_ir and args.ir_features in ("require", "approx"):
        raise SystemExit(
            "--no-llvm-ir conflicts with --ir-features require/approx: "
            "explicit IR evidence cannot be requested without IR content"
        )
    if args.max_output_tokens < 0:
        raise SystemExit("--max-output-tokens must be zero or positive")
    if args.resume and any((args.task, args.embench, args.project_root, args.kernel_file,
                            args.kernel_function, args.build_cmd, args.run_cmd, args.test_cmd)):
        raise SystemExit("--resume accepts only the prior run id and execution options, not a new workload")
    input_mode = infer_input_mode(args)
    task = args.task or ("" if args.resume else default_task(args))
    dry_run = bool(args.dry_run)
    if args.no_isa_contract and not args.no_conformance:
        print("note: --no-isa-contract implies --no-conformance "
              "(conformance consumes the ISA contract)")
    state: CfuDesignState = {
        "task": task,
        "input_kind": input_mode,
        "input_mode": input_mode,
        "benchmark_name": args.embench,
        "project_root": args.project_root,
        "kernel_file": args.kernel_file,
        "kernel_function": args.kernel_function,
        "build_cmd": args.build_cmd,
        "run_cmd": args.run_cmd,
        "test_cmd": args.test_cmd,
        "workdir": str(workdir),
        "out_dir": args.out_dir,
        "model": args.model,
        "llm_config": args.llm_config,
        "dry_run": dry_run,
        "agent_tools": args.agent_tools,
        "max_tool_rounds": args.max_tool_rounds,
        "max_tool_calls": args.max_tool_calls,
        "profile_cache": args.profile_cache,
        "ir_features": "off" if args.no_llvm_ir else args.ir_features,
        "llvm_ir": not args.no_llvm_ir,
        "require_conformance": not (args.no_conformance or args.no_isa_contract),
        "isa_contract": not args.no_isa_contract,
        "loop_optimize": args.loop_optimize,
        "loop_policy": args.loop_policy,
        "loop_patience": max(1, args.loop_patience),
        "max_evals": args.max_evals,
        "stream_llm": args.stream_llm,
        "max_output_tokens": args.max_output_tokens,
        "visibility": args.visibility,
        "apply_patch": args.apply_patch and not dry_run,
        "run_commands": args.run_commands and not dry_run,
        "mode": "dry-run" if dry_run else "autonomous",
        "skip_sim": args.skip_sim,
        "skip_yosys": args.skip_yosys,
        "skip_cost": args.skip_cost,
        "cost_backend": args.cost_backend,
        "max_yosys_cells": args.max_yosys_cells,
        "fpga_part": args.fpga_part,
        "target_frequency_mhz": args.target_frequency_mhz,
        "eda_timeout": args.eda_timeout,
        "vivado_threads": args.vivado_threads,
        "validation_mode": args.validation_mode,
        "flow_version": 2,
        "max_iters": args.max_iters,
        "max_answer_repairs": args.max_answer_repairs,
        "max_preverify_repairs": args.max_preverify_repairs,
        "iteration": 0,
        "thread_id": args.thread_id,
        "run_id": args.resume or args.run_id,
        # This agent designs the AgentCfu scaffold only; the flag keeps the
        # choice explicit instead of inferring it from an always-set run root.
        "agent_cfu": True,
    }
    if args.resume:
        explicit = argv if argv is not None else sys.argv[1:]
        names = ("cost_backend", "max_yosys_cells", "fpga_part", "target_frequency_mhz", "eda_timeout", "vivado_threads", "validation_mode")
        state["resume_eda_options"] = {
            name: getattr(args, name) for name in names
            if any(token == "--" + name.replace("_", "-") or token.startswith("--" + name.replace("_", "-") + "=")
                   for token in explicit)
        }
    graph_dir = args.export_graph or str(Path(args.out_dir) / "graph")
    graph_paths = {}
    enable_progress()
    try:
        graph_paths = export_graph_visualization(workdir / graph_dir)
        final_state = run_agent(
            state,
            thread_id=args.thread_id,
            checkpoint=args.checkpoint or bool(args.resume),
            stream=args.stream,
            resume=bool(args.resume),
        )
    except Exception as exc:
        print(f"agent-error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if _flag_env("CFU_AGENT_DEBUG"):
            traceback.print_exc(file=sys.stderr)
        try:
            final_state = report_mod.write_partial_report(state, exc)
        except Exception as report_exc:
            print(f"partial-report-error: {type(report_exc).__name__}: {report_exc}", file=sys.stderr)
            if _flag_env("CFU_AGENT_DEBUG"):
                traceback.print_exc(file=sys.stderr)
            return 1
    print(f"status={final_state.get('status')}")
    print(f"termination={final_state.get('termination_reason', '')}")
    print(f"visibility={final_state.get('visibility', '')}")
    if final_state.get("stopped_at"):
        print(f"stopped_at={final_state['stopped_at']}")
    print(f"report={final_state.get('final_report_path')}")
    print(f"graph_mermaid={graph_paths.get('mermaid')}")
    print(f"graph_ascii={graph_paths.get('ascii')}")
    if args.checkpoint or args.resume:
        print(f"thread_id={final_state.get('run_id', args.resume or args.run_id)}")
    for event in final_state.get("graph_events", []):
        print(f"event={event}")
    if final_state.get("errors"):
        print("errors:")
        for error in final_state["errors"]:
            print(f"- {error}")

    _print_run_summary(final_state)
    _print_token_summary(final_state)
    # planned/complete are successful planning or execution outcomes.
    return 0 if final_state.get("status") in {"complete", "planned"} else 1


def _sim_gate(final_state: CfuDesignState) -> dict:
    gate = (final_state.get("gates") or {}).get("sim")
    return gate if isinstance(gate, dict) else {}


def _cost_summary(final_state: CfuDesignState) -> dict:
    """The structured Yosys cost of the last successful estimator run."""
    for result in reversed(final_state.get("cost_results") or []):
        parsed = result.get("parsed") or {}
        cost = parsed.get("cost") if isinstance(parsed, dict) else None
        if isinstance(cost, dict):
            return cost
    return {}


def _print_token_usage(final_state: CfuDesignState) -> None:
    """Per-stage LLM tokens in the run summary, largest input first.

    The breakdown answers "which stage's input is big" directly in the CLI
    output: every request is attributed to the node that made it, so a stage
    with an oversized prompt (context digest, re-read design surface) shows up
    here without opening run.yaml.
    """
    by_stage = final_state.get("token_usage_by_stage") or {}
    if not by_stage:
        return

    def cached_text(bucket: dict) -> str:
        reported = int(bucket.get("cached_usage_reported", 0) or 0)
        return f"cached={int(bucket.get('cached_input', 0) or 0):>8}" if reported else "cached=n/a    "

    print("  llm_tokens:")
    for name, bucket in sorted(
        by_stage.items(), key=lambda item: (-int(item[1].get("input", 0)), item[0])
    ):
        print(
            f"    {name:<24} in={int(bucket.get('input', 0)):>8} {cached_text(bucket)} "
            f"out={int(bucket.get('output', 0)):>7} calls={int(bucket.get('calls', 0))}"
        )
    total = final_state.get("token_usage") or {}
    if total:
        print(
            f"    {'TOTAL':<24} in={int(total.get('input', 0)):>8} {cached_text(total)} "
            f"out={int(total.get('output', 0)):>7} calls={int(total.get('calls', 0))}"
        )


def _print_run_summary(final_state: CfuDesignState) -> None:
    """Headline results: reference/CFU cycles, speedup, and Yosys cost."""
    sim = _sim_gate(final_state)
    if final_state.get("run_root"):
        view = provenance.performance_view(final_state)
        if not view["current_performance"]["performance_valid"]:
            sim = {"reason": view["current_performance"]["reason"]}
        historical = view["last_observed_valid_performance"]
        if historical:
            print(f"last_observed_valid_performance={historical}")
    reference = int(sim.get("reference_cycles", 0) or 0)
    cfu = int(sim.get("cfu_cycles", 0) or 0)
    speedup_x100 = int(sim.get("speedup_x100", 0) or 0)
    cost = _cost_summary(final_state)

    print()
    print("summary:")
    ablation = report_mod.ablation_lines(final_state)
    if ablation:
        print("  ablation:")
        for line in ablation:
            print(f"    {line.removeprefix('- ')}")
    measurement = report_mod.measurement_lines(final_state)
    if measurement:
        print("  embench_measurement:")
        for line in measurement:
            print(f"    {line.removeprefix('- ')}")
    if reference and cfu:
        print(f"  cycles_before = {reference}")
        print(f"  cycles_after  = {cfu}")
        print(f"  speedup       = {speedup_x100 / 100:.2f}x")
    else:
        reason = str(sim.get("reason", "") or ("skipped" if sim.get("skipped") else "not measured"))
        print("  cycles_before = n/a")
        print("  cycles_after  = n/a")
        print(f"  speedup       = n/a ({reason})")
    if final_state.get("cost_backend") == "vivado":
        for line in ppa.lines(final_state):
            print("  " + line)
    elif cost:
        print(
            "  yosys_cost    = top={top} cells={cells} wire_bits={wire_bits} "
            "memories={memories} memory_bits={memory_bits}".format(**cost)
        )
    else:
        print("  yosys_cost    = n/a")
    cache = (final_state.get("gates") or {}).get("profile") or {}
    cache = cache.get("profile_cache") if isinstance(cache, dict) else None
    if isinstance(cache, dict) and cache.get("status"):
        key = str(cache.get("key", ""))[:12]
        suffix = f" (key {key})" if key else ""
        print(f"  profile_cache = {cache['status']}{suffix}")
    _print_token_usage(final_state)
    rounds = final_state.get("optimization_rounds") or []
    if len(rounds) > 1 or final_state.get("loop_optimize"):
        best = next((item for item in reversed(rounds) if item.get("improved")), None)
        best_round = (best or {}).get("round")
        final_round = rounds[-1].get("round") if rounds else None
        print("  loop:")
        for index, entry in enumerate(rounds):
            marker = ""
            if entry.get("round") == best_round:
                marker = " (best)"
            elif entry.get("improved") is False and entry.get("measured"):
                marker = " (not improved)"
            print(
                f"    round {entry.get('round')}: {entry.get('target')} "
                + flow_optimize._round_metrics_text(entry)
                + flow_optimize._metric_deltas(entry, flow_optimize._previous_measured(rounds, index))
                + marker
            )
        if (
            best_round is not None
            and final_round is not None
            and best_round != final_round
            and rounds[-1].get("measured")
        ):
            print(
                f"    WARNING: the final round is not the best (round {best_round}); "
                "the workspace is the final experiment (not necessarily a strict regression)"
            )
    candidate = final_state.get("best_candidate") or {}
    if candidate:
        print(f"  selected_source = round {candidate.get('round')}: {candidate.get('manifest')}")


def _print_token_summary(final_state: CfuDesignState) -> None:
    usage = final_state.get("token_usage") or {}
    audit = final_state.get("request_usage") or {}
    if audit:
        usage = {"calls": audit["calls"], "input": audit["input_tokens"], "output": audit["output_tokens"],
                 "total": audit["total_tokens"]}
        print(f"request_usage_missing={audit['missing_usage_requests']}")
        print(f"request_usage_known={audit['known_usage']}")
    if usage:
        print()
        print("token_usage:")
        print(f"  calls       = {usage.get('calls', 0)}")
        print(f"  input_tokens= {usage.get('input', 0)}")
        print(f"  output_tokens={usage.get('output', 0)}")
        print(f"  total_tokens= {usage.get('total', 0)}")
    progress = final_state.get("progress_summary")
    if progress:
        print()
        print("progress:")
        print(f"  stages_run   = {progress.get('stages_run')}")
        print(f"  stages_ok    = {progress.get('stages_ok')}")
        print(f"  stages_failed= {progress.get('stages_failed')}")


TEST_API_PROMPT = "Reply with exactly the single word: OK"
TEST_API_FALLBACK = "API_TEST_FALLBACK"


def run_api_test(workdir: Path, args: argparse.Namespace) -> int:
    config_name = args.llm_config or os.environ.get("CFU_AGENT_LLM_CONFIG", "default")
    try:
        config = effective_config(workdir, config_name)
    except LlmConfigError as exc:
        print(f"llm-config-error: {exc}")
        print(f"available-configs: {', '.join(list_configs(workdir)) or 'none'}")
        return 1

    client = OpenAiTextClient(args.model, config=config)
    print(f"config={config['name']}")
    print(f"model={client.model}")
    print(f"base_url={client.base_url or '(default)'}")
    print(f"api_mode={client.api_mode}")
    print(f"api_key_set={bool(client.api_key)}")
    print(f"thinking={getattr(client, 'thinking', False)}")
    print(f"reasoning_effort={getattr(client, 'reasoning_effort', None)}")
    print(f"reasoning_summary={getattr(client, 'reasoning_summary', None)}")

    if not client.available:
        print("api-key-missing: no API key resolved; set the config's api_key_env var or OPENAI_API_KEY")
        return 1

    try:
        result = client.complete(
            "You are a helpful assistant.",
            TEST_API_PROMPT,
            fallback=TEST_API_FALLBACK,
        )
    except Exception as exc:
        print(f"api-error: {exc}")
        return 1

    used = result.used_llm
    reply = (result.text or "").strip()
    print(f"used_llm={used}")
    print(f"api_mode_used={result.api_mode_used or '(none)'}")
    print(f"finish_reason={result.finish_reason or '(none)'}")
    print(f"reasoning_source={result.reasoning_source or '(none)'}")
    print(f"reasoning_chars={len(result.reasoning or '')}")
    print(f"response_extra_keys={list(result.extra_keys)}")
    print(f"reply={reply!r}")
    if used and reply and reply != TEST_API_FALLBACK:
        print("api-status=OK")
        return 0
    print("api-status=FAILED")
    return 1


def infer_input_mode(args: argparse.Namespace) -> str:
    if getattr(args, "embench", ""):
        return "embench"
    if args.project_root or args.kernel_file or args.kernel_function:
        return "c_project"
    return "natural_language"


def default_task(args: argparse.Namespace) -> str:
    if getattr(args, "embench", ""):
        return f"Accelerate the embench-iot benchmark {args.embench} with a CFU."
    if args.kernel_file:
        function = f" function {args.kernel_function}" if args.kernel_function else ""
        return f"Design a CFU for C workload {args.kernel_file}{function}"
    raise SystemExit("--task is required unless --kernel-file is provided")
