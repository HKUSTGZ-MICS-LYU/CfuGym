"""CLI for the CFU design LangGraph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .graph import enable_progress, export_graph_visualization, run_agent
from .llm import OpenAiTextClient
from .llm_config import LlmConfigError, effective_config, list_configs
from .state import CfuDesignState


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
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Force read-only: never mutate source or run external commands.")
    parser.add_argument("--apply-patch", action="store_true", help="Allow an LLM patch to be applied to CFU/software allowlisted paths.")
    parser.add_argument("--run-commands", action="store_true", help="Allow allowlisted build/simulator/Yosys commands to execute.")
    parser.add_argument("--no-dry-run", action="store_true", help="Allow mutation and command execution (implies apply/run unless --dry-run).")
    parser.add_argument("--skip-sim", action="store_true", help="Skip SBT generation/simulation.")
    parser.add_argument("--skip-yosys", action="store_true", help="Skip Yosys cost estimation.")
    parser.add_argument("--export-graph", default="", help="Directory for Mermaid/ASCII graph visualization.")
    parser.add_argument("--checkpoint", action="store_true", help="Compile with an in-memory LangGraph checkpointer.")
    parser.add_argument("--thread-id", default="", help="Thread id used when checkpointing a graph run.")
    parser.add_argument("--stream", action="store_true", help="Run through LangGraph streaming state snapshots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()

    if args.test_api:
        return run_api_test(workdir, args)

    input_mode = infer_input_mode(args)
    task = args.task or default_task(args)
    dry_run = True if args.dry_run else not (args.apply_patch or args.run_commands or args.no_dry_run)
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
        "apply_patch": args.apply_patch and not dry_run,
        "run_commands": args.run_commands and not dry_run,
        "mode": "dry-run" if dry_run else "autonomous",
        "skip_sim": args.skip_sim,
        "skip_yosys": args.skip_yosys,
        "max_iters": args.max_iters,
        "iteration": 0,
        "thread_id": args.thread_id,
        "run_id": args.run_id,
        # This agent designs the AgentCfu scaffold only; the flag keeps the
        # choice explicit instead of inferring it from an always-set run root.
        "agent_cfu": True,
    }
    graph_dir = args.export_graph or str(Path(args.out_dir) / "graph")
    graph_paths = export_graph_visualization(workdir / graph_dir)
    enable_progress()
    final_state = run_agent(
        state,
        thread_id=args.thread_id,
        checkpoint=args.checkpoint,
        stream=args.stream,
    )
    print(f"status={final_state.get('status')}")
    print(f"report={final_state.get('final_report_path')}")
    print(f"graph_mermaid={graph_paths.get('mermaid')}")
    print(f"graph_ascii={graph_paths.get('ascii')}")
    if args.checkpoint:
        print(f"thread_id={(args.thread_id or 'cfu-design-agent')[:255]}")
    for event in final_state.get("graph_events", []):
        print(f"event={event}")
    if final_state.get("errors"):
        print("errors:")
        for error in final_state["errors"]:
            print(f"- {error}")

    _print_token_summary(final_state)
    # planned/complete are successful planning or execution outcomes.
    return 0 if final_state.get("status") in {"complete", "planned"} else 1


def _print_token_summary(final_state: CfuDesignState) -> None:
    usage = final_state.get("token_usage") or {}
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
