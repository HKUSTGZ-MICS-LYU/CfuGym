"""CLI for the CFU design LangGraph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .graph import export_graph_visualization, run_agent
from .llm import DEFAULT_MODEL
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
    parser.add_argument("--model", default=os.environ.get("CFU_AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Do not mutate CFU source or run mutating commands.")
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
    input_mode = infer_input_mode(args)
    task = args.task or default_task(args)
    state: CfuDesignState = {
        "task": task,
        "input_kind": input_mode,
        "input_mode": input_mode,
        "project_root": args.project_root,
        "kernel_file": args.kernel_file,
        "kernel_function": args.kernel_function,
        "build_cmd": args.build_cmd,
        "run_cmd": args.run_cmd,
        "test_cmd": args.test_cmd,
        "workdir": str(workdir),
        "out_dir": args.out_dir,
        "model": args.model,
        "dry_run": args.dry_run,
        "mode": "dry-run" if args.dry_run else "autonomous",
        "skip_sim": args.skip_sim,
        "skip_yosys": args.skip_yosys,
        "max_iters": args.max_iters,
        "iteration": 0,
        "thread_id": args.thread_id,
    }
    graph_dir = args.export_graph or str(Path(args.out_dir) / "graph")
    graph_paths = export_graph_visualization(workdir / graph_dir)
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
    return 0 if final_state.get("status") == "complete" else 1


def infer_input_mode(args: argparse.Namespace) -> str:
    if args.project_root or args.kernel_file or args.kernel_function:
        return "c_project"
    return "natural_language"


def default_task(args: argparse.Namespace) -> str:
    if args.kernel_file:
        function = f" function {args.kernel_function}" if args.kernel_function else ""
        return f"Design a CFU for C workload {args.kernel_file}{function}"
    raise SystemExit("--task is required unless --kernel-file is provided")
