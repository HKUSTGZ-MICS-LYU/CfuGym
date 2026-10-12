#!/usr/bin/env python3
"""Sweep every embench-iot benchmark once, in order, for the CFU design agent.

A thin preset over the shared experiment infrastructure in
``agents/experiments/harness/`` (sbt lock shim, infra retries, collection,
failure classification, aggregation -- all flags accepted unchanged).  What
differs from the pass@k matrix (``run_matrix.py``) and the ablation ladder
(``run_ablation.py``):

* ``--benchmarks`` defaults to EVERY benchmark of ``benchmarks/embench-iot``
  (discovered via the framework's ``benchmark.embench.list_benchmarks``, i.e.
  the suite's alphabetical order); an explicit ``--benchmarks`` subset wins.
* Each benchmark runs exactly ONCE per model (``--repeats`` defaults to 1).
* Runs execute strictly in that order (no scheduling shuffle).  Strict order
  needs ``--jobs 1`` (the default); a larger ``--jobs`` overlaps runs.
* ``--models`` defaults to a single config (``deepseek-pro``); pass a comma
  list to sweep the suite once per model.
* Everything after a bare ``--`` is forwarded verbatim to
  ``agents/run_cfu_agent.py``, so that CLI's original parameters apply to
  every run (e.g. ``--skip-sim``, ``--skip-yosys``, ``--task``,
  ``--no-conformance``).  Flags the harness owns (``--embench``/
  ``--llm-config``/``--run-id``/``--max-iters``/``--visibility``/
  ``--profile-cache``/``--workdir``/``--out-dir``) are rejected there;
  express those through ``--models``/``--max-iters``/``--visibility``/
  ``--profile-cache``.  Ablation switches passed here are folded into each
  run's recorded ablation flags; ``--layers`` stays the way to schedule the
  capability ladder.

Usage (see --help for everything):

    # one full run per suite benchmark, in order:
    python3 agents/experiments/run_sweep.py --exp-id sweep-20260213

    # subset / explicit order / agent parameters passed through:
    python3 agents/experiments/run_sweep.py --exp-id sweep-quick \\
        --benchmarks crc32,matmult-int -- --skip-sim --skip-yosys

    # no-conformance mode for every run (keep ISA contract + LLVM IR):
    python3 agents/experiments/run_sweep.py --exp-id sweep-noconf -- --no-conformance
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.experiments.harness import cli


def all_embench_benchmarks(workdir: Path) -> list[str]:
    """Every benchmark in the embench-iot suite, in suite (sorted) order.

    The suite knowledge belongs to the framework's benchmark adapter, so this
    reaches through its owning module instead of re-listing ``src/`` here.
    (Importing the adapter pulls the agent's dependencies, so run this preset
    in the agent environment, e.g. ``conda run -n mico_env python3 ...``.)
    """
    try:
        from agents.cfu_design_agent.benchmark import embench
    except ImportError as exc:  # missing agent deps (langgraph, ...)
        raise SystemExit(
            f"cannot import the agent's benchmark adapter ({exc}); "
            "run this preset in the agent environment (conda run -n mico_env) "
            "or pass --benchmarks explicitly"
        ) from exc

    names = embench.list_benchmarks(workdir)
    if not names:
        raise SystemExit(
            f"no embench-iot benchmarks found under {workdir / embench.EMBENCH_ROOT / 'src'}; "
            "check --workdir or pass --benchmarks explicitly"
        )
    return names


def main(argv: list[str] | None = None) -> int:
    return cli.main(
        argv,
        default_layers="full",
        description=__doc__,
        default_models=("deepseek-pro",),
        default_benchmarks=all_embench_benchmarks,
        default_repeats=1,
        ordered=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
