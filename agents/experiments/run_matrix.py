#!/usr/bin/env python3
"""LLM x benchmark pass@1/3/5 experiment matrix harness for the CFU design agent.

Design: docs/agent_pass_at_k_experiment_matrix.md

This is the pass@k preset over the shared experiment infrastructure in
``agents/experiments/harness/`` (matrix expansion, sbt lock shim, infra
retries, collection, failure classification, aggregation); only the
``--layers`` default differs from ``run_ablation.py`` (here: the single
un-ablated ``full`` layer).

One harness run drives a full matrix of ``--run-cfu-agent`` runs:

    <exp_id>-<model>-<benchmark>-r<RR>-mi<K>            e.g. pk5-20260212-glm52-crc32-r01-mi5

Each cell (model x benchmark) is repeated R times with ``--max-iters K`` (= at most
K candidate submissions to the verification ladder, including the first). A run's
``attempt_index`` is its number of verification submissions and therefore contributes

    pass@k = 1  iff the run is ``complete`` and attempt_index <= k        (k = 1, 3, 5)

so one ``--max-iters 5`` run yields the whole pass@1..pass@5 curve.  Aggregation
reports both this budget estimator and the i.i.d. sampling estimator
``pass@k = 1 - C(R-c, k) / C(R, k)`` over the R independent runs.

Every run is collected from stable machine-readable sources (CLI stdout,
``run.yaml``, ``report.md`` gates, ``10_validation/*.yaml``, ``11_cost/result.yaml``)
into ``record.json`` + ``runs.csv``; every failure is classified into a
``failure_class`` (which rung broke and why) and archived as ``failure.json`` +
``failures.csv``.

Ablation layers (``--layers``): the matrix can carry a third dimension besides
model x benchmark -- a *layered capability ladder* that adds exactly one
capability per layer, in the order ISA Contract -> Conformance -> LLVM IR
(``--no-isa-contract`` implies ``--no-conformance``, so conformance cannot
exist without the contract):

    l0-none   --no-isa-contract --no-conformance --no-llvm-ir   (nothing)
    l1-isa    --no-conformance --no-llvm-ir                     (+ ISA Contract)
    l2-conf   --no-llvm-ir                                      (+ Conformance)
    full      (no flags)                                        (+ LLVM IR)

``--layers ladder`` runs all four layers per cell (4x the runs); the default is
the single un-ablated ``full`` layer, which keeps the canonical run name.  Every
ablated layer carries its layer token in the run name; every record and
aggregated table gets a ``layer`` column, and the summary compares the layers
side by side.  What each run actually ablated is cross-checked against the
report's ``## Ablation`` section, so silently ignored switches are flagged.

Concurrency / locking: all heavy commands of the agent (SoC generation,
``MiCoSocSim`` RTL simulation, ``GenAgentCfu``) are invoked as ``sbt`` and share
repo-level state (``target/`` incremental compile, ``simWorkspace/`` for the
Verilator sim, ivy/coursier caches).  The harness therefore prepends a shim
directory to the child PATH in which ``sbt`` is a wrapper that holds an exclusive
``flock`` around the real ``sbt`` for the duration of each call.  LLM calls,
``make`` software builds and Yosys still overlap across runs; only sbt/RTL-simulation
calls are serialized.  The lock file is shared across harness invocations by
default so two batches cannot collide either.

Usage (see --help for everything; args after a bare ``--`` are forwarded
verbatim to agents/run_cfu_agent.py):

    python3 agents/experiments/run_matrix.py \
        --exp-id pk5-20260212 \
        --models deepseek-pro,glm-5.2,gpt-5.6-terra \
        --benchmarks crc32,matmult-int,md5sum \
        --repeats 5 --max-iters 5 --temperature 0.7 \
        --conda-env mico_env --jobs 2

    # layered ablation (same harness; --layers defaults to the full ladder):
    python3 agents/experiments/run_ablation.py \
        --exp-id abl-20260212 \
        --models deepseek-pro,glm-5.2,gpt-5.6-terra \
        --benchmarks crc32,matmult-int,md5sum \
        --repeats 3 --max-iters 5 --temperature 0.7 \
        --conda-env mico_env --jobs 2
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Public surface of the experiment harness, re-exported for compatibility: the
# pass@k harness historically lived in this module and callers (including the
# unit tests) reach its helpers as ``run_matrix.<name>``.  To PATCH a seam in a
# test, patch its OWNING module (e.g. ``harness.execution.run_logged``): an
# attribute assigned on this facade is a separate binding and is not observed
# by the owning module's code.
from agents.experiments.harness import *  # noqa: F401,F403
from agents.experiments.harness import cli


def build_parser():
    """The CLI parser of this preset (``--layers`` defaults to ``full``)."""
    return cli.build_parser(description=__doc__, default_layers="full")


def main(argv: list[str] | None = None) -> int:
    return cli.main(argv, default_layers="full", description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
