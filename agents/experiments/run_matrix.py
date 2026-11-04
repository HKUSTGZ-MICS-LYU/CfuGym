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

Ablation layers (``--layers``): scheme ``isa-visibility-devconf-ir-v3`` adds
ISA Contract -> visibility -> dev conformance -> LLVM IR (L0-L4, no X groups):

    layer        ISA    visibility  conformance-mode  LLVM IR
    l0-none      off    basic       off               off
    l1-isa       on     basic       off               off
    l2-vis       on     extended    off               off
    l3-devconf   on     extended    dev               off
    l4-ir        on     extended    dev               on

Layers explicitly set ``--conformance-mode off/dev``, never mixed with
``--no-conformance``; promotion is disabled. Dev conformance is evidence, not a
YAML-oracle gate. Harness ``--max-conformance-runs`` defaults to 2, stage-local,
NOT two conformance calls for a whole run; place it before any bare ``--``.
The lowest tier ``basic`` means no EXTRA repo/skills/SpinalHDL visibility,
not zero source access: workload, interfaces and necessary scaffolds remain.
This cumulative design estimates conditional additions, not independent
main effects/interactions. IR is static evidence, not measured cycles.

``--layers ladder`` runs five layers per cell: one model x two benchmarks x
R=3 = 30 planned runs, excluding infrastructure retries. The default ``full``
remains the separate legacy un-ablated matrix/sweep preset (agent default
conformance ``dev``, unsuffixed run names), NOT l4-ir or the new final layer.
Aliases none/l0, isa/l1 and visibility/vis/l2 select L0/L1/L2;
l3/conf/conformance/devconf -> l3-devconf, l4/ir/llvm-ir -> l4-ir.
Old l3-ir and l2-conf are archive-only: no new schedules, but old batches
remain readable via ``--aggregate-only``.
New layers reject global ``--visibility`` and passthrough ``--no-isa-contract``,
``--no-llvm-ir``, ``--ir-features``, ``--no-conformance``, ``--conformance-mode``,
``--conformance-promote`` and ``--max-conformance-runs``. Do not add global
conformance flags. Use a NEW unique experiment ID: metadata records
``ablation_scheme='isa-visibility-devconf-ir-v3'`` and layer controls;
v3 schema cannot resume into or mix with old v2 batches.

Each new layer carries its token in the run name; records and aggregate tables
retain the layer coordinate, and the summary compares layers side by side.
Ablation flags are cross-checked against the report's ``## Ablation`` section,
so silently ignored switches are flagged.

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

    # Generic new-ladder command; set variables before use (not executed here).
    EXP_ID="<new-unique-v3-experiment-id>"
    MODELS="<config-name[,config-name...]>"
    BENCHMARKS="<benchmark[,benchmark...]>"
    python3 agents/experiments/run_ablation.py \
        --exp-id "$EXP_ID" --models "$MODELS" --benchmarks "$BENCHMARKS" \
        --layers ladder --repeats 3 --max-iters 5 --max-conformance-runs 2 \
        --temperature 1.0 --jobs 1 --conda-env mico_env \
        -- --cost-backend vivado

First add ``--dry-run`` BEFORE the bare ``--`` to print/write the plan only:
no actual LLM calls or toolchain execution (30 runs for one model and two
benchmarks). The matrix entry is equivalent with explicit ``--layers ladder``.
Without ``--dry-run`` the command starts autonomous execution. Remove the
``-- --cost-backend vivado`` tail to use default Yosys, without Vivado resources.
Summaries report speedup min/max/mean/median over valid successful samples and
actual Vivado LUT/FF/BRAM/DSP resources. Missing data stays empty; Yosys cells
are never substituted for Vivado resources.
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
