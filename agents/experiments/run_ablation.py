#!/usr/bin/env python3
"""Layered ablation experiment for the CFU design agent.

Scheme ``isa-visibility-devconf-ir-v3`` is a cumulative L0-L4 ladder:
ISA Contract -> visibility -> dev conformance -> LLVM IR, with no X cross groups.

    layer        ISA    visibility  conformance-mode  LLVM IR
    l0-none      off    basic       off               off
    l1-isa       on     basic       off               off
    l2-vis       on     extended    off               off
    l3-devconf   on     extended    dev               off
    l4-ir        on     extended    dev               on

Layers explicitly select ``--conformance-mode off/dev`` (never mix it with
``--no-conformance``); promotion is disabled. Dev conformance is agent-authored
C-test evidence, not the legacy YAML-oracle gate. The harness option
``--max-conformance-runs`` defaults to 2: a stage-local budget, NOT two calls
for an entire run. Pass it before any bare ``--``, never as agent passthrough.

``basic`` is visibility-none only in the sense of no EXTRA
repo/skills/SpinalHDL visibility; workload, interfaces and scaffolds remain.
Adjacent contrasts are conditional additions, not independent main effects or
interactions. Small samples have wide uncertainty; IR is static evidence,
not exact-target instructions, measured cycles or a speedup guarantee.

This thin preset shares matrix expansion, sbt locks, infra retries, collection
and aggregation with ``run_matrix.py``; only ``--layers`` defaults to ``ladder``.
Aliases: none/l0 -> l0-none, isa/l1 -> l1-isa, visibility/vis/l2 -> l2-vis,
l3/conf/conformance/devconf -> l3-devconf, l4/ir/llvm-ir -> l4-ir.
Old ``l3-ir`` and ``l2-conf`` are archive-only, rejected for new schedules.
Legacy ``full`` is separate: un-ablated matrix/sweep flow, agent default ``dev``
conformance, NOT the new ladder's last layer or a synonym for l4-ir.

Do not supply a global ``--visibility`` or conformance flag for the ladder.
After ``--``, new layers reject ``--no-isa-contract``, ``--no-llvm-ir``,
``--ir-features``, ``--no-conformance``, ``--conformance-mode``,
``--conformance-promote`` and ``--max-conformance-runs``.
Metadata records ``ablation_scheme='isa-visibility-devconf-ir-v3'`` and layer
controls. Use a NEW unique experiment ID: v3 schema must not resume into or
mix with old v2 batches. Old batches remain available via ``--aggregate-only``.

Generic command (set the variables to configuration names / benchmark list):

    EXP_ID="<new-unique-v3-experiment-id>"
    MODELS="<config-name[,config-name...]>"
    BENCHMARKS="<benchmark[,benchmark...]>"
    python3 agents/experiments/run_ablation.py \
        --exp-id "$EXP_ID" --models "$MODELS" --benchmarks "$BENCHMARKS" \
        --layers ladder --repeats 3 --max-iters 5 --max-conformance-runs 2 \
        --temperature 1.0 --jobs 1 --conda-env mico_env \
        -- --cost-backend vivado

First add ``--dry-run`` BEFORE the bare ``--`` to print commands/write planning
artifacts only: no actual LLM calls or toolchain execution. Five layers x one
model x two benchmarks x R=3 plans 30 runs, excluding infrastructure retries.
Omitting ``--dry-run`` starts autonomous execution; these examples are not run.
Remove ``-- --cost-backend vivado`` to use default Yosys, without Vivado resources.
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

from agents.experiments.harness import cli


def main(argv: list[str] | None = None) -> int:
    # The ladder is the whole point of this entry point; an explicit --layers
    # still wins (and stays visible in experiment.yaml either way).
    return cli.main(argv, default_layers="ladder", description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
