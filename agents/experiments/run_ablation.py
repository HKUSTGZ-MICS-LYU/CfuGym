#!/usr/bin/env python3
"""Layered ablation experiment for the CFU design agent.

One harness run drives the *capability ladder* over the model x benchmark
matrix, adding exactly one capability per layer, in the order

    ISA Contract -> Conformance -> LLVM IR

    l0-none   --no-isa-contract --no-conformance --no-llvm-ir   (nothing)
    l1-isa    --no-conformance --no-llvm-ir                     (+ ISA Contract)
    l2-conf   --no-llvm-ir                                      (+ Conformance)
    full      (no flags)                                        (+ LLVM IR)

(``--no-isa-contract`` implies ``--no-conformance``, so l0-none spells both out
and conformance cannot exist without the contract.)  Each (layer, model,
benchmark) cell is repeated ``--repeats`` times with the same controls as the
pass@k matrix, so layer differences are attributable to the added capability.

This is a thin preset over the shared experiment infrastructure in
``agents/experiments/harness/``: every flag of the harness is accepted
unchanged (sbt lock shim, infra retries, collection, failure classification,
aggregation); only ``--layers`` defaults to ``ladder`` instead of the single
un-ablated ``full`` layer.  Aggregation writes a ``layer`` column into
``runs.csv`` / ``failures.csv`` / ``pass_at_k.csv`` and a "## Ablation ladder"
comparison section into ``summary.md``.

Usage (see --help for everything; args after a bare ``--`` are forwarded
verbatim to agents/run_cfu_agent.py):

    python3 agents/experiments/run_ablation.py \
        --exp-id abl-20260213 \
        --models deepseek-pro,glm-5.2,gpt-5.6-terra \
        --benchmarks crc32,matmult-int,md5sum \
        --repeats 3 --max-iters 5 --temperature 0.7 \
        --conda-env mico_env --jobs 2

    # single layer / subset (aliases: none, isa, conf, llvm-ir):
    python3 agents/experiments/run_ablation.py --exp-id abl-l2 --layers l2-conf ...
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
