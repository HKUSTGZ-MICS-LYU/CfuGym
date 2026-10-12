"""Matrix expansion and scheduling: which runs exist and in which order they execute."""

from __future__ import annotations

import random

from agents.experiments.harness import naming
# The ``layers`` parameter of ``expand_matrix`` shadows the module name, so the
# vocabulary module is imported under an alias.
from agents.experiments.harness import layers as layer_vocab


def expand_matrix(models, benchmarks, repeats: int, layers=None) -> list[dict]:
    """One entry per run (cell x repeat), repeat-major order before scheduling.

    A cell is (layer, model, benchmark); ``layers`` defaults to the single
    un-ablated ``full`` layer so the classic pass@k matrix is unchanged.
    """
    layer_names = list(layers) if layers else list(layer_vocab.DEFAULT_LAYERS)
    jobs = []
    for repeat in range(1, int(repeats) + 1):
        for model in models:
            for benchmark in benchmarks:
                for layer in layer_names:
                    jobs.append(
                        {
                            "model": model,
                            "benchmark": benchmark,
                            "repeat": repeat,
                            "infra_retry": 0,
                            "layer": layer,
                            "ablation_flags": layer_vocab.layer_flags(layer),
                        }
                    )
    return jobs


def schedule_jobs(jobs: list[dict], seed: int) -> list[dict]:
    """Round-robin over repeats with a seeded shuffle of the cells each round.

    Replicates of the same cell land in different time slots, so gateway load or
    time-of-day drift is spread over all cells instead of concentrating on one.
    """
    rng = random.Random(seed)
    by_repeat: dict[int, list[dict]] = {}
    for job in jobs:
        by_repeat.setdefault(int(job["repeat"]), []).append(dict(job))
    ordered: list[dict] = []
    for repeat in sorted(by_repeat):
        round_jobs = by_repeat[repeat]
        rng.shuffle(round_jobs)
        ordered.extend(round_jobs)
    return ordered


def filter_jobs(jobs: list[dict], only: list[str]) -> list[dict]:
    """Keep jobs whose run name contains every ``--only`` substring."""
    if not only:
        return list(jobs)
    tokens = [t.strip().lower() for t in only if t.strip()]
    kept = []
    for job in jobs:
        name = naming.job_run_name(
            job, job["exp_id"], job["max_iters"], job.get("infra_retry", 0)
        ).lower()
        if all(token in name for token in tokens):
            kept.append(job)
    return kept
