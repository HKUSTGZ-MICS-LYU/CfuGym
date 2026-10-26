"""pass@k estimators and per-cell metrics."""

from __future__ import annotations

import math
import random
import statistics

from agents.experiments.harness import classification

PASS_AT_K = (1, 3, 5)

ESTIMAND_HELP = (
    "attempt_index = verify_submissions (one --max-iters K run allows K candidate submissions to verification). "
    "pass@k_budget = mean over R repeats of 1[complete and attempt_index <= k]; "
    "pass@k_sample = 1 - C(R-c,k)/C(R,k) with c complete runs (i.i.d. estimator)."
)


def _pass_flag(record: dict, k: int) -> int | None:
    """1/0 for a complete run, None (CSV-empty) for a failure."""
    if record.get("status") != "complete":
        return None
    attempt = record.get("attempt_index")
    if not isinstance(attempt, int):
        return None
    return 1 if attempt <= k else 0


def budget_pass_at_k(records: list[dict], k: int) -> float | None:
    if not records:
        return None
    hits = sum(1 for r in records if _pass_flag(r, k) == 1)
    return hits / len(records)


def sample_pass_at_k(records: list[dict], k: int) -> float | None:
    """1 - C(n-c, k) / C(n, k): i.i.d. 'at least one of k samples' estimator."""
    n = len(records)
    c = sum(1 for r in records if r.get("status") == "complete")
    if n == 0 or k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_budget_ci(records: list[dict], k: int, seed: int = 20260212, iters: int = 10000) -> tuple[float | None, float | None]:
    if not records:
        return None, None
    values = [1 if _pass_flag(r, k) == 1 else 0 for r in records]
    rng = random.Random(seed + k)
    means = []
    for _ in range(int(iters)):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _median(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(nums) if nums else None


def _mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return (sum(nums) / len(nums)) if nums else None


def cell_metrics(records: list[dict], bootstrap_iters: int = 10000) -> dict:
    # Retries are audit records, not independent model samples or model failures.
    all_records = records
    records = [r for r in records if not (r.get("infra") or {}).get("infra")]
    n = len(records)
    complete = [r for r in records if r.get("status") == "complete"]
    failure_tally: dict[str, int] = {}
    for record in records:
        failure = record.get("failure") or {}
        cls = failure.get("failure_class")
        if cls:
            failure_tally[cls] = failure_tally.get(cls, 0) + 1
    protocols = sorted({str(r.get("measurement_protocol") or "legacy") for r in complete})
    mixed_protocols = len(protocols) > 1
    speedup_warning = (
        "Speedup median unavailable: complete runs use different measurement protocols "
        f"({', '.join(protocols)}); new and legacy measurements are not comparable."
        if mixed_protocols else ""
    )
    row: dict = {
        "runs": n,
        "complete_runs": len(complete),
        "attempt_index_median": _median([r.get("attempt_index") for r in complete]),
        "first_try_rate": budget_pass_at_k(records, 1),
        # Effort / verification load, mean over ALL non-infra runs (a failed run
        # still spent calls and tokens and may have failed verification several
        # times).  Unknown values (older records) are skipped, never counted 0.
        "calls_mean": _mean([r.get("token_calls") for r in records]),
        "tokens_mean": _mean([r.get("token_total") for r in records]),
        "usage_missing_runs": sum(r.get("token_usage_complete") is False or r.get("token_total") is None for r in records),
        "cost_observed_runs": len(records),
        "infra_cost_calls": sum(r.get("token_calls") or 0 for r in all_records if (r.get("infra") or {}).get("infra")),
        "infra_cost_known_tokens": sum((r.get("request_usage") or {}).get("known_usage", {}).get("total_tokens", r.get("token_total") or 0)
                                      for r in all_records if (r.get("infra") or {}).get("infra")),
        "verify_failures_mean": _mean([r.get("verify_failures") for r in records]),
        "tokens_per_complete_mean": _mean([r.get("token_total") for r in complete]),
        "wall_h_per_complete_mean": (
            (_mean([r.get("wall_seconds") for r in complete]) or 0) / 3600.0 if complete else None
        ),
        "measurement_protocols": ", ".join(protocols),
        "speedup_warning": speedup_warning,
        "speedup_x100_median": (
            None if mixed_protocols else _median([r.get("speedup_x100") for r in complete
                                                   if classification.performance_is_valid(r)])
        ),
        "yosys_cells_median": _median([r.get("yosys_cells") for r in complete]),
        "failure_classes": ", ".join(f"{k}:{v}" for k, v in sorted(failure_tally.items())),
    }
    for k in PASS_AT_K:
        row[f"pass_at_{k}_budget"] = budget_pass_at_k(records, k)
        row[f"pass_at_{k}_sample"] = sample_pass_at_k(records, k)
        lo, hi = bootstrap_budget_ci(records, k, iters=bootstrap_iters)
        row[f"pass_at_{k}_ci95"] = f"[{_fmt(lo)}, {_fmt(hi)}]"
    return row
