"""Aggregation: records -> runs.csv / failures.csv / pass_at_k.csv / summary.md."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from agents.experiments.harness import layers, metrics
from agents.cfu_design_agent.runtime import provenance

RUNS_COLUMNS = [
    "exp_id", "run_name", "run_id", "model_config", "model", "benchmark", "suite",
    "layer", "ablation_flags", "ablation_observed",
    "repeat", "max_iters", "status", "exit_code", "stopped_at",
    "failure_class", "failure_rung", "failure_detail",
    "repairs_used", "attempt_index", "pass_at_1", "pass_at_3", "pass_at_5",
    "sim_ok", "sim_attempts", "verify_failures", "reference_cycles", "cfu_cycles", "speedup_x100",
    "sim_terminal_reason", "correctness_ok", "timing_ok", "performance_valid", "observed_speedup_x100",
    "measurement_protocol", "measurement", "benchmark_integrity", "measurement_notice",
    "yosys_cells", "yosys_wire_bits",
    "token_calls", "token_in", "token_out", "token_total", "token_cached_in",
    "token_usage_source", "token_usage_complete", "token_usage_warnings", "current_performance", "last_observed_valid_performance",
    "wall_seconds", "profile_cache_status", "visibility", "infra_retries",
    "parse_warnings", "run_dir", "framework_run_dir", "report_path",
    "started_at", "finished_at",
]

FAILURES_COLUMNS = [
    "exp_id", "run_name", "model", "benchmark", "layer", "repeat", "kind", "attempt",
    "failure_class", "failure_rung", "trigger_rung", "repair_target",
    "decision_source", "budget_exhausted", "detail", "evidence",
]

PASS_AT_K_COLUMNS = [
    "model", "benchmark", "layer", "runs", "complete_runs",
    "pass_at_1_budget", "pass_at_3_budget", "pass_at_5_budget",
    "pass_at_1_sample", "pass_at_3_sample", "pass_at_5_sample",
    "pass_at_1_ci95", "pass_at_3_ci95", "pass_at_5_ci95",
    "attempt_index_median", "first_try_rate",
    "calls_mean", "tokens_mean", "verify_failures_mean", "usage_missing_runs",
    "infra_cost_calls", "infra_cost_known_tokens",
    "tokens_per_complete_mean", "wall_h_per_complete_mean",
    "speedup_x100_median", "measurement_protocols", "speedup_warning",
    "yosys_cells_median", "failure_classes",
]


def load_records(exp_dir: Path) -> list[dict]:
    records = []
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return records
    for record_path in sorted(runs_dir.glob("*/record.json")):
        try:
            data = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("terminal"):
            data.setdefault("layer", "full")
            data["measurement_protocol"] = str(data.get("measurement_protocol") or "legacy")
            # Old archive C/T/P=true describes an observed simulation, not a
            # final candidate that later failed its implementation/build rung.
            if not data.get("last_observed_valid_performance"):
                try:
                    data["last_observed_valid_performance"] = provenance.archived_performance(
                        data, evidence=provenance.archive_evidence(record_path))
                except OSError:
                    pass
            if data.get("status") in {"failed", "blocked", "needs_iteration", "initialized"} and data.get("stopped_at") not in {"sim", "cost", "review", "final_report"}:
                data["current_performance"] = {"performance_valid": False, "candidate_matches": False,
                    "reason": "final candidate stopped before simulation", "scope": "current_candidate"}
                data["performance_valid"] = False
                for field in ("speedup_x100", "reference_cycles", "cfu_cycles", "correctness_ok", "timing_ok"):
                    data[field] = None
            records.append(data)
    records.sort(key=lambda r: (str(r.get("model_config")), str(r.get("benchmark")),
                                layers._layer_order(r.get("layer")), int(r.get("repeat") or 0)))
    return records


def write_runs_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {key: metrics._fmt(record.get(key)) for key in RUNS_COLUMNS}
            row["measurement_protocol"] = str(record.get("measurement_protocol") or "legacy")
            for key in ("measurement", "benchmark_integrity", "current_performance", "last_observed_valid_performance"):
                if isinstance(record.get(key), dict):
                    row[key] = json.dumps(record[key], sort_keys=True)
            writer.writerow(row)


def write_failures_csv(records: list[dict], path: Path) -> None:
    rows = []
    for record in records:
        failure = record.get("failure") or {}
        if failure:
            rows.append({
                "exp_id": record.get("exp_id"), "run_name": record.get("run_name"),
                "model": record.get("model_config"), "benchmark": record.get("benchmark"),
                "layer": record.get("layer", "full"),
                "repeat": record.get("repeat"), "kind": "terminal",
                "attempt": record.get("attempt_index"),
                "failure_class": failure.get("failure_class"),
                "failure_rung": failure.get("failure_rung"),
                "trigger_rung": "", "repair_target": "", "decision_source": "",
                "budget_exhausted": failure.get("budget_exhausted"),
                "detail": failure.get("failure_detail"),
                "evidence": (record.get("evidence") or {}).get("report", ""),
            })
        for episode in record.get("episodes") or []:
            rows.append({
                "exp_id": record.get("exp_id"), "run_name": record.get("run_name"),
                "model": record.get("model_config"), "benchmark": record.get("benchmark"),
                "layer": record.get("layer", "full"),
                "repeat": record.get("repeat"), "kind": episode.get("kind", "repair"),
                "attempt": record.get("attempt_index"),
                "failure_class": "", "failure_rung": "",
                "trigger_rung": episode.get("trigger_rung", ""),
                "repair_target": episode.get("repair_target", ""),
                "decision_source": episode.get("decision_source", ""),
                "budget_exhausted": "", "detail": episode.get("detail", ""),
                "evidence": episode.get("evidence", ""),
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURES_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: metrics._fmt(row.get(key)) for key in FAILURES_COLUMNS})


def write_pass_at_k_csv(cells: list[tuple[str, str, str, dict]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PASS_AT_K_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for layer, model, benchmark, metrics_row in cells:
            row = {"model": model, "benchmark": benchmark, "layer": layer}
            row.update(metrics_row)
            writer.writerow({key: metrics._fmt(row.get(key)) for key in PASS_AT_K_COLUMNS})


def _ablation_summary_lines(cells: list[tuple[str, str, str, dict]]) -> list[str]:
    """The capability-ladder section: what each layer adds, compared side by side."""
    layers_present = [name for name in layers.ABLATION_LADDER if any(c[0] == name for c in cells)]
    layers_present += sorted({str(c[0]) for c in cells} - set(layers.ABLATION_LADDER))
    lines = ["## Ablation ladder", ""]
    lines.append("Layers add exactly one capability each, in the order "
                 "ISA Contract -> Conformance -> LLVM IR.")
    lines.append("")
    lines.append("| layer | adds | agent flags |")
    lines.append("| --- | --- | --- |")
    for layer in layers_present:
        entry = layers.ABLATION_LAYERS.get(layer, {})
        flags = " ".join(entry.get("flags", ())) or "(none)"
        lines.append(f"| {layer} | {entry.get('adds', '?')} | `{flags}` |")
    lines.append("")
    lines.append("### Layer comparison (pass@k budget, k=1/3/5; complete/runs)")
    lines.append("")
    by_key = {(layer, model, benchmark): m for layer, model, benchmark, m in cells}
    pairs = sorted({(model, benchmark) for _, model, benchmark, _ in cells})
    lines.append("| model | benchmark | " + " | ".join(layers_present) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in layers_present) + " |")
    for model, benchmark in pairs:
        row = [model, benchmark]
        for layer in layers_present:
            m = by_key.get((layer, model, benchmark))
            if not m:
                row.append("—")
                continue
            triple = "/".join(metrics._fmt(m.get(f"pass_at_{k}_budget")) or "—" for k in metrics.PASS_AT_K)
            row.append(f"{triple} ({m['complete_runs']}/{m['runs']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def write_summary_md(exp_dir: Path, meta: dict, records: list[dict], cells: list[tuple[str, str, str, dict]]) -> Path:
    lines = [f"# Experiment {meta.get('exp_id', exp_dir.name)}", ""]
    lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- estimand: {metrics.ESTIMAND_HELP}")
    lines.append(f"- git: {meta.get('git_commit', '?')} (dirty={meta.get('git_dirty')})")
    lines.append(f"- runs: {len(records)}")
    lines.append("")

    # The layer column only earns its space when the experiment is ablated.
    layers_seen = {str(c[0]) for c in cells}
    show_layer = bool(layers_seen - {"full"})
    head = "| model | benchmark | " + ("layer | " if show_layer else "") \
        + "runs | complete | pass@%d | CI95 | sample | attempts(med) |"

    for k in metrics.PASS_AT_K:
        lines.append(f"## pass@{k} (budget estimator)")
        lines.append("")
        lines.append(head % k)
        if show_layer:
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        else:
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for layer, model, benchmark, m in cells:
            row = f"| {model} | {benchmark} | "
            if show_layer:
                row += f"{layer} | "
            row += (
                f"{m['runs']} | {m['complete_runs']} "
                f"| {metrics._fmt(m.get(f'pass_at_{k}_budget'))} | {m.get(f'pass_at_{k}_ci95', '')} "
                f"| {metrics._fmt(m.get(f'pass_at_{k}_sample'))} | {metrics._fmt(m.get('attempt_index_median'))} |"
            )
            lines.append(row)
        lines.append("")

    if show_layer:
        lines.extend(_ablation_summary_lines(cells))

    lines.append("## Failure breakdown (failure_class)")
    lines.append("")
    lines.append("| model | benchmark | " + ("layer | " if show_layer else "") + "failure classes |")
    lines.append("| --- | --- | " + ("--- | " if show_layer else "") + "--- |")
    for layer, model, benchmark, m in cells:
        row = f"| {model} | {benchmark} | " + (f"{layer} | " if show_layer else "")
        lines.append(row + f"{m.get('failure_classes') or '—'} |")
    lines.append("")

    lines.extend([
        "## Embench measurement protocols", "",
        "Cold single invocation (`cold_single_invocation_v1`) uses no warmup and local/global "
        "scale factors 1. Timing includes original preparation, memory and CFU configuration; "
        "it is not pure math-kernel latency. Missing protocol evidence is labelled `legacy`.",
        "New and legacy measurements are not comparable. A mixed-protocol cell has no speedup "
        "median; pass@k, effort and cost statistics are unaffected. Integrity gate status is "
        "not proof of true operation equivalence or an arbitrary semantic anti-cheat guarantee.",
        "", "| model | benchmark | layer | complete-run protocols | notice |",
        "| --- | --- | --- | --- | --- |",
    ])
    for layer, model, benchmark, m in cells:
        lines.append(f"| {model} | {benchmark} | {layer} | "
                     f"{m.get('measurement_protocols') or '—'} | {m.get('speedup_warning') or '—'} |")
    lines.append("")

    lines.append("## Cost of successful runs")
    lines.append("")
    lines.append("| model | benchmark | " + ("layer | " if show_layer else "")
                 + "tokens/run | wall h/run | speedup_x100 | yosys cells |")
    lines.append("| --- | --- | " + ("--- | " if show_layer else "") + "--- | --- | --- | --- |")
    for layer, model, benchmark, m in cells:
        row = f"| {model} | {benchmark} | " + (f"{layer} | " if show_layer else "")
        lines.append(
            row + f"{metrics._fmt(m.get('tokens_per_complete_mean'))} "
            f"| {metrics._fmt(m.get('wall_h_per_complete_mean'))} | {metrics._fmt(m.get('speedup_x100_median'))} "
            f"| {metrics._fmt(m.get('yosys_cells_median'))} |"
        )
    lines.append("")

    lines.append("## Request usage completeness")
    lines.append("")
    lines.append("Costs use one final known usage per request ID, including failed requests. Unknown usage is missing, not zero; known subtotals are lower bounds. Infrastructure attempts remain outside pass@k, but their observed cost is retained.")
    lines.append("")
    lines.append("| model | benchmark | layer | missing-usage model runs | infra calls | infra known tokens (lower bound) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for layer, model, benchmark, m in cells:
        lines.append(f"| {model} | {benchmark} | {layer} | {m.get('usage_missing_runs', 0)} | {m.get('infra_cost_calls', 0)} | {m.get('infra_cost_known_tokens', 0)} |")
    lines.append("")
    lines.append("## Run effort & verification load (means over all runs)")
    lines.append("")
    lines.append("calls/run and tokens/run are LLM effort per attempted run; "
                 "verify failures/run counts simulation attempts where correctness "
                 "verification (the benchmark's verify oracle / fail symbol / the "
                 "harness result comparison) rejected the accelerated build.")
    lines.append("")
    lines.append("| model | benchmark | " + ("layer | " if show_layer else "")
                 + "calls/run | tokens/run | verify failures/run |")
    lines.append("| --- | --- | " + ("--- | " if show_layer else "") + "--- | --- | --- |")
    for layer, model, benchmark, m in cells:
        row = f"| {model} | {benchmark} | " + (f"{layer} | " if show_layer else "")
        lines.append(
            row + f"{metrics._fmt(m.get('calls_mean'))} | {metrics._fmt(m.get('tokens_mean'))} "
            f"| {metrics._fmt(m.get('verify_failures_mean'))} |"
        )
    lines.append("")

    infra_runs = [r for r in records if (r.get("infra") or {}).get("infra")]
    lines.append("## Infrastructure incidents")
    lines.append("")
    if not infra_runs:
        lines.append("none")
    else:
        lines.append("| run | kind | reason | retry |")
        lines.append("| --- | --- | --- | --- |")
        for record in infra_runs:
            infra = record["infra"]
            lines.append(
                f"| {record.get('run_name')} | {infra.get('infra_kind')} "
                f"| {str(infra.get('infra_reason', ''))[:80]} | infra{record.get('infra_retries', 0)} |"
            )
    lines.append("")
    lines.append("Details per run: `runs/<run_name>/record.json` (+ `failure.json`); "
                 "tables: `results/runs.csv`, `results/failures.csv`, `results/pass_at_k.csv`.")
    lines.append("")

    path = exp_dir / "results" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def aggregate(exp_dir: Path, meta: dict, bootstrap_iters: int = 10000) -> dict:
    records = load_records(exp_dir)
    # One cell per (layer, model, benchmark): ablation layers of the same
    # model x benchmark pair must never pool into one number.
    by_cell: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        key = (str(record.get("layer") or "full"),
               str(record.get("model_config")), str(record.get("benchmark")))
        by_cell.setdefault(key, []).append(record)
    cells = []
    for (layer, model, benchmark), cell_records in sorted(
            by_cell.items(), key=lambda kv: (layers._layer_order(kv[0][0]), kv[0][1], kv[0][2])):
        cells.append((layer, model, benchmark, metrics.cell_metrics(cell_records, bootstrap_iters)))

    results = exp_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_runs_csv(records, results / "runs.csv")
    write_failures_csv(records, results / "failures.csv")
    write_pass_at_k_csv(cells, results / "pass_at_k.csv")
    summary = write_summary_md(exp_dir, meta, records, cells)
    return {"records": len(records), "cells": len(cells), "summary": str(summary)}
