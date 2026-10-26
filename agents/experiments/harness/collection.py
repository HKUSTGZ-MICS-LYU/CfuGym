"""Collection: turn one run's artifacts into a flat record + nested evidence.

Every run is collected from stable machine-readable sources (CLI stdout,
``run.yaml``, ``report.md`` gates, ``10_validation/*.yaml``, ``11_cost/result.yaml``)
into ``record.json``; every failure is archived as ``failure.json``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from agents.experiments.harness import classification, layers, metrics, parsing
from agents.cfu_design_agent.runtime import provenance, request_usage

SUITE = "embench-iot"

# --------------------------------------------------------------------------
# Simulation attempts and verification failures
# --------------------------------------------------------------------------
#
# "Verify" is the benchmark's correctness oracle (embench ``verify_benchmark``):
# a *verify failure* is a simulation attempt where correctness verification
# rejected the accelerated build's observable result (verify oracle said fail,
# the CPU reached the fail symbol, or the harness-side reference/CFU result
# comparison disagreed).  Timeouts, speedup-floor and harness crashes are sim
# failures but NOT verify failures.
#
# Evidence: every sim attempt writes its ladder artifact ``10_validation/sim.yaml``
# and the artifact store keeps one content-addressed
# ``versions/sim.yaml.<sha256>`` file per write, carrying the attempt's
# ``parsed.verify`` verdict and ``failure_observation.category``.  run.yaml is
# NOT usable for counting attempts: its manifest merges artifact refs by path
# and keeps only the newest entry per path.


def collect_sim_attempts(framework_run_dir: Path, warnings: list[str]) -> list[dict]:
    """One dict per simulation attempt this run actually executed.

    The version files enumerate the attempts (an attempt's bytes embed its
    commands' ``log_path`` and output tails, so successive attempts do not
    collide in practice); the latest ``sim.yaml`` alone is the fallback for
    stores that kept no versions.
    """
    paths = sorted((framework_run_dir / "10_validation" / "versions").glob("sim.yaml.*"))
    if not paths:
        latest = framework_run_dir / "10_validation" / "sim.yaml"
        if latest.exists():
            paths = [latest]
    attempts = []
    for path in paths:
        attempt = parsing.load_yaml(path, warnings, "sim attempt")
        commands = attempt.get("commands")
        if isinstance(commands, list) and not any(
                isinstance(cmd, dict) and cmd.get("kind") not in {"host_gate_check", "semantic_check"}
                and not cmd.get("skipped") and cmd.get("returncode") is not None
                for cmd in commands):
            continue
        attempts.append(attempt)
    return attempts


def verify_failure_in_attempt(attempt: dict, reference_result: int | None) -> bool:
    """True when this simulation attempt failed correctness verification."""
    if attempt.get("terminal_reason"):
        return attempt["terminal_reason"] in {"verify_failed", "fail_symbol", "result_mismatch", "result_missing"}
    for cmd in attempt.get("commands") or []:
        if not isinstance(cmd, dict):
            continue
        parsed = cmd.get("parsed") or {}
        verify = parsed.get("verify") or {}
        if str(verify.get("status") or "") == "fail":
            return True
        observation = cmd.get("failure_observation") or {}
        if str(observation.get("category") or "") == "fail_symbol":
            return True
        text = f"{cmd.get('stdout_tail') or ''}\n{cmd.get('stderr_tail') or ''}".lower()
        if "embench verify failed" in text or "fail symbol" in text:
            return True
        # Harness-side independent check: the reference run and the CFU run
        # must compute the same observable result value.
        if reference_result is not None:
            bench = str(verify.get("benchmark") or "")
            profile = parsed.get("embench_profile") or {}
            entry = profile.get(bench) if bench else (next(iter(profile.values()), {}) or {})
            got = (entry or {}).get("result")
            if got is not None and int(got) != int(reference_result):
                return True
    return False


def reference_result_value(framework_run_dir: Path, warnings: list[str]) -> int | None:
    """The scalar reference run's observable result (the profile stage's ground truth).

    It is produced before any CFU work, so it is what the harness-side result
    comparison compares every accelerated build against.
    """
    profile_yaml = parsing.load_yaml(framework_run_dir / "05_profile" / "embench.yaml", warnings, "profile embench.yaml")
    bench = str(profile_yaml.get("benchmark") or "")
    profiles = (profile_yaml.get("profiles") or {}).get("profile") or {}
    entry = profiles.get(bench) if bench else (next(iter(profiles.values()), {}) or {})
    result = (entry or {}).get("result")
    return int(result) if isinstance(result, (int, float)) else None


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO timestamp, or None when the text is not one."""
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def framework_artifacts_are_stale(framework_run_dir: Path, started_at: str) -> bool:
    """True when the framework directory was last written before this attempt.

    Run ids are deterministic, so a re-run appends into the same directory. The
    merged ``run.yaml`` keeps the previous attempt's ``created_at`` *and* its
    terminal ``status`` when the new attempt dies early, so file mtime -- not
    ``created_at`` -- is what says whether this attempt wrote anything at all.
    Unparseable timestamps (test fixtures, hand-written records) never count as
    stale.
    """
    started = _parse_iso(started_at)
    if started is None:
        return False
    newest: float | None = None
    for name in ("run.yaml", "report.md"):
        path = framework_run_dir / name
        if path.is_file():
            mtime = path.stat().st_mtime
            newest = mtime if newest is None else max(newest, mtime)
    if newest is None:
        return False
    written = datetime.fromtimestamp(newest, timezone.utc)
    if (written.tzinfo is None) != (started.tzinfo is None):
        return False
    return written < started


def collect_record(
    exp_id: str,
    job: dict,
    run_name: str,
    run_dir: Path,
    framework_run_dir: Path,
    meta: dict | None = None,
) -> dict:
    """Collect one run into a flat record (runs.csv row) + nested evidence."""
    meta = dict(meta or {})
    warnings: list[str] = []
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    if not stdout:
        warnings.append("stdout.log missing or empty")

    cli = parsing.parse_cli_stdout(stdout) if stdout else {}
    report_path = framework_run_dir / "report.md"
    report = parsing.parse_report(report_path.read_text(errors="replace"), warnings) if report_path.exists() else {}
    if not report:
        warnings.append("report.md missing")
    run_yaml = parsing.load_yaml(framework_run_dir / "run.yaml", warnings, "run.yaml")
    # Belt for the archive step below: if the framework evidence predates this
    # attempt it belongs to an earlier round (a merged run.yaml keeps its
    # status/gates), so it must never decide this record's outcome.
    stale = framework_artifacts_are_stale(framework_run_dir, str(meta.get("started_at") or ""))
    if stale and not _has_status_line(stdout):
        # The attempt itself never reported an outcome, so the status/gates on
        # disk belong to an earlier attempt in the same (reused) run directory:
        # trusting them collected a dead analysis attempt as `complete`
        # (2026-10-04, 29 of 30 records mixed two rounds).
        warnings.append("stale framework artifacts: last written before this attempt, which "
                        "produced no status= line; run evidence ignored")
        run_yaml = {}
        report = {}
    embench = parsing.load_yaml(framework_run_dir / "10_validation/embench.yaml", warnings, "embench.yaml")
    cost_yaml = parsing.load_yaml(framework_run_dir / "11_cost/result.yaml", warnings, "cost result.yaml")

    gates = report.get("gates") or {}
    review = parsing.load_yaml(framework_run_dir / "12_review/decision.yaml", warnings, "decision.yaml")
    if not gates and isinstance(review.get("gates"), dict):
        gates = review["gates"]

    status = cli.get("status") or run_yaml.get("status") or report.get("status") or "missing"
    stopped_at = cli.get("stopped_at") or report.get("stopped_at") or ""
    repairs_used = report.get("repairs_used")
    if repairs_used is None:
        warnings.append("repairs_used not found in report.md")
    verify_submissions = report.get("verify_submissions")
    # Historical reports predate explicit verification-submission accounting.
    attempt_index = (
        int(verify_submissions) if isinstance(verify_submissions, int)
        else (int(repairs_used) + 1) if isinstance(repairs_used, int) else None
    )

    # Ablation bookkeeping: the layer this run was supposed to execute versus
    # what the framework actually recorded in the report's "## Ablation"
    # section.  A mismatch means a --no-* switch never took effect (or was
    # applied without being scheduled) and must not pass silently.
    layer = str(job.get("layer") or "full")
    expected_flags = layers.job_layer_flags(job)
    observed_flags = tuple(report.get("ablation_flags") or ())
    if report and (expected_flags or observed_flags) and set(observed_flags) != set(expected_flags):
        warnings.append(
            "ablation mismatch: scheduled "
            f"[{', '.join(expected_flags) or 'none'}] but report recorded "
            f"[{', '.join(observed_flags) or 'none'}]"
        )

    sim_gate = gates.get("sim") or {}
    reference_cycles = embench.get("reference_cycles", sim_gate.get("reference_cycles"))
    cfu_cycles = embench.get("cfu_cycles", sim_gate.get("cfu_cycles"))
    speedup_x100 = embench.get("speedup_x100", sim_gate.get("speedup_x100", cli.get("speedup_x100")))
    # Only measured validation evidence identifies the protocol of these cycles;
    # never infer it from the current default or a newly prepared project.
    sim_yaml = parsing.load_yaml(framework_run_dir / "10_validation/sim.yaml", warnings, "sim.yaml")
    measurement = embench.get("measurement") or sim_gate.get("measurement") or sim_yaml.get("measurement") or {}
    if not isinstance(measurement, dict):
        measurement = {}
    measurement_protocol = str(
        embench.get("measurement_protocol") or sim_gate.get("measurement_protocol")
        or sim_yaml.get("measurement_protocol") or measurement.get("protocol") or "legacy"
    )
    benchmark_integrity = (embench.get("benchmark_integrity") or sim_gate.get("benchmark_integrity")
                           or sim_yaml.get("benchmark_integrity") or {})
    if measurement_protocol == "legacy":
        measurement_notice = ("Legacy measurement: protocol evidence absent; not comparable with "
                              "cold_single_invocation_v1. Start a new run, not a legacy resume.")
    elif measurement_protocol == "cold_single_invocation_v1":
        measurement_notice = ("Cold single invocation; no warmup; local/global scale factors 1. "
                              "Includes original preparation, memory and CFU configuration; "
                              "not pure math-kernel latency. Not comparable with legacy measurements.")
    else:
        measurement_notice = "Do not compare or pool speedups from different measurement protocols."
    measurement_notice += (" Integrity gate status is not proof of true operation equivalence "
                           "or an arbitrary semantic anti-cheat guarantee.")
    yosys = cli.get("yosys_cost") or {}

    # Simulation attempts and correctness-verification ("Verify") failures:
    # one content-addressed version file per sim attempt.
    sim_attempts = collect_sim_attempts(framework_run_dir, warnings)
    ref_result = reference_result_value(framework_run_dir, warnings)
    verify_failures = sum(1 for attempt in sim_attempts if verify_failure_in_attempt(attempt, ref_result))

    # Token accounting: the CLI block wins; report.md "## Token Usage" fills in
    # what a crashed process never printed.
    token = dict(report.get("token_usage") or {})
    token.update({key: value for key, value in cli.items()
                  if key in ("token_calls", "token_in", "token_out", "token_total")
                  and value is not None})
    usage_warnings: list[str] = []
    audit_usage = request_usage.load_request_usage(framework_run_dir / "request_audit.jsonl", usage_warnings,
                                                  started_at=str(meta.get("started_at") or ""))
    if audit_usage:
        token.update(token_calls=audit_usage["calls"], token_in=audit_usage["input_tokens"],
                     token_out=audit_usage["output_tokens"], token_total=audit_usage["total_tokens"],
                     token_cached_in=audit_usage["cached_input_tokens"])
    elif not token:
        warnings.append("token usage missing from audit, CLI and report")
    termination = (report.get("termination") or run_yaml.get("termination")
                   or parsing.load_json(framework_run_dir / "12_review/termination.json", warnings, "termination"))

    llm_request = run_yaml.get("llm_request") or {}
    visibility = (run_yaml.get("visibility") or {}).get("level") or cli.get("visibility") or ""
    profile_cache = (run_yaml.get("profile_cache") or {}).get("status") or ""

    record: dict = {
        "exp_id": exp_id,
        "run_name": run_name,
        "run_id": run_name,
        "model_config": job.get("model", ""),
        "model": str(llm_request.get("model") or job.get("model", "")),
        "benchmark": job.get("benchmark", ""),
        "suite": SUITE,
        "layer": layer,
        "ablation_flags": " ".join(expected_flags),
        "ablation_observed": " ".join(observed_flags),
        "repeat": job.get("repeat"),
        "max_iters": job.get("max_iters"),
        "status": status,
        "exit_code": meta.get("exit_code"),
        "stopped_at": stopped_at,
        "repairs_used": repairs_used,
        "max_repairs": report.get("max_repairs"),
        "verify_submissions": verify_submissions,
        "max_verify_submissions": report.get("max_verify_submissions"),
        "answer_repairs_used": report.get("answer_repairs_used"),
        "max_answer_repairs": report.get("max_answer_repairs"),
        "preverify_repairs_used": report.get("preverify_repairs_used"),
        "max_preverify_repairs": report.get("max_preverify_repairs"),
        "attempt_index": attempt_index,
        "sim_ok": (sim_gate.get("ok") if sim_gate else None),
        "sim_attempts": len(sim_attempts),
        "verify_failures": verify_failures,
        "reference_cycles": reference_cycles,
        "cfu_cycles": cfu_cycles,
        "speedup_x100": speedup_x100,
        "measurement_protocol": measurement_protocol,
        "measurement": measurement,
        "benchmark_integrity": benchmark_integrity,
        "measurement_notice": measurement_notice,
        "min_speedup_x100": embench.get("min_speedup_x100"),
        "sim_attribution": embench.get("attribution", sim_gate.get("attribution", "")),
        "sim_terminal_reason": "",
        "correctness_ok": None,
        "timing_ok": None,
        "performance_valid": None,
        "yosys_cells": yosys.get("cells"),
        "yosys_wire_bits": yosys.get("wire_bits"),
        "yosys_cost": yosys,
        "token_calls": token.get("token_calls"),
        "token_in": token.get("token_in"),
        "token_out": token.get("token_out"),
        "token_total": token.get("token_total"),
        "token_cached_in": token.get("token_cached_in"),
        "token_usage_source": "request_audit" if audit_usage else "cli/report" if token else "missing",
        "token_usage_complete": audit_usage.get("usage_complete") if audit_usage else None,
        "token_usage_warnings": usage_warnings,
        "request_usage": audit_usage,
        "termination": termination,
        "termination_reason": cli.get("termination_reason") or "",
        "wall_seconds": meta.get("wall_seconds"),
        "profile_cache_status": profile_cache,
        "visibility": visibility,
        "infra_retries": job.get("infra_retry", 0),
        "run_dir": str(run_dir),
        "framework_run_dir": str(framework_run_dir),
        "report_path": cli.get("report_path") or (str(report_path) if report_path.exists() else ""),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "gates": gates,
        "stale_framework_artifacts": stale,
        "stale_evidence_ignored": bool(stale and not _has_status_line(stdout)),
        "stage_gate_rejected": classification._stage_gate_rejected(f"{stdout}\n{stderr}"),
        "stage_gate_detail": "", "stage_hardware_integrity": False,
        "cost_status": cost_yaml.get("status"),
        "evidence": {
            "run_yaml": str(framework_run_dir / "run.yaml") if run_yaml else "",
            "report": str(report_path) if report_path.exists() else "",
            "embench": str(framework_run_dir / "10_validation/embench.yaml") if embench else "",
            "hw_repair_feedback": str(framework_run_dir / "09_implementation/hw_repair_feedback.txt"),
            "sw_repair_feedback": str(framework_run_dir / "09_implementation/sw_repair_feedback.txt"),
            "conformance": str(framework_run_dir / "09_implementation/conformance"),
        },
    }

    if record["stage_gate_rejected"]:
        rejection_lines = re.findall(r"native stage failed:[^\n]*gate rejected[^\n]*",
                                     f"{stdout}\n{stderr}", re.I)
        record["stage_gate_detail"] = rejection_lines[-1] if rejection_lines else ""
        record["stage_hardware_integrity"] = bool(re.search(
            r"hardware_integrity/|hardware integrity(?:\s*:|\s+cannot read)",
            record["stage_gate_detail"], re.I))
        if not _has_status_line(stdout):
            record["status"] = "failed"
            record["stopped_at"] = "implementation"

    # A typed source is one coherent current verdict. Do not fill its null
    # measurements from a previous YAML or the CLI's rounded summary.
    current = next((source for source in (sim_gate, sim_yaml, embench)
                    if source.get("terminal_reason")), {})
    if current:
        record["sim_terminal_reason"] = current["terminal_reason"]
        for key in ("reference_cycles", "cfu_cycles", "speedup_x100", "reference_result",
                    "cfu_result", "min_speedup_x100", "correctness_ok", "timing_ok", "performance_valid"):
            record[key] = current.get(key)
        record["sim_attribution"] = current.get("attribution", "")
        # The report gate may be old or missing; publish the typed source there
        # too so classification and failure.json see the exact terminal reason.
        sim_gate = {**sim_gate, **current}
        if "ok" not in current and current.get("status") in {"passed", "failed"}:
            sim_gate["ok"] = current["status"] == "passed"
        record["sim_ok"] = sim_gate.get("ok")
        record["gates"] = {**gates, "sim": sim_gate}
    elif sim_yaml:
        # Legacy command evidence is current, unlike report-wide Errors/history.
        sim_gate = {**sim_gate, "commands": sim_yaml.get("commands") or []}
        if sim_yaml.get("errors"):
            sim_gate["errors"] = sim_yaml["errors"]
        record["gates"] = {**gates, "sim": sim_gate}
        if verify_failure_in_attempt(sim_yaml, ref_result):
            record["correctness_ok"] = False
        record["reference_result"] = ref_result
        for command in sim_yaml.get("commands") or []:
            parsed = command.get("parsed") or {}
            verify = parsed.get("verify") or {}
            profile = parsed.get("embench_profile") or {}
            bench = str(verify.get("benchmark") or job.get("benchmark") or "")
            entry = profile.get(bench) or {}
            got = entry.get("result", verify.get("result"))
            if got is not None:
                record["cfu_result"] = got
    for key in ("reference_cycles", "cfu_cycles"):
        value = record.get(key)
        if not classification._positive(value):
            if value is not None:
                record["timing_ok"] = False
            record[key] = None
    record["observed_speedup_x100"] = record.get("speedup_x100")
    sim_failure = classification._classify_sim(sim_gate, (
        classification._gate_errors(sim_gate) + " " + str(record.get("sim_attribution") or "")
    ).lower(), record)[0]
    invalid_failure = sim_failure in {"sim_correctness", "sim_timeout", "sim_assertion", "sim_no_timing", "sim_no_change"}
    record["performance_valid"] = bool(
        not record["stale_evidence_ignored"] and not invalid_failure
        and classification.performance_is_valid(record)
    )
    if not record["performance_valid"]:
        record["speedup_x100"] = None

    # Final candidate performance is distinct from old passing sim evidence.
    observed = report.get("last_observed_valid_performance") or provenance.observed_performance(sim_gate)
    historical = provenance.load_last_observed_performance(framework_run_dir, warnings)
    if not (historical or observed):
        archive_path = run_dir / "record.json"
        archive = parsing.load_json(archive_path, warnings, "archived record")
        if archive:
            try:
                historical = provenance.archived_performance(archive, evidence=provenance.archive_evidence(archive_path))
            except OSError as exc:
                warnings.append(f"performance archive unavailable: {exc}")
    record["last_observed_valid_performance"] = historical or observed
    current_view = report.get("current_performance") or {}
    candidate = current_view.get("candidate_provenance") or {}
    if not candidate:
        accepted = record.get("gates") or {}
        candidate = {"candidate_hashes": {key: value for key, value in {
            "hardware": (accepted.get("implementation_hw") or {}).get("candidate_hash"),
            "software": (accepted.get("implementation_sw") or {}).get("candidate_hash"),
            "isa": (accepted.get("implementation_sw") or {}).get("isa_hash"),
        }.items() if value}}
    upstream_stop = record["status"] in {"failed", "blocked", "needs_iteration"} and record["stopped_at"] not in {"", "sim", "cost", "review", "final_report"}
    unreported_candidate = record["status"] in {"initialized", "missing"}
    nonsemantic_failure = record["sim_terminal_reason"] in {"sim_harness_crash", "probe_failure", "probe_bookkeeping_failure",
        "probe_bookkeeping_desync", "driver_reference_failure", "reference_failure"}
    if current_view or sim_gate.get("performance_provenance") or upstream_stop or unreported_candidate or nonsemantic_failure or record["stale_evidence_ignored"]:
        current_view = provenance.current_performance(sim_gate, candidate,
            status=record["status"], stopped_at=record["stopped_at"])
        if record["stale_evidence_ignored"] or unreported_candidate:
            current_view.update(performance_valid=False, reason="stale framework evidence" if record["stale_evidence_ignored"] else "final candidate outcome not reported")
        record["performance_valid"] = current_view["performance_valid"]
        if not record["performance_valid"]:
            for key in ("reference_cycles", "cfu_cycles", "speedup_x100", "correctness_ok", "timing_ok"):
                record[key] = None
            if sim_gate:
                record["gates"]["sim"] = {**sim_gate, "performance_valid": False,
                                          "current_candidate_matches": current_view.get("candidate_matches")}
    else:
        current_view = {"performance_valid": False, "candidate_matches": False,
                        "reason": "legacy candidate provenance unavailable", "scope": "current_candidate"}
    record["current_performance"] = current_view
    record["infra"] = meta.get("infra") or classification.detect_infra(
        stdout, stderr, bool(meta.get("timed_out")), meta.get("exit_code"), record["gates"], termination, record["stopped_at"]
    )
    record["failure"] = classification.classify_failure(record)
    if record["failure"]:
        for field in ("failure_class", "failure_rung", "failure_detail"):
            record[field] = record["failure"][field]
    record["episodes"] = classification.reconstruct_episodes(record, framework_run_dir)
    record["parse_warnings"] = "; ".join(warnings)
    record["pass_at_1"] = metrics._pass_flag(record, 1)
    record["pass_at_3"] = metrics._pass_flag(record, 3)
    record["pass_at_5"] = metrics._pass_flag(record, 5)
    record["terminal"] = True
    return record


def _has_status_line(text: str) -> bool:
    """True when the CLI log carries a ``status=`` outcome line."""
    return re.search(r"^status=", text, re.M) is not None


def write_failure_json(record: dict, run_dir: Path) -> Path | None:
    failure = record.get("failure")
    if not failure:
        return None
    payload = {
        "run_name": record.get("run_name"),
        "model": record.get("model"),
        "benchmark": record.get("benchmark"),
        "layer": record.get("layer", "full"),
        "repeat": record.get("repeat"),
        "terminal": {
            "status": record.get("status"),
            "stopped_at": record.get("stopped_at"),
            "budget_exhausted": failure.get("budget_exhausted"),
            "verify_submissions": record.get("verify_submissions"),
            "max_verify_submissions": record.get("max_verify_submissions"),
            "attempt_index": record.get("attempt_index"),
            "repairs_used": record.get("repairs_used"),
            "answer_repairs_used": record.get("answer_repairs_used"),
            "preverify_repairs_used": record.get("preverify_repairs_used"),
            "failure_class": failure.get("failure_class"),
            "failure_rung": failure.get("failure_rung"),
            "failure_detail": failure.get("failure_detail"),
            "attribution": record.get("sim_attribution"),
            "reason": record.get("sim_terminal_reason"),
            "correctness_ok": record.get("correctness_ok"),
            "performance_valid": record.get("performance_valid"),
            "termination": record.get("termination"),
            "current_performance": record.get("current_performance"),
            "last_observed_valid_performance": record.get("last_observed_valid_performance"),
            "gate": (record.get("gates") or {}).get(failure.get("failure_rung", ""), {}),
        },
        "episodes": record.get("episodes") or [],
        "evidence": record.get("evidence") or {},
        "spent": {
            "token_total": record.get("token_total"),
            "token_calls": record.get("token_calls"),
            "token_in": record.get("token_in"), "token_out": record.get("token_out"),
            "token_cached_in": record.get("token_cached_in"),
            "request_usage": record.get("request_usage"),
            "wall_seconds": record.get("wall_seconds"),
        },
    }
    path = run_dir / "failure.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


def save_record(record: dict, run_dir: Path) -> Path:
    """Persist one collected run: ``record.json`` plus ``failure.json`` (if any)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "record.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n")
    write_failure_json(record, run_dir)
    return path
