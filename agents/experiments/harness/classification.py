"""Failure / infrastructure classification: which rung broke and why.

``classify_failure`` maps a non-complete run onto (failure_class, failure_rung,
detail); ``detect_infra`` separates harness/endpoint incidents from model
failures; ``reconstruct_episodes`` rebuilds the best-effort repair ledger.
"""

from __future__ import annotations

import re

from agents.experiments.harness import parsing

# Ladder/report order; also the order used to pick the first failing rung when
# ``stopped_at`` is missing from the CLI log.
RUNG_ORDER = (
    "load_context",
    "prepare_benchmark",
    "profile",
    "workload_spec",
    "isa",
    "implementation_hw",
    "hw_build",
    "hw_conformance",
    "implementation_sw",
    "sw_build",
    "sim",
    "cost",
)

# Build rung guards: hw_build/sw_build fail WITHOUT running any command when the
# implementation rung that feeds them applied nothing. The guard is not the real
# failure rung -- the implementation rejection is (observed 2026-09-26: a
# missing-EDIT_END delivery failure was recorded as stopped_at=hw_build).
GUARDED_BUILD_RUNGS = {"hw_build": "implementation_hw", "sw_build": "implementation_sw"}

FAILURE_CLASSES = (
    "infra_timeout",
    "infra_rate_limit",
    "infra_error",
    "input_or_policy",
    "precondition",
    "spec_invalid",
    "isa_contract",
    "no_deliverable",
    "repair_route_budget",
    "repair_route",
    "hw_build",
    "hardware_integrity",
    "hw_conformance",
    "sw_build",
    "sim_correctness",
    "sim_timeout",
    "sim_assertion",
    "sim_no_timing",
    "sim_speedup_floor",
    "sim_no_change",
    "sim_error",
    "cost",
    "unknown",
)

# Harness-level (non-model) failure signatures: API transport problems and
# missing toolchain executables.  A crashed framework is detected differently
# (see detect_infra): a bare "Traceback" is too noisy because benchmark build
# logs (Python-based) legitimately contain tracebacks.
INFRA_PATTERNS = (
    re.compile(r"api-key-missing"),
    re.compile(r"api-error:"),
    re.compile(r"urlopen error", re.I),
    re.compile(r"Connection (error|refused|reset|aborted)", re.I),
    re.compile(r"Max retries exceeded"),
    re.compile(r"SSLError", re.I),
    re.compile(r"\b(?:HTTP(?: Error| status)?\s*[:=]?\s*500|upstream\b.{0,60}\b500)\b", re.I),
    # Endpoint quota exhaustion: the ablation matrix of 2026-10-04 lost 30 of 43
    # attempts to `upstream error: 429`, and without this pattern each one was
    # reported as "framework crash or launcher failure".
    re.compile(r"\b(?:upstream error|HTTP(?: Error| status)?)\s*[:=]?\s*429\b", re.I),
    re.compile(r"\b429\s+Too Many Requests\b", re.I),
    re.compile(r"\b(?:502|503|504)\b.{0,40}(?:Bad Gateway|Service Unavailable|Gateway Time-out)", re.I),
    re.compile(r"\b(?:LLM|API|endpoint|gateway)\b.{0,80}\b(?:timed? out|timeout)\b", re.I),
    # The SDK's own wording carries no role word, so the rule above missed it and
    # eight 2026-10-04 ablation runs were filed as "framework crash or launcher
    # failure" (they were endpoint read stalls: `APITimeoutError: Request timed out.`).
    re.compile(r"APITimeoutError|Request timed out\b|Read timed out\b", re.I),
    re.compile(r"conformance oracle request got no model answer[^\n]{0,160}\b(?:TimeoutError|APIConnectionError|APIError|HTTP\s+(?:429|5\d\d))\b", re.I),
    re.compile(r"No such file or directory: '?(?:sbt|make|yosys|conda)", re.I),
    # A misfiring lock shim is a harness problem, never a model failure.
    re.compile(r"sbt lock shim:"),
)


def _gate_errors(gate: dict) -> str:
    parts = [str(x) for x in (gate.get("errors") or []) if x]
    for key in ("reason", "attribution", "verdict", "detail"):
        value = gate.get(key)
        if value:
            parts.append(str(value))
    return " | ".join(parts)


def classify_failure(record: dict) -> dict | None:
    """Map a non-complete run onto (failure_class, failure_rung, detail).

    The class answers "which rung broke and in what way"; ``failure_rung`` keeps
    the exact ladder position so the coarser class never loses information.
    """
    status = str(record.get("status") or "")
    if status == "complete":
        return None
    typed = record.get("termination") or {}
    if typed.get("kind") == "model_delivery_failure":
        return {"failure_class": "no_deliverable", "failure_rung": typed.get("stage") or record.get("stopped_at") or "implementation",
                "failure_detail": typed.get("reason") or "no accepted delivery", "budget_exhausted": None}
    if typed.get("kind") == "precondition":
        return {"failure_class": "precondition", "failure_rung": typed.get("stage") or record.get("stopped_at") or "unknown",
                "failure_detail": typed.get("reason") or "precondition failure", "budget_exhausted": None}
    gates = record.get("gates") or {}
    stopped = str(record.get("stopped_at") or "")
    if record.get("stage_gate_rejected"):
        if record.get("stage_hardware_integrity"):
            return {
                "failure_class": "hardware_integrity", "failure_rung": "implementation",
                "failure_detail": str(record.get("stage_gate_detail") or "hardware integrity rejected delivery"),
                "budget_exhausted": None,
            }
        # No ladder verdict at all: the stage gate refused every delivery the
        # model submitted (missing/stale/unbuildable oracle, no applied change).
        return {
            "failure_class": "no_deliverable",
            "failure_rung": "implementation",
            "failure_detail": "the independent stage gate rejected every submitted delivery",
            "budget_exhausted": None,
        }
    failing_rung = stopped
    failing_gate: dict = {}
    if not failing_rung:
        for rung in RUNG_ORDER:
            gate = gates.get(rung)
            if isinstance(gate, dict) and gate.get("ok") is False:
                failing_rung = rung
                failing_gate = gate
                break
    else:
        failing_gate = gates.get(failing_rung) if isinstance(gates.get(failing_rung), dict) else {}

    # Attribute a guarded build failure to the implementation rejection behind
    # it: the build rung ran zero commands, so recording it as a build failure
    # would misplace the failure stage in every experiment table.
    guard_note = ""
    impl_rung = GUARDED_BUILD_RUNGS.get(failing_rung)
    if impl_rung and not failing_gate.get("terminal_reason") and not (failing_gate.get("commands") or []):
        impl_gate = gates.get(impl_rung)
        if isinstance(impl_gate, dict) and (
            impl_gate.get("ok") is False or impl_gate.get("applied") is False
        ):
            guard_note = (
                f"[{failing_rung} never ran: it refuses to build with no applied "
                f"{impl_rung} change]"
            )
            failing_rung = impl_rung
            failing_gate = impl_gate

    errors = _gate_errors(failing_gate)
    sim_errors = _gate_errors(gates.get("sim") or {})
    attribution = str(record.get("sim_attribution") or "")
    detail_text = f"{errors} {sim_errors} {attribution}".lower()

    verify_submissions = record.get("verify_submissions")
    try:
        max_iters = int(record.get("max_iters")) if record.get("max_iters") is not None else None
    except (TypeError, ValueError):
        max_iters = None
    repairs_used = record.get("repairs_used")
    max_repairs = (int(max_iters) - 1) if isinstance(max_iters, int) else record.get("max_repairs")
    budget_exhausted = None
    if isinstance(verify_submissions, int) and isinstance(max_iters, int):
        budget_exhausted = verify_submissions >= max_iters
    elif isinstance(repairs_used, int) and isinstance(max_repairs, int):
        # Historical records used repairs_used + 1 as their attempt index.
        budget_exhausted = repairs_used >= max_repairs

    failure_class, detail = _classify_rung(failing_rung, failing_gate, detail_text, record, status)
    infra = record.get("infra") or {}
    if infra.get("infra"):
        failure_class = infra.get("infra_kind") or "infra_error"
        detail = infra.get("infra_reason") or detail
        budget_exhausted = None  # An endpoint incident did not exhaust the candidate submission budget.
    detail = " ".join(part for part in ((detail or errors or status), guard_note) if part)
    return {
        "failure_class": failure_class,
        "failure_rung": failing_rung or "unknown",
        "failure_detail": detail,
        "budget_exhausted": budget_exhausted,
    }


SIM_REASON_CLASSES = {
    "timeout": "sim_timeout", "liveness_timeout": "sim_timeout",
    "assertion": "sim_assertion", "no_timing": "sim_no_timing",
    "verify_failed": "sim_correctness", "fail_symbol": "sim_correctness",
    "result_mismatch": "sim_correctness", "result_missing": "sim_correctness",
    "no_verification": "sim_correctness", "no_design_change": "sim_no_change",
    "speedup_floor": "sim_speedup_floor",
    "sim_harness_crash": "infra_error", "probe_failure": "infra_error",
    "probe_bookkeeping_failure": "infra_error", "probe_bookkeeping_desync": "infra_error",
    "driver_reference_failure": "precondition", "reference_failure": "precondition",
}


def _positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def performance_is_valid(record: dict) -> bool:
    """Reject known invalid results; preserve successful pre-schema records."""
    gate = (record.get("gates") or {}).get("sim") or {}
    valid = record.get("performance_valid", gate.get("performance_valid"))
    if valid is False or record.get("correctness_ok", gate.get("correctness_ok")) is False:
        return False
    if record.get("timing_ok", gate.get("timing_ok")) is False:
        return False
    if (record.get("benchmark_integrity") or gate.get("benchmark_integrity") or {}).get("ok") is False:
        return False
    reason = str(gate.get("terminal_reason") or record.get("sim_terminal_reason") or "")
    if reason and reason not in {"passed", "speedup_floor"}:
        return False
    for key in ("reference_cycles", "cfu_cycles"):
        value = record.get(key, gate.get(key))
        if value is not None and not _positive(value):
            return False
    current = record.get("current_performance")
    if isinstance(current, dict) and current.get("candidate_provenance") is not None:
        if current.get("performance_valid") is not True or current.get("candidate_matches") is not True:
            return False
    if gate.get("stale"):
        return False
    if record.get("status") in {"failed", "blocked", "needs_iteration"} and record.get("stopped_at") not in {None, "", "sim", "cost", "review", "final_report"}:
        return False
    if valid is True:
        return True
    if record.get("status") == "complete" and record.get("sim_ok", gate.get("ok")) is not False:
        return True  # Old completed records may store only the speedup headline.
    # A legacy failed speed gate must contain actual positive measurements, not
    # the old zero defaults. Its wording alone is not measured performance.
    return (all(_positive(record.get(k, gate.get(k))) for k in ("reference_cycles", "cfu_cycles"))
            and record.get("correctness_ok", gate.get("correctness_ok")) is not False
            and ("not meaningfully faster" in _gate_errors(gate).lower()
                 or record.get("correctness_ok", gate.get("correctness_ok")) is True))


def _classify_sim(gate: dict, detail_text: str, record: dict) -> tuple[str, str]:
    reason = str(gate.get("terminal_reason") or record.get("sim_terminal_reason") or "")
    if reason:
        # Current typed terminal evidence wins over stale/historical errors.
        return SIM_REASON_CLASSES.get(reason, "sim_error"), _gate_errors(gate) or reason
    commands = gate.get("commands") or []
    commands = commands if isinstance(commands, list) else []
    categories = {(c.get("failure_observation") or {}).get("category")
                  for c in commands if isinstance(c, dict)}
    if categories & {"timeout", "liveness_timeout"} or re.search(
            r"timed\s+out|timeout|\bhang(?:s|ing)?\b|\bhung\b", detail_text):
        return "sim_timeout", _gate_errors(gate) or "simulation timed out"
    if "assertion" in categories or re.search(r"hdl assertion|assertion fail|\bassert(?:ion)?\b", detail_text):
        return "sim_assertion", _gate_errors(gate)
    if categories & {"sim_harness_crash", "environment", "policy"}:
        return "sim_error", _gate_errors(gate)
    command_verify_failed = any(
        ((c.get("parsed") or {}).get("verify") or {}).get("status") == "fail"
        for c in commands if isinstance(c, dict))
    if not command_verify_failed and "fail_symbol" not in categories and any(
            c.get("returncode") not in (0, None) for c in commands if isinstance(c, dict)):
        return "sim_error", _gate_errors(gate)
    correctness = re.search(
        r"verify|fail symbol|wrong[ -]result|result[_ -]mismatch|computed result=.*reference|observable result",
        detail_text,
    ) or "fail_symbol" in categories or any(
        ((c.get("parsed") or {}).get("verify") or {}).get("status") == "fail"
        for c in commands if isinstance(c, dict))
    ref_result, cfu_result = record.get("reference_result"), record.get("cfu_result")
    if correctness or (ref_result is not None and cfu_result is not None and ref_result != cfu_result):
        return "sim_correctness", _gate_errors(gate)
    if record.get("timing_ok", gate.get("timing_ok")) is False or re.search(
            r"no (?:valid |reference |CFU )*cycle|no (?:CFU )?profile|no timing", detail_text, re.I):
        return "sim_no_timing", _gate_errors(gate)
    if any(c.get("returncode") not in (0, None) for c in commands if isinstance(c, dict)):
        return "sim_error", _gate_errors(gate)
    if "no design change" in detail_text:
        return "sim_no_change", _gate_errors(gate)
    if performance_is_valid(record):
        if "not meaningfully faster" in detail_text:
            return "sim_speedup_floor", _gate_errors(gate)
        speedup, floor = record.get("speedup_x100"), record.get("min_speedup_x100")
        if speedup is not None and floor is not None and speedup < floor:
            return "sim_speedup_floor", _gate_errors(gate)
    return "sim_error", _gate_errors(gate)


def _classify_rung(rung: str, gate: dict, detail_text: str, record: dict, status: str) -> tuple[str, str]:
    if gate.get("failure_kind") == "precondition_failure" or gate.get("terminal_reason") in {"profile_precondition", "profile_build_or_reference"}:
        return "precondition", _gate_errors(gate) or str(gate.get("terminal_reason") or "precondition failure")
    if rung in ("load_context", "prepare_benchmark", "profile", "profile_embench", "instrument_and_profile"):
        if gate.get("terminal_reason") in {"driver_reference_failure", "reference_failure"}:
            return "precondition", _gate_errors(gate)
        return "input_or_policy", _gate_errors(gate)
    if rung == "workload_spec":
        return "spec_invalid", _gate_errors(gate)
    if rung == "isa":
        return "isa_contract", _gate_errors(gate)
    if rung in ("implementation", "implementation_hw", "implementation_sw"):
        if gate.get("terminal_reason") == "hardware_integrity" or re.search(
                r"hardware_integrity/|hardware integrity(?:\s*:|\s+cannot read)", _gate_errors(gate), re.I):
            return "hardware_integrity", _gate_errors(gate)
        return "no_deliverable", _gate_errors(gate) or "no usable EDIT/FILE/PATCH deliverable"
    if rung == "implementation_route":
        detail = _gate_errors(gate)
        limit = record.get("max_repairs")
        if limit is None and isinstance(record.get("max_iters"), int):
            limit = record["max_iters"] - 1
        used = record.get("repairs_used")
        exhausted = isinstance(used, int) and isinstance(limit, int) and used >= limit
        cls = "repair_route_budget" if exhausted or re.search(r"budget|exhaust", detail, re.I) else "repair_route"
        return cls, detail or "implementation repair route stopped"
    if rung == "hw_build":
        if gate.get("terminal_reason") == "hardware_integrity" or re.search(
                r"hardware_integrity/|hardware integrity(?:\s*:|\s+cannot read)", _gate_errors(gate), re.I):
            return "hardware_integrity", _gate_errors(gate)
        return "hw_build", _gate_errors(gate)
    if rung == "hw_conformance":
        sub = "oracle" if gate.get("oracle_failure") else "design"
        return "hw_conformance", f"{sub}: {_gate_errors(gate)}"
    if rung == "sw_build":
        return "sw_build", _gate_errors(gate)
    if rung == "cost":
        return "cost", _gate_errors(gate)
    if rung == "sim":
        return _classify_sim(gate, detail_text, record)
    if status == "blocked":
        if isinstance((record.get("gates") or {}).get("isa"), dict) and not (record["gates"]["isa"] or {}).get("ok", True):
            return "isa_contract", _gate_errors(record["gates"]["isa"] or {})
        return "spec_invalid", "run blocked before the validation ladder"
    return "unknown", _gate_errors(gate) or f"no failing rung identified (status={status})"


#: A native stage loop that exhausted its submission retries raises in the
#: framework; the ladder never produces a verdict, so the harness has to read the
#: message to know the model (not the harness) failed.
_STAGE_GATE_REJECTED_RE = re.compile(
    r"native stage failed:.*?gate rejected delivery", re.I | re.S
)


def _stage_gate_rejected(text: str) -> bool:
    """True when the run died because every submitted delivery was rejected."""
    return _STAGE_GATE_REJECTED_RE.search(text or "") is not None


def detect_infra(
    stdout: str, stderr: str, timed_out: bool, exit_code: int | None,
    gates: dict | None = None, termination: dict | None = None, stopped_at: str = "",
) -> dict:
    """Separate harness/endpoint incidents from model failures, including report gates."""
    if timed_out:
        return {"infra": True, "infra_kind": "infra_timeout", "infra_reason": "harness wall-clock timeout"}
    # The ladder finished and reported: any incident text in the log (a retried
    # HTTP 500 echoed into stdout, a benchmark log line) is not this run's
    # outcome. 2026-10-04: one *complete* l1 run was marked infra by a stray
    # "upstream error: 500" and burned a redundant retry.
    if exit_code == 0 and re.search(r"^status=complete\b", stdout, re.M):
        return {"infra": False, "infra_kind": "", "infra_reason": ""}
    typed = termination or {}
    if typed.get("kind") in {"model_delivery_failure", "precondition"}:
        return {"infra": False, "infra_kind": "", "infra_reason": typed.get("reason") or typed["kind"]}
    if typed.get("kind") in {"infrastructure", "runtime_exception"}:
        detail = str(typed.get("reason") or "")
        kind = ("infra_rate_limit" if re.search(r"\b429\b|rate.?limit", detail, re.I) else
                "infra_timeout" if re.search(r"timed? out|timeout", detail, re.I) or typed.get("exception_type") in {"TimeoutError", "APITimeoutError"}
                else "infra_error")
        return {"infra": True, "infra_kind": kind,
                "infra_reason": f"{typed.get('exception_type') or typed['kind']}: {detail}"}
    # Current typed simulation/probe evidence wins over historical design logs.
    for rung, gate in (gates or {}).items():
        if stopped_at and rung != stopped_at:
            continue
        if not isinstance(gate, dict) or gate.get("stale") or gate.get("ok") is not False:
            continue
        reason = str(gate.get("terminal_reason") or "")
        if reason in {"sim_harness_crash", "probe_failure", "probe_bookkeeping_failure", "probe_bookkeeping_desync"}:
            return {"infra": True, "infra_kind": "infra_error", "infra_reason": _gate_errors(gate) or reason}
        if reason in {"driver_reference_failure", "reference_failure", "profile_precondition", "profile_build_or_reference"} or gate.get("failure_kind") == "precondition_failure":
            return {"infra": False, "infra_kind": "", "infra_reason": "profile/reference precondition failure"}
    # A stage gate that rejected every delivery is the model's failure: the run
    # died with an unhandled RuntimeError and therefore has no `status=` line,
    # which used to look like a framework crash and was excluded from pass@k
    # (2026-10-04, `native stage failed: independent submit_stage gate rejected
    # delivery (conformance oracle is missing/stale/invalid ...)`).
    if _stage_gate_rejected(f"{stdout}\n{stderr}"):
        return {"infra": False, "infra_kind": "",
                "infra_reason": "stage gate rejected every submitted delivery (model failure)"}
    conformance = ((gates or {}).get("hw_conformance") or {}) if stopped_at in {"", "hw_conformance"} else {}
    if isinstance(conformance, dict) and conformance.get("ok") is False:
        detail = _gate_errors(conformance)
        endpoint_error = re.search(
            r"\b(?:APITimeoutError|APIConnectionError|APIStatusError|APIError|ConnectionError|TimeoutError)\b|"
            r"conformance oracle request got no model answer[^\n]{0,160}\b(?:timeout|timed out)\b|"
            r"\b(?:LLM|API|endpoint|gateway|upstream)\b.{0,80}\b(?:timed? out|timeout|(?:HTTP\s*)?500)\b|"
            r"\bHTTP Error 500\b", detail, re.I,
        )
        if endpoint_error or conformance.get("transport_error") or conformance.get("failure_kind") == "infrastructure":
            kind = "infra_timeout" if re.search(r"timed? out|timeout", detail, re.I) else "infra_error"
            return {"infra": True, "infra_kind": kind,
                    "infra_reason": f"conformance oracle endpoint: {detail or 'no model answer'}"}
    text = f"{stdout}\n{stderr}"
    for pattern in INFRA_PATTERNS:
        m = pattern.search(text)
        if m:
            matched = m.group(0)[:120]
            if re.search(r"\b429\b|Too Many Requests", matched, re.I):
                kind = "infra_rate_limit"
            elif re.search(r"timed? out|timeout", matched, re.I):
                kind = "infra_timeout"
            else:
                kind = "infra_error"
            return {
                "infra": True,
                "infra_kind": kind,
                "infra_reason": f"matched {pattern.pattern!r}: {matched}",
            }
    if exit_code is None:
        return {"infra": True, "infra_kind": "infra_error", "infra_reason": "no exit code recorded"}
    # The CLI always prints ``status=`` before exiting; its absence on a failing
    # process means the framework itself crashed before producing an outcome.
    if exit_code != 0 and not re.search(r"^status=", stdout, re.M):
        return {
            "infra": True,
            "infra_kind": "infra_error",
            "infra_reason": "agent produced no status= line (framework crash or launcher failure)",
        }
    return {"infra": False, "infra_kind": "", "infra_reason": ""}


def reconstruct_episodes(record: dict, run_root) -> list[dict]:
    """Best-effort repair/episode ledger from gate evidence.

    The framework does not persist the full iteration history, so each repair
    marker that does survive (gate ``repair`` flags, the sim repair decision,
    repair-feedback files, the conformance repair target) becomes one episode.
    """
    episodes: list[dict] = []
    gates = record.get("gates") or {}

    sim_gate = gates.get("sim") or {}
    decision = sim_gate.get("repair_decision") or parsing.load_yaml(
        run_root / "10_validation/sim_repair_decision.yaml", [], "sim_repair_decision"
    )
    if decision:
        episodes.append({
            "kind": "route_repair",
            "trigger_rung": "sim",
            "repair_target": decision.get("target", ""),
            "decision_source": decision.get("source", ""),
            "detail": f"{decision.get('root_cause', '')} {decision.get('reason', '')}".strip(),
            "evidence": "10_validation/sim_repair_decision.yaml",
        })

    for rung, target, feedback in (
        ("implementation_hw", "implement_hw", "09_implementation/hw_repair_feedback.txt"),
        ("implementation_sw", "implement_sw", "09_implementation/sw_repair_feedback.txt"),
    ):
        gate = gates.get(rung) or {}
        feedback_path = run_root / feedback
        if gate.get("repair") or feedback_path.exists():
            episodes.append({
                "kind": "repair",
                "trigger_rung": rung,
                "repair_target": target,
                "decision_source": "fixed",
                "detail": str(gate.get("repair_diagnosis") or ""),
                "evidence": feedback if feedback_path.exists() else "",
            })

    conf = gates.get("hw_conformance") or {}
    if conf.get("ok") is False and conf.get("repair_target"):
        episodes.append({
            "kind": "repair",
            "trigger_rung": "hw_conformance",
            "repair_target": str(conf.get("repair_target")),
            "decision_source": "fixed",
            "detail": _gate_errors(conf),
            "evidence": "09_implementation/conformance/",
        })
    return episodes
