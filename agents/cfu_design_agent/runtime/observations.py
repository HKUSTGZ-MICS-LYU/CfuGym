"""Typed, stable observations derived from deterministic command failures."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, TypedDict


FAILURE_OBSERVATION_SCHEMA_VERSION = "1"

FailureCategory = str


class FailureObservation(TypedDict, total=False):
    schema_version: str
    id: str
    rung: str
    category: FailureCategory
    command: str
    cwd: str
    returncode: int
    timed_out: bool
    summary: str
    source_locations: list[str]
    signature: str
    log_path: str


_DIAGNOSTIC_RE = re.compile(
    r"(?:\berror\b|\bfail(?:ed|ure)?\b|exception|mismatch|not found|undefined reference|"
    r"timed out|assert(?:ion)?|denied|rejected|no such file)",
    re.IGNORECASE,
)
_SOURCE_LOCATION_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\s:()]+\.(?:c|cc|cpp|h|hpp|scala|S|v|sv|py))"
    # The extension must end a token: without this guard a JVM stack frame like
    # "scala.collection.TraversableLike" is matched as the source "scala.c".
    r"(?![\w.])"
    r"(?::(?P<line>\d+))?(?::(?P<column>\d+))?"
)
# Runtime/library frames that appear in JVM stack traces but are never useful
# repair targets. Only stack-trace internals belong here, never project files.
_NOISE_SOURCE_BASENAMES = frozenset({
    "TraversableLike.scala", "IndexedSeqOptimized.scala", "ArrayOps.scala",
    "ResizableArray.scala", "AbstractFunction0.scala", "AbstractFunction1.scala",
    "AbstractFunction2.scala", "App.scala", "SimManager.scala",
    "SimBootstraps.scala", "package.scala",
    # SpinalHDL elaboration internals: they frame every PhaseNormalize failure
    # and pushed the actual AgentCfu location out of the bounded list.
    "Misc.scala", "Phase.scala", "ArrayBuffer.scala", "ScopeProperty.scala",
    "StateMachine.scala", "AsyncCtrl.scala", "AsyncThread.scala", "Fiber.scala",
    "Expression.scala", "Literal.scala", "EngineContext.scala", "BitVectorLiteralFactory.scala",
})
_HDL_ASSERTION_RE = re.compile(
    r"hdl assertion failure|^\s*(?:\[[^\]]+\]\s*)?FAILURE\b",
    re.MULTILINE | re.IGNORECASE,
)
# A commit watchdog is a progress failure (long CFU latency or deadlock), not
# automatically a crashed testbench merely because the probe raises SimFailure.
_LIVENESS_RE = re.compile(r"(?:hasn['’]t|has not) committed anything for too long", re.I)

# Other JVM exceptions in testbench/probe/simulator frames may be harness faults.
_HARNESS_FRAME_RE = re.compile(r"(?:vexiiriscv\.test\.|spinal\.sim\.|spinal\.core\.sim\.)")

# Failure categories that a design repair cannot act on: the command produced
# no verdict and no design-local diagnostic.
HARNESS_FAILURE_CATEGORIES = frozenset({"sim_harness_crash", "environment", "policy", "cfu_status_missing"})


def failure_is_design_actionable(category: str) -> bool:
    """True when a failure with this category is actionable by a design repair."""
    return category not in HARNESS_FAILURE_CATEGORIES


_RUN_PATH_RE = re.compile(r"agents/generated/runs/[^/\s]+")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_TIME_RE = re.compile(r"\b(?:time|cycle|tick)s?\s*[=:]\s*\d+\b", re.IGNORECASE)
_LINE_COLUMN_RE = re.compile(r"(?<=:)(?:\d+)(?::\d+)?")
_LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
_SPACE_RE = re.compile(r"\s+")


def classify_failure(rung: str, result: dict[str, Any], text: str) -> FailureCategory:
    """Classify a failed command without changing the owning domain gate."""
    returncode = result.get("returncode")
    lower = text.lower()
    if result.get("timed_out") or returncode == -9:
        return "timeout"
    if returncode == -2 or "access policy" in lower or "not allowlisted" in lower:
        return "policy"
    if returncode == -1 or "could not start" in lower or "no such file or directory" in lower:
        return "environment"
    if rung in {"hw_conformance", "sim"}:
        # Strongest domain evidence first: MiCoSocSim reports a real fail
        # symbol through the same "FAILURE <msg>" channel as HDL assertions.
        if "fail symbol" in lower:
            return "fail_symbol"
        if _LIVENESS_RE.search(text):
            return "liveness_timeout"
        # An HDL assertion names a violated hardware contract (design/platform
        # actionable); a JVM exception inside the probe or simulator kernel is a
        # harness crash that the design cannot be repaired against.
        if _HDL_ASSERTION_RE.search(text):
            return "assertion"
        if "exception in thread" in lower and _HARNESS_FRAME_RE.search(text):
            return "sim_harness_crash"
    if rung in {"sw_build", "sw_check"}:
        return "compile"
    if rung in {"hw_build", "hw_check"}:
        if any(token in lower for token in ("spinalexit", "phase", "width mismatch", "hierarchy violation")):
            return "elaboration"
        return "compile"
    if rung == "hw_conformance":
        return "conformance"
    if rung == "sim":
        return "simulation"
    if rung == "cost":
        return "synthesis"
    return "command"


def source_locations(text: str, *, workdir: str = "", limit: int = 8) -> list[str]:
    """Return bounded, de-duplicated file locations from a command log.

    Real project paths (they carry a directory) rank above bare stack-frame
    filenames, and runtime-internals frames are dropped entirely: otherwise the
    SpinalHDL phase frames fill the budget before ``AgentCfu.scala`` appears.
    """
    found: list[tuple[int, bool, str]] = []
    seen: set[str] = set()
    root = Path(workdir).resolve() if workdir else None
    for order, match in enumerate(_SOURCE_LOCATION_RE.finditer(text)):
        path = match.group("path")
        if root:
            try:
                path = Path(path).resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                pass
        location = path
        if match.group("line"):
            location += f":{match.group('line')}"
            if match.group("column"):
                location += f":{match.group('column')}"
        # Bare stack-frame filenames from known runtime internals are noise;
        # a real source location carries a directory or is a project file.
        if "/" not in path and Path(path).name in _NOISE_SOURCE_BASENAMES:
            continue
        if location in seen:
            continue
        seen.add(location)
        found.append((order, "/" in path, location))
    found.sort(key=lambda item: (0 if item[1] else 1, item[0]))
    return [location for _, _, location in found[:limit]]


# Lines that name the concrete repair target rank above generic phase banners:
# a summary that keeps "Elaboration failed" and drops "WIDTH MISMATCH (32 <- 24)"
# sends the repair after the wrong symptom (observed in crc32-run-0005).
_ACTIONABLE_RE = re.compile(
    r"width mismatch|value .+ is not a member|overloaded method|type mismatch|"
    r"not found|undefined reference|negative and cannot|assertion|"
    r"requirement failed|literal value",
    re.IGNORECASE,
)
_PHASE_BANNER_RE = re.compile(
    r"elaboration failed|error detected in phase|exception in thread|"
    r"^\W*$|spinal exit|\*{5,}",
    re.IGNORECASE,
)


def _diagnostic_score(line: str) -> int:
    score = 0
    if _ACTIONABLE_RE.search(line):
        score += 6
    if _SOURCE_LOCATION_RE.search(line):
        score += 3
    if "width mismatch" in line.lower():
        score += 4
    if _PHASE_BANNER_RE.search(line):
        score -= 4
    return score


def _diagnostic_lines(text: str, *, limit: int = 12) -> list[str]:
    """Most actionable diagnostics first, restored to log order in the answer."""
    candidates: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not _DIAGNOSTIC_RE.search(line):
            continue
        line = line[:500]
        if line not in candidates:
            candidates.append(line)
    ranked = sorted(enumerate(candidates), key=lambda item: (-_diagnostic_score(item[1]), item[0]))
    return [line for _, line in sorted(ranked[:limit], key=lambda item: item[0])]


def _stable_text(text: str, *, workdir: str = "") -> str:
    value = text
    if workdir:
        value = value.replace(str(Path(workdir).resolve()), "<workdir>")
    value = _RUN_PATH_RE.sub("agents/generated/runs/<run>", value)
    value = _HEX_RE.sub("0x#", value)
    value = _TIME_RE.sub("time=#", value)
    value = _LINE_COLUMN_RE.sub("#", value)
    value = _LONG_NUMBER_RE.sub("#", value)
    return _SPACE_RE.sub(" ", value).strip().lower()


def failure_signature(
    rung: str,
    category: str,
    result: dict[str, Any],
    text: str,
    *,
    workdir: str = "",
) -> str:
    """Hash stable failure features, excluding run-specific path and counters."""
    diagnostics = _diagnostic_lines(text)
    if not diagnostics:
        diagnostics = [text[-1200:]] if text else [str(result.get("reason", ""))]
    payload = {
        "rung": rung,
        "category": category,
        "returncode": result.get("returncode"),
        "command": _stable_text(str(result.get("command", "")), workdir=workdir),
        "diagnostics": [_stable_text(line, workdir=workdir) for line in diagnostics],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return "failure-v1:" + hashlib.sha256(encoded).hexdigest()


def build_failure_observation(
    rung: str,
    result: dict[str, Any],
    *,
    stdout: str = "",
    stderr: str = "",
    log_path: str = "",
    workdir: str = "",
) -> FailureObservation | None:
    """Build one observation for a failed command; successful/skipped calls return None."""
    returncode = result.get("returncode")
    if returncode in (0, None) or result.get("skipped"):
        return None
    text = "\n".join(part for part in (stdout, stderr) if part)
    category = classify_failure(rung, result, text)
    signature = failure_signature(rung, category, result, text, workdir=workdir)
    diagnostics = _diagnostic_lines(text, limit=4)
    summary = "\n".join(diagnostics)
    if not summary:
        summary = str(result.get("reason") or f"command exited with return code {returncode}")
    observation: FailureObservation = {
        "schema_version": FAILURE_OBSERVATION_SCHEMA_VERSION,
        "id": f"fail_{rung}_{signature.removeprefix('failure-v1:')[:12]}",
        "rung": rung,
        "category": category,
        "command": str(result.get("command", ""))[:500],
        "cwd": str(result.get("cwd", "")),
        "returncode": int(returncode),
        "timed_out": bool(result.get("timed_out", False)),
        "summary": summary[:2000],
        "source_locations": source_locations(text, workdir=workdir),
        "signature": signature,
    }
    if log_path:
        observation["log_path"] = log_path
    return observation


def record_failure_result(
    index: dict[str, Any], result: dict[str, Any], *, rung: str = "", workdir: str = "",
) -> dict[str, Any]:
    """Publish a command/check result to an existing live inspection index.

    Keep the index object shared by inspection and diagnosis tools. Semantic
    failures (e.g. missing RTL despite command success) also need inspectable IDs.
    Only an explicit, non-skipped success resolves earlier failures of this rung;
    an absent return code is not evidence that the fault was fixed.
    """
    result = dict(result)
    rung = str(result.get("rung") or rung)
    if rung:
        result["rung"] = rung
    failures = index.setdefault("failures", {})
    if result.get("skipped"):
        return result
    if result.get("returncode") == 0:
        result.pop("failure_observation", None)
        for item in failures.values():
            if isinstance(item, dict) and item.get("rung") == rung:
                item["status"] = "resolved"
        return result
    observation = result.get("failure_observation")
    if not isinstance(observation, dict) or not observation.get("id"):
        observation = build_failure_observation(
            rung, result, stdout=str(result.get("stdout_tail") or ""),
            stderr=str(result.get("stderr_tail") or ""),
            log_path=str(result.get("full_log_path") or result.get("log_path") or ""),
            workdir=workdir,
        )
    if observation:
        result["failure_observation"] = observation
        failures[str(observation["id"])] = {**observation, "status": "current"}
    return result


def failure_index(state: dict[str, Any]) -> dict[str, Any]:
    """Index validation/cost failure observations for stage tools.

    Each entry carries a ``status``: ``current`` while the failing rung has not
    produced a later success, ``resolved`` once it has. Resolved IDs stay
    available for context but must not be cited as current repair targets, so a
    repair never re-fixes an already-fixed failure.
    """
    observations: dict[str, dict[str, Any]] = {}
    status: dict[str, str] = {}
    epoch = state.get("repair_evidence_epoch") or {}
    for key in ("validation_results", "cost_results"):
        for position, result in enumerate(state.get(key) or []):
            if not isinstance(result, dict):
                continue
            item = result.get("failure_observation")
            if isinstance(item, dict) and item.get("id"):
                ident = str(item["id"])
                observations[ident] = dict(item)
                status[ident] = "stale" if position < int(epoch.get(key, 0)) else "current"
                continue
            rung = str(result.get("rung") or "")
            if rung and result.get("returncode") == 0 and not result.get("skipped"):
                for ident, item in observations.items():
                    if item.get("rung") == rung and status.get(ident) == "current":
                        status[ident] = "resolved"
    failures = {
        ident: {**item, "status": status.get(ident, "current")}
        for ident, item in observations.items()
    }
    return {"failures": failures}
