"""Bounded run-local memory with read-time byte/range provenance.

The graph owns publication: only after a node returns successfully does it fold
captured reads and accepted domain facts into state. A file reference alone is
not evidence that the model saw its content. Rendered file excerpts come from
read-time snapshots, never a fresh unobserved file head.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

from . import source_visibility
from .tools import RepoToolbox

MEMORY_VERSION = 2
MAX_ENTRIES = 24
MAX_FILE_BYTES = 2_000_000
_EXCERPT_CHARS = 600
_MAX_PATH = 240
_SAFE_PATH = re.compile(r"[A-Za-z0-9_./+ -]+\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVE_TRANSCRIPT: ContextVar[list | None] = ContextVar("flow_memory_transcript", default=None)
_ROLES = frozenset({"source", "hardware", "analysis", "contract", "profile", "workload"})
_ORIGINS = frozenset({"state", "tool"})
_STAGE_ROLES = {
    "analysis": frozenset({"source", "profile", "workload"}),
    "isa": frozenset({"analysis", "profile", "workload", "source"}),
    "implementation": frozenset({"contract", "analysis", "hardware", "source", "workload"}),
    "implement": frozenset({"contract", "analysis", "hardware", "source", "workload"}),
    "review": _ROLES,
}
_STAGE_ALIASES = {
    "analyze_workload": "analysis", "design_contract": "isa", "extend_isa": "isa",
    "implement": "implementation", "check_conformance": "implementation",
    "route_software": "implementation", "optimize": "review",
}


def memory_stage(stage: str) -> str:
    return _STAGE_ALIASES.get(stage, stage)


@contextmanager
def collect_tool_transcripts() -> Iterator[list]:
    token = _ACTIVE_TRANSCRIPT.set([])
    try:
        yield _ACTIVE_TRANSCRIPT.get()
    finally:
        _ACTIVE_TRANSCRIPT.reset(token)


def record_tool_transcript(calls: Sequence[Mapping[str, Any]] | None) -> None:
    collector = _ACTIVE_TRANSCRIPT.get()
    if collector is not None and isinstance(calls, (list, tuple)):
        collector.extend(item for item in calls if isinstance(item, Mapping))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _identity(state: Mapping[str, Any], toolbox: RepoToolbox) -> dict[str, Any] | None:
    rid, root = state.get("run_id"), state.get("run_root")
    if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", rid):
        return None
    if not isinstance(root, str) or not re.fullmatch(
        rf"agents/generated(?:/[A-Za-z0-9_.-]+)*/runs/{re.escape(rid)}", root
    ):
        return None
    if toolbox.run_root != root or toolbox.policy is None:
        return None
    if Path(str(state.get("workdir", ""))).resolve() != toolbox.workdir:
        return None
    if state.get("visibility") != toolbox.policy.visibility:
        return None
    try:
        policy_sha = _digest(asdict(toolbox.policy))
    except (TypeError, ValueError):
        return None
    # ISA is deliberately NOT envelope identity: changing it must not forget
    # unchanged methodology/source knowledge. Dependent entries carry its hash.
    return {"version": MEMORY_VERSION, "run_id": rid, "run_root": root,
            "visibility": toolbox.policy.visibility, "policy_sha256": policy_sha}


def _file_bytes(toolbox: RepoToolbox, candidate: Any) -> tuple[str, bytes] | None:
    if not isinstance(candidate, str) or not 0 < len(candidate) <= _MAX_PATH:
        return None
    if not _SAFE_PATH.fullmatch(candidate) or candidate.startswith("/") or "//" in candidate:
        return None
    if any(part in {".", ".."} for part in candidate.split("/")):
        return None
    try:
        rel = toolbox.normalize(candidate)
        if rel != candidate or not toolbox.is_read_allowed(rel):
            return None
        path = toolbox.resolve(rel)
        if not path.is_file() or not path.is_relative_to(toolbox.workdir):
            return None
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        return (rel, data) if len(data) <= MAX_FILE_BYTES else None
    except (OSError, ValueError):
        return None


def _file_hash(toolbox: RepoToolbox, candidate: Any) -> tuple[str, str] | None:
    value = _file_bytes(toolbox, candidate)
    return (value[0], hashlib.sha256(value[1]).hexdigest()) if value else None


def capture_read(toolbox: RepoToolbox, resolved_path: str,
                 arguments: Mapping[str, Any], numbered_result: str) -> None:
    """Capture a successful read *at execution time*, using resolved overlay path.

    Each numbered complete result line must match the current raw or masked
    file view; stored excerpts use only the masked view with the raw file hash.
    Partial/truncated output records only the lines actually returned, not
    the full file; captures are discarded on end-of-node hash mismatch. No-op
    outside a graph-node collector. The caller never needs to pass source prose.
    """
    collector = _ACTIVE_TRANSCRIPT.get()
    if collector is None:
        return
    value = _file_bytes(toolbox, resolved_path)
    if value is None or not isinstance(numbered_result, str):
        return
    rel, data = value
    text = data.decode("utf-8", errors="replace")
    raw_lines = text.split("\n")
    lines = source_visibility.mask_file_view(text, rel).split("\n")
    observed: list[tuple[int, str]] = []
    for row in numbered_result.split("\n"):
        match = re.fullmatch(r"\s*(\d+)\t(.*)", row)
        if not match:
            break
        number, content = int(match[1]), match[2]
        if not 1 <= number <= len(lines) or content not in (lines[number - 1], raw_lines[number - 1]):
            break  # partial last line is not exact evidence
        if observed and number != observed[-1][0] + 1:
            break
        observed.append((number, lines[number - 1]))
    if not observed:
        return
    first, last = observed[0][0], observed[-1][0]
    complete = first == 1 and last == len(lines) and "...[truncated" not in numbered_result
    excerpt = "\n".join(f"{number}: {content}" for number, content in observed)
    collector.append({"name": "_memory_read", "entry": {
        "path": rel, "sha256": hashlib.sha256(data).hexdigest(), "role": "source", "origin": "tool",
        "coverage": {"first_line": first, "last_line": last, "full": complete},
        "excerpt": excerpt[:_EXCERPT_CHARS],
    }})


def _range_excerpt(lines: list[str], first: int, last: int) -> str:
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(first, last + 1))[:_EXCERPT_CHARS]


def _contract_dependent(path: str, role: str, root: str) -> bool:
    return role in {"hardware", "contract"} or (
        path.startswith(root + "/") and any(segment in path for segment in
            ("/workspace/design/", "/workspace/", "/08_isa/", "/09_implementation/", "/10_validation/"))
    )


def validate_memory(memory: Any, state: Mapping[str, Any], toolbox: RepoToolbox) -> dict[str, Any]:
    identity = _identity(state, toolbox)
    if identity is None:
        return {}
    clean = {**identity, "entries": [], "facts": []}
    if not isinstance(memory, dict) or any(memory.get(key) != val for key, val in identity.items()):
        return clean
    items = memory.get("entries")
    if not isinstance(items, list) or len(items) > MAX_ENTRIES:
        return clean
    contract = _digest(state.get("isa_spec") or {})
    seen: set[str] = set()
    for item in items:
        allowed = {"path", "sha256", "role", "origin", "contract_sha256", "coverage", "excerpt"}
        if (not isinstance(item, dict) or not set(item) <= allowed
                or item.get("role") not in _ROLES or item.get("origin") not in _ORIGINS):
            continue
        gate = {"analysis": "analysis", "contract": "isa", "profile": "profile"}.get(item["role"])
        if gate and not _gate_ok(state.get("gates"), gate):
            continue
        sha = item.get("sha256")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            continue
        snapshot = _file_bytes(toolbox, item.get("path"))
        if snapshot is None:
            continue
        current = (snapshot[0], hashlib.sha256(snapshot[1]).hexdigest())
        if current[1] != sha or current[0] in seen:
            continue
        if item.get("contract_sha256") is not None and item["contract_sha256"] != contract:
            continue
        entry = {key: item[key] for key in ("path", "sha256", "role", "origin")}
        if item.get("contract_sha256"):
            entry["contract_sha256"] = item["contract_sha256"]
        coverage, excerpt = item.get("coverage"), item.get("excerpt")
        if isinstance(coverage, dict) and isinstance(excerpt, str) and len(excerpt) <= _EXCERPT_CHARS:
            first, last = coverage.get("first_line"), coverage.get("last_line")
            if type(first) is int and type(last) is int and 1 <= first <= last and type(coverage.get("full")) is bool:
                text = snapshot[1].decode("utf-8", errors="replace")
                raw_lines = text.split("\n")
                if last <= len(raw_lines):
                    visible_lines = source_visibility.mask_file_view(text, snapshot[0]).split("\n")
                    raw_excerpt = _range_excerpt(raw_lines, first, last)
                    visible_excerpt = _range_excerpt(visible_lines, first, last)
                    # Restored old captures may contain only verifier-body lines,
                    # with no signature to mask. Rebuild from the complete,
                    # hash-matched file, not by masking that stored fragment.
                    if excerpt in (raw_excerpt, visible_excerpt):
                        entry.update(coverage=dict(coverage), excerpt=visible_excerpt)
        seen.add(current[0])
        clean["entries"].append(entry)
    # Rebuild deterministic accepted facts from current domain state rather than
    # trusting serialized model prose or an artifact's current filename/hash.
    clean["facts"] = _facts(state)
    return clean


def _paths(state: Mapping[str, Any]) -> list[tuple[Any, str]]:
    root = state["run_root"]
    project, gates = state.get("benchmark_project") or {}, state.get("gates") or {}
    paths: list[tuple[Any, str]] = []
    if isinstance(project, dict):
        paths.extend((project.get(key), "source") for key in ("source_path", "header_path"))
    paths.extend(((state.get("workload_spec_path"), "workload"), (state.get("kernel_file"), "source")))
    for key, role in (("design_files", "hardware"), ("changed_design_files", "hardware"), ("changed_sw_files", "source")):
        if isinstance(state.get(key), list):
            paths.extend((path, role) for path in state[key][:MAX_ENTRIES])
    for gate, relative, role in (("analysis", "07_analysis/workload_analysis.yaml", "analysis"),
                                  ("isa", "08_isa/cfu_isa.yaml", "contract"),
                                  ("profile", "05_profile/result.yaml", "profile")):
        if _gate_ok(gates, gate):
            paths.append((f"{root}/{relative}", role))
    return paths


def _gate_ok(gates: Any, key: str) -> bool:
    gate = gates.get(key) if isinstance(gates, Mapping) else None
    return isinstance(gate, Mapping) and gate.get("ok") is True and not gate.get("stale")


def _facts(state: Mapping[str, Any]) -> list[dict[str, str]]:
    facts = []
    for gate, key, role in (("workload_spec", "workload_spec", "workload"),
                             ("analysis", "workload_analysis", "analysis"),
                             ("isa", "isa_spec", "contract"),
                             ("profile", "profile_result", "profile")):
        value = state.get(key)
        if value and _gate_ok(state.get("gates"), gate):
            visible = source_visibility.mask_model_payload(value)
            body = visible if isinstance(visible, str) else json.dumps(visible, sort_keys=True, ensure_ascii=False)
            facts.append({"role": role, "source": f"state.{key} (accepted {gate} gate)",
                          "sha256": _digest(value), "excerpt": body[:1000]})
    # Domain records are already source-fingerprinted by the graph wrapper.
    # Surface only concise historical intent/check outcomes; never treat these
    # memo entries as proof that a formal verification gate passed.
    for gate, key, role in (("implementation_hw", "hardware_plan", "hardware"),
                            ("implementation_sw", "software_plan", "source")):
        if state.get(key) and _gate_ok(state.get("gates"), gate):
            facts.append({"role": role, "source": f"state.{key} (intent; formal verification independent)",
                          "sha256": _digest(state[key]),
                          "excerpt": str(source_visibility.mask_model_payload(state[key]))[:600]})
    recent = [item for item in (state.get("validation_results") or []) if isinstance(item, dict)][-3:]
    for item in recent:
        observation = item.get("failure_observation") or {}
        body = {"rung": item.get("rung"), "returncode": item.get("returncode"),
                "failure_id": observation.get("id"),
                "summary": str(source_visibility.mask_model_payload(observation.get("summary") or ""))[:500],
                "log_path": item.get("log_path")}
        facts.append({"role": "hardware", "source": "validation history (not a current gate pass)",
                      "sha256": _digest(body), "excerpt": json.dumps(body, ensure_ascii=False)[:800]})
    delivery = state.get("conformance_delivery")
    if isinstance(delivery, dict) and delivery.get("oracle_sha"):
        facts.append({"role": "contract", "source": "state.conformance_delivery (approved test, simulation pending)",
                      "sha256": _digest(delivery),
                      "excerpt": json.dumps({key: delivery.get(key) for key in
                          ("oracle_path", "source_path", "isa_sha", "oracle_sha", "source_sha")}, ensure_ascii=False)[:900]})
    return facts


def derive_memory(state: Mapping[str, Any], node_update: Mapping[str, Any],
                  tool_calls: Sequence[Mapping[str, Any]], toolbox: RepoToolbox,
                  *, node_name: str = "") -> dict[str, Any]:
    if not isinstance(node_update, Mapping):
        return {}
    merged = {**state, **node_update}
    clean = validate_memory(state.get("flow_memory"), merged, toolbox)
    if not clean:
        return {}
    # The wrapper may call this on a normal-return failure update. Keep valid
    # old reads but do not publish this failed attempt's new read/edit evidence.
    new_gates = node_update.get("gates") or {}
    own_gate = {
        "analyze_workload": ("analysis",), "design_contract": ("isa",), "extend_isa": ("isa",),
        "implement": ("implementation_hw", "implementation_sw"),
        "implement_with_tools": ("implementation_hw", "implementation_sw"),
    }.get(node_name, ())
    checked = [new_gates.get(key) for key in own_gate] if own_gate else []
    accepted = not node_update.get("errors") and all(
        not isinstance(gate, Mapping) or gate.get("ok") is not False for gate in checked
    )
    contract = _digest(merged.get("isa_spec") or {})
    candidates = []
    for path, role in _paths(merged):
        current = _file_hash(toolbox, path)
        if current:
            candidates.append({"path": current[0], "sha256": current[1], "role": role, "origin": "state"})
    if accepted and isinstance(tool_calls, (list, tuple)):
        # Only read-time captures are observation evidence. Generic transcript
        # read_file records have no byte snapshot and cannot certify end bytes.
        for call in tool_calls:
            if isinstance(call, Mapping) and call.get("name") == "_memory_read" and isinstance(call.get("entry"), dict):
                entry = dict(call["entry"])
                current = _file_hash(toolbox, entry.get("path"))
                if current and current[1] == entry.get("sha256"):
                    candidates.append(entry)
    by_path = {entry["path"]: entry for entry in clean["entries"]}
    for entry in candidates:
        if _contract_dependent(entry["path"], entry["role"], merged["run_root"]):
            entry["contract_sha256"] = contract
        old = by_path.get(entry["path"])
        if old and old["sha256"] == entry["sha256"]:
            # Do not let unobserved state references overwrite precise captures.
            entry = {**entry, **old} if not entry.get("coverage") else {**old, **entry}
        by_path[entry["path"]] = entry
    clean["entries"] = list(by_path.values())[-MAX_ENTRIES:]
    clean["facts"] = _facts(merged)
    return clean


def render_memory(memory: Any, state: Mapping[str, Any], toolbox: RepoToolbox,
                  stage: str, *, max_chars: int = 1200) -> str:
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    roles = _STAGE_ROLES.get(memory_stage(stage))
    if roles is None or max_chars == 0:
        return ""
    clean = validate_memory(memory, state, toolbox)
    chunks = ["Run-local memory: accepted facts and hash-matched observed ranges below may be reused. "
              "References alone are not prior reads; re-read missing ranges or exact edit text."]
    for fact in clean.get("facts", []):
        if fact["role"] in roles:
            chunk = f"- {fact['source']} [sha256:{fact['sha256'][:12]}]\n{fact['excerpt']}"
            if len("\n".join(chunks + [chunk])) <= max_chars:
                chunks.append(chunk)
    for entry in clean.get("entries", []):
        if entry["role"] not in roles:
            continue
        coverage = entry.get("coverage")
        label = (f"observed lines {coverage['first_line']}-{coverage['last_line']}"
                 + (" (full read)" if coverage["full"] else " (partial read)")
                 if coverage else "reference only; not observed")
        chunk = f"- {entry['role']}: {entry['path']} [sha256:{entry['sha256'][:12]}] {label}"
        if entry.get("excerpt"):
            chunk += "\n" + entry["excerpt"]
        if len("\n".join(chunks + [chunk])) <= max_chars:
            chunks.append(chunk)
    return "\n".join(chunks) if len(chunks) > 1 else ""
