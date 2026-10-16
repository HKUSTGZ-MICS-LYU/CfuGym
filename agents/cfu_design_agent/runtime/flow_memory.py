"""Bounded run-local memory with read-time byte/range provenance.

The graph owns publication: after a node returns it folds hash-matched reads
and independently accepted domain facts into state. A rejected delivery does
not invalidate an unchanged read, nor does a read certify that delivery. Active
reads may also be reused by later calls in the same node without changing state.
A file reference alone is not evidence that the model saw its content.
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

MEMORY_VERSION = 2  # Additive ranges; legacy coverage/excerpt remains readable.
MAX_ENTRIES = 24
MAX_FILE_BYTES = 2_000_000
MAX_RANGE_CHARS = 4000
MAX_RANGES_PER_FILE = 4
MAX_FILE_EXCERPT_CHARS = 12000
MAX_RANGES = 48
MAX_TOTAL_EXCERPT_CHARS = 48000
_EXCERPT_CHARS = 600  # Backward-compatible alias, not the range storage budget.
_RENDER_RANGE_CHARS = 1600
_MAX_PATH = 240
_SAFE_PATH = re.compile(r"[A-Za-z0-9_./+ -]+\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVE_TRANSCRIPT: ContextVar[list | None] = ContextVar("flow_memory_transcript", default=None)
_READ_CONTEXT: ContextVar[dict | None] = ContextVar("flow_memory_read_context", default=None)
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
def collect_tool_transcripts(state: Mapping[str, Any] | None = None) -> Iterator[list]:
    token = _ACTIVE_TRANSCRIPT.set([])
    context = {"run_root": state.get("run_root") if isinstance(state, Mapping) else None,
               "contract_sha256": (_digest(state.get("isa_spec") or {})
                                   if isinstance(state, Mapping) else None)}
    context_token = _READ_CONTEXT.set(context)
    try:
        yield _ACTIVE_TRANSCRIPT.get()
    finally:
        _READ_CONTEXT.reset(context_token)
        _ACTIVE_TRANSCRIPT.reset(token)


def record_tool_transcript(calls: Sequence[Mapping[str, Any]] | None) -> None:
    # Audit transcripts are written separately by llm_step. Generic tool results
    # cannot establish byte provenance, and retaining them here would make the
    # active read cache unbounded across independent calls in one graph node.
    # Keep the public seam for existing callers, but capture_read is its sole
    # evidence producer (including for reserved-looking tool names).
    return None


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


def _copy_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(entry)
    if "coverage" in result:
        result["coverage"] = dict(result["coverage"])
    if "ranges" in result:
        result["ranges"] = [{"coverage": dict(part["coverage"]), "excerpt": part["excerpt"]}
                            for part in result["ranges"]]
    return result


def _sync_alias(entry: dict[str, Any]) -> None:
    """Old readers retain a small alias, never a second unbounded text copy."""
    if "ranges" not in entry:
        return
    if entry["ranges"]:
        first = entry["ranges"][0]
        entry.update(coverage=dict(first["coverage"]), excerpt=first["excerpt"][:_EXCERPT_CHARS])
    else:
        entry.pop("coverage", None)
        entry.pop("excerpt", None)


def _observations(entry: Mapping[str, Any]) -> list:
    if "ranges" in entry:
        return entry["ranges"]
    return [entry] if entry.get("coverage") else []


def _bound_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stable FIFO trimming; identical re-reads never refresh insertion order.

    The active collector may contain several byte versions of one path so a
    rollback can recover a previous observation. Those versions share the same
    per-file budget. Legacy aliases add at most MAX_ENTRIES * 600 characters.
    """
    result = [_copy_entry(entry) for entry in entries[-MAX_ENTRIES:]]

    def cost(group):
        parts = [part for entry in group for part in _observations(entry)]
        return len(parts), sum(len(part["excerpt"]) for part in parts)

    def trim(group, count, chars):
        while True:
            size, length = cost(group)
            if size <= count and length <= chars:
                return
            oldest = next(entry for entry in group if _observations(entry))
            if "ranges" in oldest:
                oldest["ranges"].pop(0)
                _sync_alias(oldest)
            else:
                oldest.pop("coverage", None)
                oldest.pop("excerpt", None)

    for path in dict.fromkeys(entry["path"] for entry in result):
        trim([entry for entry in result if entry["path"] == path],
             MAX_RANGES_PER_FILE, MAX_FILE_EXCERPT_CHARS)
    trim(result, MAX_RANGES, MAX_TOTAL_EXCERPT_CHARS)
    for entry in result:
        _sync_alias(entry)
    return result


def _merge_entry(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> dict[str, Any]:
    if not old or old["sha256"] != new["sha256"]:
        return _copy_entry(new)
    # Neither references nor duplicate reads replace earlier observed metadata.
    result = _copy_entry(old if old.get("coverage") else new)
    ranges = []
    seen = set()
    for entry in (old, new):
        for part in entry.get("ranges", []):
            coverage = part["coverage"]
            key = (coverage["first_line"], coverage["last_line"])
            if key not in seen:
                ranges.append({"coverage": dict(coverage), "excerpt": part["excerpt"]})
                seen.add(key)
    if ranges:
        result["ranges"] = ranges
        _sync_alias(result)
    return result


def _capture_ranges(observed: list[tuple[int, str]], line_count: int, complete: bool) -> list[dict]:
    """Split only at complete lines; skip overlarge lines, never bridge the gap."""
    ranges, rows = [], []
    chars = 0

    def flush():
        if rows:
            first, last = rows[0][0], rows[-1][0]
            ranges.append({"coverage": {"first_line": first, "last_line": last,
                                        "full": complete and first == 1 and last == line_count},
                           "excerpt": "\n".join(row for _, row in rows)})
            rows.clear()

    for number, content in observed:
        row = f"{number}: {content}"
        if rows and (number != rows[-1][0] + 1 or chars + 1 + len(row) > MAX_RANGE_CHARS):
            flush()
            chars = 0
        if len(row) > MAX_RANGE_CHARS:
            continue
        chars += len(row) + bool(rows)
        rows.append((number, row))
    flush()
    return ranges


def capture_read(toolbox: RepoToolbox, resolved_path: str,
                 arguments: Mapping[str, Any], numbered_result: str) -> None:
    """Capture exact returned lines at read time, under the resolved overlay path."""
    collector = _ACTIVE_TRANSCRIPT.get()
    if collector is None:
        return
    value = _file_bytes(toolbox, resolved_path)
    if value is None or not isinstance(numbered_result, str):
        return
    rel, data = value
    context = _READ_CONTEXT.get()
    if context is not None and context["run_root"] is not None and context["run_root"] != toolbox.run_root:
        return
    root = str(toolbox.run_root or "")
    identity = _identity({"run_id": root.rsplit("/", 1)[-1], "run_root": root,
                          "workdir": str(toolbox.workdir),
                          "visibility": getattr(toolbox.policy, "visibility", None)}, toolbox)
    if identity is None or (context is not None and context.get("identity", identity) != identity):
        return
    if context is not None:
        context["identity"] = identity
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
            break
        if observed and number != observed[-1][0] + 1:
            break
        observed.append((number, lines[number - 1]))
    complete = bool(observed and observed[0][0] == 1 and observed[-1][0] == len(lines)
                    and "...[truncated" not in numbered_result)
    ranges = _capture_ranges(observed, len(lines), complete)
    if not ranges:
        return
    entry = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
             "role": "source", "origin": "tool", "ranges": ranges}
    if (context is not None and context.get("contract_sha256") is not None
            and _contract_dependent(rel, "source", toolbox.run_root)):
        entry["contract_sha256"] = context["contract_sha256"]
    # Normalize this read's retained window BEFORE merging it with old reads.
    # Otherwise repeating an oversized read reintroduces evicted head chunks
    # after its retained tail and rotates the cache on every identical call.
    entry = _bound_entries([entry])[0]
    captures = [call["entry"] for call in collector
                if isinstance(call, Mapping) and call.get("name") == "_memory_read"
                and isinstance(call.get("entry"), dict) and call.get("identity", identity) == identity]
    key = (rel, entry["sha256"], entry.get("contract_sha256"))
    for index, old in enumerate(captures):
        if (old["path"], old["sha256"], old.get("contract_sha256")) == key:
            captures[index] = _merge_entry(old, entry)
            break
    else:
        captures.append(entry)
    collector[:] = [{"name": "_memory_read", "entry": item, "identity": dict(identity)}
                    for item in _bound_entries(captures) if _observations(item)]


def _range_excerpt(lines: list[str], first: int, last: int,
                   limit: int | None = _EXCERPT_CHARS) -> str:
    text = "\n".join(f"{number}: {lines[number - 1]}" for number in range(first, last + 1))
    return text if limit is None else text[:limit]


def _contract_dependent(path: str, role: str, root: str) -> bool:
    return role in {"hardware", "contract"} or (
        path.startswith(root + "/") and any(segment in path for segment in
            ("/workspace/design/", "/workspace/", "/08_isa/", "/09_implementation/", "/10_validation/"))
    )


def _valid_coverage(coverage: Any, count: int) -> bool:
    if not isinstance(coverage, dict) or set(coverage) != {"first_line", "last_line", "full"}:
        return False
    first, last, full = coverage["first_line"], coverage["last_line"], coverage["full"]
    return (type(first) is int and type(last) is int and 1 <= first <= last <= count
            and type(full) is bool and (not full or (first == 1 and last == count)))


def _validate_entry(item: Any, state: Mapping[str, Any], toolbox: RepoToolbox,
                    snapshots: dict) -> dict[str, Any] | None:
    allowed = {"path", "sha256", "role", "origin", "contract_sha256", "coverage", "excerpt", "ranges"}
    if (not isinstance(item, dict) or not set(item) <= allowed
            or not isinstance(item.get("role"), str) or item["role"] not in _ROLES
            or not isinstance(item.get("origin"), str) or item["origin"] not in _ORIGINS):
        return None
    gate = {"analysis": "analysis", "contract": "isa", "profile": "profile"}.get(item["role"])
    if gate and not _gate_ok(state.get("gates"), gate):
        return None
    sha, path = item.get("sha256"), item.get("path")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha) or not isinstance(path, str):
        return None
    # Match the tool's existing live-path semantics without changing physical
    # hashing helpers. A formerly direct scaffold name may now select an overlay;
    # its unchanged tracked bytes are no longer the bytes read_file would show.
    from . import tools as tools_mod
    redirected = tools_mod.redirect_target(path, str(toolbox.run_root or ""), toolbox.embench_benchmark)
    if (redirected and (toolbox.workdir / redirected).is_file()
            and toolbox.is_read_allowed(redirected)):
        return None
    if path not in snapshots:
        snapshot = _file_bytes(toolbox, path)
        if snapshot:
            text = snapshot[1].decode("utf-8", errors="replace")
            snapshots[path] = (hashlib.sha256(snapshot[1]).hexdigest(), text.split("\n"),
                               source_visibility.mask_file_view(text, path).split("\n"))
        else:
            snapshots[path] = None
    snapshot = snapshots[path]
    if snapshot is None or snapshot[0] != sha:
        return None
    contract = item.get("contract_sha256")
    dependent = _contract_dependent(path, item["role"], str(state.get("run_root") or ""))
    if dependent and (not isinstance(contract, str) or not _SHA.fullmatch(contract)):
        return None
    if contract is not None and contract != _digest(state.get("isa_spec") or {}):
        return None
    entry = {key: item[key] for key in ("path", "sha256", "role", "origin")}
    if item.get("contract_sha256"):
        entry["contract_sha256"] = item["contract_sha256"]
    raw_lines, visible_lines = snapshot[1:]
    coverage, excerpt = item.get("coverage"), item.get("excerpt")
    legacy_range = None
    if (_valid_coverage(coverage, len(raw_lines)) and isinstance(excerpt, str)
            and len(excerpt) <= _EXCERPT_CHARS):
        first, last = coverage["first_line"], coverage["last_line"]
        raw = _range_excerpt(raw_lines, first, last, None)
        visible = _range_excerpt(visible_lines, first, last, None)
        # A body-only legacy fragment cannot mask itself. Always derive its
        # visible view from the complete, raw-hash-matched file first.
        if excerpt in (raw[:_EXCERPT_CHARS], visible[:_EXCERPT_CHARS]):
            entry.update(coverage=dict(coverage), excerpt=visible[:_EXCERPT_CHARS])
            # Never expand a truncated old excerpt to its wider coverage claim.
            if excerpt in (raw, visible) and len(visible) <= MAX_RANGE_CHARS:
                legacy_range = {"coverage": dict(coverage), "excerpt": visible}
    ranges = item.get("ranges")
    if isinstance(ranges, list) and len(ranges) <= MAX_RANGES_PER_FILE:
        valid, seen = [], set()
        for part in ranges:
            if not isinstance(part, dict) or set(part) != {"coverage", "excerpt"}:
                continue
            coverage, excerpt = part["coverage"], part["excerpt"]
            if (not _valid_coverage(coverage, len(raw_lines)) or not isinstance(excerpt, str)
                    or not 0 < len(excerpt) <= MAX_RANGE_CHARS):
                continue
            first, last = coverage["first_line"], coverage["last_line"]
            raw = _range_excerpt(raw_lines, first, last, None)
            visible = _range_excerpt(visible_lines, first, last, None)
            if (first, last) not in seen and excerpt in (raw, visible) and len(visible) <= MAX_RANGE_CHARS:
                valid.append({"coverage": dict(coverage), "excerpt": visible})
                seen.add((first, last))
        if valid:
            entry["ranges"] = valid
    elif "ranges" not in item and legacy_range:
        entry["ranges"] = [legacy_range]
    # Invalid new ranges cannot be laundered through an otherwise valid alias.
    if "ranges" in item and not entry.get("ranges"):
        entry.pop("coverage", None)
        entry.pop("excerpt", None)
    return entry


def validate_memory(memory: Any, state: Mapping[str, Any], toolbox: RepoToolbox) -> dict[str, Any]:
    identity = _identity(state, toolbox)
    if identity is None:
        return {}
    clean = {**identity, "entries": [], "facts": []}
    if not isinstance(memory, dict) or any(memory.get(key) != val for key, val in identity.items()):
        return clean
    bindings = _fact_bindings(memory)
    if bindings is not None:
        clean["fact_sha256"] = bindings
    items = memory.get("entries")
    if not isinstance(items, list) or len(items) > MAX_ENTRIES:
        return clean
    seen, snapshots = set(), {}
    for item in items:
        entry = _validate_entry(item, state, toolbox, snapshots)
        if entry is not None and entry["path"] not in seen:
            clean["entries"].append(entry)
            seen.add(entry["path"])
    clean["entries"] = _bound_entries(clean["entries"])
    # Serialized facts are never authoritative; domain gates still own them.
    # Fingerprints additionally prevent a rejected value inheriting an old pass.
    clean["facts"] = _facts_for_memory(state, bindings)
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


_FACT_GATES = {
    "workload_spec": "workload_spec", "workload_analysis": "analysis", "isa_spec": "isa",
    "profile_result": "profile", "hardware_plan": "implementation_hw",
    "software_plan": "implementation_sw", "conformance_delivery": None,
}


def _fact_key(fact: Mapping[str, Any]) -> str | None:
    return next((key for key in _FACT_GATES if fact["source"].startswith(f"state.{key} (")), None)


def _fact_bindings(memory: Mapping[str, Any]) -> dict[str, str | None] | None:
    """Fixed-size acceptance constraints, not serialized model prose.

    Legacy memories without constraints still derive facts from domain state.
    An explicit null blocks an unaccepted field until a fresh successful gate.
    """
    if "fact_sha256" not in memory:
        return None
    value = memory["fact_sha256"]
    if not isinstance(value, dict):
        return dict.fromkeys(_FACT_GATES)
    return {key: value[key] if isinstance(value[key], str) and _SHA.fullmatch(value[key]) else None
            for key in _FACT_GATES if key in value}


def _facts_for_memory(state: Mapping[str, Any], bindings: Mapping[str, Any] | None) -> list[dict[str, str]]:
    facts = _facts(state)
    if bindings is None:
        return facts
    return [fact for fact in facts if _fact_key(fact) not in bindings
            or bindings[_fact_key(fact)] == fact["sha256"]]


def _updated_fact_bindings(state: Mapping[str, Any], update: Mapping[str, Any],
                           previous: Mapping[str, Any] | None) -> dict[str, str | None]:
    bindings = dict(previous or {})
    for fact in _facts(state):
        key = _fact_key(fact)
        if key:
            bindings.setdefault(key, fact["sha256"])
    merged = {**state, **update}
    gates = update.get("gates")
    for key, gate in _FACT_GATES.items():
        fresh_pass = (not update.get("errors") and
                      ((gate is None and key in update) or (gate is not None and _gate_ok(gates, gate))))
        if fresh_pass:
            bindings[key] = _digest(merged[key]) if merged.get(key) else None
        elif key in update:
            # No fresh acceptance: retain the previous accepted digest, or an
            # explicit denial if no accepted value existed. Merely rendering or
            # running an unrelated node must not bless the rejected replacement.
            bindings.setdefault(key, None)
    return bindings


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
    contract = _digest(merged.get("isa_spec") or {})
    candidates = []
    for path, role in _paths(merged):
        current = _file_hash(toolbox, path)
        if current:
            candidates.append({"path": current[0], "sha256": current[1], "role": role, "origin": "state"})
    # Normal failed deliveries reach here only after transaction rollback. Read
    # observations are independent of acceptance and still require exact current
    # hashes. Exceptions never reach this publication seam. Generic transcripts
    # and edit/check output are not read evidence.
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            if (isinstance(call, Mapping) and call.get("name") == "_memory_read"
                    and isinstance(call.get("entry"), dict)
                    and call.get("identity", _identity(merged, toolbox)) == _identity(merged, toolbox)):
                candidates.append(dict(call["entry"]))
    by_path = {entry["path"]: entry for entry in clean["entries"]}
    snapshots = {}
    for item in candidates:
        path, role = item.get("path"), item.get("role")
        if (isinstance(path, str) and isinstance(role, str)
                and _contract_dependent(path, role, merged["run_root"])):
            # A captured dependency is immutable. Missing fingerprints occur
            # only for legacy callers without a node-bound collector context.
            item.setdefault("contract_sha256", contract)
        entry = _validate_entry(item, merged, toolbox, snapshots)
        if entry is not None:
            by_path[entry["path"]] = _merge_entry(by_path.get(entry["path"]), entry)
    clean["entries"] = _bound_entries(list(by_path.values()))
    bindings = _updated_fact_bindings(state, node_update, clean.get("fact_sha256"))
    clean["fact_sha256"] = bindings
    clean["facts"] = _facts_for_memory(merged, bindings)
    return clean


def memory_for_prompt(state: Mapping[str, Any], toolbox: RepoToolbox) -> dict[str, Any]:
    """Read-only view of persisted memory plus this node's still-current reads.

    Never publish, consume the collector, accept a pending answer, or synthesize
    observations from state references. Revalidate on every independent call so
    edits, rollbacks, policy changes and contract revisions take effect at once.
    """
    return derive_memory(state, {}, tuple(_ACTIVE_TRANSCRIPT.get() or ()), toolbox)


def _entry_header(entry: Mapping[str, Any]) -> str:
    return f"- {entry['role']}: {entry['path']} [sha256:{entry['sha256'][:12]}] "


def _render_observation(entry: Mapping[str, Any], part: Mapping[str, Any],
                        budget: int, *, legacy: bool = False) -> str:
    coverage, excerpt = part["coverage"], part["excerpt"]
    first, last = coverage["first_line"], coverage["last_line"]
    heading = (_entry_header(entry) + f"observed lines {first}-{last}"
               + (" (full read)" if coverage["full"] else " (partial read)"))
    if legacy:
        # Legacy coverage can exceed the 600 characters actually retained.
        chunk = heading + "; provided legacy excerpt prefix only\n" + excerpt
        return chunk if len(chunk) <= budget else ""
    chunk = heading + "\n" + excerpt
    if len(chunk) <= budget and len(excerpt) <= _RENDER_RANGE_CHARS:
        return chunk
    rows, selected = excerpt.split("\n"), []
    best = ""
    for index, row in enumerate(rows):
        body = "\n".join([*selected, row])
        # One complete long line may use more than the usual fair-share budget;
        # never pretend that a character-truncated line is exact edit text.
        if selected and len(body) > _RENDER_RANGE_CHARS:
            break
        provided_last = first + index
        label = ("" if provided_last == last else
                 f"; provided lines {first}-{provided_last} (bounded excerpt)")
        candidate = heading + label + "\n" + body
        if len(candidate) > budget:
            break
        selected.append(row)
        best = candidate
    return best


def render_memory(memory: Any, state: Mapping[str, Any], toolbox: RepoToolbox,
                  stage: str, *, max_chars: int = 1200) -> str:
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    roles = _STAGE_ROLES.get(memory_stage(stage))
    if roles is None or max_chars == 0:
        return ""
    clean = validate_memory(memory, state, toolbox)
    heading = ("Run-local memory: accepted facts and hash-matched observed ranges below may be reused. "
               "References alone are not prior reads; re-read missing ranges or exact edit text.")
    available = max_chars - len(heading)
    entries = [entry for entry in clean.get("entries", []) if entry["role"] in roles]
    observed = [entry for entry in entries if _observations(entry)]
    # Facts cannot starve usable source. Unused fact space is given back to reads;
    # reference-only rows have the lowest priority. Ordering has no timestamps or
    # recency refreshes, and rendering itself does not mutate either memory view.
    fact_budget = min(2000, available // 3) if observed else available
    facts = []
    for fact in clean.get("facts", []):
        if fact["role"] in roles:
            chunk = f"- {fact['source']} [sha256:{fact['sha256'][:12]}]\n{fact['excerpt']}"
            if len(chunk) + 1 <= fact_budget:
                facts.append(chunk)
                fact_budget -= len(chunk) + 1
    remaining = available - sum(len(chunk) + 1 for chunk in facts)
    sources = []
    # Fair, stable selection: one range per file before each file's next range.
    for index in range(MAX_RANGES_PER_FILE):
        for entry in observed:
            parts = _observations(entry)
            if index < len(parts):
                chunk = _render_observation(entry, parts[index], remaining - 1,
                                            legacy="ranges" not in entry)
                if chunk:
                    sources.append(chunk)
                    remaining -= len(chunk) + 1
    references = []
    for entry in entries:
        if not _observations(entry):
            chunk = _entry_header(entry) + "reference only; not observed"
            if len(chunk) + 1 <= remaining:
                references.append(chunk)
                remaining -= len(chunk) + 1
    chunks = sources + facts + references
    return "\n".join([heading, *chunks]) if chunks else ""
