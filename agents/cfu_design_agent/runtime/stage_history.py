"""Bounded execution handoffs in native parent state, never tool replay.

Only canonical artifact references and source fingerprints are retained. Summaries
are rebuilt from hash-matched, policy-visible artifacts before model use. The
node wrapper publishes final completed/rejected outcomes; the in-node collector
also supports independent calls before that boundary without mutating state.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import re
from typing import Any, Iterator, Mapping, Sequence

import yaml

from . import flow_memory, source_visibility
from .tools import RepoToolbox

VERSION = 1
MAX_ENTRIES = 12
MAX_PER_STAGE = 3
MAX_DEPENDENCIES = 32
MAX_ARTIFACT_BYTES = 128_000
MAX_SUMMARY_CHARS = 1200
_STAGES = ("ingest_input", "analysis", "isa", "implementation", "review")
_ACTIVE: ContextVar[list | None] = ContextVar("stage_execution_handoffs", default=None)
_LABEL = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
# Strip complete AND unterminated delivery blocks before applying any text cap.
_DELIVERY = re.compile(r"(?ms)^\s*(EDIT_BEGIN|FILE_BEGIN|PATCH_BEGIN)\b.*?(?:^\s*(?:EDIT_END|FILE_END|PATCH_END)\b[^\n]*(?:\n|$)|\Z)")
_FENCE = re.compile(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?(?:^\s*(?:`{3,}|~{3,})[^\n]*(?:\n|$)|\Z)")


@contextmanager
def collect_executions() -> Iterator[list]:
    token = _ACTIVE.set([])
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


def collecting() -> bool:
    return _ACTIVE.get() is not None


def _identity(state: Mapping[str, Any], toolbox: RepoToolbox) -> dict | None:
    identity = flow_memory._identity(state, toolbox)
    if identity is None:
        return None
    task = {key: state.get(key) for key in
            ("task", "workload_spec", "benchmark_name", "isa_contract", "require_conformance")}
    try:
        return {**identity, "version": VERSION, "task_sha256": flow_memory._digest(task)}
    except (TypeError, ValueError):
        return None


def _summary(data: bytes, path: str, kind: str) -> str:
    text = source_visibility.mask_file_view(data.decode("utf-8", errors="replace"), path)
    if kind == "tools":
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            return ""
        if not isinstance(document, dict) or not isinstance(document.get("transcript"), list):
            return ""
        notes = []
        for call in document["transcript"]:
            if not isinstance(call, dict):
                continue
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            if call.get("name") == "submit_stage" and call.get("ok") is False:
                notes.append("Rejected submission: " + str(args.get("delivery") or ""))
                notes.append(str(call.get("result") or ""))
            elif call.get("name") == "(tool loop failed)":
                notes.append("Tool loop failed: " + str(args.get("error") or "unknown error"))
        text = "\n".join(notes[-4:])
    text = str(source_visibility.mask_model_payload(text))
    text = _FENCE.sub("[historical code omitted]\n", _DELIVERY.sub("[historical delivery omitted]\n", text))
    return " ".join(text.split())[:MAX_SUMMARY_CHARS]


def _dependencies(state: Mapping[str, Any], toolbox: RepoToolbox, stage: str) -> list[dict] | None:
    paths = []
    memory = flow_memory.memory_for_prompt(state, toolbox)
    for entry in memory.get("entries", []):
        if entry.get("coverage") or entry.get("ranges"):
            paths.append(entry["path"])
    project = state.get("benchmark_project") or {}
    if isinstance(project, dict):
        paths.extend(project.get(key) for key in ("source_path", "header_path"))
    paths.extend(state.get(key) for key in ("kernel_file", "workload_spec_path"))
    if stage in {"implementation", "review"}:
        paths.extend(state.get("design_files") or [])
        paths.extend(state.get("changed_sw_files") or [])
    root = str(state.get("run_root") or "") + "/"
    result = {}
    for path in paths:
        if not isinstance(path, str):
            continue
        # Mutable stage output files are not source inputs. In particular an ISA
        # answer must not depend on the old contract artifact it is replacing.
        if path.startswith(root) and not path.startswith(root + "workspace/"):
            continue
        current = flow_memory._file_hash(toolbox, path)
        if current:
            result[current[0]] = {"path": current[0], "sha256": current[1]}
    return list(result.values()) if len(result) <= MAX_DEPENDENCIES else None


def _entry(value: Any, state: Mapping[str, Any], toolbox: RepoToolbox) -> dict | None:
    fields = {"stage", "label", "kind", "path", "sha256", "dependencies",
              "contract_sha256", "capture_identity", "status", "summary"}
    if not isinstance(value, dict) or not set(value) <= fields:
        return None
    identity = _identity(state, toolbox)
    if identity is None or value.get("capture_identity") != identity:
        return None
    stage, label = value.get("stage"), value.get("label")
    if stage not in _STAGES or not isinstance(label, str) or not _LABEL.fullmatch(label):
        return None
    if value.get("kind") not in ("model", "tools") or value.get("status") not in ("pending", "completed", "rejected"):
        return None
    path, sha = value.get("path"), value.get("sha256")
    if (not isinstance(path, str) or not path.startswith(str(state.get("run_root")) + "/")
            or not isinstance(sha, str) or not _SHA.fullmatch(sha)):
        return None
    if value["kind"] == "model":
        # Answers only: never prompt/reasoning sidecars or arbitrary source files.
        if not re.search(r"/llm_raw/[A-Za-z0-9_-]+_[0-9]{3,}\.txt\Z", path):
            return None
    elif not path.endswith((".yaml", ".yml")):
        return None
    contract = value.get("contract_sha256")
    if (stage in ("isa", "implementation", "review") or "contract_sha256" in value):
        if contract != flow_memory._digest(state.get("isa_spec") or {}):
            return None
    deps = value.get("dependencies")
    if not isinstance(deps, list) or len(deps) > MAX_DEPENDENCIES:
        return None
    clean_deps = []
    for dep in deps:
        if not isinstance(dep, dict) or set(dep) != {"path", "sha256"}:
            return None
        current = flow_memory._file_hash(toolbox, dep.get("path"))
        if current is None or current != (dep["path"], dep["sha256"]):
            return None
        clean_deps.append(dict(dep))
    current = flow_memory._file_bytes(toolbox, path)
    if current is None or len(current[1]) > MAX_ARTIFACT_BYTES:
        return None
    if current[0] != path or hashlib.sha256(current[1]).hexdigest() != sha:
        return None
    summary = _summary(current[1], path, value["kind"])
    if not summary:
        return None
    return {**{key: value[key] for key in fields - {"summary", "contract_sha256", "dependencies", "capture_identity"}},
            "dependencies": clean_deps, "summary": summary, "capture_identity": dict(identity),
            **({"contract_sha256": contract} if contract is not None else {})}


def validate_history(history: Any, state: Mapping[str, Any], toolbox: RepoToolbox) -> dict:
    identity = _identity(state, toolbox)
    if identity is None:
        return {}
    clean = {**identity, "entries": []}
    if not isinstance(history, dict) or any(history.get(key) != val for key, val in identity.items()):
        return clean
    entries = history.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        return clean
    seen = set()
    for value in entries:
        entry = _entry(value, state, toolbox)
        if entry is not None and (entry["path"], entry["sha256"]) not in seen:
            seen.add((entry["path"], entry["sha256"]))
            clean["entries"].append(entry)
    return clean


def record_execution(state: Mapping[str, Any], toolbox: RepoToolbox, stage: str,
                     label: str, raw_path: str, *, kind: str = "model") -> None:
    collector = _ACTIVE.get()
    if collector is None:
        return
    identity = _identity(state, toolbox)
    if identity is None:
        return
    stage = flow_memory.memory_stage(stage)
    if stage not in _STAGES:
        return
    current = flow_memory._file_hash(toolbox, raw_path)
    deps = _dependencies(state, toolbox, stage)
    if current is None or deps is None:
        return
    value = {"stage": stage, "label": label, "kind": kind, "capture_identity": identity,
             "path": current[0], "sha256": current[1], "dependencies": deps, "status": "pending"}
    if stage in {"isa", "implementation", "review"}:
        value["contract_sha256"] = flow_memory._digest(state.get("isa_spec") or {})
    entry = _entry(value, state, toolbox)
    if entry is not None and not any(old["path"] == entry["path"] and old["sha256"] == entry["sha256"] for old in collector):
        collector.append(entry)
        own = [i for i, old in enumerate(collector) if (old["stage"], old["label"]) == (stage, label)]
        drop = set(own[:-MAX_PER_STAGE])
        collector[:] = [old for i, old in enumerate(collector) if i not in drop][-MAX_ENTRIES:]


def _bind_outputs(event: Mapping[str, Any], state: Mapping[str, Any],
                  toolbox: RepoToolbox, *, outputs: Mapping[str, Any]) -> dict:
    """Bind a fresh summary to committed output bytes, never rebind read evidence.

    Legacy delivery records the answer before applying it. Only workspace outputs
    explicitly reported by the graph may change this fresh summary's publication
    fingerprint. Persisted historical entries are never rebound on later visits.
    """
    value = dict(event)
    deps = value.get("dependencies")
    if not isinstance(deps, list) or any(
            not isinstance(dep, dict) or set(dep) != {"path", "sha256"}
            or not isinstance(dep["path"], str) for dep in deps):
        return value
    by_path = {dep["path"]: dict(dep) for dep in deps}
    root = str(state.get("run_root") or "") + "/workspace/"
    for key in ("changed_design_files", "changed_sw_files"):
        paths = outputs.get(key) or []
        if not isinstance(paths, (list, tuple)):
            continue
        for path in paths:
            if not isinstance(path, str) or not path.startswith(root):
                continue
            current = flow_memory._file_hash(toolbox, path)
            if current:
                by_path[current[0]] = {"path": current[0], "sha256": current[1]}
    return {**value, "dependencies": list(by_path.values())}


def derive_history(state: Mapping[str, Any], node_update: Mapping[str, Any],
                   events: Sequence[Mapping[str, Any]], toolbox: RepoToolbox,
                   *, node_name: str = "", accepted: bool = True) -> dict:
    merged = {**state, **node_update}
    clean = validate_history(state.get("stage_history"), merged, toolbox)
    if not clean:
        return {}
    entries = list(clean["entries"])
    for event in events:
        value = _bind_outputs(event, merged, toolbox, outputs=node_update) if accepted else dict(event)
        value["status"] = "completed" if accepted else "rejected"
        if value.get("stage") == "isa" and (accepted or "isa_spec" in node_update):
            # A lint-rejected candidate is still the next ISA repair's context,
            # not an accepted fact or reusable read of the old contract.
            value["contract_sha256"] = flow_memory._digest(merged.get("isa_spec") or {})
        entry = _entry(value, merged, toolbox)
        if entry is None or any(old["path"] == entry["path"] and old["sha256"] == entry["sha256"] for old in entries):
            continue
        entries.append(entry)
        # Replacement channel: prune here, never append-reduce an entire ledger.
        own = [i for i, old in enumerate(entries) if (old["stage"], old["label"]) == (entry["stage"], entry["label"])]
        drop = set(own[:-MAX_PER_STAGE])
        entries = [old for i, old in enumerate(entries) if i not in drop][-MAX_ENTRIES:]
    return {**clean, "entries": entries}


def render_history(state: Mapping[str, Any], toolbox: RepoToolbox, stage: str,
                   *, label: str = "", max_chars: int = 900) -> str:
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    stage = flow_memory.memory_stage(stage)
    if stage not in _STAGES or max_chars == 0:
        return ""
    clean = validate_history(state.get("stage_history"), state, toolbox)
    entries = list(clean.get("entries", []))
    for event in _ACTIVE.get() or ():
        entry = _entry(event, state, toolbox)
        if entry is not None and not any(old["path"] == entry["path"] and old["sha256"] == entry["sha256"] for old in entries):
            entries.append(entry)
    own = [i for i, entry in enumerate(entries) if entry["stage"] == stage and (not label or entry["label"] == label)]
    sibling = [i for i, entry in enumerate(entries) if entry["stage"] == stage and i not in own]
    upstream = [i for i, entry in enumerate(entries)
                if _STAGES.index(entry["stage"]) < _STAGES.index(stage) and entry["status"] == "completed"]
    selected = sorted(set(own[-2:] + sibling[-1:] + upstream[-1:]))
    if not selected:
        return ""
    header = ("Historical execution notes (model claims, not current gate evidence). "
              "Do not replay old edits or infer exact current source from these summaries.")
    if len(header) + 80 > max_chars:
        return ""
    chunks = [header]
    remaining = max_chars - len(header)
    for n, i in enumerate(selected):
        entry = entries[i]
        head = f"- {entry['stage']}/{entry['label']} [{entry['status']}] {entry['path']}\n"
        allowance = remaining // (len(selected) - n) - len(head) - 1
        if allowance < 40:
            continue
        body = entry["summary"][:allowance]
        chunk = head + body
        chunks.append(chunk)
        remaining -= len(chunk) + 1
    return "\n".join(chunks) if len(chunks) > 1 else ""
