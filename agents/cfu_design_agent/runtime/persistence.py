"""Opt-in, per-run LangGraph SQLite persistence and conservative side-effect journaling.

This module does not wire the graph or CLI. Hold ``open_run_persistence`` across
compile/invoke, and wrap *each* external side effect in ``external_operation``.
A pending record requires human reconciliation: a journal cannot prove whether a
process died before or after an external effect, and does not provide exactly-once.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterator

import yaml

from . import processes
from .artifacts import ARTIFACT_SCHEMA_VERSION, new_run_id, metadata_lock
from .policy import PolicyError, normalize_repo_path


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class PersistenceRiskError(RuntimeError):
    """Resume was refused because recorded and current evidence disagree."""


class RunWriterBusyError(RuntimeError):
    """Another writer currently owns this run's advisory lock."""


class PendingOperationError(PersistenceRiskError):
    """An external operation may already have occurred; reconcile manually."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    # Make the directory entry durable as well as the JSON bytes.
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _inside(root: Path, run_root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PersistenceRiskError("artifact path is missing")
    try:
        rel = normalize_repo_path(root, raw)
    except PolicyError as exc:
        raise PersistenceRiskError(f"unsafe artifact path {raw!r}: {exc}") from exc
    path = root / rel
    if not path.is_relative_to(run_root) or not path.is_file():
        raise PersistenceRiskError(f"run artifact is missing or outside this run: {raw}")
    return path


def _source_rel(root: Path, raw: str) -> str:
    try:
        return normalize_repo_path(root, raw)
    except (PolicyError, TypeError) as exc:
        raise PersistenceRiskError(f"unsafe recorded source path {raw!r}: {exc}") from exc


def _hash_ref(root: Path, run_root: Path, ref: dict[str, Any]) -> bytes:
    digest = ref.get("sha256")
    if not isinstance(digest, str) or not _SHA.fullmatch(digest):
        raise PersistenceRiskError("artifact ref has invalid sha256")
    immutable = _inside(root, run_root, ref.get("immutable_path"))
    data = immutable.read_bytes()
    if _digest(data) != digest:
        raise PersistenceRiskError(f"immutable artifact hash mismatch: {immutable}")
    return data


def _validate_sources(root: Path, run_root: Path, manifest: dict[str, Any],
                      candidate_source: str | None, candidate_sha256: str | None) -> None:
    if (candidate_source is None) != (candidate_sha256 is None):
        raise ValueError("candidate_source and candidate_sha256 must be supplied together")
    if candidate_source is not None and (not isinstance(candidate_source, str) or not candidate_source):
        raise ValueError("candidate_source must be a nonempty repo-relative path")
    if candidate_sha256 is not None and (not isinstance(candidate_sha256, str) or not _SHA.fullmatch(candidate_sha256)):
        raise ValueError("candidate_sha256 must be a lowercase SHA-256 hex digest")
    refs = manifest.get("artifacts", [])
    if not isinstance(refs, list) or any(not isinstance(ref, dict) for ref in refs):
        raise PersistenceRiskError("run.yaml artifacts must be a list of mappings")
    by_path = {ref.get("path"): ref for ref in refs if isinstance(ref.get("path"), str)}
    # Only the latest candidate manifest is authoritative. Do not silently fall
    # back to an older verified candidate when the latest ref is damaged.
    candidates = [ref for ref in refs if str(ref.get("path", "")).endswith("/candidate.yaml")
                  and ref.get("kind") == "yaml"]
    candidate_files: list[dict[str, Any]] = []
    if candidates:
        candidate_bytes = _hash_ref(root, run_root, candidates[-1])
        try:
            candidate = yaml.safe_load(candidate_bytes)
        except yaml.YAMLError as exc:
            raise PersistenceRiskError("latest candidate.yaml is malformed") from exc
        if not isinstance(candidate, dict) or not isinstance(candidate.get("files"), list):
            raise PersistenceRiskError("latest candidate.yaml has no files list")
        candidate_files = candidate["files"]
        if not candidate_files:
            raise PersistenceRiskError("latest candidate.yaml lists no source snapshots")
    matched = candidate_source is None
    candidate_sources: set[str] = set()
    for record in candidate_files:
        if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ("source", "snapshot", "sha256")):
            raise PersistenceRiskError("candidate source record is malformed")
        source = record["source"]
        candidate_sources.add(source)
        snapshot = record["snapshot"]
        ref = by_path.get(snapshot)
        if ref is None or ref.get("kind") != "validated_source":
            raise PersistenceRiskError(f"candidate source snapshot ref missing: {snapshot}")
        inputs = ref.get("inputs")
        if (record["sha256"] != ref.get("sha256") or not isinstance(inputs, list)
                or source not in inputs):
            raise PersistenceRiskError(f"candidate source/hash differs from artifact ref: {source}")
        _hash_ref(root, run_root, ref)  # Compare immutable bytes, NEVER mutable latest ref.
        if source == candidate_source:
            matched = record["sha256"] == candidate_sha256
        # Generated overlays are intentionally mutable, not original source inputs.
        normalized = _source_rel(root, source)
        if not normalized.startswith("agents/generated/"):
            live = root / normalized
            if not live.is_file() or _digest(live.read_bytes()) != record["sha256"]:
                raise PersistenceRiskError(f"non-generated candidate source changed: {source}")
    # A standalone changed_source ref can be recorded without a candidate snapshot.
    # Validate its newest version only, never an older version or its latest alias.
    changed = [ref for ref in refs if ref.get("kind") == "changed_source"]
    latest: dict[str, dict[str, Any]] = {}
    for ref in changed:
        inputs = ref.get("inputs")
        source = ref.get("source") or (inputs[0] if isinstance(inputs, list) and inputs else None)
        if not isinstance(source, str) or not source:
            raise PersistenceRiskError("changed_source artifact has no source path")
        latest[source] = ref
    for source, ref in latest.items():
        normalized = _source_rel(root, source)
        if normalized.startswith("agents/generated/"):
            continue
        _hash_ref(root, run_root, ref)
        live = root / normalized
        if not live.is_file() or _digest(live.read_bytes()) != ref["sha256"]:
            raise PersistenceRiskError(f"non-generated changed source changed: {source}")
        if source == candidate_source and source not in candidate_sources:
            matched = ref["sha256"] == candidate_sha256
    if not matched:
        raise PersistenceRiskError("candidate source/hash does not match the latest recorded source artifact")
    if candidate_source is not None:
        normalized = _source_rel(root, candidate_source)
        if normalized.startswith("agents/generated/"):
            raise PersistenceRiskError("candidate_source must be a non-generated source")
        live = root / normalized
        if not live.is_file() or _digest(live.read_bytes()) != candidate_sha256:
            raise PersistenceRiskError("candidate source bytes changed since recording")


@dataclass
class OperationJournal:
    path: Path
    run_id: str

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": "1", "run_id": self.run_id, "operations": {}}
        try:
            value = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise PersistenceRiskError(f"operation journal unreadable: {self.path}") from exc
        if (not isinstance(value, dict) or value.get("schema_version") != "1"
                or value.get("run_id") != self.run_id or not isinstance(value.get("operations"), dict)):
            raise PersistenceRiskError("operation journal schema/run_id mismatch")
        return value

    def assert_no_pending(self) -> None:
        for operation_id, entry in self._load()["operations"].items():
            if not isinstance(entry, dict) or entry.get("status") not in {"pending", "committed"}:
                raise PersistenceRiskError(f"invalid journal entry: {operation_id}")
            if entry["status"] == "pending":
                raise PendingOperationError(
                    f"operation {operation_id!r} is pending; external effect may have happened. "
                    "Reconcile manually before resume; it will not be retried automatically."
                )

    @contextmanager
    def external_operation(self, node: str, operation_id: str,
                           details: dict[str, Any] | None = None) -> Iterator[None]:
        """Write pending before an effect; commit normal or confirmed-cancelled exits.

        Call only while the owning run context holds its exclusive writer lock.
        A duplicate id is refused, including an already committed operation.
        """
        if not node or not isinstance(node, str) or not _RUN_ID.fullmatch(operation_id):
            raise ValueError("node and operation_id must be nonempty; id uses [A-Za-z0-9_-]")
        with metadata_lock(self.path):
            journal = self._load()
            if operation_id in journal["operations"]:
                raise PersistenceRiskError(f"operation id already recorded: {operation_id}")
            entry = {"node": node, "status": "pending", "details": details or {}}
            journal["operations"][operation_id] = entry
            _atomic_json(self.path, journal)
        # Never hold the metadata lock across the actual external operation.
        # Exceptions deliberately leave pending evidence for reconciliation.
        try:
            yield
        except processes.ValidationCancelled:
            # The managed runner raises cancellation only after terminating and
            # reaping the process group (or before spawning). Its outcome is
            # known, unlike an interrupted/uncertain external operation.
            self._commit_operation(operation_id, cancelled=True)
            raise
        else:
            self._commit_operation(operation_id)

    def _commit_operation(self, operation_id: str, *, cancelled: bool = False) -> None:
        with metadata_lock(self.path):
            journal = self._load()
            entry = journal["operations"][operation_id]
            entry["status"] = "committed"
            if cancelled:
                entry["outcome"] = "cancelled"
            _atomic_json(self.path, journal)


@dataclass
class RunPersistence:
    saver: Any
    run_id: str
    run_root: Path
    checkpoint_path: Path
    journal: OperationJournal

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.run_id, "checkpoint_ns": ""}}


@contextmanager
def open_run_persistence(workdir: Path, run_id: str = "", *,
                         out_dir: str = "agents/generated", resume: bool = False,
                         candidate_source: str | None = None,
                         candidate_sha256: str | None = None) -> Iterator[RunPersistence]:
    """Acquire one nonblocking run writer; own and close a SQLite saver connection.

    Missing checkpoint dependency raises ImportError; never falls back to memory.
    This is opt-in and must be held across graph compile AND invocation.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    root = Path(workdir).resolve()
    rid = run_id or ("" if resume else new_run_id())
    if not isinstance(rid, str) or not _RUN_ID.fullmatch(rid):
        raise PolicyError(f"invalid run id: {rid!r}")
    out_rel = normalize_repo_path(root, out_dir)
    if not (out_rel == "agents/generated" or out_rel.startswith("agents/generated/")):
        raise PolicyError("artifact output must be under agents/generated")
    expected_rel = f"{out_rel}/runs/{rid}"
    run_rel = normalize_repo_path(root, expected_rel)
    if run_rel != expected_rel:
        raise PolicyError(f"run path redirects through a symlink: {expected_rel}")
    run_root = root / run_rel
    checkpoint = run_root / "checkpoint.sqlite"
    manifest_file = run_root / "run.yaml"
    if resume:
        if not manifest_file.is_file() or not checkpoint.is_file():
            raise PersistenceRiskError("resume requires existing run.yaml and checkpoint.sqlite; no run created")
    elif run_root.exists() and (not run_root.is_dir() or any(run_root.iterdir())):
        raise FileExistsError(f"new run refuses existing nonempty directory: {run_root}")
    if not resume:
        run_root.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(run_root / ".writer.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunWriterBusyError(f"run {rid!r} already has a writer") from exc
        if resume:
            try:
                manifest = yaml.safe_load(manifest_file.read_bytes())
            except (OSError, yaml.YAMLError) as exc:
                raise PersistenceRiskError("run.yaml unreadable or malformed") from exc
            if (not isinstance(manifest, dict) or manifest.get("run_id") != rid
                    or str(manifest.get("schema_version")) != ARTIFACT_SCHEMA_VERSION):
                raise PersistenceRiskError("run.yaml run_id/schema_version mismatch")
            _validate_sources(root, run_root, manifest, candidate_source, candidate_sha256)
        else:
            if any(child.name != ".writer.lock" for child in run_root.iterdir()):
                raise FileExistsError(f"new run refuses existing nonempty directory: {run_root}")
            manifest_file.write_text(yaml.safe_dump({"schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": rid, "status": "initialized", "artifacts": []}))
        journal = OperationJournal(run_root / "operation_journal.json", rid)
        journal.assert_no_pending()
        connection = sqlite3.connect(checkpoint, check_same_thread=False)
        try:
            saver = SqliteSaver(connection)
            saver.setup()
            yield RunPersistence(saver, rid, run_root, checkpoint, journal)
        finally:
            connection.close()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
