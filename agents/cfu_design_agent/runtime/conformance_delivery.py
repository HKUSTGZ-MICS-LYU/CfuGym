"""Host-owned, integrity-checked oracle/C delivery at two fixed run paths.

This service never builds, executes model C, or selects arbitrary artifact refs.
Callers own syntax checks and acceptance; only an accepted record belongs in
``state['conformance_delivery']``. Pending writes can be inspected locally before
acceptance, and the entire attempt can be restored without a manifest.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import yaml

from .. import conformance
from . import commands
from .policy import PolicyError

_SCHEMA_VERSION = 1
_HASH_FIELDS = ("isa_sha", "oracle_sha", "source_sha", "generator_sha")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def delivery_paths(state: dict[str, Any]) -> dict[str, str]:
    """Return the sole current-run oracle metadata and generated C locations."""
    root = str(state.get("run_root", "") or "").rstrip("/")
    path = PurePosixPath(root)
    if (not root or path.is_absolute() or ".." in path.parts
            or path.as_posix() != root or not root.startswith("agents/generated/runs/")
            or len(path.parts) != 4):
        raise ValueError("conformance delivery requires a canonical current run_root")
    base = f"{root}/{commands.CONFORMANCE_SUBDIR}"
    return {"oracle_path": f"{base}/oracle.yaml",
            "source_path": f"{base}/cfu_conformance.c"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _generator_sha() -> str:
    # Hash the owning module's whole source, not a frozen imported binding.
    return _digest(Path(conformance.__file__).read_bytes())


def _safe_path(toolbox: Any, path: str, *, write: bool = False) -> Path:
    if toolbox.normalize(path) != path or not toolbox.is_read_allowed(path):
        raise PolicyError(f"conformance path is not policy-readable: {path}")
    if not toolbox.is_current_run_path(path):
        raise PolicyError(f"conformance path belongs to another run: {path}")
    if write and not toolbox.is_write_allowed(path):
        raise PolicyError(f"conformance path is not policy-writable: {path}")
    full = Path(toolbox.workdir) / path
    # normalize resolves links; explicitly reject even links within this run.
    for part in (full, *full.parents):
        if part == Path(toolbox.workdir):
            break
        if part.is_symlink():
            raise PolicyError(f"conformance path contains a symlink: {path}")
    return full


def _snapshot(toolbox: Any, paths: dict[str, str], *, write: bool = False) -> dict[str, bytes | None]:
    result = {}
    for path in paths.values():
        full = _safe_path(toolbox, path, write=write)
        try:
            result[path] = full.read_bytes() if full.is_file() else None
        except FileNotFoundError:
            result[path] = None
        if full.exists() and not full.is_file():
            raise PolicyError(f"conformance path is not a regular file: {path}")
    return result


def _restore(toolbox: Any, snapshot: dict[str, bytes | None]) -> None:
    errors = []
    for path, data in snapshot.items():
        try:
            full = _safe_path(toolbox, path, write=True)
            if data is None:
                if full.exists():
                    full.unlink()
            else:
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_bytes(data)
            if (full.exists() if data is None else full.read_bytes() != data):
                raise OSError(f"rollback verification failed: {path}")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise OSError("conformance delivery rollback failed: " + "; ".join(errors))


def _record_problems(record: Any, paths: dict[str, str]) -> list[str]:
    if not isinstance(record, dict):
        return ["conformance delivery record must be a mapping"]
    errors = []
    if type(record.get("schema_version")) is not int or record.get("schema_version") != _SCHEMA_VERSION:
        errors.append("conformance delivery schema_version must be 1")
    for field in _HASH_FIELDS:
        if not isinstance(record.get(field), str) or not _SHA.fullmatch(record[field]):
            errors.append(f"conformance delivery {field} must be a SHA-256 digest")
    for field, fixed in paths.items():
        if record.get(field) != fixed:
            errors.append(f"conformance delivery {field} must name the current fixed path")
    header = record.get("header_path")
    root = paths["oracle_path"].split("/" + commands.CONFORMANCE_SUBDIR + "/")[0]
    if (not isinstance(header, str) or not header.startswith(root + "/workspace/")
            or PurePosixPath(header).as_posix() != header or ".." in PurePosixPath(header).parts
            or PurePosixPath(header).suffix != ".h"):
        errors.append("conformance delivery header_path must be a canonical current workspace header")
    if not isinstance(record.get("oracle"), dict):
        errors.append("conformance delivery oracle must be a mapping")
    return errors


def inspect_delivery(state: dict[str, Any], toolbox: Any) -> tuple[dict, list[str]]:
    """Validate fixed live files against ISA, generator and host approval hashes.

    Metadata is loaded only from the policy-approved current fixed oracle path;
    neither its path fields nor state artifact references are ever followed.
    A missing host record is a failure even when on-disk hashes are consistent.
    """
    try:
        paths = delivery_paths(state)
        snapshot = _snapshot(toolbox, paths)
    except PolicyError:
        raise
    except ValueError as exc:
        return {}, [f"cannot inspect conformance delivery: {exc}"]
    errors = []
    oracle_bytes = snapshot[paths["oracle_path"]]
    source_bytes = snapshot[paths["source_path"]]
    if not oracle_bytes:
        errors.append("conformance delivery oracle record is missing or empty")
    if not source_bytes:
        errors.append("conformance delivery generated C source is missing or empty")
    record: Any = {}
    if oracle_bytes:
        try:
            record = yaml.safe_load(oracle_bytes.decode("utf-8"))
        except (UnicodeError, yaml.YAMLError) as exc:
            errors.append(f"cannot parse conformance delivery record: {exc}")
    errors.extend(_record_problems(record, paths))
    delivery = dict(record) if isinstance(record, dict) else {}
    # Never return attacker-supplied paths for a caller to build or read.
    delivery.update(paths)
    try:
        isa_sha = _digest(_canonical(state.get("isa_spec")))
        generator_sha = _generator_sha()
        if record.get("isa_sha") != isa_sha:
            errors.append("conformance delivery ISA hash does not match current isa_spec")
        if record.get("generator_sha") != generator_sha:
            errors.append("conformance delivery generator hash does not match current generator")
        oracle = record.get("oracle")
        oracle_errors = conformance.validate_oracle(state.get("isa_spec"), oracle)
        errors.extend(oracle_errors)
        if record.get("oracle_sha") != _digest(_canonical(oracle)):
            errors.append("conformance delivery oracle hash mismatch")
        if source_bytes is not None and record.get("source_sha") != _digest(source_bytes):
            errors.append("conformance delivery generated C source hash mismatch")
        if not oracle_errors:
            expected = conformance.render_conformance(state.get("isa_spec"), oracle).encode("utf-8")
            if source_bytes != expected:
                errors.append("conformance delivery generated C differs from host regeneration")
    except (TypeError, ValueError, RecursionError, AttributeError) as exc:
        errors.append(f"cannot validate conformance delivery: {exc}")
    approved = state.get("conformance_delivery")
    if not isinstance(approved, dict):
        errors.append("host state conformance_delivery approval record is missing")
    else:
        if approved.get("header_path") != delivery.get("header_path"):
            errors.append("host state conformance_delivery header_path mismatch")
        for field in _HASH_FIELDS:
            if (not isinstance(approved.get(field), str)
                    or not _SHA.fullmatch(approved[field])
                    or approved[field] != delivery.get(field)):
                errors.append(f"host state conformance_delivery {field} mismatch")
    return delivery, errors


class ConformanceDelivery:
    """One in-process transaction over the fixed oracle/C pair.

    ``write`` and ``current`` return ``(delivery, problems)``. Successful pending
    writes provide local host approval; external inspection still requires the
    parent to persist the accepted record. Diagnosis gating can be unlocked by
    assigning the parent's validated structured diagnosis to ``diagnosis``.
    """

    def __init__(self, state: dict[str, Any], toolbox: Any, header_path: str, *,
                 editable: bool = True, diagnosis_required: bool = False):
        self.state = state
        self.toolbox = toolbox
        self.header_path = header_path  # Parent alone owns syntax/build commands.
        self.editable = editable
        self.diagnosis_required = diagnosis_required
        self.diagnosis: dict[str, Any] | None = None
        self.paths = delivery_paths(state)
        self.snapshots = _snapshot(toolbox, self.paths, write=editable)
        self._pending: dict[str, Any] | None = None
        self._before_write: dict[str, bytes | None] | None = None
        self.committed = False
        self.rolled_back = False

    @property
    def unlocked(self) -> bool:
        return not self.diagnosis_required or self.diagnosis is not None

    @property
    def changed(self) -> bool:
        return _snapshot(self.toolbox, self.paths) != self.snapshots

    def current(self) -> tuple[dict, list[str]]:
        state = dict(self.state)
        if self._pending is not None:
            state["conformance_delivery"] = self._pending
        return inspect_delivery(state, self.toolbox)

    def inspect(self) -> tuple[dict, list[str]]:
        return self.current()

    @property
    def record(self) -> dict:
        """Latest valid pending or approved record; never grants external approval."""
        delivery, errors = self.current()
        return deepcopy(delivery) if not errors else {}

    def write(self, oracle_text: str) -> tuple[dict, list[str]]:
        if self.committed or self.rolled_back:
            return {}, ["conformance delivery transaction is closed"]
        if not self.editable:
            return {}, ["conformance delivery is frozen; oracle edits are not allowed"]
        # A diagnosis-locked session still accepts a *staged* revision: the model
        # may author the corrected oracle first and file the structured diagnosis
        # afterwards. Refusing the write outright burned a round in run
        # 20261005-073745, where the already-correct oracle was rejected for
        # ordering alone. ``commit`` keeps the hard gate.
        if not isinstance(oracle_text, str):
            return {}, ["oracle_text must be a string"]
        oracle, errors = conformance.parse_oracle(oracle_text)
        if not errors:
            errors = conformance.validate_oracle(self.state.get("isa_spec"), oracle)
        if errors:
            return {}, errors
        # Finish validation, rendering and serialization before touching either file.
        try:
            source = conformance.render_conformance(self.state.get("isa_spec"), oracle)
            record = {"schema_version": _SCHEMA_VERSION, **self.paths,
                      "header_path": self.header_path,
                      "isa_sha": _digest(_canonical(self.state.get("isa_spec"))),
                      "oracle_sha": _digest(_canonical(oracle)),
                      "source_sha": _digest(source.encode("utf-8")),
                      "generator_sha": _generator_sha(), "oracle": oracle}
            record_errors = _record_problems(record, self.paths)
            if record_errors:
                return {}, record_errors
            metadata = yaml.safe_dump(record, sort_keys=False)
            before = _snapshot(self.toolbox, self.paths, write=True)
        except (OSError, PolicyError):
            raise
        except (TypeError, ValueError, RecursionError, yaml.YAMLError) as exc:
            return {}, [f"cannot prepare conformance delivery: {exc}"]
        try:
            self.toolbox.write_text(self.paths["oracle_path"], metadata)
            self.toolbox.write_text(self.paths["source_path"], source)
            if _snapshot(self.toolbox, self.paths) != {
                    self.paths["oracle_path"]: metadata.encode("utf-8"),
                    self.paths["source_path"]: source.encode("utf-8")}:
                raise OSError("conformance delivery writes did not persist exact host bytes")
        except Exception:
            _restore(self.toolbox, before)
            raise
        self._pending = deepcopy(record)
        self._before_write = before
        return self.current()

    def revert_pending(self) -> None:
        """Undo the last accepted write, keeping the transaction open.

        The parent may reject a pending pair *after* ``write`` accepted it, for
        example when the host-owned compile preflight finds a redefinition in
        the rendered C. That is not a failed stage: the model has to be able to
        deliver a corrected oracle, so the transaction stays open while the
        previously approved pair (or the empty state) returns to disk.
        """
        if self.committed or self.rolled_back or self._pending is None:
            return
        _restore(self.toolbox, self._before_write or self.snapshots)
        self._pending = None
        self._before_write = None

    def commit(self) -> None:
        if self.rolled_back:
            raise RuntimeError("cannot commit a rolled-back conformance delivery")
        if not self.unlocked:
            raise ValueError("submit a structured repair diagnosis before committing the oracle")
        _, errors = self.current()
        if errors:
            raise ValueError("cannot commit invalid conformance delivery: " + "; ".join(errors))
        self.committed = True

    def rollback(self) -> None:
        if self.committed or self.rolled_back:
            return
        if self.changed:
            _restore(self.toolbox, self.snapshots)
        self._pending = None
        self.rolled_back = True
