"""Transactional state shared by bounded software and hardware coding tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from .tools import RepoToolbox, redirect_target


@dataclass
class CodingSession:
    """Snapshot approved targets and commit or restore one model coding attempt.

    This is an in-process transaction. It prevents provider/tool failures and
    rejected implementation attempts from contaminating the next graph rung.
    RunArtifacts remain the durable audit log; a future slice may add crash
    recovery from the snapshot manifest.
    """

    toolbox: RepoToolbox
    allowed_targets: frozenset[str]
    failure_index: dict[str, Any] | None = None
    diagnosis_required: bool = False
    snapshots: dict[str, bytes | None] = field(init=False)
    touched_paths: list[str] = field(default_factory=list)
    diagnosis: dict[str, Any] | None = None
    edits: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False

    def __post_init__(self) -> None:
        self.snapshots = {}
        for path in sorted(self.allowed_targets):
            full = self.toolbox.workdir / path
            self.snapshots[path] = full.read_bytes() if full.exists() else None

    @property
    def unlocked(self) -> bool:
        return not self.diagnosis_required or self.diagnosis is not None

    def resolve_target(self, path: str) -> str | None:
        """Map a model-named path onto an approved target, or None if no mapping.

        ``read_file`` reads the run overlay through the tracked repo path and
        the final EDIT/FILE/PATCH delivery maps tracked paths onto the overlay;
        the edit tools must agree, or the model reads one file and is denied
        when editing exactly the bytes it just read. The write boundary does not
        change: only the graph-approved targets are ever written.
        """
        if path in self.allowed_targets:
            return path
        normalize = getattr(self.toolbox, "normalize", None)
        run_root = getattr(self.toolbox, "run_root", "") or ""
        if normalize is None or not run_root:
            return None
        try:
            normalized = normalize(path)
        except Exception:
            return None
        redirected = redirect_target(
            normalized, run_root, str(getattr(self.toolbox, "embench_benchmark", "") or "")
        )
        return redirected if redirected in self.allowed_targets else None

    def candidate_hash(self) -> str:
        payload: list[tuple[str, str]] = []
        for path in sorted(self.allowed_targets):
            text = self.toolbox.read_text(path, limit=2_000_000)
            payload.append((path, hashlib.sha256(text.encode("utf-8")).hexdigest()))
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def submit_diagnosis(self, value: dict[str, Any]) -> str:
        observed = value.get("observed_errors")
        root_cause = str(value.get("root_cause", "") or "").strip()
        mapping = value.get("repair_mapping")
        risks = value.get("regression_risks")
        raw_ids = value.get("failure_ids")
        ids = (
            [item.strip() for item in raw_ids if isinstance(item, str) and item.strip()]
            if isinstance(raw_ids, list)
            else []
        )
        failures = (self.failure_index or {}).get("failures", {})
        known = {
            ident for ident, item in failures.items()
            if not isinstance(item, dict) or item.get("status", "current") == "current"
        }
        resolved = set(failures) - known
        problems: list[str] = []
        if (
            not isinstance(observed, list)
            or not observed
            or any(not isinstance(item, str) or not item.strip() for item in observed)
        ):
            problems.append("observed_errors must be a non-empty array of non-empty strings")
        if not root_cause:
            problems.append("root_cause must be non-empty")
        if (
            not isinstance(mapping, list)
            or not mapping
            or any(not isinstance(item, str) or not item.strip() for item in mapping)
        ):
            problems.append("repair_mapping must be a non-empty array of non-empty strings")
        if (
            not isinstance(risks, list)
            or not risks
            or any(not isinstance(item, str) or not item.strip() for item in risks)
        ):
            problems.append("regression_risks must be a non-empty array of non-empty strings (use 'none' explicitly)")
        if known and (not ids or any(ident not in known for ident in ids)):
            message = "failure_ids must cite only current failure IDs"
            if resolved:
                message += (
                    "; these are already resolved and must not be re-fixed: "
                    + ", ".join(sorted(resolved))
                )
            problems.append(message)
        if problems:
            return "diagnosis rejected: " + "; ".join(problems)
        self.diagnosis = {
            "failure_ids": ids,
            "observed_errors": observed,
            "root_cause": root_cause,
            "repair_mapping": mapping,
            "regression_risks": risks,
        }
        return "diagnosis accepted; edit capability unlocked"

    def apply_edits(self, edits: list[tuple[str, list[tuple[str, str]]]]) -> dict[str, Any]:
        if not self.unlocked:
            return {
                "returncode": 1,
                "stderr_tail": "submit a structured repair diagnosis before editing",
                "parsed": {"touched_paths": []},
            }
        for path, _pairs in edits:
            if path not in self.allowed_targets:
                return {
                    "returncode": 1,
                    "stderr_tail": "path is not an approved coding-session target",
                    "parsed": {"touched_paths": []},
                }
        before = self.candidate_hash()
        result = dict(self.toolbox.apply_edit_blocks(edits))
        touched = list((result.get("parsed") or {}).get("touched_paths") or [])
        if result.get("returncode") == 0:
            for path in touched:
                if path not in self.touched_paths:
                    self.touched_paths.append(path)
            self.edits.append({
                "before_hash": before,
                "after_hash": self.candidate_hash(),
                "paths": touched,
            })
        return result

    def record_check(self, result: dict[str, Any]) -> None:
        self.checks.append({
            "candidate_hash": self.candidate_hash(),
            "returncode": result.get("returncode"),
            "failure_id": (result.get("failure_observation") or {}).get("id", ""),
            "full_log_path": result.get("full_log_path", result.get("log_path", "")),
        })

    def current_check(self) -> dict[str, Any] | None:
        current = self.candidate_hash()
        for item in reversed(self.checks):
            if item.get("candidate_hash") == current:
                return item
        return None

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        if self.committed or self.rolled_back:
            return
        restored: list[str] = []
        try:
            for path, data in self.snapshots.items():
                full = self.toolbox.workdir / path
                if data is None:
                    if full.exists():
                        full.unlink()
                else:
                    # Rollback is a harness-owned restoration operation, not a
                    # new model write; restore exact bytes including line endings.
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_bytes(data)
                restored.append(path)
            for path, data in self.snapshots.items():
                full = self.toolbox.workdir / path
                if data is None and full.exists():
                    raise OSError(f"rollback failed to remove {path}")
                if data is not None and full.read_bytes() != data:
                    raise OSError(f"rollback verification failed for {path}")
        except OSError as exc:
            raise OSError(
                f"coding-session rollback failed after restoring {restored}: {exc}"
            ) from exc
        self.rolled_back = True
        self.touched_paths.clear()
