"""Run-scoped artifact storage and manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any

import yaml

from .policy import AccessPolicy, PolicyError, normalize_repo_path


ARTIFACT_SCHEMA_VERSION = "1"


@dataclass
class RunArtifacts:
    workdir: Path
    run_id: str
    root_rel: str
    policy: AccessPolicy
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        workdir: Path,
        policy: AccessPolicy,
        *,
        out_dir: str = "agents/generated",
        run_id: str = "",
    ) -> "RunArtifacts":
        root = workdir.resolve()
        out_rel = normalize_repo_path(root, out_dir)
        if not out_rel.startswith("agents/generated"):
            raise PolicyError("artifact output must be under agents/generated")
        safe_run_id = run_id or new_run_id()
        if not _safe_run_id(safe_run_id):
            raise PolicyError(f"invalid run id: {safe_run_id!r}")
        root_rel = f"{out_rel}/runs/{safe_run_id}"
        manifest_path = root / root_rel / "run.yaml"
        manifest = None
        if manifest_path.exists():
            try:
                loaded = yaml.safe_load(manifest_path.read_text(errors="replace"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except (OSError, yaml.YAMLError):
                manifest = None
        if manifest is None:
            manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": safe_run_id,
                "status": "initialized",
                "policy_version": policy.version,
                "created_at": _now(),
                "artifacts": [],
            }
            manifest.setdefault("artifacts", [])
        instance = cls(
            workdir=root,
            run_id=safe_run_id,
            root_rel=root_rel,
            policy=policy,
            manifest=manifest,
        )
        instance._write_manifest("load_context")
        return instance

    @property
    def root(self) -> Path:
        return self.workdir / self.root_rel

    def path(self, relative: str) -> Path:
        return self.workdir / normalize_repo_path(self.workdir, f"{self.root_rel}/{relative}")

    def write_text(
        self,
        stage: str,
        relative: str,
        text: str,
        *,
        kind: str = "text",
        status: str = "produced",
        inputs: list[str] | None = None,
        visible_files: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(
            stage,
            relative,
            text.encode("utf-8"),
            kind=kind,
            status=status,
            inputs=inputs or [],
            visible_files=visible_files or [],
            metadata=metadata or {},
        )

    def write_yaml(
        self,
        stage: str,
        relative: str,
        value: dict[str, Any],
        *,
        kind: str = "yaml",
        status: str = "produced",
        inputs: list[str] | None = None,
        visible_files: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = yaml.safe_dump(value, sort_keys=False, allow_unicode=False)
        return self.write_text(
            stage,
            relative,
            text,
            kind=kind,
            status=status,
            inputs=inputs,
            visible_files=visible_files,
            metadata=metadata,
        )

    def write_json(
        self,
        stage: str,
        relative: str,
        value: Any,
        *,
        kind: str = "json",
        status: str = "produced",
        inputs: list[str] | None = None,
        visible_files: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = json.dumps(value, indent=2, sort_keys=False, default=str) + "\n"
        return self.write_text(
            stage,
            relative,
            text,
            kind=kind,
            status=status,
            inputs=inputs,
            visible_files=visible_files,
            metadata=metadata,
        )

    def set_status(self, stage: str, status: str, *, error: str = "") -> None:
        self.manifest["status"] = status
        self.manifest["last_stage"] = stage
        if error:
            self.manifest.setdefault("errors", []).append(error)
        self._write_manifest(stage)

    def _write(
        self,
        stage: str,
        relative: str,
        data: bytes,
        *,
        kind: str,
        status: str,
        inputs: list[str],
        visible_files: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        rel_path = normalize_repo_path(self.workdir, f"{self.root_rel}/{relative}")
        self.policy.check_write(stage, rel_path)
        path = self.workdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, data)
        ref: dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "stage": stage,
            "path": rel_path,
            "kind": kind,
            "status": status,
            "inputs": inputs,
            "visible_files": visible_files,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "created_at": _now(),
        }
        ref.update(metadata)
        self.manifest.setdefault("artifacts", []).append(ref)
        self.manifest["last_stage"] = stage
        self._write_manifest(stage)
        return ref

    def _write_manifest(self, stage: str) -> None:
        rel_path = f"{self.root_rel}/run.yaml"
        self.policy.check_write(stage, rel_path)
        path = self.workdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.manifest, sort_keys=False, allow_unicode=False).encode("utf-8")
        _atomic_write(path, text)


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _safe_run_id(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "-_" for char in value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
