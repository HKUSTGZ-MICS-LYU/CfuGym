"""Content-addressed hardware filesets; never accept simulator or historical RTL."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from . import dependencies, stage_services

_SUFFIXES = {".v", ".sv", ".vh", ".svh", ".bin", ".hex", ".mem", ".dat"}
_REFERENCES = re.compile(r'(?:\$readmem[bh]\s*\(\s*|`include\s*)"([^"]+)"')


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze(state: dict[str, Any]) -> dict[str, Any]:
    """Freeze generator output and referenced ROM images under this run only."""
    root = Path(state.get("workdir", ".")).resolve()
    run = str(state.get("run_root") or "")
    if not run.startswith("agents/generated/runs/"):
        raise ValueError("hardware snapshot requires a run-scoped workspace")
    source = (root / run / "workspace/soc").resolve()
    if not source.is_relative_to(root / run) or not (source / "MiCoSoc.v").is_file():
        raise ValueError("hardware snapshot requires generated workspace/soc/MiCoSoc.v")
    contents: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("hardware fileset contains a symlink or escaped path")
        rel = path.relative_to(source).as_posix()
        if "simworkspace" in rel.lower() or path.stem.lower().endswith("sim"):
            raise ValueError("simulator RTL cannot enter the hardware fileset")
        contents[rel] = path.read_bytes()
    for rel, data in list(contents.items()):
        if Path(rel).suffix.lower() not in {".v", ".sv", ".vh", ".svh"}:
            continue
        for name in _REFERENCES.findall(data.decode("utf-8", errors="replace")):
            # Vivado runs with the fileset as cwd. Never silently resolve an old
            # memory image elsewhere in the repository.
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"nonlocal RTL include/memory image: {name}")
            target = name if name in contents else (Path(rel).parent / name).as_posix()
            if target not in contents:
                raise ValueError(f"missing RTL include/memory image: {name}")
    files = [{"path": rel, "sha256": _digest(data)} for rel, data in sorted(contents.items())]
    hw_input = dependencies.input_fingerprints(state)["hw_build"]
    payload = {"schema_version": 1, "top": "MiCoSoc", "files": files,
               "hardware_input_sha256": hw_input}
    digest = _digest(json.dumps(payload, sort_keys=True).encode())
    base = f"{run}/10_validation/rtl_snapshots/{digest}"
    policy = stage_services._policy(state)
    for rel, data in contents.items():
        destination = f"{base}/rtl/{rel}"
        policy.check_write("validation", destination)
        path = root / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("immutable RTL snapshot was modified")
        else:
            with path.open("xb") as handle:
                handle.write(data)
    manifest = f"{base}/fileset.json"
    stage_services._artifacts_for_stage(state, "validation").write_text(
        "validation", f"10_validation/rtl_snapshots/{digest}/fileset.json",
        json.dumps({**payload, "sha256": digest}, indent=2) + "\n", kind="rtl_fileset")
    return {**payload, "sha256": digest, "root": f"{base}/rtl", "manifest": manifest}


def validate(state: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(state.get("rtl_snapshot") or {})
    root = Path(state.get("workdir", ".")).resolve()
    run = str(state.get("run_root") or "")
    directory = (root / str(snapshot.get("root") or "")).resolve()
    allowed = (root / run / "10_validation/rtl_snapshots").resolve()
    if not run or not snapshot or not allowed.is_relative_to(root) or not directory.is_relative_to(allowed):
        raise ValueError("Vivado requires this run's immutable hw_build RTL snapshot")
    if snapshot.get("hardware_input_sha256") != dependencies.input_fingerprints(state)["hw_build"]:
        raise ValueError("RTL snapshot belongs to a different hardware candidate/configuration")
    manifest = root / str(snapshot.get("manifest") or "")
    if not manifest.resolve().is_relative_to(allowed):
        raise ValueError("RTL fileset manifest escapes snapshot root")
    recorded = json.loads(manifest.read_text())
    payload = {key: recorded[key] for key in ("schema_version", "top", "files", "hardware_input_sha256")}
    digest = _digest(json.dumps(payload, sort_keys=True).encode())
    if digest != snapshot.get("sha256") or recorded.get("sha256") != digest or recorded["files"] != snapshot.get("files"):
        raise ValueError("RTL fileset manifest hash mismatch")
    for entry in recorded["files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or _digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"RTL snapshot hash mismatch: {entry['path']}")
    if not any(item["path"] == "MiCoSoc.v" for item in recorded["files"]):
        raise ValueError("hardware snapshot lacks MiCoSoc.v")
    return snapshot
