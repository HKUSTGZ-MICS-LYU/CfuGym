"""Cache of the embench reference/profile measurement, keyed by its inputs.

The ``reference`` and ``profile`` configurations are plain scalar builds that
are simulated on the *seeded* scaffold SoC: profiling runs before any agent
hardware edit, so both are independent of the agent's CFU design. Their cycle
counts come from a cycle-accurate simulator and are therefore deterministic for
a fixed benchmark, build command, CPU configuration, scaffold and toolchain.

This module computes a key over exactly those inputs and (de)serializes the
cached measurement so a later run with the same inputs can skip both builds and
both simulations. It is deliberately conservative: anything that can change the
generated RTL or the benchmark code is part of the key.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


CACHE_SUBDIR = "agents/generated/cache/embench_profile"
SCHEMA_VERSION = "1"
# Bump when the key composition changes so old entries cannot be reused.
KEY_VERSION = "8"  # versioned workload patches and protected semantic harness inputs
# The reference build's disassembly, stored beside the entry under the same key:
# a reused measurement has no build outputs of its own, and the analysis reads
# this file as the ASM candidate for a measured loop.
DISASSEMBLY_NAME = "embench_main.asm"
# A cache key is a hex digest; an artifact name is a plain file name.
_NAME = re.compile(r"[0-9a-f]{16,128}")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")
# Placeholder substituted for the run root before hashing, so the same
# benchmark measured in two different runs shares one cache entry.
RUN_PLACEHOLDER = "<run>"
# Build outputs, never inputs: excluded from a source-tree hash.
_SKIP_SUFFIXES = {
    ".elf",
    ".o",
    ".a",
    ".so",
    ".asm",
    ".map",
    ".vcd",
    ".fst",
    ".bin",
    ".hex",
}
_SKIP_DIRS = {"build", "target", "__pycache__", ".git"}
# Repo trees whose content changes the generated SoC that runs the profile.
SOC_SOURCE_ROOTS = (
    "src/main/scala/vexiiriscv/soc/mico",
    "src/main/scala/vexiiriscv/soc/cfu",
    "src/main/scala/vexiiriscv/execute/cfu",
)


def _hash_file(path: Path, digest: "hashlib._Hash") -> None:
    digest.update(path.name.encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)


def tree_hash(root: Path) -> str:
    """Content hash of a file or directory, ignoring build outputs."""
    digest = hashlib.sha256()
    if root.is_file():
        _hash_file(root, digest)
        return digest.hexdigest()
    if not root.is_dir():
        return ""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        _hash_file(path, digest)
    return digest.hexdigest()


def _version(argv: list[str]) -> str:
    executable = shutil.which(argv[0])
    if not executable:
        return "missing"
    try:
        proc = subprocess.run(
            [executable, *argv[1:]], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    return lines[0][:200] if lines else "unknown"


def toolchain_fingerprint() -> dict[str, str]:
    """Versions of the tools whose codegen the cycle counts depend on."""
    return {
        "gcc": _version(["riscv64-unknown-elf-gcc", "--version"]),
        "verilator": _version(["verilator", "--version"]),
    }


def repo_fingerprint(workdir: Path) -> dict[str, str]:
    """Commit and submodule state, which fix the simulator and toolchain build."""
    executable = shutil.which("git")

    def git(*args: str) -> str:
        if not executable:
            return "missing"
        try:
            proc = subprocess.run(
                [executable, *args],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        if proc.returncode != 0:
            return "unknown"
        return (proc.stdout or "").strip()[:1000]

    return {"head": git("rev-parse", "HEAD"), "submodules": git("submodule", "status")}


def _normalize_run_root(value: Any, run_root: str) -> Any:
    """Replace the run-root path inside a value with a stable placeholder.

    The build argv names the run's own workspace, so hashing it verbatim would
    make every run's key unique and defeat cross-run reuse. The workspace layout
    is identical across runs, so the path is normalized before hashing.
    """
    if not run_root:
        return value
    if isinstance(value, str):
        return value.replace(run_root, RUN_PLACEHOLDER)
    if isinstance(value, list):
        return [_normalize_run_root(item, run_root) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_run_root(item, run_root) for key, item in value.items()}
    return value


def build_key(
    workdir: Path,
    *,
    run_root: str,
    benchmark: str,
    configs: list[str],
    build_commands: dict[str, list[str]],
    cpu_flags: list[str],
    observation_plan: Any,
    measurement_protocol: str | None = None,
) -> dict[str, Any]:
    """Return ``{"key": <sha256>, "inputs": <auditable breakdown>}``.

    Existing callers use the current Embench protocol; callers hashing older
    evidence can supply its protocol explicitly without changing the API.
    """
    if measurement_protocol is None:
        from .benchmark import embench

        measurement_protocol = embench.MEASUREMENT_PROTOCOL
    inputs: dict[str, Any] = {
        "key_version": KEY_VERSION,
        "measurement_protocol": measurement_protocol,
        "benchmark": benchmark,
        "configs": list(configs),
        # Normalized: the run root is a placeholder, so two runs of the same
        # benchmark+config share a key.
        "build_commands": _normalize_run_root(
            {config: list(build_commands[config]) for config in configs}, run_root
        ),
        # CPU/SoC knobs: the profile sim uses these even when the datapath flags
        # are still empty because the ISA contract has not been drafted yet.
        "cpu_flags": list(cpu_flags),
        "observation_plan": _normalize_run_root(observation_plan, run_root) if observation_plan else "default",
        "instrumentation_hash": {
            name: tree_hash(workdir / name) for name in (
                "agents/cfu_design_agent/benchmark/generic.py",
                "agents/cfu_design_agent/benchmark/embench.py",
                "agents/cfu_design_agent/benchmark/embench_patchers.py",
                "agents/cfu_design_agent/benchmark/patches",
                "agents/cfu_design_agent/benchmark/embench_integrity.py",
                "agents/cfu_design_agent/benchmark/cfu_execution.py",
                "agents/cfu_design_agent/benchmark/embench_instrument.py",
                "agents/cfu_design_agent/benchmark/embench_loops.py",
                "sw/embench/boardsupport.c",
                "sw/embench/embench_obs.h",
            )
        },
        "sources_hash": tree_hash(
            workdir / run_root / "workspace" / f"embench_{benchmark}" / "reference"
        ),
        # The seeded design overlay: what the profile SoC actually compiles.
        "design_hash": tree_hash(workdir / run_root / "workspace" / "design"),
        "soc_hash": {
            name: tree_hash(workdir / name) for name in SOC_SOURCE_ROOTS
        },
        "toolchain": toolchain_fingerprint(),
        "repo": repo_fingerprint(workdir),
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"key": hashlib.sha256(canonical.encode()).hexdigest(), "inputs": inputs}


def cache_relpath(key: str) -> str:
    return f"{CACHE_SUBDIR}/{key}.json"


def artifact_relpath(key: str, name: str) -> str:
    """Cache-side copy of one reference build artifact, under the same key.

    The key already hashes the sources, commands, flags and toolchain, so an
    artifact stored with it is exactly as reusable as the measurement itself.
    """
    if not _NAME.fullmatch(key or "") or not _ARTIFACT_NAME.fullmatch(name or ""):
        raise ValueError("unsafe cache key or artifact name")
    return f"{CACHE_SUBDIR}/{key}.{name}"


def parse(text: str, key: str) -> dict[str, Any] | None:
    """Parse a cached entry, or return None when it is unusable."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != SCHEMA_VERSION or data.get("key") != key:
        return None
    if not data.get("verified") or not data.get("profiles"):
        return None
    return data


def serialize(
    key: str,
    *,
    benchmark: str,
    configs: list[str],
    profiles: dict[str, dict],
    observation: dict,
    inputs: dict[str, Any],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark": benchmark,
        "configs": list(configs),
        "verified": True,
        "profiles": profiles,
        "observation": observation or {},
        "inputs": inputs,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
