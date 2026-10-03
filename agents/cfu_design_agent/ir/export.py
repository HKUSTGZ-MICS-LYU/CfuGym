"""P0 Embench LLVM IR export; no repository writes, static evidence only.

The adapter's reference make recipe is dry-run to recover the *actual* GCC
compile flags. Each C translation unit is then compiled to LLVM text on stdout
by a trusted local Clang. The caller owns writing the returned artifact texts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

from ..benchmark import embench

SCHEMA_VERSION = "1"
TARGET = "riscv32-unknown-elf"
OPT_LEVEL = "O1"
TIMEOUT_SECONDS = 30
MAX_IR_BYTES = 4_000_000
MAX_COMMAND_BYTES = 2_000_000
MAX_STDERR_BYTES = 8192
_RUN_ROOT = re.compile(r"agents/generated(?:/[A-Za-z0-9_-]+)?/runs/[A-Za-z0-9_-]+\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_TRUSTED_COMPILER_ROOTS = ("/usr/bin", "/usr/local/bin", "/usr/lib", "/opt/llvm", "/usr/local/swift")
# Never invoke a GCC discovered through PATH (or selected through an environment variable).
# Only these explicitly approved toolchain installations may provide headers.
_APPROVED_RISCV_GCC = ("/home/jzj/riscv/bin/riscv64-unknown-elf-gcc",)
_APPROX_ARCH = "rv32imc"
_APPROX_CONFIDENCE = "approximate_static_ir"
_APPROX_CAVEAT = (
    "Approximate Clang IR uses rv32imc and RISC-V GCC headers; it is not faithful "
    "GCC machine code or evidence of instruction counts, cycles, or loop mapping."
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _discover_clang() -> str | None:
    """Ignore executables from workspaces or writable PATH entries."""
    for name in ("clang", "clang-18", "clang-17", "clang-16", "clang-15", "clang-14"):
        candidate = shutil.which(name)
        if not candidate:
            continue
        real = Path(candidate).resolve()
        if real.is_file() and os.access(real, os.X_OK) and any(
            real.is_relative_to(root) for root in _TRUSTED_COMPILER_ROOTS
        ):
            return str(real)
    return None


def _execute(argv: list[str], *, cwd: Path, limit: int) -> dict[str, Any]:
    """Capture bounded output through OS temporary files outside the repo."""
    result: dict[str, Any] = {"argv": argv, "cwd": str(cwd), "returncode": None, "stderr": ""}
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                completed = subprocess.run(
                    argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout,
                    stderr=stderr, timeout=TIMEOUT_SECONDS, check=False,
                    env={**os.environ, "MAKEFLAGS": "", "GNUMAKEFLAGS": ""},
                )
                result["returncode"] = completed.returncode
            except subprocess.TimeoutExpired:
                result["stderr"] = f"timed out after {TIMEOUT_SECONDS}s"
            stdout.seek(0)
            data = stdout.read(limit + 1)
            stderr.seek(0)
            diagnostic = stderr.read(MAX_STDERR_BYTES + 1)
            if diagnostic:
                result["stderr"] += diagnostic[:MAX_STDERR_BYTES].decode("utf-8", "replace")
                if len(diagnostic) > MAX_STDERR_BYTES:
                    result["stderr"] += " [stderr truncated]"
            if len(data) > limit:
                result["stderr"] += f" [stdout exceeded {limit} bytes]"
                result["overflow"] = True
            else:
                result["stdout"] = data
    except (OSError, ValueError) as exc:
        result["stderr"] = f"{type(exc).__name__}: {exc}"[:MAX_STDERR_BYTES]
    return result


def _approx_headers(sw: Path, commands: list[dict[str, Any]]) -> tuple[str, str]:
    """Query only the approved installed GCC; reject escaped or noncanonical headers."""
    gcc = Path(_APPROVED_RISCV_GCC[0])
    prefix = gcc.parent.parent
    if (not gcc.is_file() or gcc.is_symlink() or gcc.resolve() != gcc
            or not os.access(gcc, os.X_OK)):
        raise ValueError("approved RISC-V GCC executable not found or unsafe")
    discovered: list[Path] = []
    for option in ("-print-sysroot", "-print-file-name=include"):
        result = _execute([str(gcc), option], cwd=sw, limit=4096)
        commands.append({k: v for k, v in result.items() if k != "stdout"})
        if result["returncode"] != 0 or "stdout" not in result or result.get("overflow"):
            raise ValueError(f"approved RISC-V GCC {option} probe failed")
        try:
            output = result["stdout"].decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"approved RISC-V GCC {option} returned invalid text") from exc
        if "\n" in output or "\r" in output or not output:
            raise ValueError(f"approved RISC-V GCC {option} returned invalid path")
        path = Path(output)
        if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
                or path.resolve() != path or not path.is_relative_to(prefix)):
            raise ValueError(f"approved RISC-V GCC {option} returned unsafe directory")
        discovered.append(path)
    sysroot, internal = discovered
    headers = sysroot / "include"
    if (sysroot != prefix / "riscv64-unknown-elf"
            or not headers.is_dir() or headers.is_symlink()
            or headers.resolve() != headers
            or not re.fullmatch(
                re.escape(str(prefix / "lib/gcc/riscv64-unknown-elf"))
                + r"/[0-9]+(?:\.[0-9]+){1,3}/include", str(internal)
            )):
        raise ValueError("approved RISC-V GCC sysroot/internal headers outside expected install")
    return str(headers), str(internal)


def _failed(summary: dict[str, Any], reason: str, *, mode: str) -> tuple[dict[str, Any], dict[str, str]]:
    summary["status"] = "failed" if mode == "require" else "unavailable"
    summary["reason"] = reason
    return summary, {}


def _recipe_commands(text: str, sources: dict[str, Path], sw: Path) -> dict[str, list[str]]:
    """Select compiler invocations whose -c source is exactly a reference TU."""
    found: dict[str, list[str]] = {}
    source_by_path = {path.resolve(): key for key, path in sources.items()}
    for line in text.splitlines():
        try:
            argv = shlex.split(line.strip())
        except ValueError:
            continue
        if not argv or Path(argv[0]).name != "riscv64-unknown-elf-gcc" or "-c" not in argv:
            continue
        try:
            # The Makefile emits '-c -o object.c.o source.c', not '-c source.c'.
            src = (sw / argv[argv.index("-o") + 2]).resolve()
        except (ValueError, IndexError, OSError):
            continue
        key = source_by_path.get(src)
        if key is not None:
            if key in found:
                raise ValueError(f"duplicate compile commands for {key}")
            found[key] = argv
    return found


def _clang_argv(gcc: list[str], source: Path, sw: Path, clang: str) -> list[str]:
    """Translate only known compile options; never carry arbitrary GCC switches."""
    flags: list[str] = []
    observed: dict[str, str] = {}
    i = 1
    while i < len(gcc):
        value = gcc[i]
        if value in ("-c", "-o"):
            if value == "-o":
                i += 2
            else:
                i += 1
            continue
        if value in ("-I", "-isystem", "-include", "-D", "-U"):
            if i + 1 >= len(gcc):
                raise ValueError(f"option {value} missing operand")
            operand = gcc[i + 1]
            if value in ("-I", "-isystem", "-include"):
                header = (sw / operand).resolve()
                if not header.is_relative_to(sw.parent.resolve()):
                    raise ValueError("compile include escapes repository")
                flags.extend((value, operand))
            elif value in ("-D", "-U"):
                flags.extend((value, operand))
            i += 2
            continue
        if (sw / value).resolve() == source.resolve() and value.endswith(".c"):
            i += 1
            continue
        if value.startswith("-march=") or value.startswith("-mabi="):
            flag, setting = value.split("=", 1)
            if flag in observed:
                raise ValueError(f"duplicate compiler option: {flag}")
            observed[flag] = setting
        elif value.startswith(("-I", "-isystem", "-include")):
            if value.startswith("-I") and value != "-I":
                header = (sw / value[2:]).resolve()
                if not header.is_relative_to(sw.parent.resolve()):
                    raise ValueError("compile include escapes repository")
                flags.append(value)
            else:
                raise ValueError(f"unsupported compiler option: {value}")
        elif value.startswith(("-D", "-U", "-std=", "-Wno-")) or value in (
            "-ffunction-sections", "-fdata-sections", "-fno-common", "-fno-inline"
        ):
            flags.append(value)
        elif re.fullmatch(r"-O(?:[0-3sSz]|fast)", value) or value == "-mcmodel=medany":
            pass  # -O1 is deliberate; Clang target defaults to medany for analysis.
        else:
            raise ValueError(f"unsupported compiler option: {value}")
        i += 1
    if observed != {"-march": embench.DEFAULT_MARCH, "-mabi": "ilp32"}:
        raise ValueError(f"unexpected GCC ISA/ABI: {observed}")
    return [clang, "-target", TARGET, "-march=" + observed["-march"],
            "-mabi=" + observed["-mabi"], "-O1", "-g", "-S", "-emit-llvm",
            *flags, "-o", "-", str(source)]


def export_embench_ir(
    workdir: Path, run_root: str, benchmark: str, mode: str = "auto"
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``(audit_summary, {artifact_relative_path: LLVM_text})``.

    ``off`` does not inspect inputs or invoke tools. ``require`` reports failed
    for missing or partial IR; no exceptions for tool failure. ``approx`` uses
    rv32imc plus approved GCC system headers; it never claims the requested GCC
    ISA or faithful code.

    ``auto`` prefers the exact reference target, but the requested ISA
    (``rv32imc_zicsr_zifencei``) is GCC-only, so the installed clang often
    rejects it. In that case ``auto`` retries the approximate export and returns
    it **labelled** (``status: approximate``, ``target_match: false``, the strict
    reason under ``fallback``) instead of handing the analysis no evidence at
    all. A precondition failure (unsafe input, missing clang) is never retried,
    because changing the target ISA cannot fix it. Files are never persisted
    here; the caller writes via RunArtifacts.
    """
    if mode not in ("off", "auto", "require", "approx"):
        raise ValueError("mode must be off, auto, require, or approx")
    summary, files = _export_target_ir(workdir, run_root, benchmark, mode)
    if mode != "auto" or files:
        return summary, files
    if not any(
        isinstance(unit, dict) and unit.get("status") == "failed"
        for unit in summary.get("units", []) or []
    ):
        return summary, files
    approximate, approx_files = _export_target_ir(workdir, run_root, benchmark, "approx")
    if not approx_files:
        return summary, files
    approximate["fallback"] = {
        "from": "auto",
        "strict_status": str(summary.get("status", "")),
        "strict_reason": str(summary.get("reason", ""))[:512],
    }
    approximate["reason"] = (
        "exact target export unavailable ("
        + (str(summary.get("reason", "")) or str(summary.get("status", "")))[:256]
        + "); reported the approximate rv32imc export instead"
    )
    return approximate, approx_files


def _export_target_ir(
    workdir: Path, run_root: str, benchmark: str, mode: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """One export attempt for exactly one mode; the caller owns any fallback."""
    if mode not in ("off", "auto", "require", "approx"):
        raise ValueError("mode must be off, auto, require, or approx")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": "off" if mode == "off" else "unavailable",
        "reason": "IR export disabled" if mode == "off" else "", "mode": mode,
        "toolchain": {"target": TARGET, "opt_level": OPT_LEVEL, "clang": None,
                      "requested_arch": embench.DEFAULT_MARCH,
                      "actual_arch": embench.DEFAULT_MARCH if mode != "approx" else _APPROX_ARCH,
                      "target_match": mode != "approx"},
        "inputs": {}, "commands": [], "units": [],
        "limitations": "Clang IR is static evidence; GCC machine instructions, cycles and loop mapping are not inferred.",
    }
    if mode == "off":
        return summary, {}
    if mode == "approx":
        summary.update(
            requested_arch=embench.DEFAULT_MARCH, actual_arch=_APPROX_ARCH,
            target_match=False, confidence=_APPROX_CONFIDENCE,
            limitations=_APPROX_CAVEAT,
        )
        summary["toolchain"].update(
            requested_arch=embench.DEFAULT_MARCH, actual_arch=_APPROX_ARCH,
            target_match=False,
        )
    root = Path(workdir).resolve()
    if not isinstance(run_root, str) or not isinstance(benchmark, str) or not _RUN_ROOT.fullmatch(run_root) or not _NAME.fullmatch(benchmark):
        return _failed(summary, "invalid run root or benchmark name", mode=mode)
    cfg = root / embench.config_workspace(run_root, benchmark, "reference")
    if not cfg.resolve().is_relative_to((root / run_root).resolve()) or cfg.is_symlink() or (cfg / "src").is_symlink():
        return _failed(summary, "reference workspace escapes run root or is symlinked", mode=mode)
    raw_sources = embench.config_sources(root, run_root, benchmark, "reference")
    sources: dict[str, Path] = {}
    for relative in raw_sources:
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(cfg.resolve() / "src"):
            return _failed(summary, "invalid or symlinked reference source", mode=mode)
        sources[path.name] = path
    if not sources:
        return _failed(summary, "no uninstrumented reference C sources", mode=mode)
    try:
        summary["inputs"]["sources"] = {
            key: {"path": str(path.relative_to(root)), "sha256": _digest(path.read_bytes())}
            for key, path in sorted(sources.items())
        }
    except OSError as exc:
        return _failed(summary, f"cannot read reference source: {exc}", mode=mode)
    clang = _discover_clang()
    if clang is None:
        return _failed(summary, "trusted local clang executable not found", mode=mode)
    sw = root / "sw"
    if not sw.is_dir():
        return _failed(summary, "sw build directory missing", mode=mode)
    # A rogue build directory could change Makefile dry-run commands. This
    # exporter trusts the checkout Makefiles, not arbitrary symlinked trees.
    if sw.is_symlink() or (sw / "Makefile").is_symlink() or not (sw / "Makefile").is_file():
        return _failed(summary, "sw build directory or Makefile is unsafe/missing", mode=mode)
    # GNU make -n does execute $(shell ...) expansions. Reject unsafe or
    # unexpected checkout recipes instead of executing arbitrary make files.
    for mk in (sw / "Makefile", sw / "MiCo-Lib/targets/common.mk", sw / "MiCo-Lib/targets/vexii_soc.mk"):
        try:
            if mk.is_symlink() or not mk.is_file():
                return _failed(summary, "reference Makefile missing or symlinked", mode=mode)
            contents = mk.read_text(encoding="utf-8")
            if "$(shell" in contents and mk.name != "vexii_soc.mk":
                return _failed(summary, "untrusted make shell expansion", mode=mode)
            if mk.name == "vexii_soc.mk" and re.findall(r"\$\(shell[^\n]*", contents) != [
                "$(shell echo $(MARCH) | sed 's/_.*$$//')"
            ]:
                return _failed(summary, "unexpected make shell expansion", mode=mode)
        except OSError as exc:
            return _failed(summary, f"cannot read reference Makefile: {exc}", mode=mode)
    version = _execute([clang, "--version"], cwd=sw, limit=4096)
    summary["commands"].append({k: v for k, v in version.items() if k != "stdout"})
    if version["returncode"] != 0 or "stdout" not in version:
        return _failed(summary, "clang version probe failed", mode=mode)
    summary["toolchain"]["clang"] = version["stdout"].decode("utf-8", "replace").splitlines()[0]
    make = shutil.which("make")
    if not make or not Path(make).resolve().is_relative_to("/usr/bin"):
        return _failed(summary, "trusted local make executable not found", mode=mode)
    approx_headers: tuple[str, str] | None = None
    if mode == "approx":
        try:
            approx_headers = _approx_headers(sw, summary["commands"])
        except ValueError as exc:
            return _failed(summary, str(exc), mode=mode)
        summary["toolchain"]["gcc_headers"] = list(approx_headers)
    build = embench.build_command(run_root, benchmark, "reference")
    files: dict[str, str] = {}
    errors: list[str] = []
    for name, source in sorted(sources.items()):
        entry: dict[str, Any] = {"source": str(source.relative_to(root)), "status": "failed"}
        summary["units"].append(entry)
        # Restrict make -n to the exact source object, avoiding huge dry-run
        # output for all runtime objects and any recursive compile targets.
        source_rel_sw = os.path.relpath(source, sw)
        build_rel_sw = next(arg[6:] for arg in build if arg.startswith("BUILD="))
        target_obj = f"{build_rel_sw}/{source_rel_sw[:-2]}.o"
        dry_argv = [str(Path(make).resolve()), *build[1:-1], "-n", "-B", target_obj]
        dry = _execute(dry_argv, cwd=root, limit=MAX_COMMAND_BYTES)
        summary["commands"].append({k: v for k, v in dry.items() if k != "stdout"})
        if dry["returncode"] != 0 or "stdout" not in dry:
            entry["reason"] = "reference make dry-run failed"
            errors.append(f"{name}: {entry['reason']}")
            continue
        try:
            gcc = _recipe_commands(dry["stdout"].decode("utf-8", "replace"), {name: source}, sw).get(name)
        except ValueError as exc:
            entry["reason"] = str(exc)
            errors.append(f"{name}: {exc}")
            continue
        if gcc is None:
            entry["reason"] = "no exact reference GCC compile command in make dry-run"
            errors.append(f"{name}: {entry['reason']}")
            continue
        entry["gcc_argv"] = gcc
        try:
            argv = _clang_argv(gcc, source, sw, clang)
            if mode == "approx":
                assert approx_headers is not None
                arch_index = argv.index("-march=" + embench.DEFAULT_MARCH)
                argv[arch_index] = "-march=" + _APPROX_ARCH
                argv[arch_index + 1:arch_index + 1] = [
                    "-isystem", approx_headers[0], "-isystem", approx_headers[1],
                ]
        except ValueError as exc:
            entry["reason"] = str(exc)
            errors.append(f"{name}: {exc}")
            continue
        if mode == "approx":
            entry.update(requested_arch=embench.DEFAULT_MARCH, actual_arch=_APPROX_ARCH,
                         target_match=False, confidence=_APPROX_CONFIDENCE,
                         limitations=_APPROX_CAVEAT)
        entry["compile_flags_sha256"] = _digest(json.dumps(argv[1:-1], separators=(",", ":")).encode())
        result = _execute(argv, cwd=sw, limit=MAX_IR_BYTES)
        summary["commands"].append({k: v for k, v in result.items() if k != "stdout"})
        entry["returncode"] = result["returncode"]
        entry["stderr"] = result["stderr"]
        data = result.get("stdout", b"")
        if result["returncode"] != 0 or not data or result.get("overflow"):
            diagnostic = result["stderr"].strip().splitlines()
            entry["reason"] = (
                "clang compilation failed or output exceeded limit"
                + (": " + diagnostic[-1][:256] if diagnostic else "")
            )
            errors.append(f"{name}: {entry['reason']}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            entry["reason"] = "clang produced non-UTF-8 output"
            errors.append(f"{name}: {entry['reason']}")
            continue
        artifact = f"07_analysis/ir/{name[:-2]}.ll"
        files[artifact] = text
        entry.update(status="approximate" if mode == "approx" else "ok",
                     artifact=artifact, ir_sha256=_digest(data))
    summary["status"] = ("failed" if mode == "require" else "partial") if errors else (
        "approximate" if mode == "approx" else "ok"
    )
    if errors:
        summary["reason"] = "; ".join(errors)[:1024]
        if not files and mode in ("auto", "approx"):
            summary["status"] = "unavailable"
    return summary, files
