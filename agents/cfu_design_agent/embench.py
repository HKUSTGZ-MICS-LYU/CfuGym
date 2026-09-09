"""Embench-iot workload support: discovery, workspace preparation, and commands.

Embench lives in a read-only submodule (benchmarks/embench-iot). Every run copies
the selected benchmark, Embench's support code and the VexiiRiscv board
(sw/embench) into its own workspace, then builds three configurations:

    reference  plain build; correctness + baseline cycles
    profile    reference + observation points; per-function cycle table
    cfu        -DUSE_AGENT_CFU; the agent's accelerated kernel

Correctness comes from Embench's own verify_benchmark(): support/main.c returns
!correct, and sw/MiCo-Lib/targets/vexii_soc/start.S jumps to the pass/fail
symbols that MiCoSocSim watches. Benchmarks that do not verify (edn) are marked
so the run never claims correctness for them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .embench_instrument import find_definition_brace
from .tools import RepoToolbox


EMBENCH_ROOT = "benchmarks/embench-iot"
BOARD_ROOT = "sw/embench"
SUPPORT_FILES = ("main.c", "beebsc.c", "beebsc.h", "support.h")
BOARD_FILES = ("boardsupport.c", "boardsupport.h", "embench_obs.h")
CONFIGS = ("reference", "profile", "cfu")
DEFAULT_MARCH = "rv32imc_zicsr_zifencei"
DEFAULT_WARMUP_HEAT = 1
DEFAULT_GLOBAL_SCALE_FACTOR = 1


def benchmark_dir(name: str) -> str:
    return f"{EMBENCH_ROOT}/src/{name}"


def list_benchmarks(workdir: Path) -> list[str]:
    root = workdir / EMBENCH_ROOT / "src"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def benchmark_source_files(workdir: Path, name: str) -> list[str]:
    root = workdir / benchmark_dir(name)
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(workdir)) for p in root.glob("*.c"))


def has_real_verification(text: str) -> bool:
    """True unless verify_benchmark() is a stub that always returns -1."""
    brace = find_definition_brace(text, "verify_benchmark")
    if brace < 0:
        return False
    depth = 0
    i = brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[brace + 1 : i]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    body = re.sub(r"//[^\n]*", "", body)
    statements = [s.strip() for s in body.split(";") if s.strip()]
    return not (len(statements) == 1 and statements[0].startswith("return -1"))


def workspace_rel(run_root: str, name: str) -> str:
    return f"{run_root.strip('/')}/workspace/embench_{name}"


def config_workspace(run_root: str, name: str, config: str) -> str:
    """Per-config source tree.

    Each configuration needs its own copy of the sources: the sw Makefile
    derives object paths from the source path, so sharing one tree would make
    the three configurations reuse each other's objects and silently ignore
    their own defines (observed: the profile build linked reference objects and
    produced no observation counters).
    """
    return f"{workspace_rel(run_root, name)}/{config}"


def config_defines(name: str, config: str) -> list[str]:
    defines = [
        "-DHAVE_BOARDSUPPORT_H",
        f"-DWARMUP_HEAT={DEFAULT_WARMUP_HEAT}",
        f"-DGLOBAL_SCALE_FACTOR={DEFAULT_GLOBAL_SCALE_FACTOR}",
        f"-DEMBENCH_NAME={name}",
    ]
    if config == "profile":
        defines.append("-DEMBENCH_OBS")
    if config == "cfu":
        defines.append("-DUSE_AGENT_CFU")
    return defines


def build_command(run_root: str, name: str, config: str) -> list[str]:
    """make invocation that builds one Embench configuration."""
    cfg = config_workspace(run_root, name, config)
    rel = os.path.relpath(cfg, "sw")
    sources = " ".join(
        [f"{rel}/src/*.c", f"{rel}/support/beebsc.c", f"{rel}/board/boardsupport.c"]
    )
    includes = " ".join([f"{rel}/support", f"{rel}/board", f"{rel}/include"])
    defines = " ".join(config_defines(name, config))
    return [
        "make",
        "-C",
        "sw",
        "TARGET=vexii_soc",
        f"MARCH={DEFAULT_MARCH}",
        f"MAIN={rel}/embench_main",
        f"BUILD={rel}/build",
        f"EXTRA_SOURCES={sources}",
        f"EXTRA_INCLUDES={includes}",
        f"EXTRA_CFLAGS={defines}",
        "compile",
    ]


def elf_path(run_root: str, name: str, config: str) -> str:
    return f"{config_workspace(run_root, name, config)}/embench_main.elf"


def config_sources(workdir: Path, run_root: str, name: str, config: str) -> list[str]:
    """Benchmark source files in one configuration's workspace copy."""
    root = workdir / config_workspace(run_root, name, config) / "src"
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(workdir)) for p in root.glob("*.c"))


def prepare_embench_benchmark(toolbox: RepoToolbox, spec: dict) -> dict:
    """Copy the benchmark, Embench support and the board into the run workspace."""
    name = str(spec.get("benchmark_name", "") or "").strip()
    if not name:
        raise ValueError("embench workload requires benchmark_name")
    run_root = toolbox.run_root
    if not run_root:
        raise ValueError("embench workload requires a run root")

    source_files = benchmark_source_files(toolbox.workdir, name)
    if not source_files:
        raise ValueError(f"unknown embench benchmark: {name}")

    ws = workspace_rel(run_root, name)
    notes: list[str] = []
    verification = "none"
    copied: list[str] = []

    # Read every source once; Embench's own tree is never modified.
    source_texts: dict[str, str] = {}
    for rel_src in source_files:
        text = toolbox.read_text(rel_src, limit=2_000_000)
        source_texts[Path(rel_src).name] = text
        if verification == "none" and "verify_benchmark" in text and has_real_verification(text):
            verification = "bench_verify"
    support_texts = {
        fname: toolbox.read_text(f"{EMBENCH_ROOT}/support/{fname}", limit=2_000_000)
        for fname in SUPPORT_FILES
    }
    board_texts = {
        fname: toolbox.read_text(f"{BOARD_ROOT}/{fname}", limit=2_000_000)
        for fname in BOARD_FILES
    }
    main_text = support_texts.get("main.c", "")

    for config in CONFIGS:
        cfg = config_workspace(run_root, name, config)
        for fname, text in source_texts.items():
            copied.append(f"{cfg}/src/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/src/{fname}", text)
        for fname, text in support_texts.items():
            copied.append(f"{cfg}/support/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/support/{fname}", text)
        for fname, text in board_texts.items():
            copied.append(f"{cfg}/board/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/board/{fname}", text)
        copied.append(f"{cfg}/embench_main.c")
        if not toolbox.dry_run:
            toolbox.write_text(f"{cfg}/embench_main.c", main_text)

    if verification == "none":
        notes.append("verify_benchmark() is a stub; this benchmark provides no correctness oracle")

    return {
        "kind": "embench_benchmark",
        "suite": "embench-iot",
        "benchmark": name,
        "workspace": ws,
        "source_path": source_files[0] if source_files else "",
        "source_files": source_files,
        "header_path": f"{ws}/board/embench_obs.h",
        "verification": verification,
        "configs": list(CONFIGS),
        "elf_path": elf_path(run_root, name, "reference"),
        "elf_paths": {config: elf_path(run_root, name, config) for config in CONFIGS},
        "build_cmd": " ".join(build_command(run_root, name, "reference")),
        "run_cmd": "",
        "candidate_hotspots": [],
        "copied_files": copied,
        "notes": notes,
    }


def parse_profile(text: str) -> dict[str, dict[str, int]]:
    """Collect CFU_AGENT_PROFILE lines into {id: {cycles, calls}}."""
    result: dict[str, dict[str, int]] = {}
    for match in re.finditer(r"CFU_AGENT_PROFILE\s+(?P<body>.*)", text):
        fields: dict[str, int] = {}
        name = ""
        for item in match.group("body").split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                number = int(value)
            except ValueError:
                continue
            if key == "calls":
                fields["calls"] = number
            else:
                name = key
                fields["cycles"] = number
        if name:
            result[name] = fields
    return result


def hotspot_from_profile(profile: dict[str, dict[str, int]], *, exclude: tuple[str, ...] = ()) -> str:
    """Hottest observation point by cycles, ignoring the whole-benchmark entry."""
    candidates = {
        name: data.get("cycles", 0)
        for name, data in profile.items()
        if name not in exclude
    }
    if not candidates:
        return ""
    return max(candidates, key=lambda key: candidates[key])
