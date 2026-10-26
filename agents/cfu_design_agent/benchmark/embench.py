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

import hashlib
import os
import re
from pathlib import Path

from . import cfu_execution, embench_integrity, embench_patchers
from .embench_instrument import find_definition_brace
from .profile_lines import profile_lines
from ..runtime.tools import RepoToolbox, embench_cfu_rel


EMBENCH_ROOT = "benchmarks/embench-iot"
BOARD_ROOT = "sw/embench"
SUPPORT_FILES = ("main.c", "beebsc.c", "beebsc.h", "support.h")
BOARD_FILES = ("boardsupport.c", "boardsupport.h", "embench_obs.h")
CONFIGS = ("reference", "profile", "cfu")
DEFAULT_MARCH = "rv32imc_zicsr_zifencei"
DEFAULT_C_STANDARD = "-std=gnu11"
DEFAULT_WARMUP_HEAT = 1
DEFAULT_GLOBAL_SCALE_FACTOR = 1
MEASUREMENT_PROTOCOL = "cold_single_invocation_v1"

def measurement_parameters() -> dict:
    """One Host protocol shared by preparation, integrity and checkpoint checks."""
    return {"protocol": MEASUREMENT_PROTOCOL, "local_scale_factor": 1,
            "global_scale_factor": DEFAULT_GLOBAL_SCALE_FACTOR, "warmup": False,
            "cfu_status_protocol": cfu_execution.PROTOCOL}


# The SoC harness stops on the pass/fail symbol, which only says main() returned
# non-zero. The harness owns the generated embench_main.c, so it adds one
# machine-readable line that makes a verify failure attributable in the sim log
# without touching Embench's oracle or its read-only tree.
VERIFY_CALL = "correct = verify_benchmark (result);"
VERIFY_PREAMBLE = """/* Injected by the CFU agent harness (verify diagnostics). */
#include <stdio.h>
#ifndef EMBENCH_NAME
#define EMBENCH_NAME unknown
#endif
#ifndef EMBENCH_STRINGIFY
#define EMBENCH_STRINGIFY_(x) #x
#define EMBENCH_STRINGIFY(x) EMBENCH_STRINGIFY_(x)
#endif
"""
VERIFY_DIAGNOSTIC = """
  /* Harness diagnostic. Embench's verify_benchmark() returns non-zero when the
     result is CORRECT and 0 on a mismatch; main() returns !correct and the
     startup code branches to `pass` iff that return is 0. So correct != 0 is a
     PASS and correct == 0 is a FAIL. */
  printf ("EMBENCH_VERIFY_%s benchmark=%s verify=%d result=%d\\n",
          correct ? "PASS" : "FAIL", EMBENCH_STRINGIFY (EMBENCH_NAME), correct,
          (int) result);
"""


def instrument_embench_main(text: str) -> str:
    """Omit warmup entirely and add the machine-readable verification line."""
    text = embench_integrity.omit_warmup(text)
    if VERIFY_CALL not in text:
        return text
    text = cfu_execution.instrument_main(text, VERIFY_CALL)
    # Status must affect `correct` before the diagnostic prints the final verdict.
    status_verdict = VERIFY_CALL + cfu_execution.VERDICT
    anchor = status_verdict if status_verdict in text else VERIFY_CALL
    injected = text.replace(anchor, anchor + VERIFY_DIAGNOSTIC, 1)
    return VERIFY_PREAMBLE + injected


# The machine-readable verify line the harness injects into every config's
# main(). Parsing it gives the harness an oracle independent of the model's
# writable cfu copy: the reference run and the cfu run must compute the same
# ``result`` value.
VERIFY_LINE_RE = re.compile(
    r"EMBENCH_VERIFY_(?P<status>PASS|FAIL)\s+"
    r"benchmark=(?P<benchmark>\S+)\s+"
    r"verify=(?P<verify>-?\d+)\s+result=(?P<result>-?\d+)"
)


def parse_verify_result(text: str) -> dict | None:
    """Last EMBENCH_VERIFY line as {benchmark, pass, verify, result}, or None."""
    matches = list(VERIFY_LINE_RE.finditer(text or ""))
    if not matches:
        return None
    last = matches[-1]
    return {
        "benchmark": last.group("benchmark"),
        "pass": last.group("status") == "PASS",
        "verify": int(last.group("verify")),
        "result": int(last.group("result")),
    }



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


def config_cflags(name: str, config: str) -> list[str]:
    """C flags shared by every Embench build configuration."""
    return [*config_defines(name, config), DEFAULT_C_STANDARD]


def build_command(run_root: str, name: str, config: str) -> list[str]:
    """make invocation that builds one Embench configuration."""
    cfg = config_workspace(run_root, name, config)
    rel = os.path.relpath(cfg, "sw")
    sources = " ".join(
        [f"{rel}/src/*.c", f"{rel}/support/beebsc.c", f"{rel}/board/boardsupport.c",
         *(f"{rel}/support/{filename}" for filename in embench_patchers.extra_support_files(name)
           if filename.endswith(".c"))]
    )
    includes = " ".join([f"{rel}/support", f"{rel}/board", f"{rel}/include"])
    defines = " ".join(config_cflags(name, config))
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


def normalize_embench_link_symbols(name: str, sources: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Compatibility API; preparation applies the unified patcher pipeline."""
    patcher = embench_patchers.StatemateSymbolPatcher()
    if not patcher.applies_to(name):
        return sources, []
    result = patcher.apply(sources)
    return result.sources, result.notes


def cfu_source_targets(workdir: Path, run_root: str, name: str) -> list[tuple[str, str]]:
    """[(cfu workspace path, current text)] for the benchmark sources."""
    targets: list[tuple[str, str]] = []
    for rel_src in benchmark_source_files(workdir, name):
        target = embench_cfu_rel(run_root, name, Path(rel_src).name)
        path = workdir / target
        targets.append((target, path.read_text(errors="replace") if path.is_file() else ""))
    return targets


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
    patched = embench_patchers.apply_benchmark_patches(name, source_texts)
    source_texts = patched.sources
    notes.extend(patched.notes)
    source_root = toolbox.workdir / benchmark_dir(name)
    header_texts = {
        path.name: toolbox.read_text(str(path.relative_to(toolbox.workdir)), limit=2_000_000)
        for path in sorted(source_root.glob("*.h"))
    }
    support_texts = {
        fname: toolbox.read_text(f"{EMBENCH_ROOT}/support/{fname}", limit=2_000_000)
        for fname in SUPPORT_FILES
    }
    if support_texts.keys() & patched.support_files.keys():
        raise ValueError("benchmark patch conflicts with Embench support files")
    support_texts.update(patched.support_files)
    board_texts = {
        fname: toolbox.read_text(f"{BOARD_ROOT}/{fname}", limit=2_000_000)
        for fname in BOARD_FILES
    }
    main_text = instrument_embench_main(support_texts.get("main.c", ""))

    reference_sources: dict[str, str] = {}
    protected_files: dict[str, str] = {}

    def source_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    for config in CONFIGS:
        cfg = config_workspace(run_root, name, config)
        for fname, text in source_texts.items():
            if config == "reference":
                reference_sources[f"{cfg}/src/{fname}"] = source_hash(text)
            copied.append(f"{cfg}/src/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/src/{fname}", text)
        for fname, text in header_texts.items():
            protected_files[f"{cfg}/src/{fname}"] = source_hash(text)
            copied.append(f"{cfg}/src/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/src/{fname}", text)
        for fname, text in support_texts.items():
            protected_files[f"{cfg}/support/{fname}"] = source_hash(text)
            copied.append(f"{cfg}/support/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/support/{fname}", text)
        for fname, text in board_texts.items():
            protected_files[f"{cfg}/board/{fname}"] = source_hash(text)
            copied.append(f"{cfg}/board/{fname}")
            if not toolbox.dry_run:
                toolbox.write_text(f"{cfg}/board/{fname}", text)
        protected_files[f"{cfg}/embench_main.c"] = source_hash(main_text)
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
        "measurement": measurement_parameters(),
        "patches": patched.patches,
        "integrity": {
            "version": embench_integrity.INTEGRITY_VERSION,
            "reference_sources": reference_sources,
            "protected_files": protected_files,
        },
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
    """Collect CFU_AGENT_PROFILE lines into {id: {cycles, calls}}.

    The harness EMBENCH_VERIFY line is folded in as ``verify``/``result`` on the
    whole-benchmark entry so a run can compare the value the reference computed
    with the value the cfu build computed, independent of verify_benchmark().
    """
    result: dict[str, dict[str, int]] = {}
    for fields in profile_lines(text):
        calls = fields.get("calls")
        for name, cycles in fields.items():
            if name == "calls":
                continue
            result[name] = {"cycles": cycles, **({"calls": calls} if calls is not None else {})}
    verify = parse_verify_result(text)
    if verify:
        entry = result.setdefault(str(verify["benchmark"]), {})
        entry["verify"] = int(verify["verify"])
        entry["result"] = int(verify["result"])
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


def function_region(text: str, name: str) -> str:
    """The balanced-brace definition of ``name``, or "" when absent.

    Used to compare one function across two workspace copies without parsing C:
    the definition through its matching close brace is stable under renaming of
    surrounding code.
    """
    brace = find_definition_brace(text or "", name)
    if brace < 0:
        return ""
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
    return text[brace : i + 1]


def workspace_tampering(workdir, run_root: str, name: str) -> list[str]:
    """Benchmark-oracle tampering between the harness reference and cfu copies.

    The model may edit the cfu kernel sources, but the correctness oracle
    (verify_benchmark), the cycle measurement (board/*), the Embench support
    code and the instrumented main must stay byte-identical to the
    harness-owned reference copy; otherwise a hard-coded expected result or a
    faked rdcycle passes every gate. Files absent from either copy are skipped
    so fixture workspaces without a full config set still validate.
    """
    ref_root = Path(workdir) / config_workspace(run_root, name, "reference")
    cfu_root = Path(workdir) / config_workspace(run_root, name, "cfu")
    if not ref_root.is_dir() or not cfu_root.is_dir():
        return []
    problems: list[str] = []
    protected = (
        [f"board/{fname}" for fname in BOARD_FILES]
        + [f"support/{fname}" for fname in SUPPORT_FILES]
        + ["embench_main.c"]
    )
    for rel in protected:
        ref, cfu = ref_root / rel, cfu_root / rel
        if ref.is_file() and cfu.is_file() and ref.read_bytes() != cfu.read_bytes():
            problems.append(
                f"benchmark harness file '{rel}' was modified in the cfu workspace: the "
                "correctness oracle and the cycle measurement must stay byte-identical to "
                "the harness-owned reference copy"
            )
    ref_src_root, cfu_src_root = ref_root / "src", cfu_root / "src"
    if ref_src_root.is_dir() and cfu_src_root.is_dir():
        for ref_src in sorted(ref_src_root.glob("*.c")):
            cfu_src = cfu_src_root / ref_src.name
            if not cfu_src.is_file():
                continue
            ref_fn = function_region(ref_src.read_text(errors="replace"), "verify_benchmark")
            cfu_fn = function_region(cfu_src.read_text(errors="replace"), "verify_benchmark")
            if ref_fn and cfu_fn and ref_fn != cfu_fn:
                problems.append(
                    f"verify_benchmark() in 'src/{ref_src.name}' was modified in the cfu "
                    "workspace: the benchmark's correctness oracle must stay byte-identical "
                    "to the harness copy"
                )
    return problems
