"""Benchmark workspace and profiling helpers for the CFU design agent."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from .state import BenchmarkProject, CommandResult, ProfileRegion, ProfileResult, WorkloadSpec
from .tools import RepoToolbox, parse_command_output


PROFILE_LINE_RE = re.compile(r"CFU_AGENT_PROFILE\s+(?P<body>.*)")


def slugify(text: str, *, default: str = "cfu_workload") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()
    return slug[:80] or default


def workspace_rel(spec: WorkloadSpec, suffix: str, *, run_root: str = "") -> str:
    slug = slugify(spec.get("name", "cfu_workload"))
    if run_root:
        return f"{run_root}/workspace/{slug}_{suffix}"
    return f"agents/generated/workspaces/{slug}/{suffix}"


def prepare_generated_benchmark(toolbox: RepoToolbox, spec: WorkloadSpec) -> BenchmarkProject:
    root = workspace_rel(spec, "benchmark", run_root=getattr(toolbox, "run_root", ""))
    source = f"{root}/main.c"
    header = f"{root}/cfu_agent_counter.h"
    build = (
        "make -C sw TARGET=vexii_soc "
        f"MAIN=../{source[:-2]} "
        f"BUILD=../{root}/build "
        "MARCH=rv32imc_zicsr_zifencei compile"
    )
    elf = f"{source[:-2]}.elf"
    run = f"sbt runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf {elf} --with-rvc --with-rvm --with-rdtime"
    notes = ["Generated from natural-language WorkloadSpec."]
    candidates = ["scalar_kernel"]

    if not toolbox.dry_run:
        toolbox.write_text(header, render_counter_header())
        toolbox.write_text(source, render_benchmark_c(spec))
    else:
        notes.append("dry-run: benchmark source/header were planned but not written.")

    return {
        "kind": "generated_benchmark",
        "workspace": root,
        "source_path": source,
        "header_path": header,
        "build_cmd": build,
        "run_cmd": run,
        "elf_path": elf,
        "candidate_hotspots": candidates,
        "notes": notes,
    }


def prepare_c_project_profile(toolbox: RepoToolbox, spec: WorkloadSpec) -> BenchmarkProject:
    root = workspace_rel(spec, "project_profile", run_root=getattr(toolbox, "run_root", ""))
    project_root = spec.get("project_root", "")
    kernel_file = spec.get("kernel_file", "")
    copied_project = f"{root}/project"
    notes = ["Original project is copied before instrumentation."]
    code = toolbox.read_text(kernel_file, limit=200000) if kernel_file else ""
    candidates = static_hotspot_candidates(code)

    if not project_root:
        notes.append("missing project_root; profiling workspace is only planned.")
        return {
            "kind": "planned",
            "workspace": root,
            "source_path": kernel_file,
            "build_cmd": spec.get("build_cmd", ""),
            "run_cmd": spec.get("run_cmd", ""),
            "test_cmd": spec.get("test_cmd", ""),
            "candidate_hotspots": candidates,
            "notes": notes,
        }

    if not toolbox.dry_run:
        toolbox.copy_tree(project_root, copied_project)
        toolbox.write_text(f"{root}/cfu_agent_counter.h", render_counter_header())
        instrument_main_copy(toolbox, spec, copied_project, root, notes)
    else:
        notes.append("dry-run: project copy and instrumentation were planned but not written.")

    return {
        "kind": "instrumented_copy",
        "workspace": root,
        "source_path": mapped_kernel_path(spec, copied_project),
        "header_path": f"{root}/cfu_agent_counter.h",
        "build_cmd": remap_project_command(spec.get("build_cmd", ""), project_root, copied_project),
        "run_cmd": remap_project_command(spec.get("run_cmd", ""), project_root, copied_project),
        "test_cmd": remap_project_command(spec.get("test_cmd", ""), project_root, copied_project),
        "candidate_hotspots": candidates,
        "notes": notes,
    }


def instrument_main_copy(
    toolbox: RepoToolbox,
    spec: WorkloadSpec,
    copied_project: str,
    root: str,
    notes: list[str],
) -> None:
    rel = mapped_kernel_path(spec, copied_project)
    if not rel:
        notes.append("missing kernel_file; no source instrumentation was applied.")
        return
    text = toolbox.read_text(rel, limit=200000)
    if not text:
        notes.append(f"kernel file not found in copied workspace: {rel}")
        return
    replaced, count = re.subn(r"\bint\s+main\s*\(", "int cfu_agent_target_main(", text, count=1)
    if count == 0:
        notes.append("no `int main(...)` definition found; no wrapper was inserted.")
        return
    header_rel = os.path.relpath(
        toolbox.resolve(f"{root}/cfu_agent_counter.h"),
        toolbox.resolve(Path(rel).parent),
    )
    wrapper = f"""

#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif
#include "{header_rel}"

int main(void) {{
    cfu_agent_cycle_t start = cfu_agent_counter_read();
    int rc = cfu_agent_target_main();
    cfu_agent_cycle_t end = cfu_agent_counter_read();
    uint32_t cycles = cfu_agent_counter_elapsed(start, end);
    printf("CFU_AGENT_PROFILE main=%u\\n", cycles);
    return rc;
}}
"""
    toolbox.write_text(rel, replaced + wrapper)
    notes.append(f"renamed `main` and inserted top-level profile wrapper in {rel}.")


def mapped_kernel_path(spec: WorkloadSpec, copied_project: str) -> str:
    kernel = spec.get("kernel_file", "")
    project = spec.get("project_root", "")
    if not kernel:
        return ""
    try:
        rel = Path(kernel).relative_to(project)
    except ValueError:
        rel = Path(kernel).name
    return str(Path(copied_project) / rel)


def remap_project_command(command: str, project_root: str, copied_project: str) -> str:
    if not command:
        return ""
    parts = shlex.split(command)
    if parts and parts[0] == "make":
        if "-C" not in parts:
            remapped = [
                "make",
                "-C",
                copied_project,
                *[
                    remap_project_token(part, project_root, copied_project, inside_project=True)
                    for part in parts[1:]
                ],
            ]
        else:
            remapped = []
            remap_next_cdir = False
            for idx, part in enumerate(parts):
                if remap_next_cdir:
                    remapped.append(copied_project if part == project_root else part)
                    remap_next_cdir = False
                elif part == "-C":
                    remapped.append(part)
                    remap_next_cdir = True
                elif idx == 0:
                    remapped.append(part)
                else:
                    remapped.append(remap_project_token(part, project_root, copied_project, inside_project=True))
    else:
        remapped = [
            remap_project_token(part, project_root, copied_project, inside_project=False)
            for part in parts
        ]
    return " ".join(shlex.quote(part) for part in remapped)


def remap_project_token(part: str, project_root: str, copied_project: str, *, inside_project: bool) -> str:
    if not project_root:
        return part
    if "=" in part:
        key, value = part.split("=", 1)
        remapped = remap_project_path(value, project_root, copied_project, inside_project=inside_project)
        return f"{key}={remapped}"
    return remap_project_path(part, project_root, copied_project, inside_project=inside_project)


def remap_project_path(path: str, project_root: str, copied_project: str, *, inside_project: bool) -> str:
    if path == project_root:
        return "." if inside_project else copied_project
    prefix = project_root.rstrip("/") + "/"
    if not path.startswith(prefix):
        return path
    suffix = path[len(prefix):]
    return suffix if inside_project else str(Path(copied_project) / suffix)


def profile_commands(project: BenchmarkProject) -> list[list[str]]:
    commands: list[list[str]] = []
    for key in ("build_cmd", "test_cmd", "run_cmd"):
        raw = project.get(key, "").strip()
        if raw:
            commands.append(shlex.split(raw))
    return commands


def profile_result_from_commands(results: list[CommandResult], *, dry_run: bool, planned_reason: str = "dry-run") -> ProfileResult:
    if dry_run:
        return {
            "status": "planned",
            "command_results": results,
            "regions": [],
            "hot_spot": "",
            "summary": f"{planned_reason}: profiling commands were planned but not executed.",
        }
    regions: list[ProfileRegion] = []
    for result in results:
        text = "\n".join([result.get("stdout_tail", ""), result.get("stderr_tail", "")])
        regions.extend(parse_profile_regions(text))
    hot = max(regions, key=lambda item: item.get("cycles", 0), default={}).get("name", "")
    failed = any(result.get("returncode", 0) != 0 for result in results)
    return {
        "status": "failed" if failed else "profiled",
        "command_results": results,
        "regions": regions,
        "hot_spot": hot,
        "summary": summarize_profile(regions, failed),
    }


def parse_profile_regions(text: str) -> list[ProfileRegion]:
    regions: list[ProfileRegion] = []
    for match in PROFILE_LINE_RE.finditer(text):
        for item in match.group("body").split():
            if "=" not in item:
                continue
            name, value = item.split("=", 1)
            if name in {"repeat", "bytes", "speedup_x100", "speedup_fused_x100", "speedup_twopass_x100"}:
                continue
            try:
                cycles = int(value)
            except ValueError:
                continue
            regions.append({"name": name, "cycles": cycles, "source": "CFU_AGENT_PROFILE"})
    parsed = parse_command_output(text)
    for profile in parsed.get("profiles", []):
        if isinstance(profile, dict):
            for name, value in profile.items():
                if name == "name" or not isinstance(value, int):
                    continue
                if name in {"repeat", "bytes", "speedup_x100", "speedup_fused_x100", "speedup_twopass_x100"}:
                    continue
                regions.append({"name": name, "cycles": value, "source": str(profile.get("name", "PROFILE"))})
    return regions


def static_hotspot_candidates(code: str) -> list[str]:
    if not code:
        return []
    candidates: list[str] = []
    skip_names = {
        "main",
        "printf",
        "init",
        "init_data",
        "fill",
        "fill_case",
        "setup",
        "teardown",
    }
    pattern = re.compile(
        r"(?:__attribute__\s*\(\([^)]*\)\)\s*)?"
        r"(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w\s\*]*?\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(code))
    for idx, match in enumerate(matches):
        name = match.group("name")
        if name in skip_names:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(code)
        body = code[start:end]
        if re.search(r"\b(for|while)\s*\(", body):
            candidates.append(name)
    return candidates[:8]


def summarize_profile(regions: list[ProfileRegion], failed: bool) -> str:
    if failed:
        return "profiling command failed; inspect command output before CFU design."
    if not regions:
        return "no profile regions were parsed; add or fix profiling print lines."
    ordered = sorted(regions, key=lambda item: item.get("cycles", 0), reverse=True)
    return ", ".join(f"{item.get('name')}={item.get('cycles')}" for item in ordered[:5])


def render_counter_header() -> str:
    return """#ifndef CFU_AGENT_COUNTER_H
#define CFU_AGENT_COUNTER_H

#include <stdint.h>

typedef uint32_t cfu_agent_cycle_t;

#if defined(__riscv)
static inline cfu_agent_cycle_t cfu_agent_counter_read(void) {
    cfu_agent_cycle_t value;
    __asm__ volatile("rdcycle %0" : "=r"(value));
    return value;
}
#else
#include <time.h>
static inline cfu_agent_cycle_t cfu_agent_counter_read(void) {
    return (cfu_agent_cycle_t)clock();
}
#endif

static inline uint32_t cfu_agent_counter_elapsed(cfu_agent_cycle_t start, cfu_agent_cycle_t end) {
    return (uint32_t)(cfu_agent_cycle_t)(end - start);
}

#endif
"""


def render_benchmark_c(spec: WorkloadSpec) -> str:
    operation = spec.get("operation", "unknown")
    elem = "uint8_t" if "8" in spec.get("element_type", "uint8") else "uint32_t"
    bytes_count = int(spec.get("vector_bytes", 32)) * 4
    if operation in {"weighted_average", "dot_product"}:
        kernel = """
__attribute__((noinline))
static uint32_t scalar_kernel(const volatile uint8_t *x, const volatile uint8_t *w, int bytes) {
    uint32_t sum = 0;
    for(int i = 0; i < bytes; ++i) {
        sum += (uint32_t)x[i] * (uint32_t)w[i];
    }
    return sum;
}
"""
        call = "acc += scalar_kernel(x, w, BYTES);"
    elif operation == "elementwise_add":
        kernel = """
__attribute__((noinline))
static uint32_t scalar_kernel(const volatile uint8_t *x, const volatile uint8_t *y, volatile uint8_t *z, int bytes) {
    uint32_t checksum = 0;
    for(int i = 0; i < bytes; ++i) {
        z[i] = (uint8_t)(x[i] + y[i]);
        checksum += z[i];
    }
    return checksum;
}
"""
        call = "acc += scalar_kernel(x, w, out, BYTES);"
    else:
        kernel = """
__attribute__((noinline))
static uint32_t scalar_kernel(const volatile uint8_t *x, int bytes) {
    uint32_t sum = 0;
    for(int i = 0; i < bytes; ++i) {
        sum += x[i];
    }
    return sum;
}
"""
        call = "acc += scalar_kernel(x, BYTES);"
    return f"""#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdint.h>
#include <stdio.h>
#endif

#include "cfu_agent_counter.h"

enum {{ BYTES = {bytes_count}, REPEAT = 8 }};

static volatile {elem} x[BYTES] __attribute__((aligned(32)));
static volatile {elem} w[BYTES] __attribute__((aligned(32)));
static volatile {elem} out[BYTES] __attribute__((aligned(32)));
volatile uint32_t cfu_agent_sink = 0;

static void init_data(void) {{
    for(int i = 0; i < BYTES; ++i) {{
        x[i] = ({elem})((i * 17 + 3) & 0xff);
        w[i] = ({elem})((i * 29 + 91) & 0xff);
        out[i] = 0;
    }}
}}

{kernel}
int main(void) {{
    init_data();
    uint32_t acc = 0;
    cfu_agent_cycle_t start = cfu_agent_counter_read();
    for(int r = 0; r < REPEAT; ++r) {{
        {call}
    }}
    cfu_agent_cycle_t end = cfu_agent_counter_read();
    uint32_t cycles = cfu_agent_counter_elapsed(start, end);
    cfu_agent_sink = acc;
    printf("CFU_AGENT_PASS checksum=%u\\n", cfu_agent_sink);
    printf("CFU_AGENT_PROFILE scalar_kernel=%u repeat=%d bytes=%d\\n",
           cycles, REPEAT, BYTES);
    return 0;
}}
"""
