#!/usr/bin/env python3
import argparse
import csv
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SW = ROOT / "sw"
DEFAULT_OUT = ROOT / "benchmark_results" / "bncfu_v2_block_explore"


PROFILE_RE = re.compile(
    r"BNCFU_COMPARE_PROFILE\s+variant=(?P<variant>\S+)\s+"
    r"scalar=(?P<scalar>\d+)\s+cfu=(?P<cfu>\d+)\s+speedup_x100=(?P<speedup>\d+)"
)
PASS_RE = re.compile(
    r"BNCFU_COMPARE PASS\s+variant=(?P<variant>\S+)\s+expected=(?P<expected>-?\d+)\s+got=(?P<got>-?\d+)"
)
COST_RE = re.compile(
    r"cost\s+top=\S+\s+cells=(?P<cells>\d+)\s+wire_bits=(?P<wire_bits>\d+)\s+"
    r"memories=(?P<memories>\d+)\s+memory_bits=(?P<memory_bits>\d+)"
)


@dataclass(frozen=True)
class BlockCase:
    block_count: int
    total_outputs: int
    set_full_count_each_block: bool = False

    @property
    def active_count(self) -> int:
        tail = self.total_outputs % self.block_count
        return tail if tail else self.block_count

    @property
    def name(self) -> str:
        suffix = "_seteach" if self.set_full_count_each_block else ""
        return f"block{self.block_count}_tail{self.active_count}_total{self.total_outputs}{suffix}"

    @property
    def output_count(self) -> int:
        return 64 * self.total_outputs

    @property
    def c_defines(self) -> List[str]:
        defines = [
            "-DBNCFU_COMPARE_V2=1",
            "-DBNCFU_COMPARE_FUSED=1",
            "-DBNCFU_COMPARE_WAIT=1",
            "-DBNCFU_COMPARE_RESIDENT_ACT=1",
            "-DBNCFU_COMPARE_LOAD_FUSED=1",
            f"-DBNCFU_COMPARE_BLOCK_COUNT={self.block_count}",
            f"-DBNCFU_COMPARE_ACTIVE_BLOCK_COUNT={self.active_count}",
            f"-DBNCFU_COMPARE_TOTAL_OUTPUTS={self.total_outputs}",
            "-DBNCFU_COMPARE_ACCUM=1",
            "-DBNCFU_COMPARE_HOIST_FENCE=1",
        ]
        if self.set_full_count_each_block:
            defines.append("-DBNCFU_COMPARE_SET_FULL_COUNT_EACH_BLOCK=1")
        return defines


def parse_block_counts(text: str) -> List[int]:
    values = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        value = int(part, 0)
        if value <= 0:
            raise argparse.ArgumentTypeError("block counts must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one block count is required")
    return values


def quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(cmd: List[str], log_path: Path, cwd: Path, dry_run: bool) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "$ " + quote_cmd(cmd) + "\n"
    if dry_run:
        print(header, end="")
        return header

    with log_path.open("w") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        chunks = [header]
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            chunks.append(line)
        rc = proc.wait()
    text = "".join(chunks)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return text


def parse_profile(text: str) -> Dict[str, Optional[int]]:
    passed = PASS_RE.search(text)
    profile = PROFILE_RE.search(text)
    if not passed:
        raise RuntimeError("simulation log has no BNCFU_COMPARE PASS line")
    if not profile:
        raise RuntimeError("simulation log has no BNCFU_COMPARE_PROFILE line")
    return {
        "variant": profile.group("variant"),
        "expected": int(passed.group("expected")),
        "got": int(passed.group("got")),
        "scalar_cycles": int(profile.group("scalar")),
        "cfu_cycles": int(profile.group("cfu")),
        "speedup_x100": int(profile.group("speedup")),
    }


def parse_cost(text: str) -> Dict[str, Optional[int]]:
    match = COST_RE.search(text)
    if not match:
        return {"cells": None, "wire_bits": None, "memories": None, "memory_bits": None}
    return {key: int(match.group(key)) for key in ("cells", "wire_bits", "memories", "memory_bits")}


def build_compare_elf(case: BlockCase, out_dir: Path, dry_run: bool) -> None:
    build = f"build_bncfu_compare_v2_{case.name}_load_fused_accum_hoist_fence_wait_resident"
    cmd = [
        "make",
        "TARGET=vexii_soc",
        "MARCH=rv32imc_zicsr_zifencei",
        "OPT=cfu",
        "MAIN=tests/bncfu_compare_test",
        f"BUILD={build}",
        "C_DEFINES=" + " ".join(case.c_defines),
        "clean",
        "compile",
    ]
    run(cmd, out_dir / case.name / "build.log", SW, dry_run)


def simulate_case(case: BlockCase, out_dir: Path, dry_run: bool, plugin_host: bool) -> Dict[str, Optional[int]]:
    cfu_flag = "--mico-bitnet-cfu-v2-plugin-host" if plugin_host else "--mico-bitnet-cfu-v2"
    cmd = [
        "sbt",
        "runMain vexiiriscv.soc.mico.MiCoSocSim "
        "--load-elf sw/tests/bncfu_compare_test.elf "
        "--with-rvc --with-rvm --with-rdtime "
        f"{cfu_flag} "
        "--bitnet-cfu-v2-reg-depth 3 "
        "--bitnet-cfu-v2-load-queue-depth 2 "
        "--bitnet-cfu-v2-quant-queue-depth 2 "
        "--bitnet-cfu-v2-dot-queue-depth 2 "
        "--bitnet-cfu-v2-result-queue-depth 8 "
        f"--bitnet-cfu-v2-block-count {case.block_count}",
    ]
    text = run(cmd, out_dir / case.name / "sim.log", ROOT, dry_run)
    return {} if dry_run else parse_profile(text)


def yosys_case(case: BlockCase, out_dir: Path, dry_run: bool, plugin_host: bool) -> Dict[str, Optional[int]]:
    suffix = "_plugin" if plugin_host else ""
    rtl_dir = ROOT / "synth_runs" / "rtl" / f"bitnet_cfu_v2{suffix}_block{case.block_count}_q128_runtime_count_safe"
    synth_dir = ROOT / "synth_runs" / "yosys" / f"bitnet_cfu_v2{suffix}_block{case.block_count}_q128_runtime_count_safe_flat"
    main = "BitNetCfuV2PluginGen" if plugin_host else "BitNetCfuV2Gen"
    top = "BitNetCfuV2PluginHost" if plugin_host else "BitNetCfuV2"
    gen_cmd = [
        "sbt",
        f"runMain vexiiriscv.soc.mico.{main} "
        f"--target-dir {rtl_dir.relative_to(ROOT)} "
        "--reg-depth 3 "
        "--load-queue-depth 2 "
        "--quant-queue-depth 2 "
        "--dot-queue-depth 2 "
        "--result-queue-depth 8 "
        f"--block-count {case.block_count}",
    ]
    run(gen_cmd, out_dir / case.name / "rtl.log", ROOT, dry_run)
    yosys_cmd = [
        "python3",
        "skills/cfu-designer/scripts/yosys_cost_report.py",
        str((rtl_dir / f"{top}.v").relative_to(ROOT)),
        "--top",
        top,
        "--out-dir",
        str(synth_dir.relative_to(ROOT)),
        "--flatten",
    ]
    text = run(yosys_cmd, out_dir / case.name / "yosys.log", ROOT, dry_run)
    return {} if dry_run else parse_cost(text)


def write_summary(rows: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    fields = [
        "case",
        "block_count",
        "active_count",
        "total_outputs",
        "outputs",
        "variant",
        "expected",
        "got",
        "scalar_cycles",
        "cfu_cycles",
        "speedup_x100",
        "speedup",
        "cfu_cycles_per_output",
        "cells",
        "wire_bits",
        "memories",
        "memory_bits",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# BNCFUv2 Block Exploration\n\n")
        f.write("| Case | Outputs | CFU cycles | Speedup | Cycles/output | Cells | Wire bits |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| `{row['case']}` | {row['outputs']} | {row.get('cfu_cycles', '')} | "
                f"{row.get('speedup', '')} | {row.get('cfu_cycles_per_output', '')} | "
                f"{row.get('cells', '')} | {row.get('wire_bits', '')} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, simulate, and optionally cost BNCFUv2 block-count variants."
    )
    parser.add_argument("--block-counts", type=parse_block_counts, default=parse_block_counts("64,128,256"))
    parser.add_argument("--total-outputs", type=int, default=229)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--with-yosys", action="store_true", help="also generate CFU RTL and run the Yosys cost report")
    parser.add_argument("--plugin-host", action="store_true", help="use the plugin-hosted BNCFUv2 hardware variant")
    parser.add_argument("--set-full-count-each-block", action="store_true", help="benchmark the older software policy that sets full count before every full block")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args = parser.parse_args()

    if args.total_outputs <= 0:
        parser.error("--total-outputs must be positive")

    rows: List[Dict[str, object]] = []
    for block_count in args.block_counts:
        case = BlockCase(
            block_count=block_count,
            total_outputs=args.total_outputs,
            set_full_count_each_block=args.set_full_count_each_block,
        )
        print(f"[case] {case.name}")
        build_compare_elf(case, args.out_dir, args.dry_run)
        profile = simulate_case(case, args.out_dir, args.dry_run, args.plugin_host)
        cost = yosys_case(case, args.out_dir, args.dry_run, args.plugin_host) if args.with_yosys else {}

        cfu_cycles = profile.get("cfu_cycles")
        speedup_x100 = profile.get("speedup_x100")
        row: Dict[str, object] = {
            "case": case.name,
            "block_count": case.block_count,
            "active_count": case.active_count,
            "total_outputs": case.total_outputs,
            "outputs": case.output_count,
            **profile,
            **cost,
        }
        if isinstance(speedup_x100, int):
            row["speedup"] = f"{speedup_x100 / 100.0:.2f}x"
        if isinstance(cfu_cycles, int):
            row["cfu_cycles_per_output"] = f"{cfu_cycles / case.output_count:.2f}"
        rows.append(row)

    if not args.dry_run:
        write_summary(rows, args.out_dir)
        print(f"[summary] {args.out_dir / 'summary.md'}")
        print(f"[summary] {args.out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
