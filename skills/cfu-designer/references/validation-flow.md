# CFU Validation Flow

## Structural Checks

Use Scala generation/elaboration first after hardware wiring changes. Existing entry scripts are useful examples:

- `gen_soc.sh`: MiCo SoC generation pattern.
- `sim_soc.sh`: generic MiCo SoC simulation pattern.
- `sim_soc_bncfu.sh`: BNCFU/BitNet CFU SoC simulation pattern with parameterized qtype, vector length, width, register depth, pipeline, quant width, and Q8 toggles.

For direct commands, use the repo's SBT entry points:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocGen <soc args>"
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf <elf> <soc args>"
```

The CPU ISA must match the ELF ISA. The MiCo software target defaults to
`rv32imc`, while `ParamMiCo` defaults to RV32I unless extensions are enabled
on the simulator command line. For the normal bare-metal CFU tests, use:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf <elf> --with-rvc --with-rvm --with-rdtime <soc args>"
```

Omitting `--with-rvc` or `--with-rvm` does not measure CFU behavior: the ELF
will trap on compressed or multiply/divide instructions and can appear to
hang at the reset trap vector. Rebuild the ELF with the same `MARCH` when
changing the CPU ISA.

If the firmware uses `rdcycle`/counter CSRs, include a CPU counter option in `<soc args>`. In this repo, use `--with-rdtime` for the minimal `zicntr` path, or `--performance-counters <n>` when extra HPM counters are required.

For BNCFU-like validation, start from:

```bash
bash sim_soc_bncfu.sh <elf> 1.5b 256 128 5 0 1 128 1
```

Adjust arguments only when the software flags and hardware parameters match.

## Software Build Surface

For bare-metal CFU tests, the software target is usually:

```bash
TARGET=vexii_soc OPT=cfu MARCH=rv32imafc_zifencei SPRAM=1 VLEN=<vlen>
```

For BNCFU-style software, use `OPT=bncfu` when that target exists in the checkout. Keep build directories isolated when comparing hardware/software variants to avoid host/RISC-V object pollution:

```bash
BUILD=build_<case_name>
```

Check `sw/MiCo-Lib/targets/vexii_soc.mk` before adding files or flags. It owns target-specific source inclusion, `VLEN`, `MICO_ALIGN`, `SPRAM`, ABI, and ISA flags.

Do not infer generated hardware features from `MARCH` alone. `MARCH=..._zicsr_zifencei` controls software assembly/linking, while Vexii CPU features are selected through Scala parameters and simulator/generator CLI flags such as `--with-cfu`, `--with-rdtime`, `--fetch-l1`, `--lsu-l1`, `--decoders`, and `--lanes`.

## Correctness Ladder

1. Scalar unit test: compare CFU output with a simple C or Scala reference for small inputs.
2. Software intrinsic test: confirm custom instruction fields match hardware decode comments.
3. SoC simulation: run the ELF on the generated SoC with the same parameters used at compile time.
4. Stress or benchmark: test multiple vector lengths, memory alignments, RF modes, and pipeline toggles.
5. Basic hardware cost: run the Yosys cost flow for early RTL-level cell/memory comparison.
6. Synthesis/PPA: use existing exploration scripts when final area/timing/fmax matter.

Reusable scripts:

- `tools/bncfu_perf_explore.py`: benchmark orchestration, hardware presets, software variants, log parsing, CSV/summary output.
- `tools/bncfu_v2_block_explore.py`: focused BNCFUv2 block-count loop. It builds `sw/tests/bncfu_compare_test.c`, runs `MiCoSocSim` with the matching `--bitnet-cfu-v2-block-count`, parses `BNCFU_COMPARE_PROFILE`, and can generate standalone `BitNetCfuV2` RTL plus a Yosys cost report.
- `tools/bncfu_space_explore.py`: synthesis/design-space exploration pattern.
- `tools/bncfu_e2e_kivi_benchmark.py` and `tools/bncfu_e2e_llama_benchmark.py`: end-to-end benchmark runner patterns.

For quick CFU cost checks, use `references/yosys-cost-flow.md` and `scripts/yosys_cost_report.py` before a slower vendor flow.
Use RTL emitted by `MiCoSocGen` or a standalone CFU `generateVerilog` object for this cost flow. Do not synthesize `MiCoSocSim.v` or `simWorkspace` simulation RTL for hardware-cost claims.

For BNCFUv2 block-count exploration, prefer:

```bash
python3 tools/bncfu_v2_block_explore.py \
  --block-counts 64,128,256 \
  --total-outputs 229 \
  --with-yosys
```

The script derives each tail count from `total_outputs % block_count` and keeps the software macros aligned with the hardware SoC parameter. Use `--set-full-count-each-block` only to compare against the older software scheduling policy.

## Acceptance Checks

Before calling a CFU change done, report:

- Exact hardware args and software flags.
- The scalar/reference comparison result.
- The SoC simulation command and pass/fail output.
- Cycle/profile numbers if the task involved speed.
- Yosys or vendor synthesis numbers if the task involved hardware cost.
- Any unvalidated assumptions, such as missing synthesis, unavailable simulator, or skipped long benchmark.

## Failure Triage

- If output is all zero or stale, inspect command acceptance, response timing, RF read latency, and memory visibility/fences before retuning math.
- If simulation hangs, check `io.bus.cmd.ready`, one-response-per-command behavior, and FSM exit conditions.
- If a hang appears immediately after `rdcycle`/`rdtime`, check that `zicntr` counter hardware was generated, such as by adding `--with-rdtime`.
- If a custom instruction is illegal, check CSR CFU enable, decode encodings, `withCfu`, and RVLS/custom instruction settings.
- If hardware and software disagree about operands, inspect whether raw instruction fields carry register IDs while `inputs` carry register values.
- If benchmark numbers are noisy or impossible, isolate build dirs and verify the emitted ELF path before comparing cycles.
