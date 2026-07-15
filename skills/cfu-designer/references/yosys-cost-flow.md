# Yosys Hardware Cost Flow

Use this flow when a CFU design needs a fast area/cell-count proxy before a full FPGA/vendor synthesis run.

## Purpose

Yosys is useful for early iteration:

- Compare two CFU RTL variants with the same top module and synthesis options.
- Catch obvious cost regressions from extra multipliers, memories, shifters, queues, or unrolled lanes.
- Produce a basic report for the optimization loop: correctness, speed, and hardware cost.

Do not use generic Yosys numbers as final FPGA LUT/FF/BRAM/Fmax claims. Use Vivado or the target vendor flow for final PPA.

## Script

The reusable helper is:

```bash
python3 skills/cfu-designer/scripts/yosys_cost_report.py <verilog-file-or-dir> --top <TopModule> --out-dir <out-dir>
```

Outputs:

- `run.ys`: generated Yosys script.
- `yosys.log`: full synthesis log.
- `stat.txt`: human-readable Yosys `stat`.
- `stat.json`: Yosys JSON stat when available.
- `netlist.json`: synthesized netlist JSON.
- `summary.json`: parsed metrics for automation.
- `summary.md`: compact report for humans.

Default mode is `generic`, which maps RTL through generic cells and ABC. It is best for relative CFU comparison when the final FPGA family is not fixed.

Other modes:

```bash
--mode xilinx
--mode ice40
--mode ecp5
```

Use these only when the target family matches the design question.

## Basic Flow

1. Generate Verilog from SpinalHDL for the smallest meaningful top:
   - Prefer a standalone CFU component for local datapath cost.
   - Use the full SoC top only when the question is integration overhead.
   - Use generator output, not simulation wrapper RTL.
2. Run Yosys with a stable top and mode.
3. Record total cells, memory bits, estimated LUT/LC if present, and top cell types.
4. Compare against a previous variant using the exact same script options.
5. If cost is high, inspect top cell types and change the hardware/software contract.

Do not use `MiCoSocSim.v` or Verilog from `simWorkspace` for synthesis cost claims. Simulation RTL can include testbench wrappers, simulator-only memories, probes, and naming/blackbox choices that do not match the generation flow. Use a Spinal `generateVerilog` flow instead.

## Generating RTL

For standalone CFU cost, add or use a small generator for the CFU component:

```scala
object MyCfuGen extends App {
  SpinalConfig(targetDirectory = "synth_runs/rtl/my_cfu")
    .generateVerilog(new MyCfu(MyCfu.busParameter(32)))
}
```

Then run:

```bash
python3 skills/cfu-designer/scripts/yosys_cost_report.py \
  synth_runs/rtl/my_cfu/MyCfu.v \
  --top MyCfu \
  --out-dir synth_runs/yosys/my_cfu
```

For integration-level cost, use the SoC generation entry point, not `MiCoSocSim`:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocGen \
  --with-rvc --with-rvf --with-rvm \
  --with-rdtime \
  --mico-simd8-sum-cfu"
```

Use the emitted `MiCoSoc.v` or the configured `targetDirectory` output:

```bash
python3 skills/cfu-designer/scripts/yosys_cost_report.py \
  MiCoSoc.v \
  --top MiCoSoc \
  --out-dir synth_runs/yosys/mico_soc_simd8 \
  --flatten
```

Full-SoC Yosys runs can be slow and can be dominated by CPU, memory, and bus logic. For CFU optimization, compare the standalone CFU top first; use the SoC top only to estimate integration overhead.

## What To Compare

Track these fields from `summary.json`:

- `metrics.num_cells`
- `metrics.num_wire_bits`
- `metrics.num_memories`
- `metrics.num_memory_bits`
- `metrics.estimated_lut4` when present
- top entries under `metrics.cell_counts`

When comparing CFUs, report a small table:

| Variant | Cells | Wire bits | Memory bits | Speedup | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| scalar baseline | 0 | 0 | 0 | 1.00x | no CFU |
| cfu-v1 | ... | ... | ... | ... | first working version |
| cfu-v2 | ... | ... | ... | ... | reduced lanes / shared datapath |

## Cost-Reduction Heuristics

Use the cell report to guide design changes:

- Many `$mul` cells: replace multipliers with shifts, muxes, add/sub, lookup tables, or low-bit selection.
- Many `$mem*` or high memory bits: shrink RF depth/width, bank only reused data, or keep data in software memory.
- Many `$mux` cells: simplify decode, reduce function count, avoid wide one-hot crossbars, or split control by operation.
- Many `$add`/`$alu` cells in reductions: use balanced trees, reduce lane count, or accumulate across cycles.
- Many flops after `--flatten`: check queues, response buffers, status registers, and pipeline stages that do not improve throughput or Fmax.
- Large top-level SoC cost: rerun on the standalone CFU top before blaming the accelerator.

Always keep correctness and benchmark numbers next to cost. A smaller CFU that pushes too much work back into software may lose the speedup.

## Reporting Discipline

For every Yosys cost claim, include:

- Verilog source path or generation command.
- Yosys command and mode.
- Top module.
- Whether hierarchy was flattened.
- Total cells and memory bits.
- Top cell types.
- Caveat that generic Yosys is a relative estimate unless a target-specific mode/vendor flow was used.
