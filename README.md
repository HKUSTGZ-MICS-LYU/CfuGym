# CfuGym

CfuGym is a VexiiRiscv-based hardware/software co-design workspace for building, integrating, and validating custom function units (CFUs), custom RISC-V instructions, and mixed-precision accelerators. It keeps the original VexiiRiscv and MiCo context while adding practical flows for SoC generation, bare-metal software validation, simulation, profiling, and hardware-cost exploration.

## Install the CFU Designer Skill

This repo includes a Codex skill for CFU design at `skills/cfu-designer`. Install it into your local Codex skills directory with:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" skills/cfu-designer
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  skills/cfu-designer/ "${CODEX_HOME:-$HOME/.codex}/skills/cfu-designer/"
```

Restart Codex after installation so the new skill is discovered. Use it in future prompts with:

```text
Use $cfu-designer to design, validate, and estimate hardware cost for a CfuGym CFU.
```

## BNCFUv2 Block Exploration

BNCFUv2 has a focused HW/SW exploration wrapper for the block-count path used by `sw/tests/bncfu_compare_test.c`:

```bash
python3 tools/bncfu_v2_block_explore.py \
  --block-counts 64,128,256 \
  --total-outputs 229 \
  --with-yosys
```

The wrapper builds the bare-metal compare ELF, runs `MiCoSocSim` with the matching `--bitnet-cfu-v2-block-count`, parses `BNCFU_COMPARE_PROFILE`, optionally generates standalone `BitNetCfuV2` RTL, runs the Yosys cost helper, and writes `summary.md`/`summary.csv` under `benchmark_results/bncfu_v2_block_explore` by default.

## Original VexiiRiscv + MiCo Context

### VexiiRiscv + MiCo

VexiiRiscv-MiCo is a mixed-precision computing extension plugin for VexiiRiscv.

You can find the MiCo plugin in scala class `vexiiriscv.execute.MiCoPlugin`.

### MiCo Plugin

The MiCo plugin provides 10 custom insturctions, focusing on signed dot product operations between two 32/64-bit vectors. Each of the packed vectors can contain INT8/INT4/INT2/INT1 data.

#### Usage

To add the `MiCoPlugin` into `Param.scala`, you need to find the lines about the `lane0`, and add one more line for `MiCoPlugin`:
```scala
val early0 = new LaneLayer("early0", lane0, priority = 0)
plugins += lane0
plugins += new SrcPlugin(early0, executeAt = 0, relaxedRs = relaxedSrc)
plugins += new MiCoPlugin(early0) // Add MiCoPlugin Here!
plugins += new IntAluPlugin(early0, formatAt = 0)
plugins += shifter(early0, formatAt = relaxedShift.toInt)
plugins += new IntFormatPlugin(lane0)
plugins += new BranchPlugin(layer=early0, aluAt=0, jumpAt=relaxedBranchtoInt, wbAt=0)
```

Then you can generate/simulate VexiiRiscv with MiCo Plugin, check the VexiiRiscv guides below.

Due to the custom instructions added, please turn off RVLS when simulating (`--no-rvls-check`).

# VexiiRiscv

VexiiRiscv (Vex2Risc5) is the successor of VexRiscv. Work in progress, here are its currently implemented features :

- RV32/64 I[M][A][F][D][C][S][U][B]
- Up to 5.24 coremark/MHz 2.50 dhystone/MHz
- In-order execution
- early [late-alu]
- single/dual issue (can be asymmetric)
- BTB, GShare, RAS branch prediction
- cacheless fetch/load/store
- Optional I$, D$
- Optional SV32/SV39 MMU
- Can run linux / buildroot / Debian
- Pipeline visualisation in simulation via Konata
- Lock step simulation via RVLS and Spike
- AXI4, Wishbone, Tilelink memory busses (RVA is not available in some configs, see the RTD doc SoC main page)
- ... and many other things

Here is a demonstration of a quad core VexiiRiscv running debian on FPGA : https://youtu.be/dR_jqS13D2c?t=112

Overall the goal is to have a design which can stretch (through configuration) from Cortex M0 up to a Cortex A53 and potentially beyond.

Here is the online documentation : 

- https://spinalhdl.github.io/VexiiRiscv-RTD/master/VexiiRiscv/Introduction/#
- https://spinalhdl.github.io/VexiiRiscv-RTD/master/VexiiRiscv/HowToUse/index.html

Here is the VexiiRiscv's scala doc (auto-generated from the source code) :

- https://spinalhdl.github.io/VexiiRiscv/doc/vexiiriscv/index.html

A roadmap is available here : 

- https://github.com/SpinalHDL/VexiiRiscv/issues/1

# TL;DR Getting started

The quickest way for getting started is to pull the Docker image with all the dependencies installed

Please refer to the self contained tutorial for a comprehensive step by step instruction manual with
screenshots: https://spinalhdl.github.io/VexiiRiscv-RTD/master/VexiiRiscv/Tutorial/index.html

After running the generation you'll find a file named "VexiiRiscv.v" in the root
of the repository folder, which you can drag into your Quartus or whatever.

We decided to not start covering FPGA boards because there's just too many, so it's up to you
to define your pin configuration for your specific FPGA board

If you want to know what else you can do with sbt, please refer to the complete documentation.

# Rebuild the Docker container

In case you wanna rebuild leviathan's Docker container you can run

    docker build . -f docker/Dockerfile -t vexiiriscv --progress=plain
