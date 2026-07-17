# U8 Weighted Sum/Average CFU Experiment

Use this example when the workload has multiple operations over the same vectors and an intermediate vector result can be consumed by the CFU without a memory round trip.

## Workload

Reference kernel:

```c
uint32_t sum = 0;
for(int i = 0; i < bytes; ++i) {
    uint16_t product = (uint16_t)((uint32_t)x[i] * (uint32_t)w[i]);
    sum += product;
}
uint32_t avg = sum / bytes;
```

The natural optimization is not just SIMD multiply. The product vector is an intermediate that is immediately reduced, so the CFU should keep it in resident state instead of writing `uint16_t products[]` back to memory.

The first implementation used operation-local `vecX`, `vecW`, and widened `vecP` registers. The current implementation has been migrated toward the VPU pattern: `X` and `W` live in a small universal vector RF, while products are widened only inside the compute slice and accumulated immediately. Keep a widened product RF only if later operations need the product vector itself.

## ISA Contract

The working CfuGym experiment uses custom0 `func3` commands:

| `func3` | Command | Operand | Result |
| ---: | --- | --- | --- |
| `0` | reset | none | status |
| `1` | load X vector | `rs1` address | status |
| `2` | load W vector | `rs1` address | status |
| `3` | multiply resident vectors | none | status |
| `4` | sum resident products | none | sum |
| `5` | average resident products | none | sum shifted by log2(vector bytes) |

Software hides the raw instruction words behind intrinsics such as `u8_wavg_load_x`, `u8_wavg_load_w`, `u8_wavg_mul`, `u8_wavg_sum`, and `u8_wavg_avg`.

## Hardware Shape

The CfuGym implementation is `U8WeightedAvgCfu`:

- TileLink-backed `CfuBus` component selected through `U8WeightedAvgCfuSpec`.
- Shared `CfuLsu` configured with `withLoad = true`, `withStore = false`.
- RF slot 0 holds loaded `X`; RF slot 1 holds loaded `W`.
- `rfRam = false` stores the RF as `Vec(Reg(Bits(vlen bits)), regDepth)`. This is the validated default for the integrated SoC simulation.
- `rfRam = true` elaborates banked `Mem(Bits(xlen bits), regDepth)` storage, one bank per TileLink beat. Treat this as experimental until it passes integrated SoC sim.
- `maclen` controls the compute slice width. Each slice multiplies `maclen / 8` unsigned byte lanes, widens each product locally, reduces the slice, and accumulates into a scalar sum.
- `computePipe` enables a one-stage manual pipeline for slice partials. The software ISA is unchanged by this choice.
- `avg` shifts by `log2(vectorBytes)`, so the vector byte count must be a power of two.
- Burst loads issue one TileLink transfer with `size = vectorBytes`, so software buffers must be aligned to `vectorBytes`.

Useful SoC parameters:

```text
--mico-u8-wavg-cfu
--u8-wavg-cfu-len <bits>
--u8-wavg-cfu-bus-width <bits>
--u8-wavg-cfu-maclen <bits>
--u8-wavg-cfu-reg-depth <entries>
--u8-wavg-cfu-rf-reg
--u8-wavg-cfu-rf-ram
--u8-wavg-cfu-no-pipe
--u8-wavg-cfu-advanced-mem
--u8-wavg-cfu-burst-mem
```

The SoC parameter layer must also set `vexii.withCfu = true` and raise `vexii.lsuMemDataWidthMin` to at least the CFU TileLink bus width.

## Validation Commands

Generate the SoC from the normal MiCo flow:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocGen --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu"
```

Build the firmware:

```bash
cd sw
make TARGET=vexii_soc MAIN=tests/u8_wavg_cfu_test BUILD=build_u8_wavg_cfu MARCH=rv32imc_zicsr_zifencei OPT=cfu compile
```

Run integrated SoC simulation:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/u8_wavg_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu"
```

Passing output from the VPU-style RF-register default (`vlen=256`, `xlen=32`, `maclen=32`, `computePipe=true`):

```text
U8_WAVG PASS sum=16483840 avg=16097 checksum=2370482432
U8_WAVG_PROFILE fused=1181972 two_pass=1903825 cfu=320210 speedup_fused_x100=369 speedup_twopass_x100=594
```

Interpretation:

- CFU speedup vs direct scalar fused loop: `3.69x`.
- CFU speedup vs scalar two-pass product-buffer loop: `5.94x`.
- The two-pass comparison captures the value of keeping the product vector resident inside the CFU.

Useful validated variants:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/u8_wavg_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu --u8-wavg-cfu-no-pipe"
```

```text
U8_WAVG PASS sum=16483840 avg=16097 checksum=2370482432
U8_WAVG_PROFILE fused=1181972 two_pass=1903825 cfu=318162 speedup_fused_x100=371 speedup_twopass_x100=598
```

The no-pipe version is slightly faster in this functional simulator because it avoids one pipeline bubble and there is no timing/Fmax model.

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/u8_wavg_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu --u8-wavg-cfu-maclen 64"
```

```text
U8_WAVG PASS sum=16483840 avg=16097 checksum=2370482432
U8_WAVG_PROFILE fused=1181972 two_pass=1903825 cfu=312018 speedup_fused_x100=378 speedup_twopass_x100=610
```

Larger `maclen` improves this test because it reduces the number of compute slices per loaded vector. The area/timing tradeoff should be checked with synthesis before keeping the wider datapath.

With burst loads enabled and `vectorBytes`-aligned buffers:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/u8_wavg_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu --u8-wavg-cfu-burst-mem"
```

```text
U8_WAVG PASS sum=16483840 avg=16097 checksum=2370482432
U8_WAVG_PROFILE fused=1181972 two_pass=1903825 cfu=295634 speedup_fused_x100=399 speedup_twopass_x100=643
```

Burst mode improved the RF-register default from `3.69x` to `3.99x` vs the scalar fused loop, and from `5.94x` to `6.43x` vs the scalar two-pass loop.

The banked async RF mode is available for exploration:

```bash
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/u8_wavg_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu --u8-wavg-cfu-rf-ram"
```

In the current repo state, this mode elaborates but integrated SoC simulation trips a CPU stream-persistence assertion around time 10000 before firmware produces a profile line. Do not mark it validated until the banked-RF read/write timing and stream interactions are debugged. A synchronous RF backend, like the `BitNetCfu` `rfSync` style, is the safer next memory-backed implementation target.

## Hardware Cost

Use generated SoC RTL, not simulation RTL:

```bash
python3 skills/cfu-designer/scripts/yosys_cost_report.py \
  MiCoSoc.v \
  --top U8WeightedAvgCfu \
  --out-dir synth_runs/yosys/u8_wavg_cfu_vpu_rf \
  --flatten
```

Initial pre-RF generic Yosys result:

```text
top=U8WeightedAvgCfu cells=23239 wire_bits=26359 memories=0 memory_bits=0
```

VPU-style RF-register default result:

```text
top=U8WeightedAvgCfu cells=11726 wire_bits=13553 memories=0 memory_bits=0
```

The RF-register sliced-reduction design roughly halves this generic cost estimate versus the original resident-product-vector design, at the price of more compute cycles. Top cell types remain dominated by mux and boolean logic, with no inferred memories. Treat this as a relative RTL-cost estimate only; use the same generated RTL style, top, flatten mode, and Yosys mode when comparing variants.

## Lessons

- A multi-operation accelerator should optimize data lifetime, not only per-lane ALU count.
- Resident intermediates help when a vector-to-vector step feeds an immediate vector-to-scalar reduction, but the intermediate does not need a visible vector register unless software needs to read or reuse it.
- For a reusable VPU-style CFU, keep input vectors in generic RF slots and make widened products operation-local unless the product vector is part of the ISA.
- Separate `vlen`, `xlen`, `maclen`, `regDepth`, RF backend, and pipeline knobs. They affect different parts of the design and should be sweepable independently.
- Prefer RF-register mode first for correctness and debug visibility. Move to banked or synchronous RF only after the software contract and arithmetic are stable.
- Load-only TileLink CFUs may omit `a.data`, `a.mask`, `a.corrupt`, `d.denied`, or `d.corrupt`; shared LSU logic should guard optional channel fields.
- A power-of-two vector byte count makes average cheap. Non-power-of-two averages need a divider, reciprocal multiply, or software finalization.
- Burst mode needs an explicit alignment contract. A randomized test can fail while zero/max patterns pass if the burst base address is not aligned to the burst byte count.
- The reduction tree and unrolled multipliers dominate generic cost; if cost is too high, reduce lane count, accumulate across cycles, or split multiply and reduction over multiple commands.
- Always compare against both the best scalar fused loop and the memory-temporary loop that the CFU intentionally avoids.
