# Minimal Example: SIMD Int8 Sum CFU

This example shows the smallest useful vector-to-scalar CFU: sum four signed int8 lanes packed in a 32-bit register and return a scalar int32 result.

Use it when a user asks for a first CFU, a sanity-check custom instruction, or a simple example of the optimization loop.

## Workload Analysis

Scalar hot loop, assuming the int8 stream is already packed as four lanes per 32-bit word:

```c
uint32_t packed = x[word];
sum += (int32_t)(int8_t)(packed >> 0);
sum += (int32_t)(int8_t)(packed >> 8);
sum += (int32_t)(int8_t)(packed >> 16);
sum += (int32_t)(int8_t)(packed >> 24);
```

Pattern: vector-to-scalar reduction. One CFU instruction replaces four byte extracts/sign-extends and three scalar additions for each packed word.

Expected benefit:

- Best case: close to one CFU instruction per four elements plus loop overhead.
- Remaining overhead: loads, loop branch, tail handling, and CFU response latency.
- End-to-end speedup depends on how much total runtime is in this loop.

## ISA Entry

Example custom0 allocation:

| Field | Value | Meaning |
| --- | --- | --- |
| opcode | `0x0B` | custom0 |
| `func3` | `0` | `SIMD8_SUM4` |
| `func7` | `0` | reserved |
| `rs1` | packed input word | four signed int8 lanes |
| `rs2` | unused | encode as `x0` |
| `rd` | scalar sum | signed int32 result |

This is intentionally stateless, so it does not need `CfuPlugin` CSR state beyond normal CFU enable.

In CfuGym, `Param.scala` already adds `CfuPlugin` when `withCfu` is enabled. Its default custom0 encoding matches `instruction = M"-------------------------0001011"` and `functionId = List(14 downto 12)`, so this example uses `func3=0` without adding a new CPU-side decoder. The accelerator still checks `io.bus.cmd.function_id` so the hardware decode and software intrinsic remain explicit.

## Hardware

The complete starter component is in `assets/examples/simd8-sum-cfu/Simd8SumCfu.scala`.

Core datapath:

```scala
val lanes = io.bus.cmd.inputs(0).subdivideIn(8 bits).map(_.asSInt.resize(16))
val sum = Vec(lanes).reduceBalancedTree(_ +^ _).resize(32)
io.bus.rsp.outputs(0) := sum.asBits
```

Integration options:

- For the reusable MiCo-style integration, wrap it as a `DirectCfuSpec` and select it through the shared CFU container/fiber path. This keeps direct ALU-style CFUs and TileLink-backed CFUs behind the same SoC selection abstraction.
- Add a `TilelinkCfuSpec` only if the CFU needs memory. This stateless example does not need TileLink.
- Keep the direct CFU command/response data width equal to CPU XLEN. For the default MiCo configuration this is 32 bits, so the software packs four int8 lanes per instruction.
- Keep the instruction encoding in the C intrinsic and Scala decode comments synchronized.

Minimal patch shape:

```scala
case class Simd8SumCfuSpec() extends DirectCfuSpec {
  override def cfuBusParameter(xlen: Int) = Simd8SumCfu.busParameter(xlen)

  override def build(cfuParam: CfuBusParameter, cfuBus: CfuBus) = new Area {
    val cfu = new Simd8SumCfu(cfuParam)
    cfu.io.bus <> cfuBus
  }
}
```

For production integration, expose this through a SoC parameter and avoid colliding with other users of the single CPU CFU bus.

Because this CFU is a pure register-input/register-output reduction, use the same zero-latency stream style as `CfuTest`:

```scala
io.bus.rsp.arbitrationFrom(io.bus.cmd)
io.bus.rsp.response_id := io.bus.cmd.request_id
io.bus.rsp.outputs(0) := sum.asBits
```

Use registered responses for multi-cycle or memory-backed CFUs, not for this direct ALU-style example.

## C Self-Validation and Speed Measurement

The complete C example is in `assets/examples/simd8-sum-cfu/simd8_sum_test.c`.
The standalone counter helper is in `assets/examples/simd8-sum-cfu/cfu_counter.h`.

It performs:

- Scalar reference sum over packed 4x int8 words.
- CFU sum over one packed 4x int8 word per instruction.
- Correctness comparison.
- Simulator-friendly pass/fail return code.
- Measured scalar and CFU timings using the MiCo-free `cfu_counter_read()` helper.
- Parseable UART lines:
  - `SIMD8_SUM PASS scalar=<sum> cfu=<sum>`
  - `SIMD8_SUM_PROFILE scalar=<ticks> cfu=<ticks> speedup_x100=<ratio>`

When adapting to CfuGym, wire the test into `sw/tests/` or the relevant MiCo-Lib target and run it through the Vexii SoC simulator. If the CFU reads CPU-written memory in a later version, add a fence before the CFU load path.

Prefer a local, dependency-free counter wrapper for example code:

```c
uint32_t start_time = cfu_counter_read();
kernel();
long elapsed = cfu_counter_elapsed(start_time, cfu_counter_read());
```

On RISC-V, `cfu_counter_read()` uses `rdcycle` directly. That keeps the example usable in the original VexiiRiscv platform without `profile.h`, `MiCo_time()`, or MiCo-Lib target headers. In Vexii simulation, pair this with a CPU configuration that implements `zicntr`; in this repo's CLI that means adding `--with-rdtime`. Without that hardware option, a `rdcycle` instruction can compile but still trap or stall at runtime because the counter CSR path is not present. See `references/vexii-cfu-methodology.md` for the CPU-parameter vs SoC-parameter checklist.

Validated CfuGym command pair:

```bash
make TARGET=vexii_soc MAIN=tests/simd8_sum_cfu_test BUILD=build_simd8_sum_cfu_direct_onchip2 MARCH=rv32imc_zicsr_zifencei OPT=cfu compile
sbt "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/simd8_sum_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-simd8-sum-cfu"
```

Validated output:

```text
SIMD8_SUM PASS scalar=-4 cfu=-4
SIMD8_SUM_PROFILE scalar=1183632 cfu=661522 speedup_x100=178
```

That is a measured kernel speedup of `1.78x`. Treat it as a smoke-test reference point, not a portable promise: compiler flags, CPU configuration, CFU latency, memory latency, loop shape, and packing strategy all affect the result.

## Optimization Iteration

If speedup is weak:

- Increase work per instruction: sum 8 or 16 lanes using CFU vector registers or wider CPU XLEN.
- Fuse more work: sum absolute values, dot product, bias add, threshold, or quantization.
- Reduce software overhead: unroll the CFU loop, align loads, and remove redundant packing.
- Pipeline hardware only if CFU latency stalls the CPU and the instruction stream has enough independent work.
- Move from register operands to CFU-owned vector registers only when load reuse amortizes setup cost.
