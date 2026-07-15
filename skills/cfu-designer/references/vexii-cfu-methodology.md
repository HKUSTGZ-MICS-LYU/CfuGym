# VexiiRiscv CFU Methodology

## Local Architecture

CfuGym has two CFU/custom-instruction styles:

- In-core execution plugins, such as `vexiiriscv.execute.MiCoPlugin` and `BitNetPlugin`, extend an execute lane through `ExecutionUnitElementSimple`.
- Stream-bus CFUs use `vexiiriscv.execute.cfu.CfuPlugin` as the CPU-side decoder/bridge and a separate accelerator component connected through `CfuBus`.

Prefer the stream-bus CFU route for stateful accelerators, memory-backed vector registers, multi-cycle datapaths, or anything that should be reusable at the SoC level.

Primary files:

- `src/main/scala/vexiiriscv/execute/cfu/CfuBus.scala`: command/response bundle definition.
- `src/main/scala/vexiiriscv/execute/cfu/CfuPlugin.scala`: CPU custom-instruction decode, lane freeze, CSR enable/status, and writeback.
- `src/main/scala/vexiiriscv/soc/mico/VpuCfu.scala`: compact VPU-style stream CFU example.
- `src/main/scala/vexiiriscv/soc/mico/BitNetCfu.scala`: richer stateful CFU with vector RF, optional synchronous RF, pipelined compute, quantization units, TileLink loads, and FSM.
- `src/main/scala/vexiiriscv/soc/mico/TilelinkVpuCfuFiber.scala` and `TilelinkBitNetCfuFiber.scala`: TileLink wrappers that instantiate the accelerator and expose `cfuBus`.
- `src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala`: SoC integration; `Fiber patch` connects `cpu.logic.core.host[CfuPlugin].logic.bus` to the selected CFU fiber.
- `src/main/scala/vexiiriscv/soc/mico/MiCoSocParam.scala`: CLI/parameter wiring and legalization.
- `sw/MiCo-Lib/targets/vexii_soc.mk`: software source and compile-flag wiring for `TARGET=vexii_soc`.
- `sw/MiCo-Lib/targets/vexii_soc/cfu/`: CFU software kernels and custom instruction usage.

## CFU Bus Contract

`CfuBusParameter` controls visible command/response shape:

- `CFU_FUNCTION_ID_W`: width of `cmd.function_id`, usually enough for `func3`.
- `CFU_INPUTS`, `CFU_INPUT_DATA_W`: scalar operands forwarded from architectural registers or immediate.
- `CFU_OUTPUTS`, `CFU_OUTPUT_DATA_W`: current `CfuPlugin` expects one output.
- `CFU_RAW_INSN_W`: set to 32 when the accelerator needs raw `rd`/`rs1`/`rs2`/`func7` fields.
- `CFU_CFU_ID_W`, `CFU_STATE_INDEX_NUM`: optional CSR-selected CFU/state indices.
- `CFU_WITH_STATUS`: enables response status flags written back through the plugin CSR path.

`CfuPlugin` maps custom instruction encodings to bus commands. It fills:

- `cmd.function_id` from configured instruction bit ranges.
- `cmd.inputs(0)` from RS1.
- `cmd.inputs(1)` from RS2 or sign-extended I-immediate, depending on `Input2Kind`.
- `cmd.raw_insn` from the decoded 32-bit instruction.

In the current CfuGym `Param.scala`, enabling `withCfu` installs a default `CfuPlugin` on `early0` with two instruction families:

- custom0 (`opcode 0x0B`, pattern `-------------------------0001011`) uses `func3` bits `14 downto 12` as `function_id` and takes `rs2` from the integer register file.
- custom1-like (`opcode 0x2B`, pattern `-------------------------0101011`) also uses `func3` as `function_id` and supplies input 2 from the sign-extended I-immediate.

Use that existing path before modifying CPU decode.

The accelerator must:

- Drive safe defaults for `io.bus.rsp.valid`, `io.bus.rsp.outputs(0)`, and optional `status`.
- Assert `io.bus.cmd.ready` only when it can accept a new command.
- Return exactly one response for each accepted command.
- Preserve ordering unless it implements and uses request/response IDs.

## Integration Pattern

For a new stream CFU:

1. Define a parameter case class near the accelerator. Include widths, register depths, pipeline toggles, and feature toggles.
2. Implement `class MyCfu(cfuParam: CfuBusParameter, busParam: BusParameter, p: MyCfuParameter) extends Component`.
3. Add `io.bus = slave(CfuBus(cfuParam))` and, if memory is needed, `io.dBus = master(tilelink.Bus(busParam))`.
4. Decode `function_id` plus raw instruction fields in one `decode` area. Keep the decode comments synced with software intrinsics.
5. Use an FSM for long operations: `IDLE` accepts commands, work states drive datapath/memory, final states raise `rsp.valid`.
6. If a datapath needs staging, use `Node`, `Payload`, `StageLink`, and `Builder` like `BitNetCfu.compute`.
7. Wrap the component in a `Tilelink*Fiber` that creates `Node.down()`, forces M2S parameters, instantiates `CfuBus`, and connects the accelerator.
8. Add SoC param fields/options and legalize incompatible configurations.
9. In `MiCoSoc`, instantiate the fiber, connect its TileLink node to `mainBus`, and patch-connect `cfuBus` to the CPU `CfuPlugin` bus.
10. Add software intrinsics and tests under `sw/MiCo-Lib/targets/vexii_soc/...`; wire them in `vexii_soc.mk`.

## Vexii Parameter and CLI Lessons

Separate CPU features from SoC accelerator selection:

- `p.vexii` owns CPU/plugin features such as `withCfu`, ISA extensions, counters, caches, lanes, decoders, bypass, and register-file mode.
- `MiCoSocParam` owns SoC-level accelerator choices such as MiCo VPU, BitNet CFU, SIMD8 example CFU, TileLink peripherals, sparse memory, and CLI legalization.
- A SoC option that instantiates a CFU must also set `vexii.withCfu = true`; otherwise the CPU-side `CfuPlugin` bus is absent even if the accelerator component exists.
- If benchmark code reads `rdcycle`, ensure the generated CPU includes the counter CSR path. In this repo, `--with-rdtime` adds `zicntr`, which makes `Param.withRdTime` true and enables the performance counter/rdtime plumbing. Without it, `rdcycle` can assemble but trap or stall at runtime.
- `--performance-counters <n>` also adds `zicntr`/`zihpm` and is appropriate when extra HPM counters are part of the benchmark. For a minimal cycle counter, prefer `--with-rdtime`.
- Cache, dual-issue, bypass, regfile, and sparse-memory CLI flags change measured speed. Report the full SoC CLI when publishing cycle numbers.

Parameter update pattern for a new single-owner CFU option:

```scala
var useMyCfu = false

opt[Unit]("mico-my-cfu") action { (_, _) =>
  useMyCfu = true
  vexii.withCfu = true
}

if(useMyCfu) {
  require(!useMiCoVpu && !useBitNetCfu, "single CPU CFU bus: enable only one CFU owner")
}
```

In `MiCoSoc`, instantiate the accelerator outside the patch when it is a normal component, then connect it inside the build-time `Fiber patch`:

```scala
val myCfu = p.useMyCfu generate new MyCfu(MyCfu.busParameter(p.vexii.xlen))

val cfuConnect = p.vexii.withCfu generate (Fiber patch new Area {
  val cpuCfuBus = cpu.logic.core.host[CfuPlugin].logic.bus
  if(p.useMyCfu) myCfu.io.bus << cpuCfuBus
})
```

Keep software `MARCH` and hardware ISA options consistent. Modern toolchains may require explicit ISA extensions such as `zicsr`/`zifencei` in `MARCH`, while Vexii hardware generation uses its own CLI/`Param` ISA state.

## Common Pitfalls

- Forgetting `vexii.withCfu = true` when a SoC option enables the accelerator.
- Connecting more than one accelerator to the single CPU CFU bus without arbitration.
- Timing with `rdcycle` but forgetting to enable the CPU counter feature, such as `--with-rdtime`/`zicntr`, in simulation.
- Treating software `MARCH` as proof that the generated CPU has the matching CSR/plugin hardware.
- Using raw `rs1`/`rs2` register IDs in hardware but passing values in software, or vice versa.
- Advancing internal cursors on `valid` instead of command/response fire conditions.
- Missing a fence when software packs data in memory immediately before CFU loads it.
- Adding duplicate payloads for values already carried by the stage/lane.
