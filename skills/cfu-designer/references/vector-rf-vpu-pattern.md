# VPU-Style Vector RF and Parameterized Datapath Pattern

Use this pattern when a CFU becomes more than a one-off instruction and needs reusable vector state, dedicated CFU memory, or configurable performance/area tradeoffs.

## When To Use It

Prefer a VPU-style CFU over fixed operation-local registers when:

- Multiple operations reuse vectors already loaded into the CFU.
- Software should issue `load`, `config`, and `compute` commands against CFU vector-register IDs.
- The workload benefits from a flexible `vlen`, TileLink bus width, compute slice width, or pipeline toggle.
- Intermediate vectors should stay inside the CFU instead of being written to main memory.
- You need to sweep area/performance by changing RF backend, `maclen`, or pipeline depth.

Keep fixed local registers for very small examples where each command owns a dedicated state layout and the RF abstraction would only add decode/control cost.

## Local Examples

`VpuCfu` is the compact pattern:

- `VpuCfuParameter(vlen, xlen, maclen, vregs, noWaitCompute, rfRam, computePipe)` controls vector width, bus/datapath width, number of vector registers, RF backend, and optional pipeline.
- `nLoad = vlen / xlen` is the number of TileLink beats per vector load.
- `nCompute = vlen / maclen` is the number of compute slices per vector operation.
- `rfRam = true` stores each vector register as banked `Mem(Bits(xlen bits), wordCount = vregs)`, one bank per load beat.
- `rfRam = false` stores the RF as `Vec(Reg(Bits(vlen bits)), vregs)`, which is easier to read but costs flops and wide muxes.
- Raw `rs1`/`rs2` fields are reused as CFU vector-register IDs and element-width config fields.
- `computePipe` inserts a `Node`/`Payload`/`StageLink` boundary between extraction and compute.

`BitNetCfu` extends the same idea:

- It adds `rfSync`, giving banked RAM a synchronous-read option with explicit read command signals.
- It keeps per-register cursors in `vecOffsets`.
- It uses raw instruction fields for RF IDs and `func7` chunk selection.
- It keeps optional features behind parameters such as Q2T/Q8 paths and pipeline toggles.

## RF Backend Shape

A practical universal RF backend has these parameters:

```scala
case class VectorRfParameter(
  vlen: Int,
  xlen: Int,
  regDepth: Int,
  rfRam: Boolean = true,
  rfSync: Boolean = false
) {
  def beats = vlen / xlen
  def regSelWidth = log2Up(regDepth) max 1
}
```

For banked memory:

```scala
val banks = Seq.fill(beats)(Mem(Bits(xlen bits), wordCount = regDepth))
```

This layout writes one `xlen` beat into one bank during a load and reconstructs a full vector by concatenating all banks on read. It maps naturally to a TileLink LSU where `source` or an internal beat counter identifies the bank.

For register storage:

```scala
val regs = Vec(Reg(Bits(vlen bits)) init(0), regDepth)
```

This layout is good for small `vlen` or debugging because full-vector read and partial-beat write are simple. It is usually more expensive for large `vlen`/`regDepth`.

Start with register storage when migrating a fixed CFU into a VPU-style RF. It keeps read latency simple, makes waveform/debug inspection direct, and avoids conflating arithmetic bugs with RF timing bugs. After the ISA and arithmetic pass integrated SoC simulation, move to banked memory or synchronous RF for area.

Async banked `Mem.readAsync` can elaborate while still exposing simulator/assertion problems once connected to the full SoC. If an async RF backend trips CPU stream-persistence or bus assertions before firmware starts, treat the RF backend as unvalidated rather than assuming the arithmetic is wrong. A synchronous RF backend with an explicit read-command phase is the more robust next step for large vector memories.

## ISA Contract

Use commands like:

| Command | Typical encoding | Meaning |
| --- | --- | --- |
| `config` | `func3`, raw `rs1`/`rs2`, `func7` | Set element widths, quant type, mode, cursor reset, or pipeline mode. |
| `load` | `rs1` value = address, raw `rs2` or `rd` = vector register ID | Load memory into a CFU vector register. |
| `store` | `rs1` value = address, raw register field = vector register ID | Store a vector register to memory when the result must escape. |
| `compute` | raw `rs1`/`rs2` = vector register IDs | Run a vector op, dot/reduce, transform, or stateful sequence. |
| `read/status` | optional | Return scalar accumulator, status, cursor, or debug data. |

Keep a written mapping from raw register fields to RF IDs. In Vexii CFU commands, `inputs(0)` and `inputs(1)` carry architectural register values, while `raw_insn` carries the original register IDs.

## Datapath Parameters

Use separate parameters for independent design axes:

- `vlen`: architectural CFU vector-register width in bits.
- `xlen` or `busWidth`: TileLink beat width and CFU load/store granularity.
- `elemWidth`: logical element width, often configurable at runtime for int8/int4/int2/int1.
- `maclen`: number of vector bits consumed by one compute slice.
- `accWidth`: scalar accumulator/result width.
- `regDepth`/`vregs`: number of vector registers.
- `rfRam`/`rfSync`: RF implementation and read latency.
- `computePipe`: whether extraction and compute are separated by `StageLink`.
- `noWaitCompute`: whether the first compute slice can be launched in the accept cycle.

Add compile-time checks:

```scala
require(vlen % xlen == 0)
require(vlen % maclen == 0)
require(maclen % elemWidth == 0)
require(regDepth >= 2)
```

If the CFU supports multiple runtime element widths, check all supported widths at elaboration and guard illegal runtime config in hardware.

## Pipelining Pattern

Use Spinal `Node`/`Payload`/`StageLink` when compute timing, RF latency, or slice extraction should be separable:

```scala
val stages = Array.fill(if(p.computePipe) 2 else 1)(Node())
val extractStage = stages(0)
val computeStage = stages.last

val SEL = Payload(Bool())
val DONE = Payload(Bool())
val OPA = Payload(Bits(p.maclen bits))
val OPB = Payload(Bits(p.maclen bits))

new extractStage.Area {
  OPA := vecA(offset, p.maclen bits)
  OPB := vecB(offset, p.maclen bits)
  SEL := computeSel
  DONE := doneNow
}

new computeStage.Area {
  when(SEL) {
    acc := acc + dot(OPA, OPB)
  }
}

if(p.computePipe) Builder(Seq(StageLink(stages(0), stages(1))))
```

Delay control signals such as `DONE`, register IDs, mode bits, and accumulator enables through payloads. Do not advance cursors from raw `valid`; advance them from the same selected/fire condition that launches a slice.

For synchronous RF (`rfSync`), add a read-command phase before the extraction stage, or treat the first pipeline stage as the RF-read latency stage. `BitNetCfu` uses `vecReadSyncCmd` and separate read-data ports for this reason.

## Handling Widening Intermediates

Do not force a widened intermediate into a fixed `vlen` vector unless truncation is explicitly part of the ISA.

For `int8 * int8 -> int16`, choose one of these contracts:

- **Immediate reduce**: keep input RF entries at `vlen`, compute `maclen` bits at a time, widen products internally, and accumulate into `accWidth`. This is best for weighted sum, dot product, and average because no product vector needs to escape.
- **Dedicated widened RF**: add a separate product RF with `2 * vlen` bits per logical vector, or bank it as twice as many `xlen` banks. This is useful when later vector-to-vector operations consume the products.
- **Paired normal registers**: store the widened product vector across two normal vector registers and define the low/high lane packing in the ISA.
- **Half-lane operation**: process only half the input lanes so products fit back into one `vlen` register. Use this only when the software contract accepts half throughput per command.

For the weighted-sum/average experiment, the VPU-aligned design should normally use immediate reduce:

1. Load `X` into RF slot 0.
2. Load `W` into RF slot 1.
3. Configure `elemWidth = 8`, `maclen = 32/64/...`, and unsigned or signed mode.
4. For each compute slice, extract `maclen / 8` lanes from both RF entries.
5. Multiply into 16-bit lane products locally and accumulate into `accWidth`.
6. Return `sum` or `avg` as scalar output.

This keeps the RF universal and avoids a one-off `vecP` shape. If the product vector must be visible, model that as a separate widened storage resource instead of silently dropping lanes.

## Migration Checklist

When migrating a fixed CFU to the VPU style:

1. Replace operation-specific vectors such as `vecX`, `vecW`, `vecP` with RF slots plus operation-local accumulator/state.
2. Move destination/source selection into raw instruction fields.
3. Split parameters into `vlen`, `xlen`, `maclen`, `regDepth`, RF backend, and pipeline toggles.
4. Use a shared LSU or RF load/store commands to move memory data into RF banks.
5. Make element width, signedness, and cursor behavior explicit config.
6. Decide where widened intermediates live: immediate accumulator, widened RF, paired registers, or half-lane command.
7. Validate scalar equivalence across RF backends, `maclen` values, pipeline on/off, and load modes.
8. Compare Yosys cost for each RF/backend/pipeline point before keeping a more general design.
9. Keep an implementation-status note for each backend: elaborated, CFU-unit tested, integrated SoC simulated, and synthesized. Do not collapse those into one "works" state.
