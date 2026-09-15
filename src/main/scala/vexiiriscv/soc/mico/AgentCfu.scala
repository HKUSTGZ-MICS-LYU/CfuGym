// SPDX-License-Identifier: MIT
//
// AgentCfu - the checked-in, algorithm-neutral CFU design used by design agents.
//
// This file and AgentCfuFiber.scala are the ONLY hardware surface a design agent
// may change. Agent runs never edit this repository copy: each run seeds an
// isolated workspace copy under
//   agents/generated/runs/<run_id>/workspace/design/
// and the SoC build overlays that copy (see agents/README.md and
// project/AgentCfuSourceFilter.scala).
//
// The bus shell, the CfuResource/Instruction vocabulary, the vector RF, the
// response buffer, CfuLsu/FSM and the instruction dispatcher live in the
// read-only AgentCfuLib.scala. A whole-file rewrite of this file therefore cannot
// delete them, and this file stays limited to the design itself. Keep the public
// contract stable:
//   * AgentCfuParameter
//   * class AgentCfu(cfuParam, busParam, p)
//   * AgentCfuFiber.getCfuBusParameters / getM2sParameters / dummyBusParameter
//   * AgentCfuFunction.{compute, config, load, store}
// Changing those signatures breaks the SoC integration and every existing run.
package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib.bus.tilelink._

import vexiiriscv.execute.cfu._

/** Parameters for the algorithm-neutral CFU surface used by design agents. */
case class AgentCfuParameter(
    xlen: Int = 32,
    vlen: Int = 256,
    regDepth: Int = 2,
    addressWidth: Int = 32,
    withTilelink: Boolean = true,
    withLoad: Boolean = true,
    withStore: Boolean = true
) {
  require(xlen > 0 && xlen % 8 == 0 && isPow2(xlen),
    "AgentCfu xlen must be a positive power-of-two, byte-aligned width")
  require(vlen > 0 && vlen % xlen == 0, "AgentCfu vlen must be a positive multiple of xlen")
  require(regDepth > 0 && regDepth <= 32, "AgentCfu regDepth must be in the raw RISC-V register-index range 1..32")
  require(addressWidth > 0, "AgentCfu addressWidth must be positive")
  require(withTilelink || (!withLoad && !withStore),
    "AgentCfu load/store require the optional TileLink interface")

  val beatBytes = xlen / 8
  val vectorBytes = vlen / 8
  val beatCount = vlen / xlen
  val beatIndexWidth = log2Up(beatCount) max 1
  val regIndexWidth = log2Up(regDepth) max 1
  val pendingSize = beatCount max 1
}

object AgentCfuFunction {
  val compute = 1
  val config = 2
  val load = 4
  val store = 5
}

/**
  * The agent-owned CFU design.
  *
  * Everything shared lives in [[AgentCfuBase]]; this class only declares which
  * instructions exist and what each one does. Add or remove one `Instruction` entry
  * (and its local handler) to grow or shrink the ISA: decode, unsupported commands,
  * acceptance, response backpressure and writeback routing are all derived from the
  * table, so no separate supported/decode/FSM branch needs editing.
  *
  * Operand and writeback resources are explicit `CfuResource` declarations, never
  * implicit handler behavior:
  *   CpuRead(RS1)                            - the CPU X[rs1] value, not its number
  *   CpuRead(RS2) / CpuWriteRD               - the CPU X[rs2] value / CPU rd result
  *   VectorRead(VS1, CpuValue(RS1))          - CFU V[X[rs1]], vector width
  *   VectorRead(VS2, Constant(1))            - CFU V[1]
  *   VectorWrite(VD, CpuValue(RS1))          - write the result to CFU V[X[rs1]]
  *   LegacyRegisterNumber(RS2)               - instruction[24:20] with legacy
  *                                             index-width truncation, not X[rs2]
  * A vector index implicitly consumes the selected CPU input; it needs no separate
  * CpuRead. `LegacyRegisterNumber` exists only to keep the historical load/store
  * encodings, so new instructions should use `CpuValue`/`Constant`. At most one
  * data write (CPU RD or vector VD) is allowed. `resources = None` is the legacy
  * two-argument interface; `Some(Seq.empty)` acknowledges with no data write.
  *
  * Handlers read through `ctx.read(role)` and finish with `ctx.complete(result)`,
  * `ctx.complete()` for an acknowledgement, or `ctx.fail()` for an error. Use
  * `ctx.defer()` plus a `tick` handler for a multicycle operation; the context
  * captures request inputs, vector sources and the destination index on acceptance,
  * so a deferred handler never observes a later bus or register-file change. This
  * class may not use the legacy `ctx.input`/`ctx.reply` for a resource-aware entry.
  *
  * The table runs while the platform is still being constructed, so use only
  * this class's constructor parameters (`p`, `cfuParam`), the platform members, and
  * state allocated inside the table itself. Do not read a `val` declared in the
  * class body: Scala assigns those after the platform has already called the table.
  */
class AgentCfu(
    cfuParam: CfuBusParameter,
    busParam: BusParameter,
    p: AgentCfuParameter
) extends AgentCfuBase(cfuParam, busParam, p) {
  import AgentCfuBase.CfuResource._

  /** The design surface: one entry per ISA operation. */
  override protected def instructionTable: Seq[Instruction] = Seq(
    Instruction("compute", AgentCfuFunction.compute,
      start = ctx => ctx.reply(B(0, cfuParam.CFU_OUTPUT_DATA_W bits))),
    Instruction("config", AgentCfuFunction.config,
      start = ctx => ctx.reply(B(0, cfuParam.CFU_OUTPUT_DATA_W bits))),
    Instruction("load", AgentCfuFunction.load, enabled = p.withTilelink && p.withLoad,
      ready = () => lsu.cmd.ready,
      resources = Some(Seq(CpuRead(RS1), VectorWrite(VD, LegacyRegisterNumber(RS2)))),
      start = ctx => {
        lsu.cmd.address := ctx.read(RS1).asUInt.resize(p.addressWidth)
        loadDestination := ctx.index(VD)
        lsu.cmd.load := True
        fsm.goto(fsm.LOAD)
      }),
    Instruction("store", AgentCfuFunction.store, enabled = p.withTilelink && p.withStore,
      ready = () => lsu.cmd.ready,
      resources = Some(Seq(CpuRead(RS1), VectorRead(VS1, LegacyRegisterNumber(RS2)))),
      start = ctx => {
        lsu.cmd.address := ctx.read(RS1).asUInt.resize(p.addressWidth)
        storeSource := ctx.index(VS1)
        lsu.cmd.store := True
        fsm.goto(fsm.STORE)
      })
  )
}
