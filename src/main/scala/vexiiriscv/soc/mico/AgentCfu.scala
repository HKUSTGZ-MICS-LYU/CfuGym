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
    withStore: Boolean = true,
    advancedMem: Boolean = false,
    burstMem: Boolean = false,
    scalarRegDepth: Int = 4
) {
  def this(
      xlen: Int,
      vlen: Int,
      regDepth: Int,
      addressWidth: Int,
      withTilelink: Boolean,
      withLoad: Boolean,
      withStore: Boolean
  ) = this(xlen, vlen, regDepth, addressWidth, withTilelink, withLoad, withStore, false, false, 4)

  require(xlen > 0 && xlen % 8 == 0 && isPow2(xlen),
    "AgentCfu xlen must be a positive power-of-two, byte-aligned width")
  require(vlen > 0 && vlen % xlen == 0, "AgentCfu vlen must be a positive multiple of xlen")
  require(regDepth > 0 && regDepth <= 32, "AgentCfu regDepth must be in the raw RISC-V register-index range 1..32")
  require(scalarRegDepth > 0 && scalarRegDepth <= 32,
    "AgentCfu scalarRegDepth must be in the range 1..32")
  require(addressWidth > 0, "AgentCfu addressWidth must be positive")
  require(withTilelink || (!withLoad && !withStore),
    "AgentCfu load/store require the optional TileLink interface")
  require(!burstMem || isPow2(vlen / 8),
    "AgentCfu burstMem requires a power-of-two vector byte count (vlen / 8)")

  val beatBytes = xlen / 8
  val vectorBytes = vlen / 8
  val beatCount = vlen / xlen
  val beatIndexWidth = log2Up(beatCount) max 1
  val regIndexWidth = log2Up(regDepth) max 1
  val scalarIndexWidth = log2Up(scalarRegDepth) max 1
  val transferBytes = if (burstMem) vectorBytes else beatBytes
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
  * Everything shared lives in [[AgentCfuBase]]; this class declares the custom
  * hardware datapath, state/Areas and the instruction table. Add or remove one
  * `Instruction`, `PipedInstruction` or `FsmInstruction` entry (or call
  * `register(...)` inside an `Area`) to grow or shrink the ISA: decode,
  * unsupported commands, acceptance, response backpressure and writeback routing
  * are all derived from the table.
  *
  * Operand and writeback resources are explicit `CfuResource` declarations:
  *   CpuRead(RS1) / CpuRead(RS2)             - CPU X[rs1] / X[rs2] value
  *   CpuWriteRD                              - write result to CPU rd
  *   VectorRead(VS1, CpuValue(RS1))          - CFU V[X[rs1]], vector width
  *   VectorRead(VS2, Constant(1))            - CFU V[1], vector width
  *   VectorWrite(VD, CpuValue(RS1))          - write result to CFU V[X[rs1]]
  *   ScalarRead(SS1, Constant(0))            - CFU local scalar S[0], xlen width
  *   ScalarWrite(SD, BitSlice(RS1, 7, 0))    - write result to CFU S[X[rs1][7:0]]
  *   BitSlice(RS1, hi, lo) / CsrState        - packed bit-slice or CSR state index
  *   LegacyRegisterNumber(RS2)               - compatibility-only instruction[24:20]
  *
  * Execution styles supported by [[AgentCfuBase]]:
  *   1. `Instruction(...)`       - 1-cycle combinational (`ctx.complete(res)`) or
  *                                 deferred (`ctx.defer()` + `tick`)
  *   2. `PipedInstruction(...)`  - multi-stage SpinalHDL `StagePipeline` with
  *                                 automatic inter-stage registers and backpressure
  *   3. `FsmInstruction(...)`    - instruction-scoped `StateMachine` (`CfuSubFsm`)
  *   4. LSU helpers              - `lsuStartVectorLoad`, `lsuStartStreamLoad`,
  *                                 `lsuStartScalarLoad`, `lsuStartVectorStore`,
  *                                 `lsuStartScalarStore` + `lsu.loadFlow`
  *   5. SIMD/Math helpers        - `lanesSInt`, `lanesUInt`, `packLanes`,
  *                                 `dotProductSInt`, `reduceSumSInt`, `clampSInt`
  *
  * Dispatch is elaborated in `prePop` after this class's constructor body runs,
  * so top-level `val`s, `Reg`s, `Mem`s and `new Area` blocks declared in this
  * class may be freely referenced inside `instructionTable`.
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
        lsuStartVectorLoad(ctx.read(RS1).asUInt.resize(p.addressWidth), ctx.index(VD))
        fsm.goto(fsm.LOAD)
      }),
    Instruction("store", AgentCfuFunction.store, enabled = p.withTilelink && p.withStore,
      ready = () => lsu.cmd.ready,
      resources = Some(Seq(CpuRead(RS1), VectorRead(VS1, LegacyRegisterNumber(RS2)))),
      start = ctx => {
        lsuStartVectorStore(ctx.read(RS1).asUInt.resize(p.addressWidth), ctx.index(VS1))
        fsm.goto(fsm.STORE)
      })
  )
}
