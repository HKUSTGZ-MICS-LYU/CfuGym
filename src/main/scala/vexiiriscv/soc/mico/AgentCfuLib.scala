// SPDX-License-Identifier: MIT
//
// AgentCfuLib - the read-only AgentCfu platform.
//
// Design agents own only AgentCfu.scala (the instruction table / datapath design)
// and AgentCfuFiber.scala. This file holds the machinery they must not need to
// touch: the CfuResource vocabulary, Instruction/PipedInstruction/FsmInstruction,
// InstructionContext, the CfuBus shell, response buffer, vector/scalar RFs,
// decoupled CfuLsu helpers, SIMD/math primitives and the instruction dispatcher.
//
// Each run compiles an isolated workspace copy of AgentCfu.scala /
// AgentCfuFiber.scala over this tracked platform (see agents/README.md and
// project/AgentCfuSourceFilter.scala). Keeping the platform here means a
// whole-file rewrite of AgentCfu.scala cannot delete it, and the agent reads only
// the design it actually owns.
//
// Only AgentCfuBase is a top-level class name. CfuResource lives in its companion
// object so that older run workspaces, which still inline a top-level CfuResource,
// keep compiling against this repository copy.
package vexiiriscv.soc.mico

import scala.collection.mutable
import spinal.core._
import spinal.lib._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.fsm._
import spinal.lib.misc.pipeline._

import vexiiriscv.execute.cfu._
import vexiiriscv.soc.cfu.{CfuLsu, CfuLsuLoadBeat, CfuLsuParameter}

/** Elaboration-time operand/writeback resources, inspired by native RfResource.
  *
  * A resource describes what one instruction reads and writes. It adds no CfuBus
  * field: CPU roles refer to the standard command input slots, vector roles to
  * the CFU's vector register file, and scalar roles to the CFU's local scalar RF.
  * Vector and scalar accesses carry an explicit index source, so a CPU register
  * VALUE, a packed bit-slice, a CSR state index, an instruction register NUMBER
  * and a constant slot are never confused.
  */
object AgentCfuBase {
  sealed trait CfuResource

  object CfuResource {
    sealed trait Role
    sealed trait CpuSource extends Role
    case object RS1 extends CpuSource
    case object RS2 extends CpuSource
    case object RD extends Role

    sealed trait VectorRole extends Role
    case object VS1 extends VectorRole
    case object VS2 extends VectorRole
    case object VD extends VectorRole

    sealed trait ScalarRole extends Role
    case object SS1 extends ScalarRole
    case object SS2 extends ScalarRole
    case object SD extends ScalarRole

    sealed trait Index
    case class CpuValue(register: CpuSource) extends Index
    case class Constant(value: Int) extends Index
    case class BitSlice(register: CpuSource, hi: Int, lo: Int) extends Index
    case object CsrState extends Index
    case class LegacyRegisterNumber(register: CpuSource) extends Index

    case class CpuRead(role: CpuSource) extends CfuResource
    case object CpuWriteRD extends CfuResource
    case class VectorRead(role: VectorRole, index: Index) extends CfuResource
    case class VectorWrite(role: VectorRole, index: Index) extends CfuResource
    case class ScalarRead(role: ScalarRole, index: Index) extends CfuResource
    case class ScalarWrite(role: ScalarRole, index: Index) extends CfuResource
  }

  /** Attached to every platform-enforced design check.
    *
    * A repair prompt that sees a failure located in this read-only file must change
    * the design instead of trying to edit the platform.
    */
  val READ_ONLY_HINT: String =
    "AgentCfuLib.scala is a read-only platform file: do not edit, rename or " +
    "re-declare it. Fix the instruction table or datapath in AgentCfu.scala (or " +
    "AgentCfuFiber.scala); report a platform defect if the platform itself is wrong."
}

/**
  * Shared, non-design CFU platform: bus shell, vector RF, local scalar RF,
  * response buffer, CfuLsu/FSM, pipeline/sub-FSM builders, SIMD math helpers
  * and the instruction dispatcher.
  *
  * `AgentCfu` supplies the design by overriding `instructionTable` (and/or by
  * calling `register(...)` inside class-body `Area` blocks). The dispatcher is
  * elaborated at the start of `prePop()`, after the subclass constructor finishes
  * and before any `StateMachine` or `StagePipeline` builds, so `AgentCfu` may
  * freely declare top-level `val`s, `Reg`s, `Mem`s and `new Area` blocks.
  */
abstract class AgentCfuBase(
    cfuParam: CfuBusParameter,
    busParam: BusParameter,
    p: AgentCfuParameter
) extends Component {
  import AgentCfuBase.CfuResource
  import AgentCfuBase.CfuResource._

  /** Fail elaboration with an actionable, design-attributed message.
    *
    * Every platform check that a design can violate goes through this so the
    * failure names the design contract and the read-only platform boundary, instead
    * of surfacing as an opaque backend error somewhere in this file.
    */
  protected def designRequire(condition: Boolean, message: => String): Unit =
    require(condition, s"$message\n${AgentCfuBase.READ_ONLY_HINT}")

  require(!p.withTilelink || busParam.dataWidth == p.xlen,
    "AgentCfu TileLink width must match AgentCfu xlen")

  val io = new Bundle {
    val bus = slave(CfuBus(cfuParam))
    val dBus = p.withTilelink generate master(tilelink.Bus(busParam))
  }

  private val regIndexWidth = p.regIndexWidth
  private val scalarIndexWidth = p.scalarIndexWidth
  private val beatShift = log2Up(p.xlen)

  // Register-backed platform storage: simple to inspect and deterministic to
  // elaborate. A design may also allocate its own state or scratchpad Memories.
  val vectorRegs = Vec(Reg(Bits(p.vlen bits)) init (0), p.regDepth)
  val scalarRegs = Vec(Reg(Bits(p.xlen bits)) init (0), p.scalarRegDepth)

  // A single slot must be addressed statically. SpinalHDL addresses a Vec with
  // log2Up(size) bits, so a one-element Vec expects a zero-width address while the
  // generic slot index is at least 1 bit wide.
  private val singleVectorSlot = p.regDepth == 1
  private val singleScalarSlot = p.scalarRegDepth == 1

  /** Normalise and validate a design-supplied vector slot address. */
  private def slotAddress(addr: UInt): UInt = {
    if (singleVectorSlot) U(0, 1 bits)
    else {
      val needed = log2Up(p.regDepth)
      designRequire(widthOf(addr) <= needed,
        s"vector slot address has ${widthOf(addr)} bits, but regDepth=${p.regDepth} needs $needed")
      addr.resize(needed)
    }
  }

  /** Normalise and validate a design-supplied scalar slot address. */
  private def scalarSlotAddress(addr: UInt): UInt = {
    if (singleScalarSlot) U(0, 1 bits)
    else {
      val needed = log2Up(p.scalarRegDepth)
      designRequire(widthOf(addr) <= needed,
        s"scalar slot address has ${widthOf(addr)} bits, but scalarRegDepth=${p.scalarRegDepth} needs $needed")
      addr.resize(needed)
    }
  }

  def vectorRead(addr: UInt): Bits =
    if (singleVectorSlot) vectorRegs(0) else vectorRegs(slotAddress(addr))

  def vectorReadBeat(addr: UInt, index: UInt): Bits = {
    val offset = (index.resize(log2Up(p.vlen)) << beatShift).resize(log2Up(p.vlen))
    vectorRead(addr)(offset, p.xlen bits)
  }

  def vectorWrite(addr: UInt, data: Bits): Unit =
    if (singleVectorSlot) vectorRegs(0) := data else vectorRegs(slotAddress(addr)) := data

  def vectorWriteBeat(addr: UInt, index: UInt, data: Bits): Unit = {
    val offset = (index.resize(log2Up(p.vlen)) << beatShift).resize(log2Up(p.vlen))
    if (singleVectorSlot) vectorRegs(0)(offset, p.xlen bits) := data
    else vectorRegs(slotAddress(addr))(offset, p.xlen bits) := data
  }

  def scalarRead(addr: UInt): Bits =
    if (singleScalarSlot) scalarRegs(0) else scalarRegs(scalarSlotAddress(addr))

  def scalarWrite(addr: UInt, data: Bits): Unit =
    if (singleScalarSlot) scalarRegs(0) := data else scalarRegs(scalarSlotAddress(addr)) := data

  /** Allocate an SRAM/LUTRAM-backed scratchpad memory for larger tiles or tables. */
  protected def allocateScratchpad(wordWidth: Int, depth: Int): Mem[Bits] = {
    designRequire(wordWidth > 0 && depth > 0, "scratchpad dimensions must be positive")
    Mem(Bits(wordWidth bits), depth)
  }

  /** Allocate a low-latency stream FIFO for decoupled producer/consumer stages. */
  protected def allocateFifo[T <: Data](dataType: HardType[T], depth: Int): StreamFifoLowLatency[T] = {
    designRequire(depth > 0, "FIFO depth must be positive")
    new StreamFifoLowLatency(dataType(), depth)
  }

  // Common SIMD lane slicing, balanced-tree reduction and saturation helpers.
  protected def lanesSInt(vec: Bits, elemBits: Int): Vec[SInt] = {
    designRequire(elemBits > 0 && widthOf(vec) % elemBits == 0,
      s"vector width ${widthOf(vec)} must be a multiple of elemBits=$elemBits")
    Vec(vec.subdivideIn(elemBits bits).map(_.asSInt))
  }

  protected def lanesUInt(vec: Bits, elemBits: Int): Vec[UInt] = {
    designRequire(elemBits > 0 && widthOf(vec) % elemBits == 0,
      s"vector width ${widthOf(vec)} must be a multiple of elemBits=$elemBits")
    Vec(vec.subdivideIn(elemBits bits).map(_.asUInt))
  }

  protected def packLanes[T <: BitVector](lanes: Seq[T]): Bits = {
    designRequire(lanes.nonEmpty, "packLanes requires a non-empty lane sequence")
    Cat(lanes.map(_.asBits))
  }

  protected def reduceSumSInt(terms: Seq[SInt], accBits: Int = p.xlen): SInt = {
    designRequire(terms.nonEmpty, "reduceSumSInt requires a non-empty sequence")
    terms.map(_.resize(accBits)).reduceBalancedTree(_ + _)
  }

  protected def reduceSumUInt(terms: Seq[UInt], accBits: Int = p.xlen): UInt = {
    designRequire(terms.nonEmpty, "reduceSumUInt requires a non-empty sequence")
    terms.map(_.resize(accBits)).reduceBalancedTree(_ + _)
  }

  protected def dotProductSInt(a: Bits, b: Bits, elemBits: Int, accBits: Int = p.xlen): SInt = {
    designRequire(widthOf(a) == widthOf(b), "dotProductSInt operands must have equal width")
    val la = lanesSInt(a, elemBits)
    val lb = lanesSInt(b, elemBits)
    val products = (0 until la.length).map(i => (la(i) * lb(i)).resize(accBits))
    products.reduceBalancedTree(_ + _)
  }

  protected def dotProductUInt(a: Bits, b: Bits, elemBits: Int, accBits: Int = p.xlen): UInt = {
    designRequire(widthOf(a) == widthOf(b), "dotProductUInt operands must have equal width")
    val la = lanesUInt(a, elemBits)
    val lb = lanesUInt(b, elemBits)
    val products = (0 until la.length).map(i => (la(i) * lb(i)).resize(accBits))
    products.reduceBalancedTree(_ + _)
  }

  protected def clampSInt(value: SInt, minVal: Int, maxVal: Int, outBits: Int): SInt = {
    designRequire(minVal <= maxVal && outBits > 0, "invalid clampSInt bounds")
    val w = widthOf(value) max outBits
    val v = value.resize(w)
    val lo = S(minVal, w bits)
    val hi = S(maxVal, w bits)
    Mux(v < lo, lo, Mux(v > hi, hi, v)).resize(outBits)
  }

  // Param.scala custom0 decode leaves funct7 unconstrained. Keep the legacy
  // function bus and carry sub-operations in the existing raw instruction.
  require(cfuParam.CFU_FUNCTION_ID_W == 3, "AgentCfu requires the funct3 bus")
  require(cfuParam.CFU_RAW_INSN_W == 32, "AgentCfu requires the full raw instruction")
  val func3 = io.bus.cmd.function_id
  val subOp = io.bus.cmd.raw_insn(31 downto 25).asUInt

  def matches(function: Int, operation: Int = 0): Bool = {
    designRequire(function >= 0 && function < 8, "function must fit funct3")
    designRequire(operation >= 0 && operation < 128, "operation must fit funct7")
    (func3 === U(function, 3 bits)) && (subOp === U(operation, 7 bits))
  }

  val decode = new Area {
    val rs1 = io.bus.cmd.raw_insn(19 downto 15).asUInt.resize(regIndexWidth)
    val rs2 = io.bus.cmd.raw_insn(24 downto 20).asUInt.resize(regIndexWidth)
  }

  val responsePending = RegInit(False)
  val responseId = Reg(UInt(cfuParam.CFU_REQ_RESP_ID_W bits)) init (0)
  val responseData = Reg(Bits(cfuParam.CFU_OUTPUT_DATA_W bits)) init (0)
  val responseStatus = cfuParam.CFU_WITH_STATUS generate Reg(Bits(3 bits)) init (0)

  io.bus.rsp.valid := responsePending
  io.bus.rsp.response_id := responseId
  io.bus.rsp.outputs(0) := responseData
  if (cfuParam.CFU_WITH_STATUS) io.bus.rsp.status := responseStatus

  when (io.bus.rsp.fire) {
    responsePending := False
  }

  def enqueueResponse(id: UInt, data: Bits, status: Bits): Unit = {
    responseId := id
    responseData := data.resize(cfuParam.CFU_OUTPUT_DATA_W)
    if (cfuParam.CFU_WITH_STATUS) responseStatus := status
    responsePending := True
  }

  val loadDestination = Reg(UInt(regIndexWidth bits)) init (0)
  val storeSource = Reg(UInt(regIndexWidth bits)) init (0)
  val loadWriteRf = RegInit(True)
  val storeFromRf = RegInit(True)
  val storeCustomData = Reg(Bits(p.xlen bits)) init (0)
  // Custom/empty tables may omit memory entries while retaining the LSU shell.
  loadDestination := loadDestination
  storeSource := storeSource
  loadWriteRf := loadWriteRf
  storeFromRf := storeFromRf
  storeCustomData := storeCustomData
  val commandRequestId = Reg(UInt(cfuParam.CFU_REQ_RESP_ID_W bits)) init (0)

  val lsu = if (p.withTilelink) {
    new CfuLsu(
      io.dBus,
      CfuLsuParameter(
        dataWidth = p.xlen,
        vectorBytes = p.vectorBytes,
        addressWidth = p.addressWidth,
        withLoad = p.withLoad,
        withStore = p.withStore,
        advancedMem = p.advancedMem,
        burstMem = p.burstMem
      )
    )
  } else {
    null
  }

  if (p.withTilelink) {
    lsu.cmd.load := False
    lsu.cmd.store := False
    lsu.cmd.address := io.bus.cmd.inputs(0).asUInt.resize(p.addressWidth)
    lsu.store.data := storeFromRf.mux(vectorReadBeat(storeSource, lsu.store.index), storeCustomData)

    when (lsu.load.valid && loadWriteRf) {
      vectorWriteBeat(loadDestination, lsu.load.index, lsu.load.data)
    }
    when (lsu.done) {
      loadWriteRf := True
      storeFromRf := True
    }
  }

  /** Start a full or partial vector load into `vectorRegs(dstSlot)`. */
  protected def lsuStartVectorLoad(address: UInt, dstSlot: UInt, beats: Int = p.beatCount): Unit = {
    designRequire(p.withTilelink && p.withLoad, "lsuStartVectorLoad requires TileLink load support")
    designRequire(beats >= 1 && beats <= p.beatCount, s"LSU beat count $beats out of range 1..${p.beatCount}")
    lsu.cmd.address := address.resize(p.addressWidth)
    lsu.cmd.lastBeat := U(beats - 1, p.beatIndexWidth bits)
    loadDestination := dstSlot.resize(regIndexWidth)
    loadWriteRf := True
    lsu.cmd.load := True
  }

  /** Start a streaming load of `beats` words WITHOUT overwriting `vectorRegs`.
    * Consume beats via `lsu.loadFlow.valid` (`lsu.loadFlow.fragment.data`, `index`, `last`).
    */
  protected def lsuStartStreamLoad(address: UInt, beats: Int = p.beatCount): Unit = {
    designRequire(p.withTilelink && p.withLoad, "lsuStartStreamLoad requires TileLink load support")
    designRequire(beats >= 1 && beats <= p.beatCount, s"LSU beat count $beats out of range 1..${p.beatCount}")
    lsu.cmd.address := address.resize(p.addressWidth)
    lsu.cmd.lastBeat := U(beats - 1, p.beatIndexWidth bits)
    loadWriteRf := False
    lsu.cmd.load := True
  }

  /** Start a single-word (1-beat) load WITHOUT overwriting `vectorRegs`.
    * When `lsu.done` asserts, the word is on `lsu.load.data` (and latched in `lsu.load.lastData`).
    */
  protected def lsuStartScalarLoad(address: UInt): Unit =
    lsuStartStreamLoad(address, beats = 1)

  /** Start a full or partial vector store from `vectorRegs(srcSlot)`. */
  protected def lsuStartVectorStore(address: UInt, srcSlot: UInt, beats: Int = p.beatCount): Unit = {
    designRequire(p.withTilelink && p.withStore, "lsuStartVectorStore requires TileLink store support")
    designRequire(beats >= 1 && beats <= p.beatCount, s"LSU beat count $beats out of range 1..${p.beatCount}")
    lsu.cmd.address := address.resize(p.addressWidth)
    lsu.cmd.lastBeat := U(beats - 1, p.beatIndexWidth bits)
    storeSource := srcSlot.resize(regIndexWidth)
    storeFromRf := True
    lsu.cmd.store := True
  }

  /** Start a single-word (1-beat) store of `data` without reading `vectorRegs`. */
  protected def lsuStartScalarStore(address: UInt, data: Bits): Unit = {
    designRequire(p.withTilelink && p.withStore, "lsuStartScalarStore requires TileLink store support")
    designRequire(widthOf(data) == p.xlen, s"scalar store data width ${widthOf(data)} must match xlen=${p.xlen}")
    lsu.cmd.address := address.resize(p.addressWidth)
    lsu.cmd.lastBeat := U(0, p.beatIndexWidth bits)
    storeFromRf := False
    storeCustomData := data
    lsu.cmd.store := True
  }

  val fsm = new StateMachine {
    val IDLE = new State with EntryPoint
    val LOAD = new State
    val STORE = new State

    if (p.withTilelink && p.withLoad) {
      LOAD.whenIsActive {
        when (lsu.done) {
          enqueueResponse(
            commandRequestId,
            B(0, cfuParam.CFU_OUTPUT_DATA_W bits),
            lsu.error.mux(B"001", B"000")
          )
          goto(IDLE)
        }
      }
    }

    if (p.withTilelink && p.withStore) {
      STORE.whenIsActive {
        when (lsu.done) {
          enqueueResponse(
            commandRequestId,
            B(0, cfuParam.CFU_OUTPUT_DATA_W bits),
            lsu.error.mux(B"001", B"000")
          )
          goto(IDLE)
        }
      }
    }
  }

  /** One exact instruction declaration, inspired by IntAluPlugin's add/decode table.
    * resources=None retains the legacy input/reply API; Some(Seq.empty) means ACK
    * with no operands or data writeback. Prefer Some(...) plus read/complete.
    */
  protected case class Instruction(
      name: String, function: Int, subop: Int = 0, enabled: Boolean = true,
      ready: () => Bool = () => True,
      start: InstructionContext => Unit,
      tick: Option[InstructionContext => Unit] = None,
      resources: Option[Seq[CfuResource]] = None
  ) {
    private[AgentCfuBase] var elaborate: Option[InstructionContext => Unit] = None
  }

  private val registeredInstructions = mutable.ArrayBuffer[Instruction]()

  /** Imperatively register an instruction from inside an Area or helper block. */
  protected def register(instruction: Instruction): Instruction = {
    registeredInstructions += instruction
    instruction
  }

  private val acceptedInputs = Vec(Reg(Bits(cfuParam.CFU_INPUT_DATA_W bits)) init(0), cfuParam.CFU_INPUTS)
  private val acceptedRaw = Reg(Bits(cfuParam.CFU_RAW_INSN_W bits)) init(0)
  private val acceptedState = Reg(cloneOf(io.bus.cmd.state_index)) init(0)
  private val acceptedCfu = Reg(cloneOf(io.bus.cmd.cfu_index)) init(0)

  private def resourceRole(r: CfuResource): Role = r match {
    case CpuRead(role) => role
    case CpuWriteRD => RD
    case VectorRead(role, _) => role
    case VectorWrite(role, _) => role
    case ScalarRead(role, _) => role
    case ScalarWrite(role, _) => role
  }

  private def isWriteResource(r: CfuResource): Boolean = r match {
    case CpuWriteRD | VectorWrite(_, _) | ScalarWrite(_, _) => true
    case _ => false
  }

  private def expectedWriteWidth(rs: Seq[CfuResource]): Int = {
    rs.collectFirst {
      case CpuWriteRD => cfuParam.CFU_OUTPUT_DATA_W
      case VectorWrite(_, _) => p.vlen
      case ScalarWrite(_, _) => p.xlen
    }.getOrElse(cfuParam.CFU_OUTPUT_DATA_W)
  }

  private def validateResources(rs: Seq[CfuResource]): Unit = {
    designRequire(rs.map(resourceRole).distinct.size == rs.size, "duplicate resource role")
    designRequire(rs.count(isWriteResource) <= 1, "at most one resource write destination")
    rs.foreach {
      case VectorRead(role, index) =>
        designRequire(role == VS1 || role == VS2, "vector read role must be VS1 or VS2")
        validateIndex(index, p.regDepth, isVector = true)
      case VectorWrite(role, index) =>
        designRequire(role == VD, "vector write role must be VD")
        validateIndex(index, p.regDepth, isVector = true)
      case ScalarRead(role, index) =>
        designRequire(role == SS1 || role == SS2, "scalar read role must be SS1 or SS2")
        validateIndex(index, p.scalarRegDepth, isVector = false)
      case ScalarWrite(role, index) =>
        designRequire(role == SD, "scalar write role must be SD")
        validateIndex(index, p.scalarRegDepth, isVector = false)
      case _ =>
    }
  }

  private def validateIndex(index: Index, depth: Int, isVector: Boolean): Unit = index match {
    case Constant(n) =>
      if (isVector) designRequire(n >= 0 && n < depth, "constant vector index out of range")
      else designRequire(n >= 0 && n < depth, "constant scalar index out of range")
    case BitSlice(_, hi, lo) =>
      designRequire(lo >= 0 && hi >= lo && hi < cfuParam.CFU_INPUT_DATA_W,
        s"BitSlice($hi downto $lo) out of range 0..${cfuParam.CFU_INPUT_DATA_W - 1}")
    case _ =>
  }

  // Shared snapshot storage for deferred (`tick`) handlers. Because `!handlerBusy`
  // guarantees at most one deferred instruction is active at any time, sharing the
  // latched vector/scalar operands across entries avoids O(N_insn * vlen) flip-flops.
  private val sharedSavedVs1 = Reg(Bits(p.vlen bits)) init(0)
  private val sharedSavedVs2 = Reg(Bits(p.vlen bits)) init(0)
  private val sharedSavedSs1 = Reg(Bits(p.xlen bits)) init(0)
  private val sharedSavedSs2 = Reg(Bits(p.xlen bits)) init(0)
  private val sharedSavedVecIdx = Map[Role, UInt](
    VS1 -> (Reg(UInt(regIndexWidth bits)) init(0)),
    VS2 -> (Reg(UInt(regIndexWidth bits)) init(0)),
    VD  -> (Reg(UInt(regIndexWidth bits)) init(0))
  )
  private val sharedSavedScalarIdx = Map[Role, UInt](
    SS1 -> (Reg(UInt(scalarIndexWidth bits)) init(0)),
    SS2 -> (Reg(UInt(scalarIndexWidth bits)) init(0)),
    SD  -> (Reg(UInt(scalarIndexWidth bits)) init(0))
  )

  private class ResourceBinding(rs: Seq[CfuResource], capture: Bool, hasTick: Boolean) extends Area {
    def slot(role: CpuSource): Bits = io.bus.cmd.inputs(if (role == RS1) 0 else 1)
    def savedSlot(role: CpuSource): Bits = acceptedInputs(if (role == RS1) 0 else 1)

    private def resolveIndex(idx: Index, depth: Int, idxWidth: Int): (UInt, Bool) = idx match {
      case CpuValue(reg) =>
        val full = slot(reg).asUInt
        (full.resize(idxWidth), full.resize(full.getWidth + 1) < U(depth, full.getWidth + 1 bits))
      case Constant(n) =>
        (U(n, idxWidth bits), True)
      case BitSlice(reg, hi, lo) =>
        val slice = slot(reg)(hi downto lo).asUInt
        (slice.resize(idxWidth), slice.resize(slice.getWidth + 1) < U(depth, slice.getWidth + 1 bits))
      case CsrState =>
        val st = io.bus.cmd.state_index
        (st.resize(idxWidth), st.resize(st.getWidth + 1) < U(depth, st.getWidth + 1 bits))
      case LegacyRegisterNumber(reg) =>
        val lo = if (reg == RS1) 15 else 20
        (io.bus.cmd.raw_insn(lo + 4 downto lo).asUInt.resize(idxWidth), True)
    }

    val indices: Map[Role, (UInt, UInt, Bool)] = rs.collect {
      case VectorRead(role, idx) => (role: Role, idx, p.regDepth, regIndexWidth, sharedSavedVecIdx(role))
      case VectorWrite(role, idx) => (role: Role, idx, p.regDepth, regIndexWidth, sharedSavedVecIdx(role))
      case ScalarRead(role, idx) => (role: Role, idx, p.scalarRegDepth, scalarIndexWidth, sharedSavedScalarIdx(role))
      case ScalarWrite(role, idx) => (role: Role, idx, p.scalarRegDepth, scalarIndexWidth, sharedSavedScalarIdx(role))
    }.map { case (role, idx, depth, idxWidth, sharedReg) =>
      val (address, isValid) = resolveIndex(idx, depth, idxWidth)
      val saved = if (hasTick) {
        when(capture) { sharedReg := address }
        sharedReg
      } else {
        address
      }
      role -> (address, saved, isValid)
    }.toMap

    val valid: Bool = indices.values.map(_._3).foldLeft(True)(_ && _)

    val reads: Map[Role, (Bits, Bits)] = rs.collect {
      case CpuRead(role) =>
        val live = slot(role)
        val saved = if (hasTick) savedSlot(role) else live
        (role: Role) -> (live, saved)
      case VectorRead(role, _) =>
        val live = vectorRead(indices(role)._1)
        val sharedReg = if (role == VS1) sharedSavedVs1 else sharedSavedVs2
        val saved = if (hasTick) {
          when(capture && valid) { sharedReg := live }
          sharedReg
        } else {
          live
        }
        (role: Role) -> (live, saved)
      case ScalarRead(role, _) =>
        val live = scalarRead(indices(role)._1)
        val sharedReg = if (role == SS1) sharedSavedSs1 else sharedSavedSs2
        val saved = if (hasTick) {
          when(capture && valid) { sharedReg := live }
          sharedReg
        } else {
          live
        }
        (role: Role) -> (live, saved)
    }.toMap
  }

  protected class InstructionContext private[AgentCfuBase](
      private val active: Bool,
      deferred: Boolean,
      hasTick: Boolean,
      resources: Option[Seq[CfuResource]],
      binding: ResourceBinding
  ) {
    def input(index: Int): Bits = {
      designRequire(resources.isEmpty, "resource-aware instruction must use read, not input")
      if (deferred) acceptedInputs(index) else io.bus.cmd.inputs(index)
    }
    def read(role: Role): Bits = {
      designRequire(binding.reads.contains(role), s"undeclared resource read: $role")
      val pair = binding.reads(role)
      if (deferred) pair._2 else pair._1
    }
    def index(role: VectorRole): UInt = {
      designRequire(binding.indices.contains(role), s"undeclared vector index: $role")
      val triple = binding.indices(role)
      if (deferred) triple._2 else triple._1
    }
    def scalarIndex(role: ScalarRole): UInt = {
      designRequire(binding.indices.contains(role), s"undeclared scalar index: $role")
      val triple = binding.indices(role)
      if (deferred) triple._2 else triple._1
    }
    /** Error completion suppresses vector/scalar writeback; CPU receives zero irrespective of status. */
    def fail(): Unit = finish(B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"001")
    def complete(): Unit = {
      designRequire(resources.isDefined && !resources.get.exists(isWriteResource),
        "complete without result requires no write destination")
      finish(B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"000")
    }
    def complete(result: Bits, status: Bits = B"000"): Unit = {
      designRequire(resources.isDefined, "complete requires declared resources")
      designRequire(status.getWidth == 3, "completion status must be 3 bits")
      resources.get.collectFirst {
        case CpuWriteRD => CpuWriteRD
        case v: VectorWrite => v
        case s: ScalarWrite => s
      } match {
        case Some(CpuWriteRD) =>
          designRequire(result.getWidth == cfuParam.CFU_OUTPUT_DATA_W, "CPU result width mismatch")
          finish(result, status)
        case Some(VectorWrite(role, _)) =>
          designRequire(result.getWidth == p.vlen, "vector result width mismatch")
          when(status === B"000") { vectorWrite(index(role), result) }
          finish(B(0, cfuParam.CFU_OUTPUT_DATA_W bits), status)
        case Some(ScalarWrite(role, _)) =>
          designRequire(result.getWidth == p.xlen, "scalar result width mismatch")
          when(status === B"000") { scalarWrite(scalarIndex(role), result) }
          finish(B(0, cfuParam.CFU_OUTPUT_DATA_W bits), status)
        case _ => designRequire(false, "result requires a write destination")
      }
    }
    def rawInstruction: Bits = if (deferred) acceptedRaw else io.bus.cmd.raw_insn
    def stateIndex: UInt = if (deferred) acceptedState else io.bus.cmd.state_index
    def cfuIndex: UInt = if (deferred) acceptedCfu else io.bus.cmd.cfu_index
    def requestId: UInt = if (deferred) commandRequestId else io.bus.cmd.request_id
    def reply(data: Bits, status: Bits = B"000"): Unit = {
      designRequire(resources.isEmpty, "resource-aware instruction must use complete, not reply")
      finish(data, status)
    }
    private def finish(data: Bits, status: Bits): Unit = {
      enqueueResponse(requestId, data, status)
      active := False
    }
    // Context inputs/raw instruction/state/CFU/request IDs are captured on fire;
    // tick observes that immutable request even if upstream changes the bus.
    def defer(): Unit = {
      designRequire(!deferred, "defer is only valid in an instruction start handler")
      designRequire(hasTick, "defer requires a tick handler")
      active := True
    }
  }

  /** Context passed to one stage of a `PipedInstruction`. */
  protected class CfuStageContext private[AgentCfuBase](
      val pipeCtx: CfuPipelineContext,
      val stageIndex: Int,
      val node: NodeMirror
  ) {
    def apply[T <: Data](payload: Payload[T]): T = node(payload)
    def read(role: Role): Bits = {
      designRequire(pipeCtx.readPayloads.contains(role), s"undeclared resource read: $role")
      node(pipeCtx.readPayloads(role))
    }
    def index(role: VectorRole): UInt = {
      designRequire(pipeCtx.indexPayloads.contains(role), s"undeclared vector index: $role")
      node(pipeCtx.indexPayloads(role))
    }
    def scalarIndex(role: ScalarRole): UInt = {
      designRequire(pipeCtx.indexPayloads.contains(role), s"undeclared scalar index: $role")
      node(pipeCtx.indexPayloads(role))
    }
    def complete(): Unit = {
      designRequire(stageIndex == pipeCtx.stages - 1, "complete() must be called in the final pipeline stage")
      designRequire(pipeCtx.resources.isDefined && !pipeCtx.resources.get.exists(isWriteResource),
        "complete without result requires no write destination")
      node(pipeCtx.resultPayload) := B(0, pipeCtx.resultWidth bits)
      node(pipeCtx.statusPayload) := B"000"
      pipeCtx.markCompleted()
    }
    def complete(result: Bits, status: Bits = B"000"): Unit = {
      designRequire(stageIndex == pipeCtx.stages - 1, "complete(result) must be called in the final pipeline stage")
      designRequire(pipeCtx.resources.isDefined && pipeCtx.resources.get.exists(isWriteResource),
        "result requires a write destination")
      designRequire(status.getWidth == 3, "completion status must be 3 bits")
      designRequire(result.getWidth == pipeCtx.resultWidth,
        s"pipeline result width mismatch: got ${result.getWidth}, expected ${pipeCtx.resultWidth}")
      node(pipeCtx.resultPayload) := result
      node(pipeCtx.statusPayload) := status
      pipeCtx.markCompleted()
    }
    def fail(): Unit = {
      designRequire(stageIndex == pipeCtx.stages - 1, "fail() must be called in the final pipeline stage")
      node(pipeCtx.resultPayload) := B(0, pipeCtx.resultWidth bits)
      node(pipeCtx.statusPayload) := B"001"
      pipeCtx.markCompleted()
    }
  }

  /** Builder passed to `PipedInstruction` to declare inter-stage `Payload`s and per-stage logic. */
  protected class CfuPipelineContext private[AgentCfuBase](
      val pipe: StagePipeline,
      val stages: Int,
      val resources: Option[Seq[CfuResource]],
      private[AgentCfuBase] val readPayloads: Map[Role, Payload[Bits]],
      private[AgentCfuBase] val indexPayloads: Map[Role, Payload[UInt]],
      private[AgentCfuBase] val resultWidth: Int,
      private[AgentCfuBase] val resultPayload: Payload[Bits],
      private[AgentCfuBase] val statusPayload: Payload[Bits]
  ) {
    private var completedInLastStage = false
    private[AgentCfuBase] def markCompleted(): Unit = { completedInLastStage = true }
    private[AgentCfuBase] def isCompleted: Boolean = completedInLastStage

    /** Declare an inter-stage `Payload` carried across `StageLink` registers. */
    def payload[T <: Data](dataType: HardType[T]): Payload[T] = Payload(dataType)

    /** Define hardware for pipeline stage `index` (`0 .. stages - 1`). */
    def stage(index: Int)(body: CfuStageContext => Unit): Area = {
      designRequire(index >= 0 && index < stages, s"pipeline stage index $index out of range 0..${stages - 1}")
      new pipe.Area(index) {
        val stageCtx = new CfuStageContext(CfuPipelineContext.this, index, this)
        body(stageCtx)
      }
    }
  }

  /** Create a multi-stage pipelined instruction backed by SpinalHDL `StagePipeline`.
    *
    * Stage `0` runs combinationally on acceptance (`cmd.fire`), followed by `stages`
    * registered `StageLink` hops before completion. Operands and destination indices
    * declared in `resources` are automatically injected as pipeline `Payload`s.
    */
  protected def PipedInstruction(
      name: String,
      function: Int,
      subop: Int = 0,
      stages: Int = 2,
      enabled: Boolean = true,
      ready: () => Bool = () => True,
      resources: Option[Seq[CfuResource]]
  )(buildPipe: CfuPipelineContext => Unit): Instruction = {
    designRequire(stages >= 1 && stages <= 16, s"PipedInstruction '$name' stages must be in 1..16")
    designRequire(resources.isDefined, s"PipedInstruction '$name' requires explicit Some(resources)")
    val rs = resources.get
    validateResources(rs)

    val entryValid = Bool()
    entryValid := False
    val startReads = mutable.Map[Role, Bits]()
    val startIndices = mutable.Map[Role, UInt]()

    val readPayloads: Map[Role, Payload[Bits]] = rs.collect {
      case CpuRead(role) => (role: Role) -> Payload(Bits(cfuParam.CFU_INPUT_DATA_W bits))
      case VectorRead(role, _) => (role: Role) -> Payload(Bits(p.vlen bits))
      case ScalarRead(role, _) => (role: Role) -> Payload(Bits(p.xlen bits))
    }.toMap

    val indexPayloads: Map[Role, Payload[UInt]] = rs.collect {
      case VectorRead(role, _) => (role: Role) -> Payload(UInt(regIndexWidth bits))
      case VectorWrite(role, _) => (role: Role) -> Payload(UInt(regIndexWidth bits))
      case ScalarRead(role, _) => (role: Role) -> Payload(UInt(scalarIndexWidth bits))
      case ScalarWrite(role, _) => (role: Role) -> Payload(UInt(scalarIndexWidth bits))
    }.toMap

    val resWidth = expectedWriteWidth(rs)
    val hasWrite = rs.exists(isWriteResource)
    val resultPayload = Payload(Bits(resWidth bits))
    val statusPayload = Payload(Bits(3 bits))

    val pipe = new StagePipeline()
    // Ensure all nodes 0 .. stages exist in order.
    for (i <- 0 to stages) { pipe.node(i) }
    pipe.first.valid := entryValid
    for ((role, pl) <- readPayloads) {
      val wire = Bits(widthOf(pl) bits)
      wire := 0
      startReads(role) = wire
      pipe.first(pl) := wire
    }
    for ((role, pl) <- indexPayloads) {
      val wire = UInt(widthOf(pl) bits)
      wire := 0
      startIndices(role) = wire
      pipe.first(pl) := wire
    }

    val pipeCtx = new CfuPipelineContext(
      pipe, stages, resources, readPayloads, indexPayloads, resWidth, resultPayload, statusPayload
    )
    if (enabled) {
      buildPipe(pipeCtx)
      designRequire(pipeCtx.isCompleted,
        s"PipedInstruction '$name' must call complete(...) or fail() in final stage ${stages - 1}")
    } else {
      pipe.node(stages - 1)(resultPayload) := 0
      pipe.node(stages - 1)(statusPayload) := B"000"
    }

    val outValid = Bool()
    val outResult = Bits(resWidth bits)
    val outStatus = Bits(3 bits)
    outValid := pipe.last.isValid
    outResult := pipe.last(resultPayload)
    outStatus := pipe.last(statusPayload)
    pipe.build(withoutCollapse = true)

    Instruction(
      name = name,
      function = function,
      subop = subop,
      enabled = enabled,
      ready = ready,
      resources = resources,
      start = ctx => {
        entryValid := True
        for ((role, wire) <- startReads) { wire := ctx.read(role) }
        for ((role, wire) <- startIndices) {
          role match {
            case v: VectorRole => wire := ctx.index(v)
            case s: ScalarRole => wire := ctx.scalarIndex(s)
            case _ =>
          }
        }
        ctx.defer()
      },
      tick = Some(ctx => {
        when(outValid) {
          if (hasWrite) {
            ctx.complete(outResult, outStatus)
          } else {
            when(outStatus === B"000") { ctx.complete() } otherwise { ctx.fail() }
          }
        }
      })
    )
  }

  /** Instruction-scoped SpinalHDL `StateMachine` for structured multi-cycle operations.
    *
    * Declare states with `val S = workState()` (the first `workState()` becomes the
    * active state entered on the cycle after `cmd.fire`), and finish from any state
    * with `completeAndExit(result)`, `completeAndExit()`, or `failAndExit()`.
    */
  protected abstract class CfuSubFsm(val ctx: InstructionContext) extends StateMachine {
    val IDLE: State = new State with EntryPoint
    private var firstWorkState: State = null

    protected def workState(): State = {
      val s = new State
      if (firstWorkState == null) firstWorkState = s
      s
    }

    def completeAndExit(): Unit = {
      ctx.complete()
      goto(IDLE)
    }

    def completeAndExit(result: Bits, status: Bits = B"000"): Unit = {
      ctx.complete(result, status)
      goto(IDLE)
    }

    def failAndExit(): Unit = {
      ctx.fail()
      goto(IDLE)
    }

    private[AgentCfuBase] def bindStartPulse(startPulse: Bool): Unit = {
      designRequire(firstWorkState != null, "CfuSubFsm must declare at least one workState()")
      IDLE.whenIsActive {
        when(startPulse) {
          goto(firstWorkState)
        }
      }
    }
  }

  /** Create a multi-cycle instruction controlled by its own SpinalHDL `CfuSubFsm`. */
  protected def FsmInstruction(
      name: String,
      function: Int,
      subop: Int = 0,
      enabled: Boolean = true,
      ready: () => Bool = () => True,
      resources: Option[Seq[CfuResource]]
  )(buildFsm: InstructionContext => CfuSubFsm): Instruction = {
    designRequire(resources.isDefined, s"FsmInstruction '$name' requires explicit Some(resources)")
    val startPulse = Bool()
    startPulse := False
    var subFsmBuilt = false

    val insn = Instruction(
      name = name,
      function = function,
      subop = subop,
      enabled = enabled,
      ready = ready,
      resources = resources,
      start = ctx => {
        startPulse := True
        ctx.defer()
      },
      tick = Some(_ => ())
    )
    insn.elaborate = Some(ctx => {
      if (!subFsmBuilt) {
        subFsmBuilt = true
        val subFsm = buildFsm(ctx)
        subFsm.bindStartPulse(startPulse)
      }
    })
    insn
  }

  /** The design hook. `AgentCfu.scala` overrides this with its instruction table. */
  protected def instructionTable: Seq[Instruction] = Seq.empty

  private var dispatchBuilt = false

  private def buildInstructionDispatch(): Unit = {
    val declarations = registeredInstructions.toSeq ++ instructionTable
    designRequire(declarations.forall(i => i.name.nonEmpty), "instruction name must not be empty")
    designRequire(declarations.map(_.name).distinct.size == declarations.size, "duplicate instruction name")
    designRequire(declarations.forall(i => i.function >= 0 && i.function < 8 && i.subop >= 0 && i.subop < 128),
      "instruction encoding must fit funct3/funct7")
    designRequire(declarations.map(i => (i.function, i.subop)).distinct.size == declarations.size,
      "duplicate instruction encoding")

    declarations.foreach(_.resources.foreach(validateResources))
    val entries = declarations.filter(_.enabled).map { instruction => new Area {
      val selected = matches(instruction.function, instruction.subop)
      val active = RegInit(False)
      active := active // Memory handlers use the existing FSM rather than defer.
      val hasDeferred = instruction.tick.nonEmpty || instruction.elaborate.nonEmpty
      val binding = new ResourceBinding(
        instruction.resources.getOrElse(Seq.empty),
        io.bus.cmd.fire && selected,
        hasDeferred
      )
      val ready = !binding.valid || instruction.ready()
      val startContext = new InstructionContext(active, false, hasDeferred, instruction.resources, binding)
      val tickContext = new InstructionContext(active, true, hasDeferred, instruction.resources, binding)
      instruction.elaborate.foreach(_(tickContext))
      fsm.IDLE.whenIsActive {
        when(io.bus.cmd.fire && selected) {
          when(binding.valid) {
            instruction.start(startContext)
          } otherwise {
            enqueueResponse(io.bus.cmd.request_id, B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"001")
          }
        }
      }
      instruction.tick match {
        case Some(handler) => when(active) { handler(tickContext) }
        case None => assert(!active, s"${instruction.name}: defer requires a tick handler")
      }
    }}
    val supported = entries.map(_.selected).foldLeft(False)(_ || _)
    val handlerBusy = entries.map(_.active).foldLeft(False)(_ || _)
    val selectedReady = entries.map(e => !e.selected || e.ready).foldLeft(True)(_ && _)
    when(io.bus.cmd.fire) {
      commandRequestId := io.bus.cmd.request_id
      acceptedInputs := io.bus.cmd.inputs
      acceptedRaw := io.bus.cmd.raw_insn
      acceptedState := io.bus.cmd.state_index
      acceptedCfu := io.bus.cmd.cfu_index
      when(!supported) {
        enqueueResponse(io.bus.cmd.request_id, B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"001")
      }
    }
    io.bus.cmd.ready := fsm.isActive(fsm.IDLE) && !responsePending && !handlerBusy && selectedReady
  }

  /** Build the dispatcher at the start of `prePop`: after the subclass constructor
    * body has initialized all `val`s and `Area`s, and before `super.prePop()` runs
    * `StateMachine.build()` tasks.
    */
  override def prePop(): Unit = {
    if (!dispatchBuilt) {
      dispatchBuilt = true
      buildInstructionDispatch()
    }
    super.prePop()
  }
}
