// SPDX-License-Identifier: MIT
//
// AgentCfuLib - the read-only AgentCfu platform.
//
// Design agents own only AgentCfu.scala (the instruction table / datapath design)
// and AgentCfuFiber.scala. This file holds the machinery they must not need to
// touch: the CfuResource vocabulary, Instruction/InstructionContext, the CfuBus
// shell, response buffer, vector RF, CfuLsu/FSM and the instruction dispatcher.
//
// Each run compiles an isolated workspace copy of AgentCfu.scala /
// AgentCfuFiber.scala over this tracked platform (see agents/README.md and
// project/AgentCfuSourceFilter.scala). Keeping the platform here means a
// whole-file rewrite of AgentCfu.scala cannot delete it, and the agent reads only
// the design it actually owns.
//
// Only AgentCfuBase is a new top-level name. CfuResource lives in its companion
// object so that older run workspaces, which still inline a top-level CfuResource,
// keep compiling against this repository copy.
package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.fsm._

import vexiiriscv.execute.cfu._
import vexiiriscv.soc.cfu.{CfuLsu, CfuLsuParameter}

/** Elaboration-time operand/writeback resources, inspired by native RfResource.
  *
  * A resource describes what one instruction reads and writes. It adds no CfuBus
  * field: CPU roles refer to the standard command input slots and vector roles to
  * the CFU's own register file. Vector accesses carry an explicit index source, so
  * a CPU register VALUE, an instruction register NUMBER and a vector slot are never
  * confused.
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

    sealed trait Index
    case class CpuValue(register: CpuSource) extends Index
    case class Constant(value: Int) extends Index
    case class LegacyRegisterNumber(register: CpuSource) extends Index

    case class CpuRead(role: CpuSource) extends CfuResource
    case object CpuWriteRD extends CfuResource
    case class VectorRead(role: VectorRole, index: Index) extends CfuResource
    case class VectorWrite(role: VectorRole, index: Index) extends CfuResource
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
  * Shared, non-design CFU platform: bus shell, vector RF, response buffer,
  * CfuLsu/FSM and the instruction dispatcher.
  *
  * `AgentCfu` supplies the design by overriding `instructionTable`. The dispatcher
  * is built during this base constructor so the CfuLsu state machine can consume its
  * `whenIsActive` tasks; a design table may therefore use its own primary-constructor
  * parameters, but not a `val` declared in the subclass body.
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
  private val beatShift = log2Up(p.xlen)

  // Register-backed platform storage: simple to inspect and deterministic to
  // elaborate. A design may add its own state inside instructionTable.
  val vectorRegs = Vec(Reg(Bits(p.vlen bits)) init (0), p.regDepth)

  // A single slot must be addressed statically. SpinalHDL addresses a Vec with
  // log2Up(size) bits, so a one-element Vec expects a zero-width address while the
  // generic slot index is regIndexWidth = log2Up(regDepth) max 1 = 1 bit wide. That
  // mismatch is only reported at elaboration, and it also breaks the deferred path,
  // whose latched index register carries no auto-resize tag. Reading/writing slot 0
  // directly is equivalent to a plain Reg, and a one-element Vec with a static index
  // synthesises to exactly that, so every regDepth keeps the same storage shape.
  private val singleVectorSlot = p.regDepth == 1

  /** Normalise and validate a design-supplied vector slot address.
    *
    * SpinalHDL addresses a Vec with exactly log2Up(size) bits, so an over-wide
    * address otherwise fails deep inside the backend with an opaque "Vec address
    * width mismatch". Report it here, attributed to the design, and zero-extend a
    * narrower address so the checked accessors accept any legal slot index.
    */
  private def slotAddress(addr: UInt): UInt = {
    if (singleVectorSlot) U(0, 1 bits)
    else {
      val needed = log2Up(p.regDepth)
      designRequire(widthOf(addr) <= needed,
        s"vector slot address has ${widthOf(addr)} bits, but regDepth=${p.regDepth} needs $needed")
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
  // Custom/empty tables may omit memory entries while retaining the LSU shell.
  loadDestination := loadDestination
  storeSource := storeSource
  val commandRequestId = Reg(UInt(cfuParam.CFU_REQ_RESP_ID_W bits)) init (0)

  val lsu = if (p.withTilelink) {
    new CfuLsu(
      io.dBus,
      CfuLsuParameter(
        dataWidth = p.xlen,
        vectorBytes = p.vectorBytes,
        addressWidth = p.addressWidth,
        withLoad = p.withLoad,
        withStore = p.withStore
      )
    )
  } else {
    null
  }

  if (p.withTilelink) {
    lsu.cmd.load := False
    lsu.cmd.store := False
    lsu.cmd.address := io.bus.cmd.inputs(0).asUInt.resize(p.addressWidth)
    lsu.store.data := vectorReadBeat(storeSource, lsu.store.index)

    when (lsu.load.valid) {
      vectorWriteBeat(loadDestination, lsu.load.index, lsu.load.data)
    }
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
    * CPU roles refer to standard input slots (custom0: X[rs1]/X[rs2]); custom1's
    * slot 1 remains its immediate. No opcode projection is silently changed.
    * Vector indices are full CPU values or constants, except explicit legacy
    * register-number truncation. Invalid indices complete with error before start.
    * ready/start/tick are Scala elaboration callbacks, not runtime software.
    * start is generated ONLY under cmd.fire; tick only while this entry is busy.
    */
  protected case class Instruction(
      name: String, function: Int, subop: Int = 0, enabled: Boolean = true,
      ready: () => Bool = () => True,
      start: InstructionContext => Unit,
      tick: Option[InstructionContext => Unit] = None,
      resources: Option[Seq[CfuResource]] = None
  )

  private val acceptedInputs = Vec(Reg(Bits(cfuParam.CFU_INPUT_DATA_W bits)) init(0), cfuParam.CFU_INPUTS)
  private val acceptedRaw = Reg(Bits(cfuParam.CFU_RAW_INSN_W bits)) init(0)
  private val acceptedState = Reg(cloneOf(io.bus.cmd.state_index)) init(0)
  private val acceptedCfu = Reg(cloneOf(io.bus.cmd.cfu_index)) init(0)

  private def resourceRole(r: CfuResource): Role = r match {
    case CpuRead(role) => role
    case CpuWriteRD => RD
    case VectorRead(role, _) => role
    case VectorWrite(role, _) => role
  }
  private def validateResources(rs: Seq[CfuResource]): Unit = {
    designRequire(rs.map(resourceRole).distinct.size == rs.size, "duplicate resource role")
    designRequire(rs.count { case CpuWriteRD | VectorWrite(_, _) => true; case _ => false } <= 1,
      "at most one resource write destination")
    rs.foreach {
      case VectorRead(role, index) =>
        designRequire(role == VS1 || role == VS2, "vector read role must be VS1 or VS2")
        validateIndex(index)
      case VectorWrite(role, index) =>
        designRequire(role == VD, "vector write role must be VD")
        validateIndex(index)
      case _ =>
    }
  }
  private def validateIndex(index: Index): Unit = index match {
    case Constant(n) => designRequire(n >= 0 && n < p.regDepth, "constant vector index out of range")
    case _ =>
  }
  private class ResourceBinding(rs: Seq[CfuResource], capture: Bool) extends Area {
    def slot(role: CpuSource): Bits = io.bus.cmd.inputs(if(role == RS1) 0 else 1)
    val indices = rs.collect {
      case VectorRead(role, idx) => role -> idx
      case VectorWrite(role, idx) => role -> idx
    }.map { case (role, idx) =>
      val (address, valid) = idx match {
        case CpuValue(reg) =>
          val full = slot(reg).asUInt
          (full.resize(regIndexWidth), full.resize(full.getWidth + 1) < U(p.regDepth, full.getWidth + 1 bits))
        case Constant(n) => (U(n, regIndexWidth bits), True)
        case LegacyRegisterNumber(reg) =>
          val lo = if(reg == RS1) 15 else 20
          (io.bus.cmd.raw_insn(lo + 4 downto lo).asUInt.resize(regIndexWidth), True)
      }
      val saved = Reg(UInt(regIndexWidth bits)) init(0)
      when(capture) { saved := address }
      role -> (address, saved, valid)
    }.toMap
    val valid = indices.values.map(_._3).foldLeft(True)(_ && _)
    val reads = rs.collect {
      case CpuRead(role) => role -> slot(role)
      case VectorRead(role, _) => role -> vectorRead(indices(role)._1)
    }.map { case (role, live) =>
      val saved = Reg(Bits(live.getWidth bits)) init(0)
      when(capture && valid) { saved := live }
      role -> (live, saved)
    }.toMap
  }

  protected class InstructionContext private[AgentCfuBase](private val active: Bool, deferred: Boolean,
      hasTick: Boolean, resources: Option[Seq[CfuResource]], binding: ResourceBinding) {
    def input(index: Int): Bits = {
      designRequire(resources.isEmpty, "resource-aware instruction must use read, not input")
      if (deferred) acceptedInputs(index) else io.bus.cmd.inputs(index)
    }
    def read(role: Role): Bits = {
      designRequire(binding.reads.contains(role), s"undeclared resource read: $role")
      val pair = binding.reads(role)
      if(deferred) pair._2 else pair._1
    }
    def index(role: VectorRole): UInt = {
      designRequire(binding.indices.contains(role), s"undeclared vector index: $role")
      val triple = binding.indices(role)
      if(deferred) triple._2 else triple._1
    }
    /** Error completion suppresses vector writeback; CPU receives zero irrespective of status. */
    def fail(): Unit = finish(B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"001")
    def complete(): Unit = {
      designRequire(resources.isDefined && !resources.get.exists {
        case CpuWriteRD | VectorWrite(_, _) => true; case _ => false
      }, "complete without result requires no write destination")
      finish(B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"000")
    }
    def complete(result: Bits, status: Bits = B"000"): Unit = {
      designRequire(resources.isDefined, "complete requires declared resources")
      designRequire(status.getWidth == 3, "completion status must be 3 bits")
      resources.get.collectFirst { case CpuWriteRD => CpuWriteRD; case v: VectorWrite => v } match {
        case Some(CpuWriteRD) =>
          designRequire(result.getWidth == cfuParam.CFU_OUTPUT_DATA_W, "CPU result width mismatch")
          finish(result, status)
        case Some(VectorWrite(role, _)) =>
          designRequire(result.getWidth == p.vlen, "vector result width mismatch")
          when(status === B"000") { vectorWrite(index(role), result) }
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

  /** The design hook. `AgentCfu.scala` overrides this with its instruction table.
    *
    * One entry declares a name, encoding, optional Scala `enabled`, a combinational
    * `ready`, a `start` handler and an optional multicycle `tick`. The dispatcher
    * below is the single source for decode, supported commands, acceptance and
    * response backpressure, so adding or removing an entry needs no central switch.
    * A deferred operation supplies `tick`, calls `defer()` in `start`, then calls
    * `complete`/`reply` exactly once from `tick` when ready. Context request fields
    * are latched automatically; handlers only need registers for their own state.
    * ready callbacks must be combinational and must not depend on cmd.ready.
    */
  protected def instructionTable: Seq[Instruction]

  /** Built here, during construction, because `StateMachine.build()` consumes the
    * registered `whenIsActive` tasks once at prePop time and `goto` is only legal
    * while that build runs. Calling `instructionTable` here is safe: Scala assigns
    * primary-constructor parameter fields before the superclass body runs, so a
    * design table may read its own constructor parameters. It may NOT read a `val`
    * declared in the subclass body, which is assigned after this call.
    */
  private def buildInstructionDispatch(): Unit = {
    val declarations = instructionTable
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
      val binding = new ResourceBinding(instruction.resources.getOrElse(Seq.empty), io.bus.cmd.fire && selected)
      val ready = !binding.valid || instruction.ready()
      val startContext = new InstructionContext(active, false, instruction.tick.nonEmpty, instruction.resources, binding)
      val tickContext = new InstructionContext(active, true, instruction.tick.nonEmpty, instruction.resources, binding)
      fsm.IDLE.whenIsActive {
        when(io.bus.cmd.fire && selected) {
          when(binding.valid) { instruction.start(startContext) } otherwise { enqueueResponse(io.bus.cmd.request_id, B(0, cfuParam.CFU_OUTPUT_DATA_W bits), B"001") }
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

  // Build the dispatcher now: the FSM consumes its whenIsActive tasks once, at
  // prePop time, and `goto` is only legal during that build.
  buildInstructionDispatch()
}
