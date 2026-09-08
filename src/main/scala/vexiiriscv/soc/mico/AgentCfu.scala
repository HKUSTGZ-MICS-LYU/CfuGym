package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.fsm._

import vexiiriscv.execute.cfu._
import vexiiriscv.soc.cfu.{CfuLsu, CfuLsuParameter}

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
 * Minimal stateful CFU shell.
 *
 * The RF and memory protocol are functional, while compute/config are deliberate
 * no-op extension points. New agents can replace those extension points without
 * changing the CPU or SoC integration contract.
 */
class AgentCfu(
    cfuParam: CfuBusParameter,
    busParam: BusParameter,
    p: AgentCfuParameter
) extends Component {
  require(!p.withTilelink || busParam.dataWidth == p.xlen,
    "AgentCfu TileLink width must match AgentCfu xlen")

  val io = new Bundle {
    val bus = slave(CfuBus(cfuParam))
    val dBus = p.withTilelink generate master(tilelink.Bus(busParam))
  }

  private val regIndexWidth = p.regIndexWidth
  private val beatIndexWidth = p.beatIndexWidth
  private val beatShift = log2Up(p.xlen)

  // Register-backed first implementation: simple to inspect and deterministic to
  // elaborate. Agents may replace this storage with banked or synchronous RAM.
  val vectorRegs = Vec(Reg(Bits(p.vlen bits)) init (0), p.regDepth)

  def vectorRead(addr: UInt): Bits = vectorRegs(addr)

  def vectorReadBeat(addr: UInt, index: UInt): Bits = {
    val offset = (index.resize(log2Up(p.vlen)) << beatShift).resize(log2Up(p.vlen))
    vectorRegs(addr)(offset, p.xlen bits)
  }

  def vectorWrite(addr: UInt, data: Bits): Unit = {
    vectorRegs(addr) := data
  }

  def vectorWriteBeat(addr: UInt, index: UInt, data: Bits): Unit = {
    val offset = (index.resize(log2Up(p.vlen)) << beatShift).resize(log2Up(p.vlen))
    vectorRegs(addr)(offset, p.xlen bits) := data
  }

  val func3 = io.bus.cmd.function_id
  val isCompute = func3 === U(AgentCfuFunction.compute, cfuParam.CFU_FUNCTION_ID_W bits)
  val isConfig = func3 === U(AgentCfuFunction.config, cfuParam.CFU_FUNCTION_ID_W bits)
  val isLoad = if (p.withLoad) {
    func3 === U(AgentCfuFunction.load, cfuParam.CFU_FUNCTION_ID_W bits)
  } else {
    False
  }
  val isStore = if (p.withStore) {
    func3 === U(AgentCfuFunction.store, cfuParam.CFU_FUNCTION_ID_W bits)
  } else {
    False
  }
  val isSupported = isCompute || isConfig || isLoad || isStore

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

    IDLE.whenIsActive {
      when (io.bus.cmd.fire) {
        commandRequestId := io.bus.cmd.request_id
        if (p.withTilelink) {
          when (isLoad) {
            loadDestination := decode.rs2
            lsu.cmd.load := True
            goto(LOAD)
          } elsewhen (isStore) {
            storeSource := decode.rs2
            lsu.cmd.store := True
            goto(STORE)
          } otherwise {
            // Compute/config are intentionally inert. Unsupported function IDs
            // receive the generic CFU error status instead of hanging the CPU.
            val status = isSupported.mux(B"000", B"001")
            enqueueResponse(io.bus.cmd.request_id, B(0, cfuParam.CFU_OUTPUT_DATA_W bits), status)
          }
        } else {
          // Direct mode keeps the same response contract but has no memory
          // commands. Compute/config remain algorithm-neutral no-ops.
          val status = isSupported.mux(B"000", B"001")
          enqueueResponse(io.bus.cmd.request_id, B(0, cfuParam.CFU_OUTPUT_DATA_W bits), status)
        }
      }
    }

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

  val memoryReady = if (p.withTilelink) {
    !isLoad && !isStore || lsu.cmd.ready
  } else {
    True
  }
  io.bus.cmd.ready := fsm.isActive(fsm.IDLE) && !responsePending && memoryReady
}
