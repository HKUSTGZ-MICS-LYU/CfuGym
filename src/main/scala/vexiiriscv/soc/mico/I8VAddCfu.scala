package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.fiber._
import spinal.lib._
import spinal.lib.bus._
import spinal.lib.bus.misc._
import spinal.lib.bus.tilelink._
import spinal.lib.bus.tilelink.fabric._
import spinal.lib.fsm._

import vexiiriscv.execute.cfu._
import vexiiriscv.soc.cfu._

case class I8VAddCfuParameter(
  var vlen: Int = 256,
  var xlen: Int = 32,
  var advancedMem: Boolean = false,
  var burstMem: Boolean = false
) {
  def beats = vlen / xlen
  def beatBytes = xlen / 8
  def vectorBytes = vlen / 8
  def lsuParameter = CfuLsuParameter(
    dataWidth = xlen,
    vectorBytes = vectorBytes,
    withLoad = true,
    withStore = true,
    advancedMem = advancedMem,
    burstMem = burstMem
  )
  def transferBytes = lsuParameter.transferBytes
  def pendingSize = lsuParameter.pendingSize
}

object TilelinkI8VAddCfuFiber {
  def getM2sParameters(name: Nameable, width: Int = 32, pendingSize: Int = 4, transferBytes: Int = 4) = tilelink.M2sParameters(
    addressWidth = 32,
    dataWidth = width,
    masters = List(
      tilelink.M2sAgent(
        name = name,
        mapping = List(
          tilelink.M2sSource(
            id = SizeMapping(0, pendingSize),
            emits = M2sTransfers(
              get = tilelink.SizeRange(1, transferBytes),
              putFull = tilelink.SizeRange(1, transferBytes)
            )
          )
        )
      )
    )
  )

  def getCfuBusParameters(xlen: Int = 32) = CfuBusParameter(
    CFU_VERSION = 0,
    CFU_INTERFACE_ID_W = 0,
    CFU_FUNCTION_ID_W = 3,
    CFU_REORDER_ID_W = 0,
    CFU_REQ_RESP_ID_W = 0,
    CFU_INPUTS = 2,
    CFU_INPUT_DATA_W = xlen,
    CFU_OUTPUTS = 1,
    CFU_OUTPUT_DATA_W = xlen,
    CFU_FLOW_REQ_READY_ALWAYS = false,
    CFU_FLOW_RESP_READY_ALWAYS = false,
    CFU_WITH_STATUS = true,
    CFU_RAW_INSN_W = 32,
    CFU_CFU_ID_W = 4,
    CFU_STATE_INDEX_NUM = 5
  )
}

case class I8VAddCfuSpec(vaddParam: I8VAddCfuParameter) extends TilelinkCfuSpec {
  override def m2sParameters(name: Nameable) = TilelinkI8VAddCfuFiber.getM2sParameters(
    name,
    vaddParam.xlen,
    vaddParam.pendingSize,
    vaddParam.transferBytes
  )

  override def cfuBusParameter(xlen: Int) = TilelinkI8VAddCfuFiber.getCfuBusParameters(xlen)

  override def build(cfuParam: CfuBusParameter, cfuBus: CfuBus, dBus: tilelink.Bus) = new Area {
    val cfu = new I8VAddCfu(cfuParam, dBus.p, vaddParam)

    cfu.io.bus <> cfuBus
    cfu.io.dBus <> dBus
  }
}

class TilelinkI8VAddCfuFiber(vaddParam: I8VAddCfuParameter, xlen: Int) extends TilelinkCfuFiber(I8VAddCfuSpec(vaddParam), xlen)

class I8VAddCfu(cfuParam: CfuBusParameter, busParam: BusParameter, p: I8VAddCfuParameter) extends Component {
  val xlen = busParam.dataWidth
  require(p.xlen == xlen, s"I8VAddCfu xlen ${p.xlen} must match TileLink data width $xlen")
  require(p.vlen % xlen == 0, "I8VAddCfu vlen must be a multiple of the TileLink data width")
  require(xlen % 8 == 0, "I8VAddCfu TileLink data width must be byte aligned")
  if(p.burstMem) require(isPow2(p.vectorBytes), "I8VAddCfu burst mode requires a power-of-two vector byte size")

  val beatCount = p.vlen / xlen
  val beatBytes = xlen / 8
  val vectorBytes = p.vlen / 8

  val io = new Bundle {
    val bus = slave(CfuBus(cfuParam))
    val dBus = master(tilelink.Bus(busParam))
  }

  val func3 = io.bus.cmd.function_id
  val isReset = func3 === U(0, 3 bits)
  val isLoadA = func3 === U(1, 3 bits)
  val isLoadB = func3 === U(2, 3 bits)
  val isAdd = func3 === U(3, 3 bits)
  val isStoreC = func3 === U(4, 3 bits)

  val vecA = Vec(Reg(Bits(xlen bits)) init(0), beatCount)
  val vecB = Vec(Reg(Bits(xlen bits)) init(0), beatCount)
  val vecC = Vec(Reg(Bits(xlen bits)) init(0), beatCount)

  def addPackedI8(a: Bits, b: Bits): Bits = {
    val result = Bits(xlen bits)
    val aLanes = a.subdivideIn(8 bits)
    val bLanes = b.subdivideIn(8 bits)
    for(i <- 0 until beatBytes) {
      val sum = aLanes(i).asUInt + bLanes(i).asUInt
      result(i * 8, 8 bits) := sum.resize(8).asBits
    }
    result
  }

  val rspValid = RegInit(False)
  val rspResponseId = Reg(UInt(cfuParam.CFU_REQ_RESP_ID_W bits)) init(0)
  val rspData = Reg(Bits(cfuParam.CFU_OUTPUT_DATA_W bits)) init(0)
  val rspStatus = if(cfuParam.CFU_WITH_STATUS) Reg(Bits(3 bits)) init(0) else null

  def complete(status: Bits = B"000", data: Bits = B(0, cfuParam.CFU_OUTPUT_DATA_W bits)): Unit = {
    rspValid := True
    rspData := data
    if(cfuParam.CFU_WITH_STATUS) rspStatus := status
  }

  io.bus.rsp.valid := rspValid
  io.bus.rsp.response_id := rspResponseId
  io.bus.rsp.outputs(0) := rspData
  if(cfuParam.CFU_WITH_STATUS) io.bus.rsp.status := rspStatus

  when(io.bus.rsp.fire) {
    rspValid := False
  }

  val loadB = Reg(Bool()) init(False)

  val lsu = new CfuLsu(io.dBus, p.lsuParameter)
  lsu.cmd.load := False
  lsu.cmd.store := False
  lsu.cmd.address := 0
  lsu.store.data := vecC(lsu.store.index)

  when(lsu.load.valid) {
    when(loadB) {
      vecB(lsu.load.index) := lsu.load.data
    } otherwise {
      vecA(lsu.load.index) := lsu.load.data
    }
  }

  val fsm = new StateMachine {
    val IDLE = new State with EntryPoint
    val LOAD = new State
    val STORE = new State

    IDLE.whenIsActive {
      when(io.bus.cmd.fire) {
        rspResponseId := io.bus.cmd.request_id
        when(isReset) {
          for(i <- 0 until beatCount) {
            vecA(i) := 0
            vecB(i) := 0
            vecC(i) := 0
          }
          complete()
        } elsewhen(isLoadA || isLoadB) {
          loadB := isLoadB
          lsu.cmd.load := True
          lsu.cmd.address := io.bus.cmd.inputs(0).asUInt.resized
          goto(LOAD)
        } elsewhen(isAdd) {
          for(i <- 0 until beatCount) {
            vecC(i) := addPackedI8(vecA(i), vecB(i))
          }
          complete()
        } elsewhen(isStoreC) {
          lsu.cmd.store := True
          lsu.cmd.address := io.bus.cmd.inputs(0).asUInt.resized
          goto(STORE)
        } otherwise {
          complete(B"001")
        }
      }
    }

    LOAD.whenIsActive {
      when(lsu.done) {
        complete(lsu.error.mux(B"010", B"000"))
        goto(IDLE)
      }
    }

    STORE.whenIsActive {
      when(lsu.done) {
        complete(lsu.error.mux(B"010", B"000"))
        goto(IDLE)
      }
    }
  }

  io.bus.cmd.ready := fsm.isActive(fsm.IDLE) && !rspValid && lsu.cmd.ready
}
