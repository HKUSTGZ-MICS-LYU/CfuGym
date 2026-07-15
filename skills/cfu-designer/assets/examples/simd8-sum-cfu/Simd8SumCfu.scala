package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib._
import vexiiriscv.execute.cfu._

object Simd8SumCfu {
  def busParameter(xlen: Int = 32) = CfuBusParameter(
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

class Simd8SumCfu(cfuParam: CfuBusParameter) extends Component {
  val io = new Bundle {
    val bus = slave(CfuBus(cfuParam))
  }

  val func3 = io.bus.cmd.function_id.asBits
  val isSum4 = func3 === B"000"

  val lanes = io.bus.cmd.inputs(0).subdivideIn(8 bits).map(_.asSInt.resize(16))
  val sum = Vec(lanes).reduceBalancedTree(_ +^ _).resize(cfuParam.CFU_OUTPUT_DATA_W)

  val rspValid = RegInit(False)
  val rspResponseId = Reg(UInt(cfuParam.CFU_REQ_RESP_ID_W bits)) init(0)
  val rspData = Reg(Bits(cfuParam.CFU_OUTPUT_DATA_W bits)) init(0)
  val rspStatus = if(cfuParam.CFU_WITH_STATUS) Reg(Bits(3 bits)) init(0) else null

  io.bus.cmd.ready := !rspValid
  io.bus.rsp.valid := rspValid
  io.bus.rsp.response_id := rspResponseId
  io.bus.rsp.outputs(0) := rspData
  if(cfuParam.CFU_WITH_STATUS) io.bus.rsp.status := rspStatus

  when(io.bus.rsp.fire) {
    rspValid := False
  }

  when(io.bus.cmd.fire) {
    rspValid := True
    rspResponseId := io.bus.cmd.request_id
    rspData := isSum4.mux(sum.asBits, B(0, cfuParam.CFU_OUTPUT_DATA_W bits))
    if(cfuParam.CFU_WITH_STATUS) rspStatus := isSum4.mux(B"000", B"001")
  }
}
