package vexiiriscv.execute.cfu

import spinal.core._
import spinal.lib._

case class CfuBusReorderSlot(outputWidth : Int) extends Bundle {
  val valid = Bool()
  val done = Bool()
  val output = Bits(outputWidth bits)
  val status = Bits(3 bits)
}

object CfuBusReorderSlot {
  def zero(outputWidth : Int) : CfuBusReorderSlot = {
    val s = CfuBusReorderSlot(outputWidth)
    s.valid := False
    s.done := False
    s.output := B(0, outputWidth bits)
    s.status := B(0, 3 bits)
    s
  }
}

/**
  * Small bus-only reorder adapter for the classic CfuPlugin path.
  *
  * The upstream CPU side remains in-order and ID-less.  The downstream CFU
  * side receives allocated request IDs and may return responses out of order.
  * Responses are buffered and presented upstream in command issue order.
  */
class CfuBusReorderAdapter(upstreamParameter : CfuBusParameter,
                           downstreamParameter : CfuBusParameter,
                           depth : Int) extends Component {
  require(depth > 1, "CFU bus reorder depth must be greater than one")
  require((depth & (depth - 1)) == 0, "CFU bus reorder depth must be a power of two")
  require(upstreamParameter.CFU_REQ_RESP_ID_W == 0,
    "CFU bus reorder adapter expects an ID-less upstream CfuPlugin bus")
  require(downstreamParameter.CFU_REQ_RESP_ID_W >= log2Up(depth),
    "CFU bus reorder adapter requires enough downstream request/response ID bits")
  require(upstreamParameter.copy(CFU_REQ_RESP_ID_W = downstreamParameter.CFU_REQ_RESP_ID_W) == downstreamParameter,
    "CFU bus reorder adapter only supports changing the request/response ID width")
  require(upstreamParameter.CFU_OUTPUTS == 1,
    "CFU bus reorder adapter currently supports one output")

  val io = new Bundle {
    val upstream = slave(CfuBus(upstreamParameter))
    val downstream = master(CfuBus(downstreamParameter))
  }

  private val idWidth = log2Up(depth)
  private val countWidth = log2Up(depth + 1)

  val issuePtr = Reg(UInt(idWidth bits)) init(0)
  val retirePtr = Reg(UInt(idWidth bits)) init(0)
  val occupancy = Reg(UInt(countWidth bits)) init(0)
  val full = occupancy === U(depth, countWidth bits)

  val slots = Vec.fill(depth)(RegInit(CfuBusReorderSlot.zero(upstreamParameter.CFU_OUTPUT_DATA_W)))

  val head = slots(retirePtr)
  val headReady = head.valid && head.done

  // Command path: one-to-one pass-through of the ID-less upstream command,
  // throttled when the buffer is full, allocating an issue-slot ID to each
  // accepted command.
  io.downstream.cmd.translateFrom(io.upstream.cmd.haltWhen(full)) { (dst, src) =>
    dst.function_id := src.function_id
    dst.reorder_id  := src.reorder_id
    dst.inputs      := src.inputs
    dst.state_index := src.state_index
    dst.cfu_index   := src.cfu_index
    dst.raw_insn    := src.raw_insn
    dst.request_id  := issuePtr.resized
  }

  when(io.downstream.cmd.fire) {
    val slot = slots(issuePtr)
    slot.valid := True
    slot.done := False
    slot.output := B(0, upstreamParameter.CFU_OUTPUT_DATA_W bits)
    slot.status := B(0, 3 bits)
    issuePtr := issuePtr + 1
  }

  // Response path: buffer each response by request ID, then present them
  // upstream in command issue order (out-of-order arrivals are reordered).
  io.downstream.rsp.ready := True
  val responseSlot = io.downstream.rsp.response_id.resize(idWidth)
  val responseInRange = if(downstreamParameter.CFU_REQ_RESP_ID_W > idWidth)
    io.downstream.rsp.response_id < U(depth, downstreamParameter.CFU_REQ_RESP_ID_W bits)
  else True
  val responseKnown = slots(responseSlot).valid ||
    (io.downstream.cmd.fire && responseSlot === issuePtr)
  // A response targeting the current retire head is presented upstream in the
  // same cycle (combinational bypass), so the common in-order path pays no
  // buffer-register round-trip.  Out-of-order responses are still buffered.
  val rspHit = io.downstream.rsp.fire && responseSlot === retirePtr

  when(io.downstream.rsp.fire) {
    assert(responseInRange, "CFU bus reorder adapter received an out-of-range response ID")
    assert(responseKnown, "CFU bus reorder adapter received a response for a non-outstanding request")
    when(responseKnown) {
      when(slots(responseSlot).valid) {
        assert(!slots(responseSlot).done, "CFU bus reorder adapter received a duplicate response ID")
      }
      val slot = slots(responseSlot)
      slot.valid := True
      slot.done := True
      slot.output := io.downstream.rsp.outputs(0)
      if(upstreamParameter.CFU_WITH_STATUS) slot.status := io.downstream.rsp.status
    }
  }

  io.upstream.rsp.valid := headReady || rspHit
  io.upstream.rsp.outputs(0) := Mux(rspHit, io.downstream.rsp.outputs(0), head.output)
  if(upstreamParameter.CFU_WITH_STATUS)
    io.upstream.rsp.status := Mux(rspHit, io.downstream.rsp.status, head.status)

  when(io.upstream.rsp.fire) {
    head.valid := False
    head.done := False
    retirePtr := retirePtr + 1
  }

  when(io.downstream.cmd.fire && !io.upstream.rsp.fire) {
    occupancy := occupancy + 1
  } elsewhen(!io.downstream.cmd.fire && io.upstream.rsp.fire) {
    occupancy := occupancy - 1
  }
}
