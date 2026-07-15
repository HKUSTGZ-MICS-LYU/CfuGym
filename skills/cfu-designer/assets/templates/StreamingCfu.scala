package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib._
import spinal.lib.bus._
import spinal.lib.bus.tilelink._
import spinal.lib.fsm._
import spinal.lib.misc.pipeline._
import vexiiriscv.execute.cfu._

case class StreamingCfuParameter(
  var xlen: Int = 32,
  var vlen: Int = 256,
  var regDepth: Int = 2,
  var pipe: Boolean = true
) {
  def pendingSize: Int = vlen / xlen
}

class StreamingCfu(cfuParam: CfuBusParameter,
                   busParam: BusParameter,
                   p: StreamingCfuParameter) extends Component {
  val xlen = busParam.dataWidth
  val vlen = p.vlen
  val nLoad = vlen / xlen
  val regSelWidth = log2Up(p.regDepth) max 1
  val loadIdWidth = log2Up(nLoad) max 1

  val io = new Bundle {
    val bus = slave(CfuBus(cfuParam))
    val dBus = master(tilelink.Bus(busParam))
  }

  val func3 = io.bus.cmd.function_id.asBits
  val isCompute = func3 === B"001"
  val isConfig  = func3 === B"010"
  val isLoad    = func3 === B"100"

  val decode = new Area {
    val FUNC7 = io.bus.cmd.raw_insn(31 downto 25).asUInt
    val RS1_RAW = io.bus.cmd.raw_insn(19 downto 15).asUInt
    val RS2_RAW = io.bus.cmd.raw_insn(24 downto 20).asUInt
    val RS1 = RS1_RAW.resize(regSelWidth)
    val RS2 = RS2_RAW.resize(regSelWidth)
  }

  io.bus.rsp.valid := False
  io.bus.rsp.response_id := io.bus.cmd.request_id
  io.bus.rsp.outputs(0) := 0
  if(cfuParam.CFU_WITH_STATUS) io.bus.rsp.status := B"000"

  val regs = Vec(Reg(Bits(vlen bits)) init(0), p.regDepth)
  val cfg = Reg(UInt(8 bits)) init(0)

  val compute = new Area {
    val sel = Bool()
    sel := False

    val stages = Array.fill(if(p.pipe) 2 else 1)(Node())
    val extractStage = stages.head
    val computeStage = stages.last

    val SEL = Payload(Bool())
    val OPA = Payload(Bits(xlen bits))
    val OPB = Payload(Bits(xlen bits))
    val RES = Payload(Bits(xlen bits))

    val extract = new extractStage.Area {
      SEL := sel
      OPA := io.bus.cmd.inputs(0).asBits
      OPB := io.bus.cmd.inputs(1).asBits
    }

    val datapath = new computeStage.Area {
      // Replace this with the real datapath. Keep state updates gated by SEL.
      RES := (OPA.asUInt + OPB.asUInt + cfg.resize(xlen)).asBits
    }

    val result = computeStage(RES)
    val done = computeStage(SEL)

    if(p.pipe) {
      Builder(StageLink(stages(0), stages(1)))
    }
  }

  val baseAddr = Reg(UInt(32 bits)) init(0)
  val offsetAddr = Reg(UInt(32 bits)) init(0)
  val offsetNext = offsetAddr + (xlen / 8)
  val accessAddr = baseAddr + offsetAddr
  val loadDst = Reg(UInt(regSelWidth bits)) init(0)
  val memValid = RegInit(False)
  val memReady = RegInit(False)
  val memFireId = Reg(UInt(loadIdWidth bits)) init(0)
  val loadHits = Vec.fill(nLoad)(RegInit(False))
  val loadCount = loadHits.sCount(True)

  io.dBus.a.opcode  := tilelink.Opcode.A.GET
  io.dBus.a.param   := tilelink.Param.Hint.NO_ALLOCATE_ON_MISS
  io.dBus.a.source  := memFireId
  io.dBus.a.data    := 0
  io.dBus.a.address := accessAddr
  io.dBus.a.mask    := B(xlen / 8 bits, default -> True)
  io.dBus.a.size    := log2Up(xlen / 8)
  io.dBus.a.corrupt := False
  io.dBus.a.valid   := memValid
  io.dBus.d.ready   := memReady

  val fsm = new StateMachine {
    val IDLE = new State with EntryPoint
    val LOAD = new State
    val COMPUTE = new State

    IDLE.whenIsActive {
      when(io.bus.cmd.fire) {
        when(isConfig) {
          cfg := decode.RS1_RAW.resize(8)
          io.bus.rsp.valid := True
        }
        when(isLoad) {
          goto(LOAD)
        }
        when(isCompute) {
          compute.sel := True
          if(p.pipe) goto(COMPUTE)
          else {
            io.bus.rsp.valid := True
            io.bus.rsp.outputs(0) := compute.result
          }
        }
      }
    }

    COMPUTE.whenIsActive {
      when(compute.done) {
        io.bus.rsp.valid := True
        io.bus.rsp.outputs(0) := compute.result
        goto(IDLE)
      }
    }

    LOAD.onEntry {
      baseAddr := io.bus.cmd.inputs(0).asUInt.resized
      loadDst := decode.RS2
      offsetAddr := 0
      memFireId := 0
      memValid := True
      memReady := True
      loadHits.foreach(_ := False)
    }

    LOAD.whenIsActive {
      when(io.dBus.a.fire) {
        offsetAddr := offsetNext
        if(nLoad != 1) memFireId := memFireId + 1
        when(offsetNext === U(vlen / 8, 32 bits)) {
          memValid := False
        }
      }

      when(io.dBus.d.fire) {
        loadHits(io.dBus.d.source) := True
        regs(loadDst)(io.dBus.d.source.resize(loadIdWidth) << log2Up(xlen), xlen bits) := io.dBus.d.data
        when(loadCount === U(nLoad - 1, log2Up(nLoad + 1) bits)) {
          memReady := False
          io.bus.rsp.valid := True
          goto(IDLE)
        }
      }
    }

    LOAD.onExit {
      memValid := False
      memReady := False
    }
  }

  io.bus.cmd.ready := fsm.isActive(fsm.IDLE)
}
