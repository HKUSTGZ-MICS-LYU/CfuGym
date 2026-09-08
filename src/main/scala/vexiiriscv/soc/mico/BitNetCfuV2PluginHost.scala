package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib._
import spinal.lib.bus._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.misc.plugin._

import vexiiriscv.execute.cfu._
import vexiiriscv.soc.cfu.{CfuLsu, CfuLsuParameter}

import scala.collection.mutable

class BitNetCfuV2PluginHost(cfuParam: CfuBusParameter,
                            busParam: BusParameter,
                            hostParam: BitNetCfuV2PluginHostParameter) extends Component {
  def this(cfuParam: CfuBusParameter,
           busParam: BusParameter,
           p: BitNetCfuV2Parameter) = this(cfuParam, busParam, BitNetCfuV2PluginHostParameter(bitNet = p))

  val io = new Bundle {
    val bus = slave(CfuBus(cfuParam))
    val dBus = master(tilelink.Bus(busParam))
  }

  hostParam.legalize()
  val host = new PluginHost()
  host.asHostOf(hostParam.plugins(this, cfuParam, busParam))
}

case class Bnc2LsuPluginParameter(withLoad: Boolean = true,
                                  withStore: Boolean = false,
                                  advancedMem: Option[Boolean] = None,
                                  burstMem: Option[Boolean] = None) {
  def advancedMemOf(p: BitNetCfuV2Parameter): Boolean = advancedMem.getOrElse(p.advancedMem)
  def burstMemOf(p: BitNetCfuV2Parameter): Boolean = burstMem.getOrElse(p.burstMem)
}

case class Bnc2QuantizerPluginParameter(withQ2T: Option[Boolean] = None,
                                        withQ8: Option[Boolean] = None,
                                        quantStandard: Option[Boolean] = None,
                                        q8ComparePipe: Option[Boolean] = None) {
  def withQ2TOf(p: BitNetCfuV2Parameter): Boolean = withQ2T.getOrElse(p.withQ2T)
  def withQ8Of(p: BitNetCfuV2Parameter): Boolean = withQ8.getOrElse(p.withQ8)
  def quantStandardOf(p: BitNetCfuV2Parameter): Boolean = quantStandard.getOrElse(p.quantStandard)
  def q8ComparePipeOf(p: BitNetCfuV2Parameter): Boolean = q8ComparePipe.getOrElse(p.q8ComparePipe)
}

case class Bnc2DotpPluginParameter(withQ2T: Option[Boolean] = None) {
  def withQ2TOf(p: BitNetCfuV2Parameter): Boolean = withQ2T.getOrElse(p.withQ2T)
}

case class Bnc2BlockSchedulerPluginParameter(withQ2T: Option[Boolean] = None,
                                             minQueueDepth: Int = 2) {
  def withQ2TOf(p: BitNetCfuV2Parameter): Boolean = withQ2T.getOrElse(p.withQ2T)
}

case class BitNetCfuV2PluginHostParameter(bitNet: BitNetCfuV2Parameter = BitNetCfuV2Parameter(),
                                          withDecode: Boolean = true,
                                          withControl: Boolean = true,
                                          withRf: Boolean = true,
                                          withQueues: Boolean = true,
                                          withResult: Boolean = true,
                                          withState: Boolean = true,
                                          withLsu: Boolean = true,
                                          withQuantizer: Boolean = true,
                                          withDotp: Boolean = true,
                                          withBlockScheduler: Boolean = true,
                                          withBusDecode: Boolean = true,
                                          lsu: Bnc2LsuPluginParameter = Bnc2LsuPluginParameter(),
                                          quantizer: Bnc2QuantizerPluginParameter = Bnc2QuantizerPluginParameter(),
                                          dotp: Bnc2DotpPluginParameter = Bnc2DotpPluginParameter(),
                                          blockScheduler: Bnc2BlockSchedulerPluginParameter = Bnc2BlockSchedulerPluginParameter()) {
  def legalize(): Unit = {
    require(withDecode, "BNCFUv2 plugin-host needs Bnc2DecodePlugin")
    require(withControl, "BNCFUv2 plugin-host needs Bnc2ControlPlugin")
    require(withRf, "BNCFUv2 plugin-host needs Bnc2RfPlugin")
    require(withQueues, "BNCFUv2 plugin-host needs Bnc2QueuePlugin")
    require(withResult, "BNCFUv2 plugin-host needs Bnc2ResultPlugin")
    require(withState, "BNCFUv2 plugin-host needs Bnc2StatePlugin")
    if(withBusDecode) {
      require(withLsu, "Bnc2BusDecodePlugin currently routes load commands through Bnc2LsuPlugin")
      require(withQuantizer, "Bnc2BusDecodePlugin currently routes quant commands through Bnc2QuantizerPlugin")
      require(withDotp, "Bnc2BusDecodePlugin currently routes dot commands through Bnc2DotpPlugin")
      require(withBlockScheduler, "Bnc2BusDecodePlugin currently routes block commands through Bnc2BlockSchedulerPlugin")
    }
    require(lsu.withLoad, "BNCFUv2 currently requires load support in Bnc2LsuPlugin")
    require(!lsu.withStore, "BNCFUv2 does not use store support in Bnc2LsuPlugin")
    require(quantizer.withQ2TOf(bitNet) || quantizer.withQ8Of(bitNet),
      "BNCFUv2 plugin-host needs at least one quantizer mode")
    require(dotp.withQ2TOf(bitNet) == quantizer.withQ2TOf(bitNet),
      "Bnc2DotpPlugin Q2T decode must match Bnc2QuantizerPlugin Q2T decode")
    require(blockScheduler.withQ2TOf(bitNet) == quantizer.withQ2TOf(bitNet),
      "Bnc2BlockSchedulerPlugin Q2T decode must match Bnc2QuantizerPlugin Q2T decode")
  }

  def plugins(owner: BitNetCfuV2PluginHost,
              cfuParam: CfuBusParameter,
              busParam: BusParameter): Seq[FiberPlugin] = new Area {
    val plugins = mutable.ArrayBuffer[FiberPlugin]()
    plugins += new Bnc2ContextPlugin(owner, cfuParam, busParam, bitNet)
    if(withDecode) plugins += new Bnc2DecodePlugin()
    if(withControl) plugins += new Bnc2ControlPlugin()
    if(withRf) plugins += new Bnc2RfPlugin()
    if(withQueues) plugins += new Bnc2QueuePlugin()
    if(withResult) plugins += new Bnc2ResultPlugin()
    if(withState) plugins += new Bnc2StatePlugin()
    if(withLsu) plugins += new Bnc2LsuPlugin(lsu)
    if(withQuantizer) plugins += new Bnc2QuantizerPlugin(quantizer)
    if(withDotp) plugins += new Bnc2DotpPlugin(dotp)
    if(withBlockScheduler) plugins += new Bnc2BlockSchedulerPlugin(blockScheduler)
    if(withBusDecode) plugins += new Bnc2BusDecodePlugin()
  }.plugins.toSeq
}

case class Bnc2DecodeSpec(name: String,
                          func3: Int,
                          func7Eq: Option[Int] = None,
                          func7Ne: Option[Int] = None,
                          usesLoad: Boolean = false,
                          usesQuant: Boolean = false,
                          usesDot: Boolean = false,
                          isConfig: Boolean = false) {
  def hit(functionId: Bits, func7Value: UInt): Bool = {
    val ret = Bool()
    ret := functionId === U(func3, 3 bits).asBits
    for(value <- func7Eq) ret clearWhen(func7Value =/= U(value, 7 bits))
    for(value <- func7Ne) ret clearWhen(func7Value === U(value, 7 bits))
    ret
  }
}

object Bnc2Ops {
  val DOT_HOLD = "dot_hold"
  val DOT = "dot"
  val CFG_RESET = "cfg_reset"
  val CFG_QTYPE = "cfg_qtype"
  val CFG_POLL = "cfg_poll"
  val CFG_READ = "cfg_read"
  val CFG_READ_WAIT = "cfg_read_wait"
  val CFG_READ_ACC_WAIT = "cfg_read_acc_wait"
  val CFG_BLOCK_COUNT = "cfg_block_count"
  val Q2T = "q2t"
  val LOAD = "load"
  val Q8 = "q8"
  val Q2T_DOT = "q2t_dot"
  val LOAD_Q2T_DOT = "load_q2t_dot"
  val LOAD_Q2T_DOT_BLOCK = "load_q2t_dot_block"
}

class Bnc2DecodePlugin extends FiberPlugin {
  val decodingLock = spinal.core.fiber.Retainer()
  private val specs = mutable.ArrayBuffer[Bnc2DecodeSpec]()

  def add(spec: Bnc2DecodeSpec): Unit = {
    require(!specs.exists(_.name == spec.name), s"Duplicated BNCFUv2 decode spec ${spec.name}")
    specs += spec
  }

  val logic = during build new Area {
    decodingLock.await()
    val c = host[Bnc2ContextPlugin]

    val func3 = c.io.bus.cmd.function_id.asBits
    val FUNC7 = c.io.bus.cmd.raw_insn(31 downto 25).asUInt
    val RS1_RAW = c.io.bus.cmd.raw_insn(19 downto 15).asUInt
    val RS2_RAW = c.io.bus.cmd.raw_insn(24 downto 20).asUInt
    val RD_RAW = c.io.bus.cmd.raw_insn(11 downto 7).asUInt
    val RS1 = RS1_RAW.resize(c.regSelWidth)
    val RS2 = RS2_RAW.resize(c.regSelWidth)
    val RD = RD_RAW.resize(c.regSelWidth)
    val RS1_VALID = if(c.regDepth == (1 << c.regSelWidth)) True else RS1 < U(c.regDepth, c.regSelWidth bits)
    val RS2_VALID = if(c.regDepth == (1 << c.regSelWidth)) True else RS2 < U(c.regDepth, c.regSelWidth bits)
    val RD_VALID = if(c.regDepth == (1 << c.regSelWidth)) True else RD < U(c.regDepth, c.regSelWidth bits)

    private val hits = specs.map(spec => spec.name -> spec.hit(func3, FUNC7)).toMap
    def is(name: String): Bool = hits.getOrElse(name, False)
    private def group(pred: Bnc2DecodeSpec => Boolean): Bool = {
      val selected = specs.filter(pred).map(spec => is(spec.name))
      if(selected.isEmpty) False else selected.orR
    }

    val isLoad = group(_.usesLoad)
    val isQuant = group(_.usesQuant)
    val isDot = group(_.usesDot)
    val isCfg = group(_.isConfig)
    val isKnown = group(_ => true)
    val isCfgFunc = func3 === U(2, 3 bits).asBits
  }
}

class Bnc2ContextPlugin(owner: BitNetCfuV2PluginHost,
                        val cfuParam: CfuBusParameter,
                        val busParam: BusParameter,
                        val p: BitNetCfuV2Parameter) extends FiberPlugin {
  import BitNetCfuCompute._

  val xlen = busParam.dataWidth
  val vlen = p.vlen
  val regDepth = p.regDepth
  val maclen = p.maclen
  val quantWidth = p.quantWidthEffective
  val lanes = maclen / 8
  val quantLanes = quantWidth / 32
  val weightSliceBitsMax = lanes * 2
  val vlenLog2 = log2Up(vlen)
  val regSelWidth = log2Up(regDepth) max 1
  val nLoad = vlen / xlen
  val nCompute = vlen / maclen
  val quantChunks = vlen / quantWidth
  val reslen = cfuParam.CFU_OUTPUT_DATA_W

  require(cfuParam.CFU_OUTPUTS == 1)
  require(cfuParam.CFU_OUTPUT_DATA_W == 32, "BNCFUv2 plugin-host currently reports one XLEN result")
  private def isPow2(value: Int): Boolean = value > 0 && ((value & (value - 1)) == 0)
  require(isPow2(vlen), "BNCFUv2 plugin-host vlen must be a power of two")
  require(isPow2(xlen), "BNCFUv2 plugin-host xlen must be a power of two")
  require(isPow2(maclen), "BNCFUv2 plugin-host maclen must be a power of two")
  require(vlen % xlen == 0, "BNCFUv2 plugin-host vlen must be a multiple of xlen")
  require(vlen % maclen == 0, "BNCFUv2 plugin-host vlen must be a multiple of maclen")
  require(maclen % 8 == 0, "BNCFUv2 plugin-host maclen must hold complete int8 lanes")
  require(weightSliceBitsMax <= vlen, "BNCFUv2 plugin-host weight slice must fit in one vector register")
  require(regDepth >= 3, "BNCFUv2 plugin-host needs at least three RF registers")
  require(p.loadQueueDepth >= 1 && p.quantQueueDepth >= 1 && p.dotQueueDepth >= 1)
  require(p.resultQueueDepth >= 1)
  require(p.blockCount >= 1)
  require(!p.computePipe, "BNCFUv2 plugin-host first pass keeps the dot path unpipelined")
  require(p.withQ2T || p.withQ8, "BNCFUv2 plugin-host needs at least one quantizer mode")
  require(quantWidth % 32 == 0, "BNCFUv2 plugin-host quantWidth must hold complete FP32 lanes")
  require(vlen % quantWidth == 0, "BNCFUv2 plugin-host vlen must be a multiple of quantWidth")
  if(p.withQ8) require((vlen / 32) * 8 <= vlen, "BNCFUv2 plugin-host Q8 result must fit in one RF register")

  def io = owner.io
  def weightInc(qType: UInt) = qType.mux(
    U(Q1B) -> U(lanes, vlenLog2 bits),
    default -> U(weightSliceBitsMax, vlenLog2 bits)
  )
}

class Bnc2ControlPlugin extends FiberPlugin {
  val setup = during setup new Area {
    val decode = host[Bnc2DecodePlugin]
    val lock = retains(decode.decodingLock)
    import Bnc2Ops._
    decode.add(Bnc2DecodeSpec(CFG_RESET, 2, func7Eq = Some(0), isConfig = true))
    decode.add(Bnc2DecodeSpec(CFG_QTYPE, 2, func7Eq = Some(1), isConfig = true))
    decode.add(Bnc2DecodeSpec(CFG_POLL, 2, func7Eq = Some(2), isConfig = true))
    decode.add(Bnc2DecodeSpec(CFG_READ, 2, func7Eq = Some(3), isConfig = true))
    decode.add(Bnc2DecodeSpec(CFG_READ_WAIT, 2, func7Eq = Some(4), isConfig = true))
    decode.add(Bnc2DecodeSpec(CFG_READ_ACC_WAIT, 2, func7Eq = Some(5), isConfig = true))
    decode.add(Bnc2DecodeSpec(CFG_BLOCK_COUNT, 2, func7Eq = Some(6), isConfig = true))
    lock.release()
  }

  val logic = during build new Area {
    val doReset = Bool()
  }
}

class Bnc2RfPlugin extends FiberPlugin {
  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val vecRegs = Vec(Reg(Bits(c.vlen bits)) init(0), c.regDepth)
    val vecOffsets = Vec(Reg(UInt(c.vlenLog2 bits)) init(0), c.regDepth)
    val rfLoadPending = Vec(Reg(Bool()) init(False), c.regDepth)
    val rfQuantPending = Vec(Reg(Bool()) init(False), c.regDepth)
    val rfDotPending = Vec(Reg(Bool()) init(False), c.regDepth)

    def producerBusy(index: UInt, valid: Bool): Bool = {
      val ret = Bool()
      ret := False
      for(i <- 0 until c.regDepth) {
        when(valid && index === U(i, c.regSelWidth bits)) {
          ret := rfLoadPending(i) || rfQuantPending(i)
        }
      }
      ret
    }

    def writeBusy(index: UInt, valid: Bool): Bool = {
      val ret = Bool()
      ret := False
      for(i <- 0 until c.regDepth) {
        when(valid && index === U(i, c.regSelWidth bits)) {
          ret := rfLoadPending(i) || rfQuantPending(i) || rfDotPending(i)
        }
      }
      ret
    }

    val ctrl = host[Bnc2ControlPlugin].logic.get
    when(ctrl.doReset) {
      vecRegs.foreach(_ := 0)
      vecOffsets.foreach(_ := U(0, c.vlenLog2 bits))
      rfLoadPending.foreach(_ := False)
      rfQuantPending.foreach(_ := False)
      rfDotPending.foreach(_ := False)
    }
  }
}

class Bnc2QueuePlugin extends FiberPlugin {
  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val ctrl = host[Bnc2ControlPlugin].logic.get

    val loadQ = StreamFifo(BitNetCfuV2LoadJob(c.regSelWidth), c.p.loadQueueDepth)
    val quantQ = StreamFifo(BitNetCfuV2QuantJob(c.regSelWidth), c.p.quantQueueDepth)
    val dotQ = StreamFifo(BitNetCfuV2DotJob(c.regSelWidth), c.p.dotQueueDepth)

    val cmdLoad = Stream(BitNetCfuV2LoadJob(c.regSelWidth))
    val cmdQuant = Stream(BitNetCfuV2QuantJob(c.regSelWidth))
    val cmdDot = Stream(BitNetCfuV2DotJob(c.regSelWidth))
    val blockLoad = Stream(BitNetCfuV2LoadJob(c.regSelWidth))
    val blockQuant = Stream(BitNetCfuV2QuantJob(c.regSelWidth))
    val blockDot = Stream(BitNetCfuV2DotJob(c.regSelWidth))

    loadQ.io.push.valid := cmdLoad.valid || blockLoad.valid
    loadQ.io.push.payload := cmdLoad.payload
    when(blockLoad.valid) { loadQ.io.push.payload := blockLoad.payload }
    cmdLoad.ready := loadQ.io.push.ready && !blockLoad.valid
    blockLoad.ready := loadQ.io.push.ready

    quantQ.io.push.valid := cmdQuant.valid || blockQuant.valid
    quantQ.io.push.payload := cmdQuant.payload
    when(blockQuant.valid) { quantQ.io.push.payload := blockQuant.payload }
    cmdQuant.ready := quantQ.io.push.ready && !blockQuant.valid
    blockQuant.ready := quantQ.io.push.ready

    dotQ.io.push.valid := cmdDot.valid || blockDot.valid
    dotQ.io.push.payload := cmdDot.payload
    when(blockDot.valid) { dotQ.io.push.payload := blockDot.payload }
    cmdDot.ready := dotQ.io.push.ready && !blockDot.valid
    blockDot.ready := dotQ.io.push.ready

    Seq(loadQ, quantQ, dotQ).foreach(_.io.flush := ctrl.doReset)
  }
}

class Bnc2ResultPlugin extends FiberPlugin {
  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val ctrl = host[Bnc2ControlPlugin].logic.get
    val accumulatorCapacity = c.p.resultQueueDepth max c.p.blockCount
    val countWidth = log2Up(accumulatorCapacity + 1) max 1

    val resultQ = StreamFifo(Bits(c.reslen bits), c.p.resultQueueDepth)
    val resultIn = Stream(Bits(c.reslen bits))
    val accIn = Flow(Bits(c.reslen bits))

    val aggSum = Reg(SInt(c.reslen bits)) init(0)
    val aggCount = Reg(UInt(countWidth bits)) init(0)
    val aggFull = aggCount === U(accumulatorCapacity, countWidth bits)
    val accReady = !aggFull

    resultQ.io.push << resultIn

    when(accIn.valid && accReady) {
      aggSum := aggSum + accIn.payload.asSInt
      aggCount := aggCount + U(1, countWidth bits)
    }

    when(ctrl.doReset) {
      aggSum := 0
      aggCount := 0
    }
    resultQ.io.flush := ctrl.doReset
  }
}

class Bnc2LsuPlugin(param: Bnc2LsuPluginParameter = Bnc2LsuPluginParameter()) extends FiberPlugin {
  val setup = during setup new Area {
    val decode = host[Bnc2DecodePlugin]
    val lock = retains(decode.decodingLock)
    if(param.withLoad) decode.add(Bnc2DecodeSpec(Bnc2Ops.LOAD, 4, usesLoad = true))
    lock.release()
  }

  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val rf = host[Bnc2RfPlugin].logic.get
    val queues = host[Bnc2QueuePlugin].logic.get

    val lsu = new CfuLsu(c.io.dBus, CfuLsuParameter(
      dataWidth = c.xlen,
      vectorBytes = c.vlen / 8,
      withLoad = param.withLoad,
      withStore = param.withStore,
      advancedMem = param.advancedMemOf(c.p),
      burstMem = param.burstMemOf(c.p)
    ))

    lsu.cmd.load := False
    lsu.cmd.store := False
    lsu.cmd.address := 0

    val loadDst = Reg(UInt(c.regSelWidth bits)) init(0)
    queues.loadQ.io.pop.ready := lsu.cmd.ready
    when(queues.loadQ.io.pop.valid && lsu.cmd.ready) {
      lsu.cmd.load := True
      lsu.cmd.address := queues.loadQ.io.pop.payload.address
      loadDst := queues.loadQ.io.pop.payload.dst
      rf.vecOffsets(queues.loadQ.io.pop.payload.dst) := U(0, c.vlenLog2 bits)
    }
    when(lsu.load.valid) {
      rf.vecRegs(loadDst)(lsu.load.index.resize(c.vlenLog2) << log2Up(c.xlen), c.xlen bits) := lsu.load.data
    }
    when(lsu.done) {
      rf.rfLoadPending(loadDst) := False
    }
  }
}

class Bnc2QuantizerPlugin(param: Bnc2QuantizerPluginParameter = Bnc2QuantizerPluginParameter()) extends FiberPlugin {
  val setup = during setup new Area {
    val c = host[Bnc2ContextPlugin]
    val decode = host[Bnc2DecodePlugin]
    val lock = retains(decode.decodingLock)
    if(param.withQ2TOf(c.p)) decode.add(Bnc2DecodeSpec(Bnc2Ops.Q2T, 3, usesQuant = true))
    if(param.withQ8Of(c.p)) decode.add(Bnc2DecodeSpec(Bnc2Ops.Q8, 5, usesQuant = true))
    lock.release()
  }

  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val rf = host[Bnc2RfPlugin].logic.get
    val queues = host[Bnc2QueuePlugin].logic.get
    import BitNetCfuCompute._

    val busy = RegInit(False)
    val laneActive = RegInit(False)
    val finishPending = RegInit(False)
    val modeQ8 = Reg(Bool()) init(False)
    val src = Reg(UInt(c.regSelWidth bits)) init(0)
    val dst = Reg(UInt(c.regSelWidth bits)) init(0)
    val offset = Reg(UInt(c.vlenLog2 bits)) init(0)
    val absReg = Reg(Bits(32 bits)) init(0)
    val opReg = Reg(Bits(c.quantWidth bits)) init(0)
    val resultReg = Reg(Bits(c.vlen bits)) init(0)

    val laneParam = BitQuantLaneParameter(
      maxQuantBits = if(param.withQ8Of(c.p)) 8 else 2,
      comparePipe = param.q8ComparePipeOf(c.p)
    )
    val absMagnitude = absReg(30 downto 0).asUInt
    val absExponent = absReg(30 downto 23).asUInt
    val absPartsDecoded = BitQuantCompute.fp32MagnitudeParts(absMagnitude)
    val absParts = new BitQuantAbsmaxParts
    absParts.valid := absMagnitude =/= 0 && absExponent =/= U(255, 8 bits)
    absParts.effectiveExponent := absPartsDecoded.effectiveExponent
    absParts.significand := absPartsDecoded.significand

    val lanesIo = Array.tabulate(c.quantLanes) { _ =>
      if(param.quantStandardOf(c.p)) {
        val lane = new BitQuantLane(laneParam)
        lane.io
      } else {
        val lane = new BitQuantDivLane(laneParam)
        lane.io
      }
    }
    val laneDone = lanesIo.map(_.done).reduce(_ && _)
    val q2tPacked = Bits(2 * c.quantLanes bits)
    val q8Packed = Bits(8 * c.quantLanes bits)
    q2tPacked := 0
    q8Packed := 0
    for(i <- 0 until c.quantLanes) {
      lanesIo(i).qBits := U(2, laneParam.qBitsWidth bits)
      if(param.withQ8Of(c.p)) {
        when(modeQ8) {
          lanesIo(i).qBits := U(8, laneParam.qBitsWidth bits)
        }
      }
      lanesIo(i).absmax := absReg
      lanesIo(i).absParts := absParts
      lanesIo(i).value := opReg(32 * i, 32 bits)
      lanesIo(i).start := busy && !laneActive && !finishPending
      q2tPacked(2 * i, 2 bits) := lanesIo(i).result(0, 2 bits)
      if(param.withQ8Of(c.p)) {
        q8Packed(8 * i, 8 bits) := lanesIo(i).result(0, 8 bits)
      }
    }

    val nextResult = Bits(c.vlen bits)
    nextResult := resultReg
    for(chunk <- 0 until c.quantChunks) {
      when(offset === U(chunk * c.quantWidth, c.vlenLog2 bits)) {
          if(param.withQ2TOf(c.p)) {
            when(!modeQ8) {
              nextResult(2 * chunk * c.quantLanes, 2 * c.quantLanes bits) := q2tPacked
            }
          }
          if(param.withQ8Of(c.p)) {
            when(modeQ8) {
              nextResult(8 * chunk * c.quantLanes, 8 * c.quantLanes bits) := q8Packed
            }
        }
      }
    }

    val lastChunk = offset === U(c.vlen - c.quantWidth, c.vlenLog2 bits)
    val canWriteRf = !host[Bnc2LsuPlugin].logic.get.lsu.load.valid
    val loadWriteHazard = queues.loadQ.io.push.valid && queues.loadQ.io.push.payload.dst === queues.quantQ.io.pop.payload.src
    val selfOverwrite = queues.quantQ.io.pop.payload.src === queues.quantQ.io.pop.payload.dst
    val srcPending = rf.rfLoadPending(queues.quantQ.io.pop.payload.src) ||
      (rf.rfQuantPending(queues.quantQ.io.pop.payload.src) && !selfOverwrite)

    queues.quantQ.io.pop.ready := !busy && !srcPending && !loadWriteHazard
    when(queues.quantQ.io.pop.fire) {
      busy := True
      laneActive := False
      finishPending := False
      modeQ8 := queues.quantQ.io.pop.payload.modeQ8
      src := queues.quantQ.io.pop.payload.src
      dst := queues.quantQ.io.pop.payload.dst
      offset := U(0, c.vlenLog2 bits)
      absReg := queues.quantQ.io.pop.payload.absmax
      opReg := rf.vecRegs(queues.quantQ.io.pop.payload.src)(0, c.quantWidth bits)
      resultReg := 0
      rf.vecOffsets(queues.quantQ.io.pop.payload.dst) := U(0, c.vlenLog2 bits)
    }

    when(busy && !laneActive && !finishPending) {
      laneActive := True
    }

    when(busy && laneActive && laneDone) {
      laneActive := False
      resultReg := nextResult
      if(c.quantChunks == 1) {
        finishPending := True
      } else {
        when(lastChunk) {
          finishPending := True
        } otherwise {
          val nextOffset = offset + U(c.quantWidth, c.vlenLog2 bits)
          offset := nextOffset
          opReg := rf.vecRegs(src)(0, c.quantWidth bits)
          for(chunk <- 0 until c.quantChunks) {
            when(nextOffset === U(chunk * c.quantWidth, c.vlenLog2 bits)) {
              opReg := rf.vecRegs(src)(chunk * c.quantWidth, c.quantWidth bits)
            }
          }
        }
      }
    }

    when(finishPending && canWriteRf) {
      rf.vecRegs(dst) := resultReg
      rf.rfQuantPending(dst) := False
      busy := False
      finishPending := False
    }

    when(host[Bnc2ControlPlugin].logic.get.doReset) {
      busy := False
      laneActive := False
      finishPending := False
    }
  }
}

class Bnc2DotpPlugin(param: Bnc2DotpPluginParameter = Bnc2DotpPluginParameter()) extends FiberPlugin {
  val setup = during setup new Area {
    val c = host[Bnc2ContextPlugin]
    val decode = host[Bnc2DecodePlugin]
    val lock = retains(decode.decodingLock)
    decode.add(Bnc2DecodeSpec(Bnc2Ops.DOT_HOLD, 0, usesDot = true))
    decode.add(Bnc2DecodeSpec(Bnc2Ops.DOT, 1, usesDot = true))
    if(param.withQ2TOf(c.p)) {
      decode.add(Bnc2DecodeSpec(Bnc2Ops.Q2T_DOT, 6, usesQuant = true, usesDot = true))
      decode.add(Bnc2DecodeSpec(Bnc2Ops.LOAD_Q2T_DOT, 7, func7Ne = Some(2), usesLoad = true, usesQuant = true, usesDot = true))
    }
    lock.release()
  }

  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val rf = host[Bnc2RfPlugin].logic.get
    val queues = host[Bnc2QueuePlugin].logic.get
    val result = host[Bnc2ResultPlugin].logic.get
    val state = host[Bnc2StatePlugin].logic.get
    import BitNetCfuCompute._

    val busy = RegInit(False)
    val finishPending = RegInit(False)
    val srcA = Reg(UInt(c.regSelWidth bits)) init(0)
    val srcW = Reg(UInt(c.regSelWidth bits)) init(1)
    val advance = Reg(Bool()) init(True)
    val accumulate = Reg(Bool()) init(False)
    val int8Offset = Reg(UInt(c.vlenLog2 bits)) init(0)
    val lowbitOffset = Reg(UInt(c.vlenLog2 bits)) init(0)
    val acc = Reg(SInt(c.reslen bits)) init(0)
    val dotResult = Reg(Bits(c.reslen bits)) init(0)

    val quantWriteHazard = queues.quantQ.io.push.valid && queues.quantQ.io.push.payload.dst === queues.dotQ.io.pop.payload.lowbit
    queues.dotQ.io.pop.ready := !busy &&
      !rf.rfLoadPending(queues.dotQ.io.pop.payload.int8) &&
      !rf.rfQuantPending(queues.dotQ.io.pop.payload.int8) &&
      !rf.rfLoadPending(queues.dotQ.io.pop.payload.lowbit) &&
      !rf.rfQuantPending(queues.dotQ.io.pop.payload.lowbit) &&
      !quantWriteHazard
    when(queues.dotQ.io.pop.fire) {
      busy := True
      finishPending := False
      srcA := queues.dotQ.io.pop.payload.int8
      srcW := queues.dotQ.io.pop.payload.lowbit
      advance := queues.dotQ.io.pop.payload.advance
      accumulate := queues.dotQ.io.pop.payload.accumulate
      int8Offset := rf.vecOffsets(queues.dotQ.io.pop.payload.int8)
      lowbitOffset := rf.vecOffsets(queues.dotQ.io.pop.payload.lowbit)
      acc := 0
    }

    val opA = rf.vecRegs(srcA)(int8Offset, c.maclen bits)
    val opW = Bits(c.weightSliceBitsMax bits)
    opW := 0
    when(state.qType === U(Q1B, 2 bits)) {
      opW := rf.vecRegs(srcW)(lowbitOffset, c.lanes bits).resize(c.weightSliceBitsMax)
    } otherwise {
      opW := rf.vecRegs(srcW)(lowbitOffset, c.weightSliceBitsMax bits)
    }

    val partial = BitNetDot(opA, opW, c.lanes, state.qType, c.reslen, c.p.withQ2)
    val nextAcc = acc + partial
    val lastSlice = if(c.nCompute == 1) True else int8Offset === U(c.vlen - c.maclen, c.vlenLog2 bits)

    when(busy && !finishPending) {
      if(c.nCompute > 1) {
        when(lastSlice) {
          dotResult := nextAcc.asBits
          finishPending := True
        } otherwise {
          acc := nextAcc
          int8Offset := int8Offset + U(c.maclen, c.vlenLog2 bits)
          lowbitOffset := lowbitOffset + c.weightInc(state.qType)
        }
      } else {
        dotResult := nextAcc.asBits
        finishPending := True
      }
    }

    result.resultIn.valid := busy && finishPending && !accumulate
    result.resultIn.payload := dotResult
    result.accIn.valid := busy && finishPending && accumulate && result.accReady
    result.accIn.payload := dotResult

    when(busy && finishPending && accumulate && result.accReady) {
      rf.vecOffsets(srcA) := U(0, c.vlenLog2 bits)
      when(advance) {
        rf.vecOffsets(srcW) := lowbitOffset + c.weightInc(state.qType)
      }
      rf.rfDotPending(srcW) := False
      busy := False
      finishPending := False
      acc := 0
    }
    when(result.resultIn.fire) {
      rf.vecOffsets(srcA) := U(0, c.vlenLog2 bits)
      when(advance) {
        rf.vecOffsets(srcW) := lowbitOffset + c.weightInc(state.qType)
      }
      rf.rfDotPending(srcW) := False
      busy := False
      finishPending := False
      acc := 0
    }

    when(host[Bnc2ControlPlugin].logic.get.doReset) {
      busy := False
      finishPending := False
      acc := 0
    }
  }
}

class Bnc2StatePlugin extends FiberPlugin {
  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val qType = Reg(UInt(2 bits)) init(U(c.p.qTypeId, 2 bits))
    val qTypeCmd = c.io.bus.cmd.inputs(0).asUInt.resize(2)
    val qTypeValid = qTypeCmd === U(BitNetCfuCompute.Q1B, 2 bits) ||
      qTypeCmd === U(BitNetCfuCompute.Q2B, 2 bits) ||
      qTypeCmd === U(BitNetCfuCompute.Q15B, 2 bits)

    when(host[Bnc2ControlPlugin].logic.get.doReset) {
      qType := U(c.p.qTypeId, 2 bits)
    }
  }
}

class Bnc2BlockSchedulerPlugin(param: Bnc2BlockSchedulerPluginParameter = Bnc2BlockSchedulerPluginParameter()) extends FiberPlugin {
  val setup = during setup new Area {
    val c = host[Bnc2ContextPlugin]
    val decode = host[Bnc2DecodePlugin]
    val lock = retains(decode.decodingLock)
    if(param.withQ2TOf(c.p)) {
      decode.add(Bnc2DecodeSpec(
        Bnc2Ops.LOAD_Q2T_DOT_BLOCK,
        7,
        func7Eq = Some(2),
        usesLoad = true,
        usesQuant = true,
        usesDot = true
      ))
    }
    lock.release()
  }

  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val rf = host[Bnc2RfPlugin].logic.get
    val queues = host[Bnc2QueuePlugin].logic.get
    val result = host[Bnc2ResultPlugin].logic.get

    val blockActiveCount = Reg(UInt(result.countWidth bits)) init(U(c.p.blockCount, result.countWidth bits))
    val blockCountCmd = c.io.bus.cmd.inputs(0).asUInt.resize(result.countWidth)
    val blockCountCmdValid = blockCountCmd =/= 0 && blockCountCmd <= U(c.p.blockCount, result.countWidth bits)
    val blockSupported = param.withQ2TOf(c.p) &&
      c.p.loadQueueDepth >= param.minQueueDepth &&
      c.p.quantQueueDepth >= param.minQueueDepth &&
      c.p.dotQueueDepth >= param.minQueueDepth

    val busy = RegInit(False)
    val base = Reg(UInt(32 bits)) init(0)
    val absmax = Reg(Bits(32 bits)) init(0)
    val issued = Reg(UInt(result.countWidth bits)) init(0)
    val slot = Reg(UInt(c.regSelWidth bits)) init(1)
    val done = issued === blockActiveCount
    val slotBusy = rf.writeBusy(slot, True)
    val address = base + ((issued.resize(32) << log2Up(c.vlen / 8)).resize(32))
    val lastSlot = slot === U(c.regDepth - 1, c.regSelWidth bits)
    val nextSlot = lastSlot.mux(U(1, c.regSelWidth bits), slot + 1)

    queues.blockLoad.valid := False
    queues.blockLoad.payload.address := 0
    queues.blockLoad.payload.dst := 0
    queues.blockQuant.valid := False
    queues.blockQuant.payload.absmax := 0
    queues.blockQuant.payload.src := 0
    queues.blockQuant.payload.dst := 0
    queues.blockQuant.payload.modeQ8 := False
    queues.blockDot.valid := False
    queues.blockDot.payload.int8 := 0
    queues.blockDot.payload.lowbit := 0
    queues.blockDot.payload.advance := True
    queues.blockDot.payload.accumulate := False

    if(blockSupported) {
      val canIssue = busy && !done && !slotBusy &&
        queues.blockLoad.ready && queues.blockQuant.ready && queues.blockDot.ready
      when(canIssue) {
        queues.blockLoad.valid := True
        queues.blockLoad.payload.address := address
        queues.blockLoad.payload.dst := slot
        queues.blockQuant.valid := True
        queues.blockQuant.payload.absmax := absmax
        queues.blockQuant.payload.src := slot
        queues.blockQuant.payload.dst := slot
        queues.blockQuant.payload.modeQ8 := False
        queues.blockDot.valid := True
        queues.blockDot.payload.int8 := U(0, c.regSelWidth bits)
        queues.blockDot.payload.lowbit := slot
        queues.blockDot.payload.advance := True
        queues.blockDot.payload.accumulate := True
        rf.rfLoadPending(slot) := True
        rf.rfQuantPending(slot) := True
        rf.rfDotPending(slot) := True
        issued := issued + 1
        slot := nextSlot
      }
    }

    when(busy && done) {
      busy := False
    }

    when(host[Bnc2ControlPlugin].logic.get.doReset) {
      busy := False
      blockActiveCount := U(c.p.blockCount, result.countWidth bits)
    }
  }
}

class Bnc2BusDecodePlugin extends FiberPlugin {
  val logic = during build new Area {
    val c = host[Bnc2ContextPlugin]
    val decode = host[Bnc2DecodePlugin].logic.get
    val ctrl = host[Bnc2ControlPlugin].logic.get
    val rf = host[Bnc2RfPlugin].logic.get
    val queues = host[Bnc2QueuePlugin].logic.get
    val lsu = host[Bnc2LsuPlugin].logic.get
    val quant = host[Bnc2QuantizerPlugin].logic.get
    val dot = host[Bnc2DotpPlugin].logic.get
    val result = host[Bnc2ResultPlugin].logic.get
    val state = host[Bnc2StatePlugin].logic.get
    val block = host[Bnc2BlockSchedulerPlugin].logic.get

    import Bnc2Ops._
    val isDotHold = decode.is(DOT_HOLD)
    val isDotOnly = decode.is(DOT) || isDotHold
    val isQ2T = decode.is(Q2T)
    val isLoadOnly = decode.is(LOAD)
    val isQ8 = decode.is(Q8)
    val isQ2TDot = decode.is(Q2T_DOT)
    val isLoadQ2TDotSingle = decode.is(LOAD_Q2T_DOT)
    val isLoadQ2TDotBlock = decode.is(LOAD_Q2T_DOT_BLOCK)
    val isQuantOnly = isQ2T || isQ8
    val isLoad = decode.isLoad
    val isQuant = decode.isQuant
    val isDot = decode.isDot
    val isCfg = decode.isCfg

    val cfgReset = decode.is(CFG_RESET)
    val cfgQType = decode.is(CFG_QTYPE)
    val cfgPoll = decode.is(CFG_POLL)
    val cfgRead = decode.is(CFG_READ)
    val cfgReadWait = decode.is(CFG_READ_WAIT)
    val cfgReadAccWait = decode.is(CFG_READ_ACC_WAIT)
    val cfgBlockCount = decode.is(CFG_BLOCK_COUNT)

    queues.cmdLoad.valid := False
    queues.cmdLoad.payload.address := 0
    queues.cmdLoad.payload.dst := 0
    queues.cmdQuant.valid := False
    queues.cmdQuant.payload.absmax := 0
    queues.cmdQuant.payload.src := 0
    queues.cmdQuant.payload.dst := 0
    queues.cmdQuant.payload.modeQ8 := False
    queues.cmdDot.valid := False
    queues.cmdDot.payload.int8 := 0
    queues.cmdDot.payload.lowbit := 0
    queues.cmdDot.payload.advance := True
    queues.cmdDot.payload.accumulate := False

    val readAccTarget = c.io.bus.cmd.inputs(0).asUInt.resize(result.countWidth)
    val readAccLegal = readAccTarget =/= 0 && readAccTarget <= U(result.accumulatorCapacity, result.countWidth bits)
    val readAccReady = readAccLegal && result.aggCount >= readAccTarget
    val isDotUsingRd = isQ2TDot || isLoadQ2TDotSingle
    val loadIdsValid = decode.RS2_VALID
    val quantIdsValid = decode.RS2_VALID && decode.RD_VALID
    val dotIdsValid = isDotUsingRd.mux(decode.RD_VALID, decode.RS1_VALID && decode.RS2_VALID)
    val blockIdsValid = if(block.blockSupported) True else False
    val commandIdsValid = Bool()
    commandIdsValid := True
    when(isLoadOnly) { commandIdsValid := loadIdsValid }
    when(isQuantOnly) { commandIdsValid := quantIdsValid }
    when(isDotOnly) { commandIdsValid := dotIdsValid }
    when(isQ2TDot) { commandIdsValid := quantIdsValid && dotIdsValid }
    when(isLoadQ2TDotSingle) { commandIdsValid := loadIdsValid && quantIdsValid && dotIdsValid }
    when(isLoadQ2TDotBlock) { commandIdsValid := blockIdsValid }

    val busyAny = lsu.lsu.busy || quant.busy || dot.busy || block.busy
    val queuesNonEmpty = queues.loadQ.io.occupancy =/= 0 ||
      queues.quantQ.io.occupancy =/= 0 ||
      queues.dotQ.io.occupancy =/= 0 ||
      result.resultQ.io.occupancy =/= 0
    val configAllowed = !busyAny && !queuesNonEmpty && result.aggCount === 0
    val resetAllowed = !busyAny && !queuesNonEmpty
    ctrl.doReset := c.io.bus.cmd.fire && cfgReset && resetAllowed

    val rspCanComplete = (!cfgReadWait || result.resultQ.io.pop.valid) && (!cfgReadAccWait || !readAccLegal || readAccReady)
    c.io.bus.cmd.ready := c.io.bus.rsp.ready && rspCanComplete
    c.io.bus.rsp.valid := c.io.bus.cmd.valid && c.io.bus.rsp.ready && rspCanComplete
    c.io.bus.rsp.response_id := c.io.bus.cmd.request_id
    c.io.bus.rsp.outputs(0) := 0
    val rspStatus = Bits(3 bits)
    rspStatus := B"000"
    when(isLoad && !queues.cmdLoad.ready) { rspStatus := B"001" }
    when(isQuant && !queues.cmdQuant.ready) { rspStatus := B"001" }
    when(isDot && !queues.cmdDot.ready) { rspStatus := B"001" }
    when((isLoad || isQuant || isDot) && !commandIdsValid) { rspStatus := B"011" }
    when(isLoad && commandIdsValid && rf.writeBusy(decode.RS2, decode.RS2_VALID)) { rspStatus := B"100" }
    when(isQuant && commandIdsValid && rf.writeBusy(decode.RD, decode.RD_VALID)) { rspStatus := B"100" }
    when(cfgRead && !result.resultQ.io.pop.valid) { rspStatus := B"010" }
    when(cfgReadAccWait && !readAccLegal) { rspStatus := B"011" }
    when(cfgReadAccWait && readAccLegal && !readAccReady) { rspStatus := B"010" }
    when(cfgBlockCount && !block.blockCountCmdValid) { rspStatus := B"011" }
    when(cfgBlockCount && block.blockCountCmdValid && !configAllowed) { rspStatus := B"100" }
    when(decode.isCfgFunc && !isCfg) { rspStatus := B"011" }
    when(cfgReset && !resetAllowed) { rspStatus := B"100" }
    when(isLoadQ2TDotBlock && block.busy) { rspStatus := B"100" }
    when(!decode.isKnown) { rspStatus := B"011" }
    when(lsu.lsu.error) { rspStatus := B"101" }

    when(cfgPoll) {
      c.io.bus.rsp.outputs(0) := (B(0, 8 bits) ##
        B(queues.loadQ.io.occupancy =/= 0, queues.quantQ.io.occupancy =/= 0, queues.dotQ.io.occupancy =/= 0, busyAny) ##
        B(0, 10 bits) ##
        (result.aggCount =/= 0).asBits ##
        result.resultQ.io.pop.valid.asBits ##
        rspStatus).asUInt.asBits.resized
    } otherwise {
      c.io.bus.rsp.outputs(0) := rspStatus.asUInt.asBits.resized
    }

    when((cfgRead || cfgReadWait) && result.resultQ.io.pop.valid) {
      c.io.bus.rsp.outputs(0) := result.resultQ.io.pop.payload
    }
    when(cfgReadAccWait && readAccReady) {
      c.io.bus.rsp.outputs(0) := result.aggSum.asBits
    }

    if(c.cfuParam.CFU_WITH_STATUS) c.io.bus.rsp.status := rspStatus

    when(c.io.bus.cmd.fire) {
      val rs2Busy = rf.writeBusy(decode.RS2, decode.RS2_VALID)
      val rdBusy = rf.writeBusy(decode.RD, decode.RD_VALID)
      when(isLoadOnly && loadIdsValid && queues.cmdLoad.ready && !rs2Busy) {
        queues.cmdLoad.valid := True
        queues.cmdLoad.payload.address := c.io.bus.cmd.inputs(0).asUInt.resized
        queues.cmdLoad.payload.dst := decode.RS2
        rf.rfLoadPending(decode.RS2) := True
      }
      when(isQuantOnly && quantIdsValid && queues.cmdQuant.ready && !rdBusy) {
        queues.cmdQuant.valid := True
        queues.cmdQuant.payload.absmax := c.io.bus.cmd.inputs(0).resize(32)
        queues.cmdQuant.payload.src := decode.RS2
        queues.cmdQuant.payload.dst := decode.RD
        queues.cmdQuant.payload.modeQ8 := isQ8
        rf.rfQuantPending(decode.RD) := True
      }
      when(isDotOnly && dotIdsValid && queues.cmdDot.ready) {
        queues.cmdDot.valid := True
        queues.cmdDot.payload.int8 := decode.RS1
        queues.cmdDot.payload.lowbit := decode.RS2
        queues.cmdDot.payload.advance := !isDotHold
        queues.cmdDot.payload.accumulate := False
        rf.rfDotPending(decode.RS2) := True
      }
      when(isQ2TDot && quantIdsValid && dotIdsValid && queues.cmdQuant.ready && queues.cmdDot.ready && !rdBusy) {
        queues.cmdQuant.valid := True
        queues.cmdQuant.payload.absmax := c.io.bus.cmd.inputs(0).resize(32)
        queues.cmdQuant.payload.src := decode.RS2
        queues.cmdQuant.payload.dst := decode.RD
        queues.cmdQuant.payload.modeQ8 := False
        queues.cmdDot.valid := True
        queues.cmdDot.payload.int8 := U(0, c.regSelWidth bits)
        queues.cmdDot.payload.lowbit := decode.RD
        queues.cmdDot.payload.advance := True
        queues.cmdDot.payload.accumulate := decode.FUNC7 === 1
        rf.rfQuantPending(decode.RD) := True
        rf.rfDotPending(decode.RD) := True
      }
      when(isLoadQ2TDotSingle &&
        loadIdsValid &&
        quantIdsValid &&
        dotIdsValid &&
        queues.cmdLoad.ready &&
        queues.cmdQuant.ready &&
        queues.cmdDot.ready &&
        !rs2Busy &&
        !rdBusy
      ) {
        queues.cmdLoad.valid := True
        queues.cmdLoad.payload.address := c.io.bus.cmd.inputs(0).asUInt.resized
        queues.cmdLoad.payload.dst := decode.RS2
        queues.cmdQuant.valid := True
        queues.cmdQuant.payload.absmax := c.io.bus.cmd.inputs(1).resize(32)
        queues.cmdQuant.payload.src := decode.RS2
        queues.cmdQuant.payload.dst := decode.RD
        queues.cmdQuant.payload.modeQ8 := False
        queues.cmdDot.valid := True
        queues.cmdDot.payload.int8 := U(0, c.regSelWidth bits)
        queues.cmdDot.payload.lowbit := decode.RD
        queues.cmdDot.payload.advance := True
        queues.cmdDot.payload.accumulate := decode.FUNC7 === 1
        rf.rfLoadPending(decode.RS2) := True
        rf.rfQuantPending(decode.RD) := True
        rf.rfDotPending(decode.RD) := True
      }
      when(isLoadQ2TDotBlock && blockIdsValid && !block.busy) {
        block.busy := True
        block.base := c.io.bus.cmd.inputs(0).asUInt.resized
        block.absmax := c.io.bus.cmd.inputs(1).resize(32)
        block.issued := 0
        block.slot := 1
      }
      when(cfgQType && state.qTypeValid) {
        state.qType := state.qTypeCmd
      }
      when(cfgBlockCount && block.blockCountCmdValid && configAllowed) {
        block.blockActiveCount := block.blockCountCmd
      }
    }
    result.resultQ.io.pop.ready := c.io.bus.cmd.fire && (cfgRead || cfgReadWait) && result.resultQ.io.pop.valid
    when(c.io.bus.cmd.fire && cfgReadAccWait && readAccReady) {
      result.aggSum := 0
      result.aggCount := 0
    }
  }
}

class TilelinkBitNetCfuV2PluginFiber(bitNetParam: BitNetCfuV2Parameter, xlen: Int) extends Area {
  val bus = spinal.lib.bus.tilelink.fabric.Node.down()
  val dBus = bus.bus
  val cfuParam = TilelinkBitNetCfuFiber.getCfuBusParameters(xlen)
  val cfuBusParam = TilelinkBitNetCfuFiber.getM2sParameters(null, bitNetParam.xlen, bitNetParam.pendingSize).toBusParameter()
  val cfu = new BitNetCfuV2PluginHost(cfuParam, cfuBusParam, bitNetParam)

  val logic = spinal.core.fiber.Fiber build new Area {
    bus.m2s forceParameters TilelinkBitNetCfuFiber.getM2sParameters(TilelinkBitNetCfuV2PluginFiber.this, bitNetParam.xlen, bitNetParam.pendingSize)
    bus.s2m.supported load tilelink.S2mSupport.none()

    val cfuBus = CfuBus(cfuParam)
    cfu.io.bus <> cfuBus
    cfu.io.dBus <> dBus
  }
}

object BitNetCfuV2PluginGen extends App {
  val p = BitNetCfuV2Parameter()
  var targetDir = "synth_runs/rtl/bitnet_cfu_v2_plugin"

  assert(new scopt.OptionParser[Unit]("BitNetCfuV2PluginGen") {
    opt[String]("target-dir") action { (v, _) => targetDir = v }
    opt[Int]("vlen") action { (v, _) => p.vlen = v }
    opt[Int]("xlen") action { (v, _) => p.xlen = v }
    opt[Int]("maclen") action { (v, _) => p.maclen = v }
    opt[Int]("reg-depth") action { (v, _) => p.regDepth = v }
    opt[String]("qtype") action { (v, _) => p.qType = v }
    opt[Unit]("with-q2") action { (_, _) => p.withQ2 = true }
    opt[Unit]("without-q2t") action { (_, _) => p.withQ2T = false }
    opt[Unit]("with-q8") action { (_, _) => p.withQ8 = true }
    opt[Int]("quant-width") action { (v, _) => p.quantWidth = v }
    opt[Unit]("quant-standard") action { (_, _) => p.quantStandard = true }
    opt[Int]("cmd-queue-depth") action { (v, _) => p.cmdQueueDepth = v }
    opt[Int]("result-queue-depth") action { (v, _) => p.resultQueueDepth = v }
    opt[Int]("load-queue-depth") action { (v, _) => p.loadQueueDepth = v }
    opt[Int]("quant-queue-depth") action { (v, _) => p.quantQueueDepth = v }
    opt[Int]("dot-queue-depth") action { (v, _) => p.dotQueueDepth = v }
    opt[Int]("block-count") action { (v, _) => p.blockCount = v }
    opt[Unit]("advanced-mem") action { (_, _) => p.advancedMem = true }
    opt[Unit]("burst-mem") action { (_, _) => p.burstMem = true }
  }.parse(args, ()).nonEmpty)

  val cfuParam = TilelinkBitNetCfuFiber.getCfuBusParameters(p.xlen)
  val busParam = TilelinkBitNetCfuFiber.getM2sParameters(null, p.xlen, p.pendingSize).toBusParameter()

  SpinalConfig(targetDirectory = targetDir)
    .generateVerilog(new BitNetCfuV2PluginHost(cfuParam, busParam, p))
}
