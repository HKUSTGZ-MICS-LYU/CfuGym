package vexiiriscv.soc.mico

import spinal.core._
import spinal.lib._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.fsm._

case class CfuLsuParameter(
  dataWidth: Int,
  vectorBytes: Int,
  addressWidth: Int = 32,
  withLoad: Boolean = true,
  withStore: Boolean = true,
  advancedMem: Boolean = false,
  burstMem: Boolean = false
) {
  require(dataWidth % 8 == 0, "CFU LSU data width must be byte aligned")
  require(vectorBytes % beatBytes == 0, "CFU LSU vector byte count must be a multiple of the beat byte count")
  if(burstMem) require(isPow2(vectorBytes), "CFU LSU burst mode requires a power-of-two vector byte count")

  def beatBytes = dataWidth / 8
  def beatCount = vectorBytes / beatBytes
  def transferBytes = if(burstMem) vectorBytes else beatBytes
  def pendingSize = beatCount max 1
}

class CfuLsu(dBus: tilelink.Bus, p: CfuLsuParameter) extends Area {
  require(dBus.p.dataWidth == p.dataWidth, s"CFU LSU data width ${p.dataWidth} must match TileLink width ${dBus.p.dataWidth}")

  val beatIndexWidth = log2Up(p.beatCount) max 1
  val beatShift = log2Up(p.beatBytes)

  val cmd = new Area {
    val load = Bool()
    val store = Bool()
    val address = UInt(p.addressWidth bits)
    val ready = Bool()

    load := False
    store := False
    address := 0
  }

  val store = new Area {
    val index = UInt(beatIndexWidth bits)
    val data = Bits(p.dataWidth bits)

    index := 0
  }

  val load = new Area {
    val valid = Bool()
    val index = UInt(beatIndexWidth bits)
    val data = Bits(p.dataWidth bits)

    valid := False
    index := 0
    data := dBus.d.data
  }

  val done = Bool()
  val error = Bool()
  val busy = Bool()

  done := False
  error := False

  val baseAddr = Reg(UInt(p.addressWidth bits)) init(0)
  val beatIndex = Reg(UInt(beatIndexWidth bits)) init(0)
  val issueIndex = Reg(UInt(beatIndexWidth bits)) init(0)
  val issueValid = RegInit(False)
  val burstActive = RegInit(False)
  val memError = RegInit(False)
  val rspHits = Vec.fill(p.beatCount)(RegInit(False))
  val rspCount = rspHits.sCount(True)
  val rspLast = rspCount === U(p.beatCount - 1, widthOf(rspCount) bits)
  val lastBeat = beatIndex === U(p.beatCount - 1, beatIndexWidth bits)
  val lastIssue = issueIndex === U(p.beatCount - 1, beatIndexWidth bits)
  val requestIndex = if(p.advancedMem && !p.burstMem) issueIndex else beatIndex
  val accessAddr = baseAddr + ((requestIndex.resize(p.addressWidth) << beatShift).resize(p.addressWidth))
  val mask = B(p.beatBytes bits, default -> True)

  dBus.a.opcode := tilelink.Opcode.A.GET
  dBus.a.param := tilelink.Param.Hint.NO_ALLOCATE_ON_MISS
  dBus.a.source := requestIndex.resized
  dBus.a.data := 0
  dBus.a.address := accessAddr
  dBus.a.mask := mask
  dBus.a.size := log2Up(p.beatBytes)
  dBus.a.corrupt := False
  dBus.a.valid := False
  dBus.d.ready := False

  val fsm = new StateMachine {
    val IDLE = new State with EntryPoint
    val LOAD_SEND = p.withLoad generate new State
    val LOAD_WAIT = p.withLoad generate new State
    val LOAD_ADVANCED = (p.withLoad && p.advancedMem) generate new State
    val LOAD_BURST = (p.withLoad && p.burstMem) generate new State
    val STORE_SEND = p.withStore generate new State
    val STORE_WAIT = p.withStore generate new State
    val STORE_ADVANCED = (p.withStore && p.advancedMem) generate new State
    val STORE_BURST = (p.withStore && p.burstMem) generate new State

    def startLoad(): Unit = {
      baseAddr := cmd.address
      beatIndex := 0
      issueIndex := 0
      issueValid := True
      burstActive := False
      memError := False
      rspHits.foreach(_ := False)
      if(p.burstMem) {
        goto(LOAD_BURST)
      } else if(p.advancedMem) {
        goto(LOAD_ADVANCED)
      } else {
        goto(LOAD_SEND)
      }
    }

    def startStore(): Unit = {
      baseAddr := cmd.address
      beatIndex := 0
      issueIndex := 0
      issueValid := True
      burstActive := False
      memError := False
      rspHits.foreach(_ := False)
      if(p.burstMem) {
        goto(STORE_BURST)
      } else if(p.advancedMem) {
        goto(STORE_ADVANCED)
      } else {
        goto(STORE_SEND)
      }
    }

    IDLE.whenIsActive {
      if(p.withLoad && p.withStore) {
        when(cmd.load) {
          startLoad()
        } elsewhen(cmd.store) {
          startStore()
        }
      } else if(p.withLoad) {
        when(cmd.load) {
          startLoad()
        }
      } else if(p.withStore) {
        when(cmd.store) {
          startStore()
        }
      }
    }

    if(p.withLoad) {
      LOAD_SEND.whenIsActive {
        dBus.a.valid := True
        dBus.a.opcode := tilelink.Opcode.A.GET
        dBus.a.param := tilelink.Param.Hint.NO_ALLOCATE_ON_MISS
        dBus.a.data := 0
        when(dBus.a.fire) {
          goto(LOAD_WAIT)
        }
      }

      LOAD_WAIT.whenIsActive {
        dBus.d.ready := True
        when(dBus.d.fire) {
          val goodRsp = dBus.d.opcode === tilelink.Opcode.D.ACCESS_ACK_DATA && !dBus.d.denied && !dBus.d.corrupt
          when(goodRsp) {
            load.valid := True
            load.index := beatIndex
            load.data := dBus.d.data
            when(lastBeat) {
              done := True
              goto(IDLE)
            } otherwise {
              beatIndex := beatIndex + 1
              goto(LOAD_SEND)
            }
          } otherwise {
            done := True
            error := True
            goto(IDLE)
          }
        }
      }
    }

    if(p.withLoad && p.advancedMem) {
      LOAD_ADVANCED.whenIsActive {
        dBus.a.valid := issueValid
        dBus.a.opcode := tilelink.Opcode.A.GET
        dBus.a.param := tilelink.Param.Hint.NO_ALLOCATE_ON_MISS
        dBus.a.data := 0
        dBus.d.ready := True

        when(dBus.a.fire) {
          when(lastIssue) {
            issueValid := False
          } otherwise {
            issueIndex := issueIndex + 1
          }
        }

        when(dBus.d.fire) {
          val rspIndex = dBus.d.source.resize(beatIndexWidth)
          val goodRsp = dBus.d.opcode === tilelink.Opcode.D.ACCESS_ACK_DATA && !dBus.d.denied && !dBus.d.corrupt
          rspHits(rspIndex) := True
          when(!goodRsp) {
            memError := True
          } otherwise {
            load.valid := True
            load.index := rspIndex
            load.data := dBus.d.data
          }
          when(rspLast) {
            done := True
            error := memError || !goodRsp
            issueValid := False
            goto(IDLE)
          }
        }
      }
    }

    if(p.withLoad && p.burstMem) {
      LOAD_BURST.whenIsActive {
        dBus.a.valid := !burstActive
        dBus.a.opcode := tilelink.Opcode.A.GET
        dBus.a.param := tilelink.Param.Hint.NO_ALLOCATE_ON_MISS
        dBus.a.source := U(0, widthOf(dBus.a.source) bits)
        dBus.a.address := baseAddr
        dBus.a.size := log2Up(p.vectorBytes)
        dBus.a.data := 0
        dBus.d.ready := True

        when(dBus.a.fire) {
          burstActive := True
        }

        when(dBus.d.fire) {
          val rspIndex = dBus.d.beatCounter().resize(beatIndexWidth)
          val goodRsp = dBus.d.opcode === tilelink.Opcode.D.ACCESS_ACK_DATA && !dBus.d.denied && !dBus.d.corrupt
          when(!goodRsp) {
            memError := True
          } otherwise {
            load.valid := True
            load.index := rspIndex
            load.data := dBus.d.data
          }
          when(dBus.d.isLast()) {
            done := True
            error := memError || !goodRsp
            burstActive := False
            goto(IDLE)
          }
        }
      }
    }

    if(p.withStore) {
      STORE_SEND.whenIsActive {
        store.index := beatIndex
        dBus.a.valid := True
        dBus.a.opcode := tilelink.Opcode.A.PUT_FULL_DATA
        dBus.a.param := 0
        dBus.a.data := store.data
        when(dBus.a.fire) {
          goto(STORE_WAIT)
        }
      }

      STORE_WAIT.whenIsActive {
        dBus.d.ready := True
        when(dBus.d.fire) {
          val goodRsp = dBus.d.opcode === tilelink.Opcode.D.ACCESS_ACK && !dBus.d.denied && !dBus.d.corrupt
          when(goodRsp) {
            when(lastBeat) {
              done := True
              goto(IDLE)
            } otherwise {
              beatIndex := beatIndex + 1
              goto(STORE_SEND)
            }
          } otherwise {
            done := True
            error := True
            goto(IDLE)
          }
        }
      }
    }

    if(p.withStore && p.advancedMem) {
      STORE_ADVANCED.whenIsActive {
        store.index := issueIndex
        dBus.a.valid := issueValid
        dBus.a.opcode := tilelink.Opcode.A.PUT_FULL_DATA
        dBus.a.param := 0
        dBus.a.data := store.data
        dBus.d.ready := True

        when(dBus.a.fire) {
          when(lastIssue) {
            issueValid := False
          } otherwise {
            issueIndex := issueIndex + 1
          }
        }

        when(dBus.d.fire) {
          val rspIndex = dBus.d.source.resize(beatIndexWidth)
          val goodRsp = dBus.d.opcode === tilelink.Opcode.D.ACCESS_ACK && !dBus.d.denied && !dBus.d.corrupt
          rspHits(rspIndex) := True
          when(!goodRsp) {
            memError := True
          }
          when(rspLast) {
            done := True
            error := memError || !goodRsp
            issueValid := False
            goto(IDLE)
          }
        }
      }
    }

    if(p.withStore && p.burstMem) {
      STORE_BURST.whenIsActive {
        store.index := issueIndex
        dBus.a.valid := issueValid
        dBus.a.opcode := tilelink.Opcode.A.PUT_FULL_DATA
        dBus.a.param := 0
        dBus.a.source := U(0, widthOf(dBus.a.source) bits)
        dBus.a.address := baseAddr
        dBus.a.size := log2Up(p.vectorBytes)
        dBus.a.data := store.data
        dBus.d.ready := True

        when(dBus.a.fire) {
          when(lastIssue) {
            issueValid := False
          } otherwise {
            issueIndex := issueIndex + 1
          }
        }

        when(dBus.d.fire) {
          val goodRsp = dBus.d.opcode === tilelink.Opcode.D.ACCESS_ACK && !dBus.d.denied && !dBus.d.corrupt
          when(!goodRsp) {
            memError := True
          }
          when(dBus.d.isLast()) {
            done := True
            error := memError || !goodRsp
            issueValid := False
            goto(IDLE)
          }
        }
      }
    }
  }

  cmd.ready := fsm.isActive(fsm.IDLE)
  busy := !fsm.isActive(fsm.IDLE)
}
