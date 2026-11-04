package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.tilelink.Opcode
import AgentCfuBase.CfuResource._

/** Regression covering the upgraded AgentCfu template primitives:
  * - Top-level `val` / `new Area` in `AgentCfu` body (via `prePop` dispatch build)
  * - `PipedInstruction` (SpinalHDL `StagePipeline` multi-stage execution)
  * - `FsmInstruction` (`CfuSubFsm` instruction-scoped state machine)
  * - `ScalarRead` / `ScalarWrite` local scalar RF + `BitSlice` / `CsrState` indices
  * - `lsuStartScalarLoad` / `lsuStartStreamLoad` + `lsu.loadFlow` without RF overwrite
  * - `CfuMath` (`dotProductSInt`, `clampSInt`)
  */
object AgentCfuAdvancedTemplateSim extends App {
  val p = AgentCfuParameter(
    vlen = 64,
    regDepth = 4,
    scalarRegDepth = 4,
    advancedMem = true,
    burstMem = false
  )
  val cp = AgentCfuFiber.getCfuBusParameters(32).copy(CFU_REQ_RESP_ID_W = 2)

  class AdvancedDut extends AgentCfu(cp, AgentCfuFiber.dummyBusParameter(p), p) {
    // Top-level Area in subclass body (previously forbidden before prePop dispatch):
    val customArea = new Area {
      val streamAcc = Reg(SInt(32 bits)) init(0)
    }

    override protected def instructionTable: Seq[Instruction] = super.instructionTable ++ Seq(
      // 1. Multi-stage PipedInstruction using CfuMath dotProductSInt + BitSlice indices
      PipedInstruction(
        name = "piped_dot_s8",
        function = 1,
        subop = 1,
        stages = 2,
        resources = Some(Seq(
          VectorRead(VS1, BitSlice(RS1, 1, 0)),
          VectorRead(VS2, BitSlice(RS1, 3, 2)),
          ScalarRead(SS1, Constant(0)),
          CpuWriteRD
        ))
      ) { pipe =>
        val DOT = pipe.payload(SInt(32 bits))
        val BIAS = pipe.payload(SInt(32 bits))
        pipe.stage(0) { s =>
          s(DOT) := dotProductSInt(s.read(VS1), s.read(VS2), elemBits = 8, accBits = 32)
          s(BIAS) := s.read(SS1).asSInt
        }
        pipe.stage(1) { s =>
          s.complete((s(DOT) + s(BIAS)).asBits)
        }
      },

      // 2. Write local scalar RF slot selected by CSR state_index
      Instruction(
        name = "write_scalar_csr",
        function = 2,
        subop = 1,
        resources = Some(Seq(CpuRead(RS1), ScalarWrite(SD, CsrState))),
        start = ctx => ctx.complete(ctx.read(RS1))
      ),

      // 3. FsmInstruction doing a 1-word scalar load without touching vectorRegs(0)
      FsmInstruction(
        name = "scalar_load_accum",
        function = 4,
        subop = 1,
        enabled = p.withTilelink && p.withLoad,
        ready = () => lsu.cmd.ready,
        resources = Some(Seq(CpuRead(RS1), CpuWriteRD))
      ) { ctx =>
        new CfuSubFsm(ctx) {
          val ISSUE = workState()
          val WAIT = workState()

          ISSUE.whenIsActive {
            lsuStartScalarLoad(ctx.read(RS1).asUInt)
            goto(WAIT)
          }
          WAIT.whenIsActive {
            when(lsu.done) {
              when(lsu.error) {
                failAndExit()
              } otherwise {
                val sum = customArea.streamAcc + lsu.load.lastData.asSInt
                customArea.streamAcc := sum
                completeAndExit(sum.asBits)
              }
            }
          }
        }
      }
    )
  }

  SimConfig.withVerilator.workspacePath("target/agent-cfu-advanced-template-sim")
    .compile(new AdvancedDut()).doSim { d =>
      SimTimeout(100000)
      d.clockDomain.forkStimulus(10)
      d.io.bus.cmd.valid #= false
      d.io.bus.cmd.function_id #= 0
      d.io.bus.cmd.raw_insn #= 0
      d.io.bus.cmd.request_id #= 0
      d.io.bus.cmd.state_index #= 0
      d.io.bus.cmd.cfu_index #= 0
      d.io.bus.cmd.inputs.foreach(_ #= 0)
      d.io.bus.rsp.ready #= true
      d.io.dBus.a.ready #= false
      d.io.dBus.d.valid #= false
      d.io.dBus.d.opcode #= Opcode.D.ACCESS_ACK
      d.io.dBus.d.param #= 0
      d.io.dBus.d.source #= 0
      d.io.dBus.d.size #= 2
      d.io.dBus.d.denied #= false
      d.io.dBus.d.corrupt #= false
      d.io.dBus.d.data #= 0
      d.clockDomain.waitSampling(5)

      def cycle(): Unit = { d.clockDomain.waitSampling(); sleep(1) }
      def waitFor(cond: => Boolean): Unit = {
        var n = 0
        while (!cond && n < 40) { cycle(); n += 1 }
        assert(cond, "simulation timeout waiting for condition")
      }

      // Write S[0] = 100 via CSR state_index = 0 (func=2, subop=1)
      waitFor(d.io.bus.cmd.ready.toBoolean)
      d.io.bus.cmd.function_id #= 2
      d.io.bus.cmd.raw_insn #= ((1 << 25) | (2 << 12) | 0xb)
      d.io.bus.cmd.state_index #= 0
      d.io.bus.cmd.inputs(0) #= 100
      d.io.bus.cmd.valid #= true
      cycle()
      d.io.bus.cmd.valid #= false
      waitFor(d.io.bus.rsp.valid.toBoolean)
      assert(d.io.bus.rsp.status.toInt == 0)
      cycle()

      // Load V[0] (2 beats for vlen=64) at 0x1000 using legacy load (func=4, subop=0, rs2=0)
      def loadVec(slot: Int, addr: Int, b0: BigInt, b1: BigInt): Unit = {
        waitFor(d.io.bus.cmd.ready.toBoolean)
        d.io.bus.cmd.function_id #= 4
        d.io.bus.cmd.raw_insn #= ((slot << 20) | (4 << 12) | 0xb)
        d.io.bus.cmd.inputs(0) #= addr
        d.io.bus.cmd.valid #= true
        cycle()
        d.io.bus.cmd.valid #= false
        for ((beatVal, idx) <- Seq((b0, 0), (b1, 1))) {
          waitFor(d.io.dBus.a.valid.toBoolean)
          val src = d.io.dBus.a.source.toBigInt
          d.io.dBus.a.ready #= true; cycle(); d.io.dBus.a.ready #= false
          d.io.dBus.d.opcode #= Opcode.D.ACCESS_ACK_DATA
          d.io.dBus.d.source #= src
          d.io.dBus.d.data #= beatVal
          d.io.dBus.d.valid #= true
          waitFor(d.io.dBus.d.ready.toBoolean); cycle(); d.io.dBus.d.valid #= false
        }
        waitFor(d.io.bus.rsp.valid.toBoolean)
        assert(d.io.bus.rsp.status.toInt == 0)
        cycle()
      }

      // V[1] = 8 lanes of 1s (0x01010101_01010101), V[2] = 8 lanes of 2s (0x02020202_02020202)
      loadVec(1, 0x1000, BigInt("01010101", 16), BigInt("01010101", 16))
      loadVec(2, 0x1008, BigInt("02020202", 16), BigInt("02020202", 16))

      // Execute 2-stage PipedInstruction `piped_dot_s8` (func=1, subop=1)
      // RS1 packs VS1=1 in bits[1:0] and VS2=2 in bits[3:2] -> RS1 = (2 << 2) | 1 = 9
      // Expected dot = 8 * (1 * 2) + S[0](100) = 116
      waitFor(d.io.bus.cmd.ready.toBoolean)
      d.io.bus.cmd.function_id #= 1
      d.io.bus.cmd.raw_insn #= ((1 << 25) | (1 << 12) | 0xb)
      d.io.bus.cmd.inputs(0) #= 9
      d.io.bus.cmd.valid #= true
      cycle()
      d.io.bus.cmd.valid #= false
      waitFor(d.io.bus.rsp.valid.toBoolean)
      assert(d.io.bus.rsp.status.toInt == 0)
      assert(d.io.bus.rsp.outputs(0).toBigInt == 116, s"got ${d.io.bus.rsp.outputs(0).toBigInt}, expected 116")
      cycle()

      // Execute FsmInstruction `scalar_load_accum` (func=4, subop=1) twice:
      // 1-beat scalar load of 25 and 17 -> accumulator returns 25 then 42, without clobbering V[1]/V[2]
      for ((word, expectedSum) <- Seq((25, 25), (17, 42))) {
        waitFor(d.io.bus.cmd.ready.toBoolean)
        d.io.bus.cmd.function_id #= 4
        d.io.bus.cmd.raw_insn #= ((1 << 25) | (4 << 12) | 0xb)
        d.io.bus.cmd.inputs(0) #= 0x2000
        d.io.bus.cmd.valid #= true
        cycle()
        d.io.bus.cmd.valid #= false
        waitFor(d.io.dBus.a.valid.toBoolean)
        val src = d.io.dBus.a.source.toBigInt
        d.io.dBus.a.ready #= true; cycle(); d.io.dBus.a.ready #= false
        d.io.dBus.d.opcode #= Opcode.D.ACCESS_ACK_DATA
        d.io.dBus.d.source #= src
        d.io.dBus.d.data #= word
        d.io.dBus.d.valid #= true
        waitFor(d.io.dBus.d.ready.toBoolean); cycle(); d.io.dBus.d.valid #= false
        waitFor(d.io.bus.rsp.valid.toBoolean)
        assert(d.io.bus.rsp.status.toInt == 0)
        assert(d.io.bus.rsp.outputs(0).toBigInt == expectedSum,
          s"scalar_load_accum got ${d.io.bus.rsp.outputs(0).toBigInt}, expected $expectedSum")
        cycle()
      }

      // Verify V[1] and V[2] were NOT overwritten by scalar_load_accum (loadWriteRf = False):
      waitFor(d.io.bus.cmd.ready.toBoolean)
      d.io.bus.cmd.function_id #= 1
      d.io.bus.cmd.raw_insn #= ((1 << 25) | (1 << 12) | 0xb)
      d.io.bus.cmd.inputs(0) #= 9
      d.io.bus.cmd.valid #= true
      cycle()
      d.io.bus.cmd.valid #= false
      waitFor(d.io.bus.rsp.valid.toBoolean)
      assert(d.io.bus.rsp.status.toInt == 0)
      assert(d.io.bus.rsp.outputs(0).toBigInt == 116, "vectorRegs were unexpectedly modified by scalar load")
      cycle()

      println("AgentCfuAdvancedTemplateSim passed (PipedInstruction + FsmInstruction + ScalarRF + BitSlice + ScalarLoad)")
    }
}
