package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.sim._
import AgentCfuBase.CfuResource
import AgentCfuBase.CfuResource._

/** regDepth = 1: the vector RF has exactly one slot, addressed by constant 0. */
object AgentCfuSingleSlotSim extends App {
  val p = AgentCfuParameter(vlen = 64, regDepth = 1, withTilelink = false, withLoad = false, withStore = false)
  val cp = AgentCfuFiber.getCfuBusParameters(32).copy(CFU_REQ_RESP_ID_W = 2)

  class Dut extends AgentCfu(cp, AgentCfuFiber.dummyBusParameter(p), p) {
    override protected def instructionTable: Seq[Instruction] = Seq(
      // Combinational write of the single slot from two CPU words.
      Instruction("set", 1, 1,
        resources = Some(Seq(CpuRead(RS1), CpuRead(RS2), VectorWrite(VD, Constant(0)))),
        start = ctx => ctx.complete(ctx.read(RS1) ## ctx.read(RS2))),
      // Combinational read-back of the single slot.
      Instruction("get", 1, 2,
        resources = Some(Seq(VectorRead(VS1, Constant(0)), CpuWriteRD)),
        start = ctx => ctx.complete(ctx.read(VS1)(31 downto 0))),
      // Deferred: the framework's latched snapshot and saved index are addressed.
      Instruction("setd", 1, 3,
        resources = Some(Seq(CpuRead(RS1), VectorWrite(VD, Constant(0)))),
        start = ctx => ctx.defer(),
        tick = Some(ctx => ctx.complete(ctx.read(RS1).resize(64)))),
      // Index taken from a CPU value (must be 0 for a single slot).
      Instruction("setv", 1, 4,
        resources = Some(Seq(CpuRead(RS1), VectorWrite(VD, CpuValue(RS2)))),
        start = ctx => ctx.complete(ctx.read(RS1).resize(64)))
    )
  }

  SimConfig.withVerilator.workspacePath("target/agent-cfu-single-slot-sim")
    .compile(new Dut()).doSim { d =>
      SimTimeout(100000)
      d.clockDomain.forkStimulus(10)
      d.io.bus.cmd.valid #= false
      d.io.bus.cmd.function_id #= 0
      d.io.bus.cmd.raw_insn #= 0
      d.io.bus.cmd.request_id #= 0
      d.io.bus.cmd.state_index #= 0
      d.io.bus.cmd.cfu_index #= 0
      d.io.bus.cmd.inputs.foreach(_ #= 0)
      d.io.bus.rsp.ready #= false
      d.clockDomain.waitSampling(5)

      var serial = 0
      def cycle(): Unit = { d.clockDomain.waitSampling(); sleep(1) }
      def command(op: Int, a: BigInt, b: BigInt = 0): BigInt = {
        val id = serial % 4
        serial += 1
        var n = 0
        while (!d.io.bus.cmd.ready.toBoolean && n < 30) { cycle(); n += 1 }
        assert(d.io.bus.cmd.ready.toBoolean, "command not accepted")
        d.io.bus.cmd.function_id #= 1
        d.io.bus.cmd.raw_insn #= ((BigInt(op) << 25) | 0x100b)
        d.io.bus.cmd.inputs(0) #= a
        d.io.bus.cmd.inputs(1) #= b
        d.io.bus.cmd.request_id #= id
        d.io.bus.cmd.valid #= true
        cycle()
        d.io.bus.cmd.valid #= false
        // Upstream may change everything after acceptance.
        d.io.bus.cmd.inputs(0) #= 0
        d.io.bus.cmd.inputs(1) #= 0
        d.io.bus.cmd.raw_insn #= 0
        n = 0
        while (!d.io.bus.rsp.valid.toBoolean && n < 30) { cycle(); n += 1 }
        assert(d.io.bus.rsp.valid.toBoolean, s"op $op: no response")
        val data = d.io.bus.rsp.outputs(0).toBigInt
        assert(d.io.bus.rsp.status.toInt == 0, s"op $op: status")
        assert(d.io.bus.rsp.response_id.toInt == id, s"op $op: response id")
        d.io.bus.rsp.ready #= true
        cycle()
        d.io.bus.rsp.ready #= false
        data
      }

      command(1, BigInt("11223344", 16), BigInt("55667788", 16))
      assert(command(2, 0) == BigInt("55667788", 16), "constant-index read-back")
      command(3, BigInt("0badf00d", 16))
      assert(command(2, 0) == BigInt("0badf00d", 16), "deferred snapshot read-back")
      command(4, BigInt("cafebabe", 16), 0)
      assert(command(2, 0) == BigInt("cafebabe", 16), "cpu-value index read-back")
      // A nonzero slot index must be rejected, not silently aliased to slot 0.
      var n = 0
      while (!d.io.bus.cmd.ready.toBoolean && n < 30) { cycle(); n += 1 }
      d.io.bus.cmd.function_id #= 1
      d.io.bus.cmd.raw_insn #= ((BigInt(4) << 25) | 0x100b)
      d.io.bus.cmd.inputs(0) #= BigInt("dead", 16)
      d.io.bus.cmd.inputs(1) #= 7
      d.io.bus.cmd.request_id #= 0
      d.io.bus.cmd.valid #= true
      cycle()
      d.io.bus.cmd.valid #= false
      n = 0
      while (!d.io.bus.rsp.valid.toBoolean && n < 30) { cycle(); n += 1 }
      assert(d.io.bus.rsp.valid.toBoolean, "invalid index must still respond")
      assert(d.io.bus.rsp.status.toInt == 1, "invalid index must report an error")
      d.io.bus.rsp.ready #= true
      cycle()
      d.io.bus.rsp.ready #= false
      assert(command(2, 0) == BigInt("cafebabe", 16), "invalid index must not write the slot")

      println(s"AgentCfuSingleSlotSim passed ($serial commands)")
    }
}
