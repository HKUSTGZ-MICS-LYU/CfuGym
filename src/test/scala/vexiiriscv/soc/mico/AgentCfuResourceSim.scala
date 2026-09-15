package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.sim._
import AgentCfuBase.CfuResource
import AgentCfuBase.CfuResource._

/** Explicit operands and writeback regression, independent of CPU/software. */
object AgentCfuResourceSim extends App {
  val p = AgentCfuParameter(vlen = 64, regDepth = 3, withTilelink = false, withLoad = false, withStore = false)
  val cp = AgentCfuFiber.getCfuBusParameters(32).copy(CFU_REQ_RESP_ID_W = 2)
  class Dut(invalid: Int = 0) extends AgentCfu(cp, AgentCfuFiber.dummyBusParameter(p), p) {
    override protected def instructionTable: Seq[Instruction] = {
      val count = Reg(UInt(3 bits)) init(0)
      def ins(name: String, op: Int, rs: Seq[CfuResource])(fn: InstructionContext => Unit) =
        Instruction(name, 1, op, resources = Some(rs), start = fn)
      val ops = Seq(
        ins("add", 1, Seq(CpuRead(RS1), CpuRead(RS2), CpuWriteRD))(c => c.complete((c.read(RS1).asUInt + c.read(RS2).asUInt).asBits)),
        ins("set", 2, Seq(CpuRead(RS2), VectorWrite(VD, CpuValue(RS1))))(c => c.complete(c.read(RS2).resize(64))),
        ins("sum", 3, Seq(VectorRead(VS1, CpuValue(RS1)), VectorRead(VS2, CpuValue(RS2)), CpuWriteRD))(c => c.complete((c.read(VS1).asUInt + c.read(VS2).asUInt).resize(32).asBits)),
        ins("inplace", 4, Seq(VectorRead(VS1, CpuValue(RS1)), VectorRead(VS2, CpuValue(RS2)), VectorWrite(VD, CpuValue(RS1))))(c => c.complete((c.read(VS1).asUInt + c.read(VS2).asUInt).asBits)),
        ins("get", 5, Seq(VectorRead(VS1, CpuValue(RS1)), CpuWriteRD))(c => c.complete(c.read(VS1).resize(32))),
        Instruction("delayed", 1, 6, resources = Some(Seq(VectorRead(VS1, CpuValue(RS1)), VectorWrite(VD, CpuValue(RS2)))),
          start = c => { count := 3; c.defer() }, tick = Some(c => {
            // Deliberately mutate the source after acceptance: read must be a snapshot.
            vectorRegs(0) := B(999, 64 bits)
            when(count === 0) { c.complete(c.read(VS1)) } otherwise { count := count - 1 }
          })),
        ins("error", 7, Seq(VectorWrite(VD, CpuValue(RS1))))(c => c.complete(B(444, 64 bits), B"001")),
        ins("ack", 8, Seq.empty)(c => c.complete()),
        Instruction("blocked", 1, 9, resources = Some(Seq(VectorWrite(VD, CpuValue(RS1)))), ready = () => False,
          start = c => c.complete(B(55, 64 bits)))
      )
      if(invalid == 0) ops else Seq(invalid match {
        case 1 => ins("bad", 1, Seq(CpuRead(RS1), CpuRead(RS1)))(c => c.complete())
        case 2 => ins("bad", 1, Seq(CpuWriteRD, VectorWrite(VD, Constant(0))))(c => c.complete(B(0, 32 bits)))
        case 3 => ins("bad", 1, Seq(CpuWriteRD))(c => c.complete(c.read(RS1)))
        case 4 => ins("bad", 1, Seq(VectorWrite(VD, Constant(0))))(c => c.complete(B(0, 32 bits)))
        case 5 => ins("bad", 1, Seq(VectorRead(VS1, Constant(3))))(c => c.complete())
        case 6 => ins("bad", 1, Seq.empty)(c => c.reply(B(0, 32 bits)))
        // Platform fallback: an over-wide slot address is reported here instead of
        // failing later with an opaque "Vec address width mismatch".
        case 7 => ins("bad", 1, Seq(CpuWriteRD))(c => c.complete(vectorRead(U(0, 8 bits)).resize(32)))
      })
    }
  }
  SimConfig.withVerilator.workspacePath("target/agent-cfu-resource-sim").compile(new Dut()).doSim { d =>
    SimTimeout(100000)
    d.clockDomain.forkStimulus(10)
    d.io.bus.cmd.valid #= false
    d.io.bus.cmd.function_id #= 1
    d.io.bus.cmd.request_id #= 0
    d.io.bus.cmd.raw_insn #= 0
    d.io.bus.cmd.cfu_index #= 0
    d.io.bus.cmd.state_index #= 0
    d.io.bus.cmd.inputs.foreach(_ #= 0)
    d.io.bus.rsp.ready #= false
    d.clockDomain.waitSampling(5)
    var serial = 0
    def cycle(): Unit = { d.clockDomain.waitSampling(); sleep(1) }
    def cmd(op: Int, a: BigInt = 0, b: BigInt = 0, value: BigInt = 0, status: Int = 0): Unit = {
      val id = serial % 4; serial += 1
      d.io.bus.cmd.raw_insn #= ((BigInt(op) << 25) | 0x100b)
      d.io.bus.cmd.inputs(0) #= a; d.io.bus.cmd.inputs(1) #= b
      d.io.bus.cmd.request_id #= id; d.io.bus.cmd.valid #= true
      sleep(1)
      assert(d.io.bus.cmd.ready.toBoolean, "command unexpectedly blocked")
      cycle(); d.io.bus.cmd.valid #= false
      d.io.bus.cmd.inputs(0) #= 2; d.io.bus.cmd.inputs(1) #= 0
      d.io.bus.cmd.raw_insn #= 0; d.io.bus.cmd.request_id #= ((id + 1) % 4)
      var n = 0
      while(!d.io.bus.rsp.valid.toBoolean && n < 20) { cycle(); n += 1 }
      assert(d.io.bus.rsp.valid.toBoolean)
      for(_ <- 0 until 4) {
        assert(d.io.bus.rsp.outputs(0).toBigInt == value, s"op $op result")
        assert(d.io.bus.rsp.status.toInt == status, s"op $op status")
        assert(d.io.bus.rsp.response_id.toInt == id)
        assert(!d.io.bus.cmd.ready.toBoolean)
        cycle()
      }
      d.io.bus.rsp.ready #= true; cycle(); d.io.bus.rsp.ready #= false
      cycle(); assert(!d.io.bus.rsp.valid.toBoolean)
    }
    cmd(1, 0x55, 0x55, 0xaa)
    cmd(2, 0, 10); cmd(2, 1, 20); cmd(2, 2, 30)
    cmd(3, 0, 1, 30)
    cmd(4, 0, 1); cmd(5, 0, value = 30)
    cmd(2, 3, 99, status = 1); cmd(2, BigInt("10000000",16), 99, status = 1)
    cmd(5, 3, status = 1); cmd(5, 0, value = 30)
    cmd(2, 2, 77)
    cmd(6, 0, 2); cmd(5, 2, value = 30); cmd(5, 0, value = 999)
    cmd(7, 2, status = 1); cmd(5, 2, value = 30)
    cmd(8)
    d.io.bus.cmd.raw_insn #= ((BigInt(9) << 25) | 0x100b)
    d.io.bus.cmd.inputs(0) #= 0; d.io.bus.cmd.valid #= true
    sleep(1)
    for(_ <- 0 until 3) { assert(!d.io.bus.cmd.ready.toBoolean); assert(!d.io.bus.rsp.valid.toBoolean); cycle() }
    d.io.bus.cmd.valid #= false
    cmd(9, 3, status = 1)
    println(s"AgentCfuResourceSim passed ($serial commands)")
  }
  val expected = Seq("duplicate resource role", "at most one resource write", "undeclared resource read",
    "vector result width mismatch", "constant vector index out of range", "must use complete",
    "vector slot address has 8 bits")
  for(n <- 1 to 7) {
    var rejected = false
    try SpinalConfig(targetDirectory = s"target/agent-cfu-resource-rejected-$n").generateVerilog(new Dut(n))
    catch { case e: Throwable =>
      val messages = Iterator.iterate(e)(_.getCause).takeWhile(_ != null).map(_.toString).mkString(" ")
      assert(messages.contains(expected(n - 1)), messages)
      // Every platform-enforced design check carries the read-only hint.
      assert(messages.contains("read-only platform file"), messages)
      rejected = true
    }
    assert(rejected)
  }
  println("AgentCfuResourceSim resource rejection checks passed")
}
