package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.sim._

/** Standalone scaffold protocol regression; no CPU, ELF, or external memory model.
  * Run: sbt 'Test / runMain vexiiriscv.soc.mico.AgentCfuSuboperationSim'
  */
object AgentCfuSuboperationSim extends App {
  // All state is allocated inside the override, not uninitialized subclass vals.
  class TableCfu(cfu: vexiiriscv.execute.cfu.CfuBusParameter, p: AgentCfuParameter,
                 empty: Boolean = false, invalid: Int = 0)
      extends AgentCfu(cfu, AgentCfuFiber.dummyBusParameter(p), p) {
    override protected def instructionTable: Seq[Instruction] = {
      val control = Reg(Bits(32 bits)) init(0)
      val count = Reg(UInt(3 bits)) init(0)
      val ops = Seq(
        Instruction("add", 1, 1, start = ctx =>
          ctx.reply((ctx.input(0).asUInt + ctx.input(1).asUInt).asBits)),
        Instruction("xor", 1, 2, start = ctx => ctx.reply(ctx.input(0) ^ ctx.input(1))),
        Instruction("write", 2, 1, start = ctx => { control := ctx.input(0); ctx.reply(B(0, 32 bits)) }),
        Instruction("read", 2, 2, start = ctx => ctx.reply(control)),
        Instruction("delayed", 1, 3,
          start = ctx => { count := 3; ctx.defer() },
          tick = Some(ctx => {
            when(count === 0) {
              ctx.reply(ctx.input(0) ^ ctx.rawInstruction ^ ctx.stateIndex.resize(32).asBits ^ ctx.cfuIndex.resize(32).asBits)
            } otherwise { count := count - 1 }
          })),
        Instruction("disabled", 1, 4, enabled = false, start = ctx => ctx.reply(B(123, 32 bits)))
      )
      if (empty) Seq.empty
      else if (invalid == 1) ops :+ Instruction("duplicateEncoding", 1, 1, enabled = false, start = _ => ())
      else if (invalid == 2) ops :+ Instruction("add", 3, 0, start = _ => ())
      else if (invalid == 3) ops :+ Instruction("outOfRange", 8, 0, start = _ => ())
      else if (invalid == 4) ops :+ Instruction("badSubop", 1, 128, start = _ => ())
      else if (invalid == 5) ops :+ Instruction("", 3, 0, start = _ => ())
      else if (invalid == 6) ops :+ Instruction("badDeferred", 3, 0, start = _.defer())
      else ops
    }
  }

  for (withMemory <- Seq(false, true)) {
    val mode = if (withMemory) "tilelink" else "direct"
    val p = AgentCfuParameter(
      vlen = 64, regDepth = 2,
      withTilelink = withMemory, withLoad = withMemory, withStore = withMemory
    )
    // Nonzero transaction IDs make preservation observable under backpressure.
    val cfu = AgentCfuFiber.getCfuBusParameters(32).copy(CFU_REQ_RESP_ID_W = 2)
    for (tableMode <- Seq("legacy", "custom", "empty")) {
    SimConfig.withVerilator
      .workspacePath(s"target/agent-cfu-suboperation-sim/$mode/$tableMode")
      .compile(if (tableMode == "legacy") new AgentCfu(cfu, AgentCfuFiber.dummyBusParameter(p), p)
               else new TableCfu(cfu, p, empty = tableMode == "empty"))
      .doSim(s"suboperation_$mode", seed = 42) { dut =>
        SimTimeout(100000)
        dut.clockDomain.forkStimulus(10)
        dut.io.bus.cmd.valid #= false
        dut.io.bus.cmd.function_id #= 0
        dut.io.bus.cmd.request_id #= 0
        dut.io.bus.cmd.state_index #= 0
        dut.io.bus.cmd.cfu_index #= 0
        dut.io.bus.cmd.raw_insn #= 0
        dut.io.bus.cmd.inputs.foreach(_ #= 0)
        dut.io.bus.rsp.ready #= false
        if (withMemory) {
          // Deliberately block A and never answer D. An invalid memory sub-op
          // must still respond promptly rather than entering a memory state.
          dut.io.dBus.a.ready #= false
          dut.io.dBus.d.valid #= false
          dut.io.dBus.d.opcode #= spinal.lib.bus.tilelink.Opcode.D.ACCESS_ACK
          dut.io.dBus.d.param #= 0
          dut.io.dBus.d.source #= 0
          dut.io.dBus.d.size #= 0
          dut.io.dBus.d.denied #= false
          if (dut.io.dBus.d.data != null) dut.io.dBus.d.data #= 0
          if (dut.io.dBus.d.corrupt != null) dut.io.dBus.d.corrupt #= false
        }
        dut.clockDomain.waitSampling(5)

        def noMemoryRequest(): Unit = if (withMemory) {
          assert(!dut.io.dBus.a.valid.toBoolean, s"$mode: unexpected TileLink A request")
        }
        def cycle(): Unit = {
          dut.clockDomain.waitSampling()
          sleep(1)
          noMemoryRequest()
        }
        def boundedWait(label: String)(condition: => Boolean): Unit = {
          var count = 0
          while (!condition && count < 20) { cycle(); count += 1 }
          assert(condition, s"$mode: timeout waiting for $label")
        }
        var serial = 0
        def command(function: Int, subOperation: Int, expectedStatus: Int,
                    expectedData: BigInt = 0, input0: BigInt = 0x1000, opcode: Int = 0x0b): Unit = {
          boundedWait("command ready")(dut.io.bus.cmd.ready.toBoolean)
          val id = serial % 4
          serial += 1
          dut.io.bus.cmd.function_id #= function
          dut.io.bus.cmd.request_id #= id
          dut.io.bus.cmd.raw_insn #= (BigInt(subOperation) << 25 | BigInt(function) << 12 | BigInt(opcode))
          dut.io.bus.cmd.state_index #= 3
          dut.io.bus.cmd.cfu_index #= 2
          dut.io.bus.cmd.inputs(0) #= input0
          dut.io.bus.cmd.inputs(1) #= 0x55
          dut.io.bus.cmd.valid #= true
          cycle()
          dut.io.bus.cmd.valid #= false
          // Upstream is free to change all request fields after fire.
          dut.io.bus.cmd.inputs(0) #= 0xdead
          dut.io.bus.cmd.raw_insn #= 0
          dut.io.bus.cmd.state_index #= 0
          dut.io.bus.cmd.cfu_index #= 0
          dut.io.bus.cmd.request_id #= ((id + 1) % 4)
          boundedWait("response")(dut.io.bus.rsp.valid.toBoolean)
          assert(dut.io.bus.rsp.status.toInt == expectedStatus,
            s"$mode: wrong status for ($function,$subOperation)")
          assert(dut.io.bus.rsp.outputs(0).toBigInt == expectedData)
          assert(dut.io.bus.rsp.response_id.toInt == id)

          // Keep a different request asserted while the old response is stalled.
          // No second command may be accepted, and every response bit must hold.
          dut.io.bus.cmd.function_id #= 1
          dut.io.bus.cmd.raw_insn #= 0x100b
          dut.io.bus.cmd.request_id #= ((id + 1) % 4)
          dut.io.bus.cmd.valid #= true
          for (_ <- 0 until 5) {
            cycle()
            assert(dut.io.bus.rsp.valid.toBoolean)
            assert(!dut.io.bus.cmd.ready.toBoolean)
            assert(dut.io.bus.rsp.status.toInt == expectedStatus)
            assert(dut.io.bus.rsp.outputs(0).toBigInt == expectedData)
            assert(dut.io.bus.rsp.response_id.toInt == id)
          }
          dut.io.bus.cmd.valid #= false
          dut.io.bus.rsp.ready #= true
          cycle()
          dut.io.bus.rsp.ready #= false
          assert(!dut.io.bus.rsp.valid.toBoolean, "response was not consumed exactly once")
          cycle()
          assert(!dut.io.bus.rsp.valid.toBoolean, "stalled second command was wrongly accepted")
        }

        if (tableMode == "legacy") {
        command(AgentCfuFunction.compute, 0, 0)
        command(AgentCfuFunction.config, 0, 0)
        for (function <- Seq(1, 2, 4, 5); subOperation <- Seq(1, 127)) {
          command(function, subOperation, 1)
        }
        command(7, 0, 1)
        if (!withMemory) {
          command(AgentCfuFunction.load, 0, 1)
          command(AgentCfuFunction.store, 0, 1)
        }
        // Recovery: rejected operations must not poison subsequent legacy calls.
        command(AgentCfuFunction.compute, 0, 0)
        command(AgentCfuFunction.config, 0, 0)
        } else if (tableMode == "custom") {
          command(1, 1, 0, 0xaa, input0 = 0x55)
          command(1, 2, 0, 0, input0 = 0x55)
          command(1, 1, 0, 0xaa, input0 = 0x55, opcode = 0x2b) // same projected key; CPU unchanged
          command(2, 1, 0, input0 = 77)
          command(2, 2, 0, 77)
          command(1, 3, 0, BigInt(0x1000) ^ (BigInt(3) << 25 | BigInt(1) << 12 | BigInt(0x0b)) ^ 3 ^ 2)
          command(1, 4, 1) // disabled
          command(1, 0, 1) // removed legacy entry
          command(2, 2, 0, 77) // state survives errors
        } else {
          for (function <- Seq(1, 2, 4, 5); operation <- Seq(0, 1)) command(function, operation, 1)
        }
        println(s"AgentCfuSuboperationSim $mode/$tableMode passed ($serial commands)")
      }
    }
  }
  val direct = AgentCfuParameter(withTilelink = false, withLoad = false, withStore = false)
  for (invalid <- 1 to 6) {
    var rejected = false
    try {
      SpinalConfig(targetDirectory = s"target/agent-cfu-suboperation-sim/rejected-$invalid")
        .generateVerilog(new TableCfu(AgentCfuFiber.getCfuBusParameters(32), direct, invalid = invalid))
    } catch {
      case error: Throwable =>
        val messages = Iterator.iterate(error)(_.getCause).takeWhile(_ != null).map(_.toString).mkString(" ")
        val expected = if (invalid == 1) "duplicate instruction encoding" else if (invalid == 2) "duplicate instruction name"
          else if (invalid <= 4) "instruction encoding must fit" else if (invalid == 5) "instruction name must not be empty"
          else "defer requires a tick handler"
        assert(messages.contains(expected), s"unexpected elaboration failure: $messages")
        rejected = true
    }
    assert(rejected, s"invalid instruction table $invalid accepted")
  }
  println("AgentCfuSuboperationSim table declaration rejection checks passed")
}
