package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.tilelink.Opcode

/** Real legacy LSU roundtrip: CPU value is address, encoded RS2 number is RF slot.
  *
  * The two-slot case checks that the encoded RS2 register number selects the slot
  * (input1 value 0 would select slot 0 if the two were confused). The single-slot
  * case checks that regDepth = 1 still moves beats through the one-slot RF.
  */
object AgentCfuMemoryResourceSim extends App {
  for ((regDepth, rs2Number, mode) <- Seq((2, 13, "two-slot"), (1, 13, "single-slot"))) {
    val p = AgentCfuParameter(vlen = 64, regDepth = regDepth)
    val cp = AgentCfuFiber.getCfuBusParameters(32).copy(CFU_REQ_RESP_ID_W = 2)
    SimConfig.withVerilator.workspacePath(s"target/agent-cfu-memory-resource-sim/$mode")
      .compile(new AgentCfu(cp, AgentCfuFiber.dummyBusParameter(p), p)).doSim { d =>
        SimTimeout(100000)
        d.clockDomain.forkStimulus(10)
        d.io.bus.cmd.valid #= false; d.io.bus.cmd.function_id #= 0
        d.io.bus.cmd.raw_insn #= 0; d.io.bus.cmd.request_id #= 0
        d.io.bus.cmd.state_index #= 0; d.io.bus.cmd.cfu_index #= 0
        d.io.bus.cmd.inputs.foreach(_ #= 0); d.io.bus.rsp.ready #= false
        d.io.dBus.a.ready #= false; d.io.dBus.d.valid #= false
        d.io.dBus.d.opcode #= Opcode.D.ACCESS_ACK; d.io.dBus.d.param #= 0
        d.io.dBus.d.source #= 0; d.io.dBus.d.size #= 2
        d.io.dBus.d.denied #= false; d.io.dBus.d.corrupt #= false; d.io.dBus.d.data #= 0
        d.clockDomain.waitSampling(5)
        def cycle(): Unit = { d.clockDomain.waitSampling(); sleep(1) }
        def waitFor(test: => Boolean): Unit = { var n = 0; while(!test && n < 30) { cycle(); n += 1 }; assert(test) }
        def command(function: Int, id: Int, address: Int): Unit = {
          waitFor(d.io.bus.cmd.ready.toBoolean)
          d.io.bus.cmd.function_id #= function
          // Encoded RS2 register number selects the slot; input1 stays 0.
          d.io.bus.cmd.raw_insn #= ((rs2Number << 20) | (function << 12) | 0xb)
          d.io.bus.cmd.inputs(0) #= address; d.io.bus.cmd.inputs(1) #= 0
          d.io.bus.cmd.request_id #= id; d.io.bus.cmd.valid #= true
          cycle(); d.io.bus.cmd.valid #= false
          d.io.bus.cmd.inputs(0) #= 0xdead; d.io.bus.cmd.raw_insn #= 0
          d.io.bus.cmd.request_id #= 0
        }
        def beat(address: Int, data: BigInt, store: Boolean): Unit = {
          waitFor(d.io.dBus.a.valid.toBoolean)
          assert(d.io.dBus.a.address.toBigInt == address)
          assert(d.io.dBus.a.opcode.toEnum == (if(store) Opcode.A.PUT_FULL_DATA else Opcode.A.GET))
          if(store) assert(d.io.dBus.a.data.toBigInt == data)
          val source = d.io.dBus.a.source.toBigInt
          // Exercise A backpressure before responding on D.
          cycle(); assert(d.io.dBus.a.valid.toBoolean)
          d.io.dBus.a.ready #= true; cycle(); d.io.dBus.a.ready #= false
          d.io.dBus.d.opcode #= (if(store) Opcode.D.ACCESS_ACK else Opcode.D.ACCESS_ACK_DATA)
          d.io.dBus.d.source #= source; d.io.dBus.d.data #= data
          d.io.dBus.d.valid #= true
          waitFor(d.io.dBus.d.ready.toBoolean); cycle(); d.io.dBus.d.valid #= false
        }
        def response(id: Int): Unit = {
          waitFor(d.io.bus.rsp.valid.toBoolean)
          for(_ <- 0 until 4) {
            assert(d.io.bus.rsp.status.toInt == 0)
            assert(d.io.bus.rsp.response_id.toInt == id)
            assert(d.io.bus.rsp.outputs(0).toBigInt == 0)
            assert(!d.io.bus.cmd.ready.toBoolean); cycle()
          }
          d.io.bus.rsp.ready #= true; cycle(); d.io.bus.rsp.ready #= false; cycle()
        }
        command(4, 1, 0x1000)
        beat(0x1000, BigInt("11223344",16), false); beat(0x1004, BigInt("aabbccdd",16), false)
        response(1)
        command(5, 2, 0x2000)
        beat(0x2000, BigInt("11223344",16), true); beat(0x2004, BigInt("aabbccdd",16), true)
        response(2)
        println(s"AgentCfuMemoryResourceSim $mode legacy load/store roundtrip passed")
      }
  }
}
