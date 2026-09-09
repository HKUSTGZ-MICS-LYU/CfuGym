// SPDX-License-Identifier: MIT
//
// AgentCfuFiber - SoC-facing wrapper for the agent CFU shell.
//
// Like AgentCfu.scala, this is a read-only default for design agents. Agent runs
// compile an isolated workspace copy instead; see agents/README.md.
// Keep the stable surface: AgentCfuFiber.getCfuBusParameters(xlen),
// getM2sParameters(name, p), dummyBusParameter(p), and
// class AgentCfuFiber(p, xlen) exposing `logic.cfuBus` and `bus`.
package vexiiriscv.soc.mico

import spinal.core._
import spinal.core.fiber._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.bus.tilelink.fabric._
import spinal.lib.bus.misc.SizeMapping

import vexiiriscv.execute.cfu._

object AgentCfuFiber {
  def getCfuBusParameters(xlen: Int) = CfuBusParameter(
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

  def getM2sParameters(name: Nameable, p: AgentCfuParameter): tilelink.M2sParameters = {
    val get = if (p.withLoad) tilelink.SizeRange(1, p.beatBytes) else tilelink.SizeRange.none
    val putFull = if (p.withStore) tilelink.SizeRange(1, p.beatBytes) else tilelink.SizeRange.none
    tilelink.M2sParameters(
      addressWidth = p.addressWidth,
      dataWidth = p.xlen,
      masters = List(
        tilelink.M2sAgent(
          name = name,
          mapping = List(
            tilelink.M2sSource(
              id = SizeMapping(0, p.pendingSize),
              emits = M2sTransfers(get = get, putFull = putFull)
            )
          )
        )
      )
    )
  }

  // Direct mode still gives the component a type-correct bus parameter. The
  // generated component has no dBus port when withTilelink is false.
  def dummyBusParameter(p: AgentCfuParameter): BusParameter = BusParameter.simple(
    addressWidth = p.addressWidth,
    dataWidth = p.xlen,
    sizeBytes = p.beatBytes,
    sourceWidth = log2Up(p.pendingSize) max 1
  )
}

/** SoC-facing wrapper for the optional-TileLink AgentCfu shell. */
class AgentCfuFiber(p: AgentCfuParameter, xlen: Int) extends Area {
  val bus = p.withTilelink generate Node.down()

  val logic = Fiber build new Area {
    val cfuParam = AgentCfuFiber.getCfuBusParameters(xlen)
    val cfuBus = CfuBus(cfuParam)
    val cfu = if (p.withTilelink) {
      bus.m2s forceParameters AgentCfuFiber.getM2sParameters(AgentCfuFiber.this, p)
      bus.s2m.supported load tilelink.S2mSupport.none()
      new AgentCfu(cfuParam, bus.bus.p, p)
    } else {
      new AgentCfu(cfuParam, AgentCfuFiber.dummyBusParameter(p), p)
    }

    cfu.io.bus <> cfuBus
    if (p.withTilelink) cfu.io.dBus <> bus.bus
  }
}
