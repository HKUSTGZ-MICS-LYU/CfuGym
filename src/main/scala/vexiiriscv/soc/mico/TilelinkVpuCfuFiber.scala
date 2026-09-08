package vexiiriscv.soc.mico

import spinal.core
import spinal.core._
import spinal.core.fiber._

import spinal.lib._
import spinal.lib.bus._
import spinal.lib.bus.misc._
import spinal.lib.bus.tilelink._
import spinal.lib.bus.tilelink.fabric._
import spinal.lib.misc._

import scala.collection.mutable.ArrayBuffer

import vexiiriscv.execute.cfu._
import vexiiriscv.soc.cfu.{TilelinkCfuFiber, TilelinkCfuSpec}

/* A Template for a Tilelink CFU (Custom Functional Unit) Fiber.
 * This fiber connects a CFU bus to CPU, allowing custom instructions
 * to be executed in a VexiiRiscv system.
 */

object TilelinkVpuCfuFiber {

  def getM2sParameters(name: Nameable, width: Int = 32, pendingSize: Int = 4) = tilelink.M2sParameters(
        addressWidth = 32,
        dataWidth = width,
        masters = List(
          tilelink.M2sAgent(
            name = name,
            mapping = List(
              tilelink.M2sSource(
                id = SizeMapping(0, pendingSize),
                emits = M2sTransfers(
                  get = tilelink.SizeRange(1, width / 8),
                  putFull = tilelink.SizeRange(1, width / 8)
                )
              )
            )
          )
        )
      )
  
  def getCfuBusParameters(xlen: Int = 32) = CfuBusParameter(
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
}

case class VpuCfuSpec(vpuParam: VpuCfuParameter) extends TilelinkCfuSpec {
  override def m2sParameters(name: Nameable) = TilelinkVpuCfuFiber.getM2sParameters(
    name,
    vpuParam.xlen,
    vpuParam.pendingSize
  )

  override def cfuBusParameter(xlen: Int) = TilelinkVpuCfuFiber.getCfuBusParameters(xlen)

  override def build(cfuParam: CfuBusParameter, cfuBus: CfuBus, dBus: tilelink.Bus) = new Area {
    val cfu = new VpuCfu(cfuParam, dBus.p, vpuParam)
    cfu.io.bus <> cfuBus
    cfu.io.dBus <> dBus
  }
}

class TilelinkVpuCfuFiber(vpuParam: VpuCfuParameter, xlen: Int) extends TilelinkCfuFiber(VpuCfuSpec(vpuParam), xlen)
