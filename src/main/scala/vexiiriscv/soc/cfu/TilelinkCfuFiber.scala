package vexiiriscv.soc.cfu

import spinal.core._
import spinal.core.fiber._
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink._
import spinal.lib.bus.tilelink.fabric._

import vexiiriscv.execute.cfu._

trait CfuSpec {
  def cfuBusParameter(xlen: Int): CfuBusParameter
}

trait DirectCfuSpec extends CfuSpec {
  def build(cfuParam: CfuBusParameter, cfuBus: CfuBus): Area
}

trait TilelinkCfuSpec extends CfuSpec {
  def m2sParameters(name: Nameable): tilelink.M2sParameters
  def build(cfuParam: CfuBusParameter, cfuBus: CfuBus, dBus: tilelink.Bus): Area
}

class DirectCfuFiber(spec: DirectCfuSpec, xlen: Int) extends Area {
  val logic = new Area {
    val cfuParam = spec.cfuBusParameter(xlen)
    val cfuBus = CfuBus(cfuParam)
    val cfuArea = spec.build(cfuParam, cfuBus)
  }
}

class TilelinkCfuFiber(spec: TilelinkCfuSpec, xlen: Int) extends Area {
  val bus = Node.down()
  val dBus = bus.bus

  val logic = Fiber build new Area {
    bus.m2s forceParameters spec.m2sParameters(TilelinkCfuFiber.this)
    bus.s2m.supported load tilelink.S2mSupport.none()

    val cfuParam = spec.cfuBusParameter(xlen)
    val cfuBus = CfuBus(cfuParam)
    val cfuArea = spec.build(cfuParam, cfuBus, dBus)
  }
}
