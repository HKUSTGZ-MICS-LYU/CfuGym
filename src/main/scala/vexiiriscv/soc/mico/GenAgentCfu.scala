// SPDX-License-Identifier: MIT
//
// GenAgentCfu - standalone AgentCfu Verilog generator.
//
// MiCoSocGen elaborates the whole SoC and leaves AgentCfu as a hierarchical
// top for Yosys, so CFU cost estimation is coupled to a full SoC build.
// GenAgentCfu elaborates ONLY the CFU component (the same class the SoC
// overlays), so the agent cost rung can synthesize the agent's design in
// isolation.
//
// It is a tracked file (not part of the agent-editable AgentCfu*.scala
// surface), so the standalone entry point stays stable while the CFU design
// itself comes from the run workspace overlay.
package vexiiriscv.soc.mico

import spinal.core._

import java.io.File

/** CLI parameters for the standalone CFU generator. */
case class GenAgentCfuConfig(
    outDir: String = ".",
    cpuXlen: Int = 32,
    vlen: Int = 256,
    busWidth: Int = 32,
    regDepth: Int = 2,
    addressWidth: Int = 32,
    tilelink: Boolean = true,
    load: Boolean = true,
    store: Boolean = true
)

object GenAgentCfu extends App {
  private val parser = new scopt.OptionParser[GenAgentCfuConfig]("GenAgentCfu") {
    head("GenAgentCfu", "1.0")
    opt[String]("out-dir").action((v, c) => c.copy(outDir = v)).text("Verilog output directory")
    opt[Int]("cpu-xlen").action((v, c) => c.copy(cpuXlen = v)).text("CPU width used by the CFU command bus")
    opt[Int]("agent-cfu-vlen").action((v, c) => c.copy(vlen = v)).text("CFU vector register width in bits")
    opt[Int]("agent-cfu-bus-width").action((v, c) => c.copy(busWidth = v)).text("CFU data width in bits")
    opt[Int]("agent-cfu-reg-depth").action((v, c) => c.copy(regDepth = v)).text("CFU vector register count")
    opt[Int]("agent-cfu-address-width").action((v, c) => c.copy(addressWidth = v))
    opt[Unit]("agent-cfu-no-tilelink").action((_, c) => c.copy(tilelink = false)).text("Generate the direct (no TileLink) CFU")
    opt[Unit]("agent-cfu-no-load").action((_, c) => c.copy(load = false))
    opt[Unit]("agent-cfu-no-store").action((_, c) => c.copy(store = false))
  }

  val config = parser.parse(args, GenAgentCfuConfig()).getOrElse(sys.exit(1))

  val p = AgentCfuParameter(
    xlen = config.busWidth,
    vlen = config.vlen,
    regDepth = config.regDepth,
    addressWidth = config.addressWidth,
    withTilelink = config.tilelink,
    withLoad = config.tilelink && config.load,
    withStore = config.tilelink && config.store
  )

  // Same bus construction the SoC uses: the CPU-facing CfuBus width comes from
  // the CPU xlen, while the CFU datapath width is AgentCfuParameter.xlen.
  val cfuParam = AgentCfuFiber.getCfuBusParameters(config.cpuXlen)
  val busParam = AgentCfuFiber.dummyBusParameter(p)

  new File(config.outDir).mkdirs()

  val report = SpinalConfig(targetDirectory = config.outDir)
    .generateVerilog(new AgentCfu(cfuParam, busParam, p))

  println(s"[GenAgentCfu] wrote ${config.outDir}/${report.toplevel.definitionName}.v")
}
