# SpinalHDL Pipeline, Plugin, and Fiber Notes

## Pipeline Primitives

The local pipeline APIs live under `spinal.lib.misc.pipeline`.

Use `Payload[T]` as a typed key for values that move through a pipeline. A `Node` materializes the value at a specific point:

```scala
val SEL = Payload(Bool())
val OPA = Payload(Bits(width bits))

val extract = new extractStage.Area {
  SEL := doIt
  OPA := sourceBits
}

val compute = new computeStage.Area {
  when(SEL) {
    result := use(OPA)
  }
}
```

`Node.apply(payload)` reads or creates the hardware signal for that payload at the node. `Node.insert(data)` creates a new payload from a signal at that node.

Use `StageLink(up, down)` when a register boundary is needed. `Builder(links)` propagates payload demand and builds the links. `StageLink` registers matching payloads and optional `valid`; it also propagates `ready` backpressure. Use `withoutCollapse()` only when bubble collapse would be semantically wrong, and `withPayloadHold()` when payload must hold under a specific upstream-valid/ready condition.

Use `StagePipeline` or `StageCtrlPipeline` for small local pipelines; use explicit `Node()` arrays when the stage count is parameterized, as in `BitNetCfu`.

Do not mirror payloads manually just to pass a value to the next stage. Define one payload and let links carry it.

## Node Handshake

Important node status helpers:

- `isValid`: transaction is present.
- `isReady`: downstream can move it.
- `isFiring`: transaction advances (`isReady` and not removed).
- `isMoving`: transaction leaves this node next cycle.
- `isCancel`: downstream cancellation.

Use state updates on fire/move semantics, not merely `valid`, when the update must correspond to an accepted transaction.

## Plugins and Hosts

`VexiiRiscv` itself is mostly a `PluginHost`; real hardware comes from plugins. `FiberPlugin` lets plugins find services and coordinate elaboration.

Patterns:

- `val logic = during setup new Area { ... awaitBuild(); ... }`: discover services and register instruction/decode intent, then wait until build phase before using built hardware.
- `val logic = during build new Area { ... }`: instantiate hardware after setup-time dependencies are resolved.
- `Fiber patch new Area { ... }`: connect pieces that must wait until build-time handles exist, such as `cpu.logic.core.host[CfuPlugin].logic.bus`.
- `host[Type]`: require exactly one service.
- `host.get[Type]`: optional service.
- `host.find[Type](predicate)`: select one matching service, commonly by lane or register file.
- `retains(...)`: hold locks while a plugin registers micro-ops, writeback ports, CSR mappings, or pipeline logic.
- Release locks after the plugin has registered the behavior other plugins depend on.

`Handle[T]` is a delayed value. Calling `.get` waits until loaded. Use `soon(...)` when a task will eventually load a handle that depends on another handle; this helps avoid dependency-chain ambiguity.

## CFU-Specific Plugin Guidance

For CPU-side instruction plugins:

- Register the instruction with `layer.add(...)`.
- Add source specs at the execute stage that first consumes the register value.
- Set completion at the stage that produces writeback.
- Add decode payloads once, with sane defaults.
- Release `uopRetainer`/locks after all micro-op metadata is registered.

For stream CFUs through `CfuPlugin`, usually do not modify `CfuPlugin` itself. Instead configure encodings/parameters through the existing Vexii param path, then implement accelerator-side semantics behind `CfuBus`.

## Timing and Area Discipline

- Put registers at meaningful boundaries: memory RF read, operand extract, arithmetic reduction, comparison/quantization, response formatting.
- Use `reduceBalancedTree` for reduction datapaths.
- Keep low-bit AI datapaths multiplier-free when the local design principle expects mux/negate/shift/add logic.
- Keep feature toggles structural and guarded by assertions so invalid parameter combinations fail during elaboration.
- After changing pipeline depth, lane count, RF structure, or arithmetic units, run the Yosys cost flow on the standalone CFU top before assuming the change is cheaper.
