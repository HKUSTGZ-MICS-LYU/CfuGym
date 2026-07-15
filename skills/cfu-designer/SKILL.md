---
name: cfu-designer
description: Design, implement, review, optimize, and validate CFU accelerators for VexiiRiscv/CfuGym using SpinalHDL Pipeline/Stage/Plugin/Fiber methodology. Use when working on CfuGym or nearby VexiiRiscv projects involving workload hot-loop analysis, custom RISC-V instructions, CfuPlugin/CfuBus, TileLink CFU fibers, MiCo/BitNet/VPU CFUs, vector-to-scalar/vector-to-vector/scalar-on-vector acceleration patterns, Spinal Payload/Node/StageLink pipelines, FiberPlugin elaboration, software intrinsics, SoC parameter wiring, simulation, speedup estimation, Yosys hardware-cost reports, or CFU performance/area optimization loops.
---

# CFU Designer

## Core Workflow

Start from the live repo semantics, not from generic CFU theory.

Use this optimization loop:

1. Analyze the workload: find the hot loop, data layout, trip counts, memory behavior, and scalar reference semantics.
2. Extract the candidate operation: classify the loop as vector-to-scalar, vector-to-vector, scalar-on-vector, memory transform, or control/state acceleration.
3. Estimate the CFU contract before coding: instruction encoding, operands, state, result, memory access, and software intrinsic/API.
4. Choose the integration style:
   - Use an in-core execution plugin for short, combinational or tightly staged ALU-like instructions.
   - Use `CfuPlugin` plus a `CfuBus`/TileLink fiber for stateful, memory-backed, multi-cycle, or reusable accelerators.
5. Design CFU hardware: start with the smallest datapath that removes the bottleneck, add pipeline stages only when timing or throughput requires them.
6. Develop the accelerated kernel: isolate the custom instruction sequence behind intrinsics or target-specific kernels, keeping scalar fallback available.
7. Verify correctness: compare scalar and CFU outputs across edge cases, alignment, sign/overflow, and state reset behavior.
8. Estimate speedup: compare cycles for scalar vs CFU paths, account for load/config overhead, and separate kernel speedup from end-to-end speedup.
9. Estimate hardware cost: run a basic Yosys report for the generated CFU RTL and record cells, wire bits, memories, and top cell types.
10. Iterate HW/SW together: revise instruction granularity, vector width, pipeline depth, memory placement, and software blocking until the measured bottleneck moves without unacceptable hardware cost.

Keep hardware and software synchronized throughout: `func3`/`func7`, raw `rs1`/`rs2` register-id usage, CSR enable/config behavior, vector-register layout, and compile flags must match.

## What To Read

Read only the references needed for the current task:

- For CfuGym/VexiiRiscv CFU architecture, file locations, and Vexii parameter/CLI usage, read `references/vexii-cfu-methodology.md`.
- For SpinalHDL `Pipeline`/`Stage`/`Plugin`/`Fiber` usage, read `references/spinal-pipeline-plugin-fiber.md`.
- For commands, benchmarks, simulation, and acceptance checks, read `references/validation-flow.md`.
- For fast hardware-cost estimation with Yosys, read `references/yosys-cost-flow.md`.
- For acceleration-pattern selection and optimization-loop structure, read `references/acceleration-patterns.md`.
- For a minimal SIMD int8 summing CFU example, read `references/minimal-example-simd8-sum.md`.

Use the starter assets only when creating a new CFU surface:

- `assets/templates/StreamingCfu.scala`: stateful `CfuBus` accelerator skeleton.
- `assets/templates/TilelinkStreamingCfuFiber.scala`: TileLink fiber wrapper skeleton.
- `assets/templates/cfu_intrinsics.h`: custom0 enable/config/load/compute intrinsic skeleton.
- `assets/templates/cfu_smoke_test.c`: scalar-vs-CFU smoke test skeleton.
- `assets/examples/simd8-sum-cfu/`: complete minimal vector-to-scalar example with Scala CFU hardware, a MiCo-free counter header, and C validation/speed-measurement code.

## Design Rules

- Preserve local patterns before introducing abstractions. In CfuGym, mirror `BitNetCfu`, `VpuCfu`, `TilelinkBitNetCfuFiber`, and `MiCoSocParam` unless the user asks for a new architecture.
- Treat `CfuPlugin` as the CPU-side bridge: it decodes custom instructions, freezes the lane while waiting for `cmd.ready`/`rsp.valid`, and writes one integer result at `joinAt`.
- Treat the CFU component as the accelerator owner: it implements command decode, internal state/FSM, optional TileLink memory access, response timing, and status.
- Treat Vexii CPU parameters and SoC parameters as separate layers. A SoC option that needs a CPU feature must update `p.vexii`, not only instantiate hardware.
- Keep the CPU CFU bus single-owner unless the SoC explicitly adds arbitration. This repo currently legalizes MiCo VPU and BitNet CFU as mutually exclusive users of the single CFU bus.
- For CPU-written data later read by the CFU or DMA/TileLink path, check fence/cache/visibility before changing math.
- Prefer acceleration patterns that amortize instruction overhead. A CFU instruction should usually replace several scalar operations, remove loop-carried work, reduce memory traffic, or expose a better layout.
- Keep scalar fallback and measurement hooks close to the accelerated kernel until correctness and speedup are stable.
- Do not claim performance or cost from compile success. Use cycle/profile lines, simulator output, Yosys cost reports, or vendor synthesis reports.

## Output Expectations

When designing or changing a CFU, produce:

- The ISA/software contract: opcode space, function IDs, raw instruction fields, CSR/config state, and intrinsic names.
- The hardware structure: component, fiber, SoC param/CLI flags, pipeline stages, state machines, memory interface, and backpressure behavior.
- The validation plan or results: scalar reference, smoke test, SoC simulation command, benchmark command, and any unresolved hardware/toolchain caveats.
