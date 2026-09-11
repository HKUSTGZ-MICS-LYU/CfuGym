---
name: cfu-designer
description: Design, implement, review, optimize, and validate CFU accelerators for VexiiRiscv/CfuGym using SpinalHDL Pipeline/Stage/Plugin/Fiber methodology and Roofline arithmetic-intensity analysis. Use when working on CfuGym or nearby VexiiRiscv projects involving workload hot-loop analysis, memory-/compute-bound classification, custom RISC-V instructions, CfuPlugin/CfuBus, TileLink CFU fibers, MiCo/BitNet/VPU CFUs, vector-to-scalar/vector-to-vector/scalar-on-vector acceleration patterns, Spinal Payload/Node/StageLink pipelines, FiberPlugin elaboration, software intrinsics, SoC parameter wiring, simulation, speedup estimation, Yosys hardware-cost reports, or CFU performance/area optimization loops.
---

# CFU Designer

## Core Workflow

Start from the live repo semantics, not from generic CFU theory.

Use this optimization loop:

1. Analyze the workload: find the hot loop, data layout, trip counts, memory behavior, and scalar reference semantics.
2. Apply the Roofline model before choosing hardware: count operations and bytes at a named memory boundary, compute arithmetic intensity and the ridge point, identify the memory/compute bottleneck, and label every ceiling as measured, parameter-derived, or assumed. Keep missing values explicit in the unknowns.
3. Extract the candidate operation: classify the loop as vector-to-scalar, vector-to-vector, scalar-on-vector, memory transform, or control/state acceleration.
4. Estimate the CFU contract before coding: instruction encoding, operands, state, result, memory access, and software intrinsic/API.
5. Choose the integration style:
   - Use an in-core execution plugin for short, combinational or tightly staged ALU-like instructions.
   - Use `CfuPlugin` plus a direct `CfuBus` spec for stateless register-input/register-output CFUs.
   - Use `CfuPlugin` plus a `CfuBus`/TileLink fiber for stateful, memory-backed, multi-cycle, or reusable accelerators.
6. Design CFU hardware: start with the smallest datapath that addresses the Roofline bottleneck, add pipeline stages or lanes only when the compute ceiling and timing budget justify them.
7. Develop the accelerated kernel: isolate the custom instruction sequence behind intrinsics or target-specific kernels, keeping scalar fallback available.
8. Verify correctness: compare scalar and CFU outputs across edge cases, alignment, sign/overflow, and state reset behavior.
9. Measure speedup and achieved Roofline position: compare cycles for scalar vs CFU paths, account for load/config/fence overhead, and separate kernel speedup from end-to-end speedup.
10. Estimate hardware cost: run a basic Yosys report for the generated CFU RTL and record cells, wire bits, memories, and top cell types.
11. Iterate HW/SW together: revise instruction granularity, vector width, pipeline depth, memory placement, and software blocking until the measured bottleneck moves toward its relevant ceiling without unacceptable hardware cost.

Keep hardware and software synchronized throughout: `func3`/`func7`, raw `rs1`/`rs2` register-id usage, CSR enable/config behavior, vector-register layout, and compile flags must match.

When the CfuGym agent delivers a hardware or software change, prefer targeted edits over whole-file rewrites:

- **Protocol A (preferred): SEARCH/REPLACE edit blocks.** Copy the exact existing lines and give the replacement; each `SEARCH` block must match exactly one location. Keep blocks small but unambiguous. This avoids re-emitting unchanged code, prevents silent rewrites of unrelated behavior, and makes the actual repair visible in the raw output.
- **Protocol B (fallback): whole-file replacement**, only for a genuinely new file or a large restructure that cannot be expressed as a few targeted edits.
- **Protocol C: a unified git diff**, when several coherent locations must change together.

A repair attempt that emits an edit must first give a `REPAIR_DIAGNOSIS` quoting the exact failing log; the edit is then grounded in the observed error, not a wholesale redesign.

## What To Read

Read only the references needed for the current task:

- For CfuGym/VexiiRiscv CFU architecture, file locations, and Vexii parameter/CLI usage, read `references/vexii-cfu-methodology.md`.
- For SpinalHDL `Pipeline`/`Stage`/`Plugin`/`Fiber` usage, read `references/spinal-pipeline-plugin-fiber.md`.
- For commands, benchmarks, simulation, and acceptance checks, read `references/validation-flow.md`.
- For fast hardware-cost estimation with Yosys, read `references/yosys-cost-flow.md`.
- For acceleration-pattern selection and optimization-loop structure, read `references/acceleration-patterns.md`.
- For arithmetic-intensity, bottleneck, bandwidth/compute ceilings, and Roofline-guided CFU choices, read `references/roofline-analysis.md`.
- For a minimal SIMD int8 summing CFU example, read `references/minimal-example-simd8-sum.md`.
- For VPU-style universal vector register files, dedicated CFU memories, datapath-width parameters, and optional compute pipelining, read `references/vector-rf-vpu-pattern.md`.
- For a memory-backed weighted-sum/average experiment with resident product vectors, read `references/u8-weighted-average-experiment.md`.

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
- Choose the shared CFU container style by behavior: use `DirectCfuSpec`/`DirectCfuFiber` for ALU-style CFUs such as SIMD8 sum; use `TilelinkCfuSpec`/`TilelinkCfuFiber` for memory-backed CFUs such as VPU, BitNet, or I8VAdd.
- For a pure direct CFU, keep the command/response width equal to CPU XLEN and prefer the zero-latency `io.bus.rsp.arbitrationFrom(io.bus.cmd)` pattern. Use registered responses only when the CFU is genuinely multi-cycle.
- Treat Vexii CPU parameters and SoC parameters as separate layers. A SoC option that needs a CPU feature must update `p.vexii`, not only instantiate hardware.
- Keep the CPU CFU bus single-owner unless the SoC explicitly adds arbitration. This repo currently legalizes MiCo VPU and BitNet CFU as mutually exclusive users of the single CFU bus.
- For CPU-written data later read by the CFU or DMA/TileLink path, check fence/cache/visibility before changing math.
- Prefer acceleration patterns that amortize instruction overhead. A CFU instruction should usually replace several scalar operations, remove loop-carried work, reduce memory traffic, or expose a better layout.
- Use Roofline arithmetic intensity and the ridge point to justify the bottleneck label. For memory-bound work, prioritize external-byte reduction, reuse, transfer width, and command batching; for compute-bound work, prioritize useful lane throughput, initiation interval, and balanced pipelines. Do not add compute resources without showing how the memory path feeds them.
- Count setup, config, fence, wait, packing, residual, and writeback costs in end-to-end estimates, and distinguish external memory bytes from CFU RF traffic.
- Label measured, parameter-derived, and assumed ceilings separately. Missing bandwidth, frequency, operation counts, or cache behavior belong in `unknowns`, together with the experiment needed to resolve them.
- For multi-operation kernels, look for an intermediate result that can stay resident inside the CFU. Avoid writing vector temporaries back to memory when the next operation is an immediate reduction or transform.
- Use a VPU-style vector RF or dedicated CFU memory when more than one operation reuses loaded vectors, when raw `rs1`/`rs2` should name CFU registers, or when the datapath needs tunable `vlen`/`xlen`/`maclen`/pipeline tradeoffs. These are independent optional knobs: a scalar/direct CFU may leave vector dimensions unused and use `regDepth = 0`; only the concrete implementation should enforce width, alignment, or RF constraints.
- Keep scalar fallback and measurement hooks close to the accelerated kernel until correctness and speedup are stable.
- Do not claim performance or cost from compile success. Use cycle/profile lines, simulator output, Yosys cost reports, or vendor synthesis reports.

## Output Expectations

When designing or changing a CFU, produce:

- The ISA/software contract: opcode space, function IDs, raw instruction fields, CSR/config state, and intrinsic names.
- The hardware structure: component, fiber, SoC param/CLI flags, pipeline stages, state machines, memory interface, and backpressure behavior.
- The Roofline analysis: operation/byte counting convention and boundary, arithmetic intensity, compute and bandwidth ceilings with provenance, ridge-point bottleneck classification, predicted versus achieved throughput, and the design action that should move the point toward the roof.
- The validation plan or results: scalar reference, smoke test, SoC simulation command, benchmark command, cycle/profile evidence, Yosys cost evidence, and any unresolved hardware/toolchain caveats.
