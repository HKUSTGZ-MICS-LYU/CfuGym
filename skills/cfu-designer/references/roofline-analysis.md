# Roofline Analysis for CFU Design

Use the Roofline model to connect workload behavior to CFU architecture. It is a
bottleneck model and a design guide, not a substitute for cycle simulation or
synthesis.

## Model

For a kernel, define:

```text
arithmetic_intensity = operations / bytes transferred
attainable_performance <= min(compute_peak, memory_bandwidth_peak * arithmetic_intensity)
```

Use a consistent unit system. For a cycle-level CFU estimate, operations/cycle
and bytes/cycle are usually less ambiguous than FLOP/s and GB/s. If frequency is
known, report the corresponding per-second values as well.

Count the hot loop's actual work and traffic, not only the ideal algorithm:

- Count the operation definition explicitly (for example, one multiply and one
  add per element, or state whether a packed MAC counts as multiple lane ops).
- Count bytes crossing the relevant boundary: DRAM, TileLink, cache, CPU/CFU
  bus, and CFU-owned RF are different boundaries. State which boundary the
  Roofline point uses.
- Include compulsory loads/stores, writebacks, packing/unpacking, scalar
  residual work, configuration instructions, completion waits, and required
  fences when estimating end-to-end behavior.
- Account for reuse separately. Resident CFU vectors reduce external bytes, but
  RF reads and writes may still consume internal ports or cycles.

## Classify the bottleneck

Compare arithmetic intensity with the ridge point:

```text
ridge_point = compute_peak / memory_bandwidth_peak
```

- `arithmetic_intensity < ridge_point`: memory/data-movement bound. More MAC
  lanes alone will not help unless the design also reduces external traffic,
  improves reuse, widens useful transfers, or raises sustained bandwidth.
- `arithmetic_intensity > ridge_point`: compute bound. Prioritize lane count,
  MAC throughput, pipeline initiation interval, and instruction granularity;
  verify that memory and command supply can feed the datapath.
- Near the ridge point: treat it as a balance problem and compare alternatives
  using measured traffic, latency, area, and timing rather than a single label.

Always distinguish the algorithmic intensity from the achieved intensity after
packing, alignment effects, cache behavior, TileLink bursts, and software
fallback work.

## Use it to choose the CFU

The workload analysis must provide, when facts permit:

1. Hot function and loop, trip count, operation count, and external byte count.
2. Arithmetic intensity with the counting convention and boundary stated.
3. Compute and bandwidth ceilings, with each value labeled as measured,
   parameter-derived, or an explicit assumption.
4. Bottleneck classification, ridge-point comparison, and the performance gap
   to the relevant ceiling.
5. The smallest design change that should move the point toward the roof, plus
   the metric that will verify it.

Translate the result into architecture decisions:

- For memory-bound loops, consider CFU-owned vector registers, resident
  intermediates, fewer round trips, larger/burst-aligned transfers, layout
  changes, and command batching. Do not add compute lanes without showing how
  the data path feeds them.
- For compute-bound loops, consider more useful lanes, a shorter initiation
  interval, balanced pipeline stages, local accumulation, and commands that
  replace enough scalar work to amortize issue and response overhead.
- For low-intensity or short loops, include setup/configuration/fence costs and
  consider whether a direct CFU or a software/vectorized baseline is better.
- For stateful or reused data, show the reuse distance and external-byte savings
  that justify RF depth, memory, and TileLink complexity. Reset and visibility
  semantics remain part of correctness.
- Compare at least one alternative when the choice is consequential, such as
  direct versus TileLink, narrow versus wide vectors, or resident versus
  streamed data. Include expected cycles, bandwidth demand, and hardware-cost
  tradeoffs.

## Validation and reporting

A Roofline-based claim must be checked against evidence:

- Compare scalar and CFU cycle counts and separate setup, steady-state, wait,
  and residual cycles where possible.
- Recompute achieved operations/cycle and bytes/cycle from the measured result;
  do not report only a speedup.
- Use the generated RTL for Yosys cost and record cells, memories, timing
  warnings, and the parameter set. A design is not better merely because it
  moves upward on a model if its area or timing is unacceptable.
- If a peak, byte count, cache effect, frequency, or operation count is unknown,
  preserve it as an unknown and state the measurement needed to resolve it.
- Do not claim that a kernel is compute-bound or memory-bound from intuition,
  instruction count, or compile success alone.

A concise report table is useful:

| quantity | value | basis/boundary |
| --- | --- | --- |
| operations | ... | lane/op convention |
| external bytes | ... | memory boundary |
| arithmetic intensity | ... ops/byte | derived |
| compute peak | ... ops/cycle | measured/parameter/assumption |
| bandwidth peak | ... bytes/cycle | measured/parameter/assumption |
| ridge point | ... ops/byte | derived |
| predicted ceiling | ... ops/cycle | Roofline bound |
| achieved result | ... ops/cycle | cycle/profile evidence |
| next design action | ... | metric to verify |
