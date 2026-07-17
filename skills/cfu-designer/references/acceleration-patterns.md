# CFU Acceleration Patterns and Optimization Loop

## Optimization Loop

Use this loop for CFU work:

1. **Analyze workload**: identify the hot loop, instruction mix, memory stride, scalar dependencies, vector width, data type, and end-to-end percentage.
2. **Extract operation**: write the scalar operation in a compact reference form and classify the acceleration pattern.
3. **Design ISA contract**: choose custom opcode/function IDs, operand packing, state/config registers, raw field use, and result format.
4. **Design hardware**: implement the smallest datapath, then add vector RF, memory load, pipelining, or state only when overhead analysis justifies it.
5. **Develop kernel**: add intrinsics and replace only the hot loop. Keep scalar fallback for correctness and speed comparison.
6. **Verify**: run deterministic edge cases first, then randomized cases, then target SoC simulation.
7. **Estimate speedup**: measure scalar cycles vs accelerated cycles and account for setup/load/config overhead.
8. **Iterate**: if speedup is weak, change instruction granularity, software blocking, vector width, pipeline depth, memory access pattern, or state reuse.

## Common Patterns

### Vector-to-Scalar

Examples: sum, dot product, min/max, popcount, reduction over packed int8/int4/int2 lanes.

Use when a hot loop reduces many small elements into one scalar. This is usually the best first CFU candidate because one instruction can replace unpacking, sign extension, multiple ALU ops, and a reduction tree.

Design choices:

- Pack multiple lanes into `rs1`/`rs2`, or load wider vectors into CFU-owned registers.
- Return one scalar in `rd`.
- Accumulate inside the CFU only when it reduces instruction overhead without creating hidden state bugs.
- Define overflow behavior explicitly: wrap, saturate, widen, or split partial sums.

### Vector-to-Vector

Examples: element-wise add/sub/xor/compare, clamp, quantize, unpack/pack, activation transforms.

Use when the hot loop maps each packed lane independently and stores another packed vector. This works well when memory traffic is not the only bottleneck and the operation has enough per-lane work to amortize instruction overhead.

Design choices:

- Return packed vector in `rd` for register-sized operations.
- For wider vectors, use CFU vector registers plus load/store commands or TileLink memory access.
- Keep lane ordering and signedness identical to software macros.

### Scalar-on-Vector

Examples: vector scale, add bias, threshold compare, multiply by small constant, shift/round.

Use when one scalar parameter applies to many packed lanes. The scalar can be an operand, a config CSR/raw field, or accelerator state.

Design choices:

- Put frequently reused scalar parameters in CFU config state.
- Use immediate/raw fields for small constants if that avoids an extra register source.
- Decide whether each instruction processes one word or a CFU-owned vector register.

### Stateful Vector Register Operations

Examples: load vector, repeated dot with one resident operand, K/V attention blocks, packed matrix rows.

Use when reusing operands across multiple CFU instructions removes memory traffic or packing overhead.

Design choices:

- Raw `rs1`/`rs2` fields can encode CFU register IDs while `inputs` carry scalar values or addresses.
- Add explicit config/reset instructions.
- Reset cursors and accumulators on config, command completion, or software-visible boundaries.
- Check fences if software writes memory before CFU loads it.
- For reusable designs, separate `vlen`, memory beat width, compute slice width, RF depth, RF backend, and pipeline toggles. Avoid baking one workload's temporary-vector shape into the universal RF.
- If an operation widens lanes, such as `int8 * int8 -> int16`, either reduce immediately, allocate widened storage, split the result across multiple RF entries, or explicitly make the command process fewer lanes.

### Memory Transform or DMA-Like CFU

Examples: quantize a memory block, pack low-bit weights, convert layout, gather/scatter.

Use when the bottleneck is layout conversion or repeated load/pack/store overhead. Prefer TileLink-backed CFU components over in-core plugins.

Design choices:

- Define command granularity and completion response.
- Use stable memory ordering rules and alignment constraints.
- Measure setup overhead separately from sustained throughput.

## Speedup Estimation

Estimate both kernel speedup and end-to-end speedup.

For a hot loop:

```text
scalar_cycles = loop_count * scalar_cycles_per_iter
cfu_cycles = setup_cycles + cfu_issue_cycles + cfu_wait_cycles + residual_scalar_cycles
kernel_speedup = scalar_cycles / cfu_cycles
end_to_end_speedup = 1 / ((1 - hot_fraction) + hot_fraction / kernel_speedup)
```

If measured speedup is below expectation, first check:

- CFU instruction overhead vs work per instruction.
- Memory load/store overhead and fences.
- Whether software still repacks or sign-extends around the CFU.
- Pipeline stalls from `cmd.ready`/`rsp.valid`.
- Missed reuse of resident CFU state.
- Incorrect benchmark isolation or mixed build artifacts.
