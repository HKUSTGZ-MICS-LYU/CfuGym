name: u8_weighted_average
source_kind: formatted_spec
description: Weighted average over uint8 input and weight vectors.
operation: weighted_average
pattern: vector_mul_reduce
element_type: uint8_t
signedness: unsigned
output_type: uint32_t
vector_bytes: 32
memory_access: sequential
alignment_bytes: 32
cfu_style: tilelink_vector_rf
validation:
  scalar_reference: true
  soc_sim: true
  cycle_profile: true
  yosys_cost: true
unknowns:
  - overflow_policy

