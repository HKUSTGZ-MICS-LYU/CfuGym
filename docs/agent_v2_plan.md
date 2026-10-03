## Core Concept

- The agent is **free within each stage** and decides how to use the available tools.
- The framework defines **what artifact must be delivered and validated**, rather than prescribing the action sequence.
- Reuse the current run-scoped workspace:
  `agents/generated/runs/<run_id>/`.
- All artifacts are versioned and recorded in `run.yaml` with SHA-256 and dependency information.
- A failed check may trigger repair within the current stage or rewind to an upstream stage.
- Updating an upstream artifact invalidates dependent downstream results.

## Revision Guideline

+ Try your best to utilize LangGraph supported features. Don't re-invent the wheels.
+ Don't be too defensive. Always use `mico_env` as the conda env, we don't need the fallback, we always have the required packages/tools. If we don't have them, we will install them.

### Token Economy

Cache-aware, Lightweight, Indexed, Read-when-requested

- Reuse the existing `profile_cache`.
- Reuse the current `basic / standard / extended` visibility levels.
- Use existing agent tools (`read_file`, `list_dir`, `search_text`) for on-demand context.
- Keep detailed logs/files outside the active context; load them only when requested.
- Preserve tool-call and LLM-output traces for reproducibility.

### Prompt Contract

Every stage prompt is a five-part brief, and nothing else:

1. **Task** — the one-line goal for this call.
2. **Where you are** — the repository root (working directory) and the run directory; all paths are repo-relative.
3. **Tools** — the exact tool names this stage may call, one line of usage each, and the round/call budget.
4. **Deliverable** — the answer shape and the artifact path it is written to.
5. **Key paths + evidence** — a path index of what may be read, plus machine-generated evidence digests (WorkloadSpec, measured profile, observation IDs, generated intrinsic names, optimization score, bounded repair logs).

Rules:

- A stage prompt must not inline repository file bodies while the tool loop is available; the agent reads files with `read_file`/`list_dir`/`search_text`. Listing paths with the framework's own repo-relative names keeps prompt text and tool policy in agreement.
- Evidence the model cannot derive (measured cycles, intrinsic names/signatures, framework verdicts) stays inline as a bounded digest; evidence that already exists as a run artifact (analysis, ISA, plans, logs) is injected by reference.
- The `--no-agent-tools` (or no tool-capable endpoint) fallback may inline the minimum file bodies a delivery needs, because a SEARCH/REPLACE edit must be built from exact bytes. That is a degraded mode, not the default.
- Prompt size is observable: `run.yaml` records `prompt_chars`/`prompt_sha256` per call and flags `prompt_oversize` above 40k chars.
- Every stage that authors an artifact (analysis, ISA contract, conformance oracle, hardware, software, optimization decision) gets the same read-only tool loop, so a stage never has to be handed a file it could read itself.

---

### Stage 1: Profiling + Workload Analysis

**Framework Preprocess**

Reuse the current front-end flow:

1. `load_context`
2. `ingest_input`
3. `route_input`
4. `prepare_benchmark`
5. `profile_embench` / `instrument_and_profile`
6. Export profiling evidence, source mappings, ASM/LLVM IR when available

The current `extract_hotspot` result should be treated as **evidence/hint**, rather than forcing the agent to accelerate only the hottest function.

**Agent Goal**

Analyze workload characteristics and identify promising CFU acceleration candidates.

**Tools Available**

- `read_file / list_dir / search_text`
- Profiling results
- Source / ASM / LLVM IR
- `skills/cfu-designer/**`
- Existing MiCo/VPU/BitNet CFU implementations according to visibility level

**Final Delivery**

`07_analysis/workload_analysis.yaml`

Including:

- Candidate region(s)
- Runtime/profile evidence
- Operation and memory characteristics
- Roofline/bottleneck analysis
- Dependencies and side effects
- Acceleration rationale
- Assumptions / unknowns

**Stage Gate**

- Schema / evidence completeness check
- No requirement that the framework agrees with the agent's architectural choice

---

### Stage 2: CFU ISA Contract

**Agent Goal**

Design the CFU ISA and define the HW/SW contract before implementation.

**Tools Available**

- All Stage 1 read tools
- `agents/specs/cfu_isa_template.yaml`
- `CfuBus / CfuPlugin / AgentCfu` interfaces
- Existing CFU examples and CFU Designer references
- Contract validation

**Final Delivery**

`08_isa/cfu_isa.yaml`

Including the existing CfuGym fields:

- `identity`
- `commands`
- `operations_contract`
- `state`
- `memory`
- `datapath`
- `software`
- `integration`
- `validation`
- `roofline`
- `assumptions / unknowns`

**Stage Gate**

Reuse `isa.py`:

- ISA schema/lint check
- Encoding and `(function_id, subfunction_id)` consistency
- Software ABI / intrinsic consistency
- Integration requirements
- Basic feasibility against Stage 1 analysis

**Optional Exit**

`ABSTAIN` if no meaningful CFU opportunity is identified.

---

### Stage 3: HW/SW Implementation + Verification

**Agent Goal**

Implement the accepted ISA contract, integrate it into MiCoSoc, and produce a correct accelerated workload.

**Hardware Workspace**

Continue using the isolated design overlay:

```text
agents/generated/runs/<run_id>/workspace/design/
├── AgentCfu.scala
└── AgentCfuFiber.scala
```

The tracked `src/main/scala/**` platform remains read-only.

**Tools Available**

- Existing read/search tools
- Targeted edit / whole-file / patch delivery protocols
- Scala/SpinalHDL build
- Software build
- MiCoSoc simulation
- Git diff / workspace recovery
- Existing validation feedback

**Implementation Loop**

```text
implement_hw
   ↓
hw_build / SoC generation
   ↓
hw_conformance
   ↓
implement_sw
   ↓
sw_build
   ↓
simulation
```

The agent is free to inspect, edit, compile, simulate, and repair repeatedly inside this stage.

**Development Testbench**

The agent may additionally generate C tests for self-debugging.

These are **development evidence only** and do not replace the existing framework-owned `hw_conformance` or workload correctness oracle.

**Final Delivery**

1. `workspace/design/AgentCfu*.scala`
2. Generated `cfu_intrinsics.h`
3. CFU-accelerated workload source
4. Agent-generated development tests, if used
5. Existing `09_implementation/*` plans, patches, tool traces, and summaries

**Stage Gate**

Reuse the current validation ladder:

- `hw_build`
- `hw_conformance`
- `sw_build`
- workload simulation
- independent workload correctness oracle

Only a fully verified design may enter Stage 4.

---

### Stage 4: Evaluation + Optimization

**Framework Evaluation**

Reuse the existing measurement flow:

1. Simulation provides:
   - Reference cycles
   - CFU cycles
   - Speedup
2. `GenAgentCfu` generates standalone CFU RTL.
3. Yosys evaluates current hardware cost:
   - Cells
   - Wires
   - Memories

Future EDA extensions may add:

- Fmax
- Power
- Energy

**Agent Goal**

Evaluate the current HW/SW solution and decide whether another optimization round is worthwhile.

**Final Delivery**

Reuse:

```text
11_cost/result.yaml
12_review/optimization_rounds.yaml
12_review/optimization_decision.yaml
```

**Optimization Decision**

```text
ACCEPT

or

CONTINUE:
    target = implement_hw
           | implement_sw
           | extend_isa
    focus  = performance
           | cost
```

The current ISA remains frozen during implementation optimization; `extend_isa` is the controlled path for compatible ISA evolution.

Every modified candidate must traverse the normal Stage 3 validation ladder again before its QoR is accepted.

---

### Rewind

Current supported rewind paths can map naturally to the stage abstraction:

```text
Stage 4 → Stage 3
    implement_hw / implement_sw

Stage 4 → Stage 2
    extend_isa

Stage 3 → Stage 2
    contract/ISA defect discovered
```

A future extension may allow:

```text
Stage 4 → Stage 1
```

when measurements show that the selected kernel itself was a poor acceleration target.

Rewinds should carry structured evidence and invalidate downstream measurements derived from the modified artifact.

---

### Where to End?

Reuse the existing bounded exploration mechanisms:

1. `--max-iters`
   - Shared repair + optimization budget
2. `--max-evals`
    - Stop after generating N eval reports
3. `--loop-patience`
   - Stop after repeated non-improving candidates
4. Agent tool/token budgets
5. Per-command simulation/build timeouts

Add a global wall-time budget if needed for experiments.

Terminate on:

- **ACCEPT** — verified design selected
- **ABSTAIN** — no worthwhile CFU design
- **BUDGET_EXHAUSTED**
- **NO_PROGRESS**
- **UNRECOVERABLE_FAILURE**

The final result should point to the **best fully verified candidate**, while preserving all explored candidates and their measurements in the run artifacts.