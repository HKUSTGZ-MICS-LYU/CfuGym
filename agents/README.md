# CfuGym Agents

This folder contains local agent frameworks for CfuGym workflows.

## Code layout

`cfu_design_agent/` separates three responsibilities:

- `benchmark/`: workload adapters. `generic.py` prepares generated/C-project
  benchmarks; `embench.py`, `embench_instrument.py`, and `embench_loops.py`
  handle Embench workspace, observation points, and source loop discovery.
  New benchmark suites should add adapters here rather than extend the flow
  runner with benchmark-specific workspace mechanics.
- `flow/`: the shared node/edge specification and traversal for LangGraph and
  the pure-Python runner (`wiring.py`). `graph.py` currently retains node
  implementations and public `build_graph`/`run_agent` entry points, because
  existing callers patch module-level node dependencies there. Move node
  implementations incrementally only with explicit dependency injection.
- `runtime/`: low-level execution services—policy, repository tools, artifacts,
  read-only agent tools, LLM transport/config, and progress tracking. These
  enforce access and run isolation independently of benchmark selection.

The short-lived old-path alias scripts were removed; import adapters via
`cfu_design_agent.benchmark.*` and execution services via
`cfu_design_agent.runtime.*`. Imports such as `cfu_design_agent.embench` or
`cfu_design_agent.tools` are no longer supported. The CLI entry point
`agents/run_cfu_agent.py` is unchanged. The Embench reference/profile cache
remains in `cfu_design_agent/profile_cache.py` and hashes the relocated
benchmark implementation.

## Hardware design boundary

`src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala` with `AgentCfuFiber.scala` is the
hardware starting point for new designs. It provides a stable CfuBus contract, an
algorithm-neutral Vector RegFile, optional TileLink load/store channels, and MiCoSoc CLI
integration.

These two tracked files are **read-only defaults**. Every run seeds an isolated workspace copy:

```text
agents/generated/runs/<run_id>/workspace/design/AgentCfu.scala
agents/generated/runs/<run_id>/workspace/design/AgentCfuFiber.scala
```

An agent may write **only** those two files for hardware; the access policy rejects any write
to `src/main/scala/**`. A patch that names the tracked scaffold path is redirected into the run
workspace automatically, so the repository tree is never modified by a design run.

The implementation rungs deliver changes with three protocols, in this order of preference:

1. **Targeted edits (Protocol A, preferred).** A `SEARCH/REPLACE` block copies the exact existing
   lines and gives the replacement. It needs no line numbers and exposes only the real change, so
   the model cannot silently rewrite unrelated code. Each `SEARCH` block must match exactly one
   location; zero or multiple matches reject the whole response with the exact reason fed back
   into the next repair prompt. All targets in a response are validated before any file is written,
   so a stale block cannot leave the workspace half-edited. Multiple `EDIT_BEGIN` sections may
   target the same file; they are applied in order to the evolving content, so a later SEARCH
   must match the file after the earlier sections. Prefer one section per file with several
   SEARCH/REPLACE pairs, and never emit superseded drafts or a second attempt at the same region.
2. **Whole-file replacement (Protocol B, fallback).** The complete new content of a file, used only
   for a genuinely new file or a large restructure.
3. **Unified diff (Protocol C).** A `PATCH_BEGIN ... PATCH_END` diff applied with `git apply`; a
   malformed hunk header rejects the whole patch.

Every protocol routes through the same path redirect and write policy (`agent_access_policy.yaml`
and `RepoToolbox`), so the repository tree is never modified by a design run. Prompts show, and
edits apply to, the **live run-workspace copy**; an applied attempt is not silently reset to the
tracked original, so repairs compose on top of earlier changes. Inline model reasoning
(`<thinking>` tags, `` blocks, or a provider reasoning field) is split out of the
answer before parsing and persisted beside each raw output as `*_<n>.reasoning.txt`.

The SoC build then compiles the workspace copy *instead of* the tracked scaffold, using a
same-package source overlay:

```bash
sbt 'set Compile/unmanagedSourceDirectories += (Compile/baseDirectory).value / "agents/generated/runs/<run_id>/workspace/design"' \
    'set Compile/unmanagedSources := AgentCfuSourceFilter.sources((Compile/unmanagedSourceDirectories).value)' \
    'set run / baseDirectory := (Compile/baseDirectory).value / "agents/generated/runs/<run_id>/workspace/soc"' \
    'runMain vexiiriscv.soc.mico.MiCoSocGen --with-rvc --with-rvm --with-rdtime --mico-agent-cfu'
```

- `project/AgentCfuSourceFilter.scala` drops the two tracked scaffolds from the resolved source
  list; Scala cannot hold two definitions of the same class in one package, so this swap is what
  lets an agent design reach the hardware without touching the repo.
- `run / baseDirectory` keeps `MiCoSoc.v` and `soc.h` inside the run workspace instead of the
  repository root.
- Run this from an environment that provides `sbt`, e.g. `conda run -n mico_env ...`.

Preserve the public contract (`AgentCfuParameter`, `class AgentCfu`, the `AgentCfuFiber`
helpers, `AgentCfuFunction` ids) and the single CPU CFU owner rule; otherwise the SoC
integration breaks.

## CFU Design Agent

The CFU design agent turns a workload request into a run-scoped, auditable CFU design loop. It runs through the following stages, each of which writes its own artifact under the run directory:

`load_context → ingest_input → route_input → prepare_benchmark → profile → hotspot → analysis → isa → implementation_hw → hw_build → implementation_sw → sw_build → sim → cost → review → final_report`

The `analysis` stage must include a Roofline section in `07_analysis/workload_analysis.yaml`: named memory boundary, operation and byte counts, arithmetic intensity, compute/bandwidth ceilings with provenance, ridge point, bottleneck, predicted versus achieved metric, design action, and unresolved unknowns. The ISA and implementation prompts carry this conclusion forward so vector width, RF residency, memory style, pipeline depth, instruction granularity, and software blocking address the modeled bottleneck.

### Fail-fast validation ladder

The design is checked by the tool that owns each artifact **immediately after it is produced**, instead of running every stage and deciding once at the end. Each rung either continues, repairs the artifact that caused the failure, or stops the run:

| Rung | Check | Gate | Repair target |
| --- | --- | --- | --- |
| `prepare_benchmark` | workload workspace copy + instrumentation setup (policy reads must succeed) | `prepare_benchmark` | — (input/policy failure stops the run) |
| `isa` | CFU ISA/integration contract lints | `isa` | `design_contract` |
| `implementation_hw` | writes `workspace/design/AgentCfu*.scala` only | `implementation_hw` | — |
| `hw_build` | `sbt … MiCoSocGen --mico-agent-cfu` (Scala compile + SpinalHDL elaboration + Verilog) | `hw_build` | `implement_hw` |
| `hw_conformance` | harness-generated differential micro-test of every declared instruction (`make … cfu_conformance` + `MiCoSocSim --load-elf`); must print `CFU_CONFORMANCE PASS commands=n/n …` | `hw_conformance` | `implement_hw` |
| `implementation_sw` | writes the accelerated kernel + `cfu_intrinsics.h`; the CFU build must call a CFU datapath intrinsic (not `cfu_enable()` + scalar). The rung may be **reused** instead of re-run when the kernel already in the workspace still drives this exact ISA contract | `implementation_sw` | — |
| `sw_build` | `make … compile` | `sw_build` | `implement_sw` |
| `sim` | `sbt … MiCoSocSim --load-elf …` plus the correctness oracle; an embench CFU build must also show >= 1.05x reference cycles (`MIN_EMBENCH_SPEEDUP_X100` in `graph.py`) | `sim` | `implement_sw` or `implement_hw`, chosen by the agent |
| `cost` | `sbt … GenAgentCfu` elaborates the CFU alone, then Yosys cost report from that standalone RTL | `cost` | `implement_hw` |

- `--max-iters N` is the **total repair budget**: `N - 1` repair rounds for the whole run. `--max-iters 1` allows no repair, so the first failing rung stops the run immediately with `status=failed` (CLI exit code 1) and skips every later rung — no compile, simulation, or Yosys run happens on a design that already failed to generate.
- A repair re-runs the responsible node with the exact tool output (Scala compiler / elaboration errors, `make` errors, or simulator output) embedded in its prompt. An implementation rung that produced no usable patch is likewise re-prompted with its rejection reason.
- Repair feedback is labeled by source and all available sources are injected together: a hardware repair sees `hw_build`, `sim`, `cost`, and `implementation_hw` errors, and a software repair sees `sw_build`, `sim`, and `implementation_sw` errors. A repair prompt enters an explicit `*_REPAIR MODE` and must emit a `REPAIR_DIAGNOSIS` (observed errors with file:line, root cause, repair mapping, regression risks) before any file or patch output, so a retry is grounded in the log instead of being an unrelated rewrite.
- The ISA contract repair carries the previous linter problems, so a rejected contract is corrected rather than regenerated blind.
- The exact injected feedback is persisted as `09_implementation/hw_repair_feedback.txt` / `09_implementation/sw_repair_feedback.txt` for every repair attempt, so a run is auditable against the LLM raw output.
- A rejected SEARCH block is reported with its exact location: the error names the SEARCH line and the current file line that differ, quotes both lines, and says whether an earlier pair already changed the region. Previously an unlocalized "SEARCH block not found" plus a parenthetical about earlier pairs led the model to misdiagnose the file as stale and resend the same edit.
- Both implementation rungs detect a stuck retry: each attempt records a `(response_hash, error_hash)` (in `sw_attempts` / `hw_attempts` and the gate). When two consecutive attempts are byte-identical and failed identically, the next repair prompt carries a `REPEATED IDENTICAL ATTEMPT` directive to switch strategy (Protocol B whole-file, or a minimal SEARCH), instead of burning the remaining budget on the same rejected answer. (The hardware rung previously lacked this guard; a model that emitted one broken answer three times exhausted the whole repair budget on it.)
- Delivery parsing tolerates a terminator glued to the next token: a degenerate sampler occasionally restarts its answer template inside the same completion, so the response arrives twice with the second copy beginning at `EDIT_ENDREPAIR_DIAGNOSIS` instead of a newline. The parser splits that glue (only outside `identifier` continuations like `EDIT_END_foo`) and collapses byte-identical duplicate sections to the first copy, so a correct edit delivered in a doubled answer is applied once instead of rejected as "missing EDIT_END".
- Prompts render each rung's own target files exactly once: they are shown in full under "Exact current contents" and omitted from the truncated repo-context digest, so a truncated copy can never be mistaken for the text to copy. `DELIVERY_PROTOCOLS` also tells the model not to wrap long comment blocks in a SEARCH and to copy any included comment byte-for-byte.
- Before answering, the analysis and hardware rungs may run a **bounded tool loop** (`--agent-tools`, on by default; `--no-agent-tools` disables it). The budget is `--max-tool-rounds N` (default 6, model turns that may use tools) plus `--max-tool-calls N` (default 24, cumulative calls across all rounds; `0` disables the cap) and at most 4 parallel calls per round. Tools are read-only and policy-checked: `read_file`, `list_dir`, `search_text`. Every call is printed in the CLI log (`[tool] read_file(...) ok -> N chars`) and written to `09_implementation/hw_tool_calls.yaml` (hardware) or `07_analysis/analysis_tool_calls.yaml` (analysis), so the agent's reading is auditable; the effective budget is recorded as `tool_policy` in `run.yaml`. If the endpoint rejects the tools payload, the step degrades to a one-shot prompt instead of failing.
- When the budget is exhausted the tools are withdrawn and a reminder is injected that the next reply must be the final EDIT/FILE/PATCH deliverable. That reply is then parsed as the answer, and an answer with no delivery block fails `implementation_hw` with an explicit `no EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN deliverable` error (which the repair prompt carries) instead of failing silently and being misreported as a build failure downstream.
- If the endpoint refuses the tools payload the request is retried once; if it still fails the step degrades to a plain completion that explicitly tells the model tools are unavailable, so it does not answer with an unexecutable "Let me read …" plan. The degradation is recorded (`degraded_reason` on the raw-output metadata, a `(tools unavailable)` entry in the tool-calls artifact, `tools_degraded` on the gate). Separately, any answer that only announces reading a file is detected and reported as a `tool-intent preamble` error, which the repair prompt carries, instead of being accepted as a finished design.
- While tool budget remains, a reply that cannot be used gets **one corrective re-ask** before the step is finalized: either the model announced a read without calling a tool, or a rung that requires a delivery block returned none (plan-only answer). The re-ask states the requirement and keeps tools available, so a dropped or forgotten tool call does not burn a ladder repair.
- The stages that own a workload also allowlist its source (`benchmarks/embench-iot/{src,support}/**`), so a file shown in the prompt context is never denied to the tool loop. The analysis rung additionally reads the profile build's disassembly (`workspace/**/*.asm`), so the hot loop can be analyzed from real instructions rather than inferred from C.
- Hardware-target reads are redirected to the run-workspace overlay: `read_file`/`search_text` on `src/main/scala/vexiiriscv/soc/mico/AgentCfu*.scala` return the run's live copy (the same bytes the prompt shows and an edit targets), so a repair sees its own previous change instead of the pristine tracked scaffold. The redirect only applies where the stage may read the overlay, so the analysis rung still reads the tracked scaffold.
- The build rungs are guarded by the implementation rung that feeds them: if `implementation_hw` applied nothing, `hw_build` fails without invoking `sbt` (the seeded scaffold would otherwise compile and "pass" while validating nothing) and repairs `implement_hw`; if `implementation_sw` applied nothing, `sw_build` fails without invoking `make` and repairs `implement_sw`.
- A failed `sim` is routed by the agent, not by a fixed order: one extra LLM call (stage `route_repair`) receives the failure attribution, the measured reference/CFU cycles, the ISA contract, both implementation summaries, and the simulation-log tail, and returns `{"repair_target": "implement_sw"|"implement_hw", "root_cause", "reason"}`. Only those two targets are accepted; anything else (or no model at all) falls back to the deterministic escalation, which is keyed to simulation repairs only: the first `sim` failure retries software, the next escalates to hardware. An unrelated earlier repair (for example a `sw_build` retry) does not skip the software retry. `--repair-routing fixed` (env `CFU_AGENT_REPAIR_ROUTING`) pins the old deterministic behavior and never calls the router. The decision is written to `10_validation/sim_repair_decision.yaml` (`target`, `root_cause`, `reason`, `source` = `llm`/`fallback`/`fixed`, and the deterministic `advisory`), recorded as `repair_decision` on the `sim` gate, added to the iteration history, and the router's raw answer is kept as `10_validation/llm_raw/repair_route_*.txt`. The named root cause is also injected into the chosen rung's repair feedback as a routing-decision line, so the retry works on the identified defect instead of re-deriving it from the log.
- After a successful `hw_build` the ladder does not have to re-write the software. The agent may decide that the kernel already in the run workspace still matches the new hardware, in which case the `implementation_sw` rung is **reused** and the run goes straight to `sw_build` → `simulate` with the existing kernel and the freshly generated `cfu_intrinsics.h`. The decision has a deterministic honesty guard: it is only taken when the current kernel bytes call a CFU datapath intrinsic **and** the ISA contract digest recorded when that kernel was written still equals the current one — a moved ABI (renamed command, new operand layout) forces the software rung. Only real datapath calls count: `cfu_enable()`, `cfu_memory_fence()` (enable/fence only) and the generated harness's `cfu_agent_counter_read()`/`cfu_agent_counter_elapsed()` cycle helpers do not; those helper names are now reserved by the ISA linter. Declared control-only operations do not count as datapath usage either. Only when the guard passes is one LLM call (stage `route_software`) made, with the ISA contract, both design summaries and the exact intrinsic call sites, to answer `{"software_change": "required"|"not_required", "reason"}`. Any failure (no model, unusable answer, or `--software-change always`, env `CFU_AGENT_SOFTWARE_CHANGE`) runs the software rung as before. The decision is written to `09_implementation/software_decision.yaml` (`software_change`, `reason`, `source` = `llm`/`guard`/`fallback`/`fixed`, `evidence`, `files`, `calls`), recorded as `software_decision` on the `hw_build` gate and as `{"ok": true, "reused": true, ...}` on the `implementation_sw` gate, added to the iteration history, and the router's raw answer is kept as `09_implementation/llm_raw/software_route_*.txt`. Reuse is never a way to skip validation: `sw_build` still compiles the kernel and `sim` still has to pass the correctness oracle and the speedup floor.
- Before any workload software exists, the `hw_conformance` rung runs the **smallest possible program** that exercises each declared instruction and compares it against an independent scalar reference, so a contract/encoding/writeback mistake is found before a kernel is written on top of it. The C test is **harness-owned** (`conformance.render_conformance`): the model supplies only an oracle YAML, so a hardware repair can never weaken the test. A command that writes CPU `rd` is compared case-by-case against an independent scalar `reference` expression; a command with no CPU result (a vector write or an acknowledgement) cannot be read back generically, so the oracle must supply a `verify` C expression instead — that is **weaker evidence** (the model attesting its own hardware) and is recorded as such. A command that uses the vector register file must supply `setup`; the harness never guesses how to seed it. A missing `reference`/`verify`/`setup`/`cases`, an unknown command name, or an argument-count mismatch is a **coverage failure**, and a missing `CFU_CONFORMANCE` line is a failure too: silence is never evidence. The rung is purely functional — it measures no cycles and asserts no performance; correctness is proved later by the `sim` rung and the workload's own oracle. A failure repairs `implement_hw`. The generated C, the oracle and the verdict are kept under `09_implementation/conformance/`.
- Oracle `cases[i].args` values are compile-time constants in the generated C: the validator accepts only integer literals (exactly one per CPU-value role, rs1/rs2 order), and both the validator message and the prompt tell the model to move address/data preparation into `setup` (e.g. seed vectors there and give `args: [0]`). The oracle's C may call only the **generated intrinsics named in the prompt's command table** (plus `cfu_enable`/`cfu_memory_fence`); an invented name such as `cfu_load` is caught by a static lint before the build (`unknown intrinsic cfu_load (did you mean cfu_vector_load_i32?)`) instead of surfacing as a bare linker error. Conformance build/sim failures embed the compiler/linker diagnostics in the repair feedback (previously only the `make` command line reached the model, so it repaired blind), and a **recurring** rejection — validator failure or build failure — is flagged as a `REPEATED ORACLE REJECTION` with the correct shape pinned. `run.yaml` merges artifact refs across writer instances, so every oracle attempt's raw answer stays registered and numbered (`conformance_000/001/...`) instead of being dropped and overwritten.
- With `--loop` a run does not have to stop when it succeeds (see **Multi-instruction ISA evolution** below for the current exploration and delivery rules). If every gate passes **and** the `--max-iters` budget still has room, the review node asks the agent (stage `optimize`, plus at most one stop reconsideration) whether another round is worth spending on: `{"continue": true|false, "target": "implement_hw"|"implement_sw"|"extend_isa", "focus": "performance"|"cost", "reason": ...}`. A round re-enters the normal ladder (`implement_hw` → `generate_soc` → software reuse-or-rewrite → `build_sw` → `simulate` → `estimate_cost` → review, or `implement_sw` → `build_sw` → …), so it can never skip verify, the 1.05x speedup floor, or a build rung. An optimization round is budgeted like a repair: `--max-iters N` stays the single bound on the whole run, and a round with no budget left, an unusable target, or a missing "concrete change" reason stops the loop deterministically; in default explore mode a `continue: false` is reconsidered once before the framework accepts it (no model answer can extend a run). The chosen rung receives an `=== OPTIMIZATION ROUND ===` block (not a `*_REPAIR MODE` block) with the current measured result, the focus, and the rule that the accepted ISA revision is frozen during implementation; only `extend_isa` may append compatible commands before re-entering hardware and software validation. Every round that produced its own measurement is recorded in `12_review/optimization_rounds.yaml` (speedup, reference/CFU cycles, Yosys cells/wires/memories), and `simulate`/`estimate_cost` stamp their gates with the round they ran in, so a round that failed before measuring is listed as `not measured` instead of repeating the previous round's numbers. The report's `## Optimization Rounds` section and the end-of-run CLI `summary:` each print one line per round with the change against the previous measured round — for example `round 1: implement_hw speedup=2.50x cells=1300 (+0.50x, +100 cells) (best)` — so the whole progression of speedup and cost is traceable; the first round is the initial ladder. A round counts as an improvement only when it is Pareto-better than the best round so far (not slower, not larger, better on at least one axis); a final measured round that regresses is reported with an explicit warning in the report and in the CLI summary. `--loop` is off by default (env `CFU_AGENT_LOOP`); without it, review stops as before.
- The cost rung synthesizes the CFU **in isolation** rather than the whole SoC: `GenAgentCfu` (a tracked, non-agent-editable entry point) elaborates only the overlaid `AgentCfu` class into `11_cost/cfu_rtl/AgentCfu.v`, and Yosys runs `synth -top AgentCfu` on it. Cost numbers therefore track the agent's CFU design instead of the CPU/SoC, and the cost rung needs no CPU build. If the standalone generation produces no RTL, the cost rung fails and repairs `implement_hw`.
- The simulation rung also fails when no design change was applied, so a benchmark whose cfu build is byte-identical to the reference cannot be reported as an accelerated result.
- A rung that was skipped (`--dry-run`, `--skip-sim`, execution disabled) never repairs and can never produce `complete`.
- A hanging simulator (`TimeoutExpired`) or a missing toolchain is a failed rung, not a crashed run.

```bash
# Planning pass (read-only): writes run artifacts, never mutates source or runs commands.
python3 agents/run_cfu_agent.py \
  --task "Accelerate a uint8 weighted average kernel with a TileLink CFU. It reads x[i] and w[i], computes sum(x[i] * w[i]) / N, arrays are 32-byte aligned, and products should stay resident inside the CFU." \
  --skip-sim --skip-yosys
```

By default the agent is **read-only** (`--dry-run` semantics). To actually apply an LLM patch and/or run allowlisted build/simulator/Yosys commands, opt in explicitly:

```bash
# Allow the agent to apply a patch to CFU/software allowlisted paths only.
python3 agents/run_cfu_agent.py --task "..." --apply-patch

# Allow allowlisted commands (mioc-gen/sim, make TARGET=vexii_soc, yosys cost) to run.
python3 agents/run_cfu_agent.py --task "..." --run-commands

# Force read-only even if other flags might imply mutation.
python3 agents/run_cfu_agent.py --task "..." --dry-run

# Restrict what the agent may read (default: standard).
python3 agents/run_cfu_agent.py --task "..." --visibility basic
```

### Visibility levels

`--visibility {basic,standard,extended}` (env `CFU_AGENT_VISIBILITY`, default `standard`)
controls **which files the agent may read** — the prompt context, the read-only tool loop
(`read_file`/`list_dir`/`search_text`), and any stage read. It is a read-only gate: writes,
command allowlists, `--apply-patch`, and `--dry-run` are unchanged, and the write confinement
to `workspace/design/AgentCfu*.scala` still applies at every level.

| Level | Adds | Still hidden |
| --- | --- | --- |
| `basic` | workload/benchmark sources, the CFU contract (`CfuBus`/`CfuPlugin`/`CfuLib`), `AgentCfu*`/`GenAgentCfu`, `MiCoSoc*`, the ISA template, `sw/tests`, run artifacts | `skills/cfu-designer/**` and sibling CFU designs (Vpu/BitNet) |
| `standard` (default) | the cfu-designer skills and the sibling CFU designs | `ext/SpinalHDL/**` |
| `extended` | `ext/SpinalHDL/**` in the `isa` and `implementation` stages, with prompt guidance to confirm the SpinalHDL API before writing hardware | — |

- The gate is declarative: `agents/specs/agent_access_policy.yaml` has a `visibility` section
  whose `gated` globs list the paths that require `standard` or `extended`. A path a stage does
  not allowlist stays unreadable regardless of level.
- Hiding a context file is expected at a lower level, so `load_context` records it under
  `withheld_files` in `01_context/manifest.yaml` (and does **not** fail the gate); only a real
  policy denial lands in `denied_files` and fails it. The chosen level is written to the
  manifest, `run.yaml`, the CLI summary (`visibility=`), and the report's `## Visibility`
  section.
- A read blocked by the tier returns an actionable policy error
  (`requires visibility level 'standard' (current: basic)`), and the level-specific prompt
  block tells the model not to assume the contents of a document it was not shown.
- `extended` never pre-reads `ext/SpinalHDL` (925 sources); it only makes the tree readable so
  the agent can look up the exact API on demand. With `--no-agent-tools` the tree is
  unreachable, and the prompt says so.

### AgentCfu design file vs. read-only platform

The scaffold is split so an agent only reads and rewrites the design it owns:

- `src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala` — **agent-owned design file.**
  It keeps the public contract (`AgentCfuParameter`, `AgentCfuFunction`,
  `class AgentCfu(cfuParam, busParam, p)`) and the
  `override protected def instructionTable: Seq[Instruction]` design. Nothing else.
- `src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala` — **read-only platform.**
  `AgentCfuBase` (the bus shell, vector RF, response buffer, CfuLsu/FSM and the
  instruction dispatcher) plus the `CfuResource` vocabulary.
- `src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala` — agent-owned, changed
  only when the fiber contract genuinely must change.

Consequences:

- A whole-file rewrite of `AgentCfu.scala` (the `FILE_BEGIN` protocol) cannot delete
  the bus shell or the dispatcher, because they are no longer in that file.
- `AgentCfuLib.scala` is not a write target: it is absent from `DESIGN_SOURCE_FILES`,
  the run-workspace overlay, and the implementation write policy. The overlay's
  `AgentCfuSourceFilter` removes only the two tracked design files, so the platform
  is always compiled from the repository copy.
- The platform is not pre-read into the hardware prompt. The design file shows the
  working API and the prompt lists it; the agent may read
  `AgentCfuLib.scala` on demand for an exact signature.
- `CfuResource` lives in the `AgentCfuBase` companion object, so an older run
  workspace that still inlines a top-level `CfuResource` keeps compiling against the
  current repository platform.

The dispatcher is built during the platform constructor. It has to be: the CfuLsu
state machine consumes its registered `whenIsActive` tasks exactly once, at prePop
time, and `fsm.goto` is only legal while that build runs. Scala assigns
primary-constructor parameter fields before the superclass body executes, so
`AgentCfu.instructionTable` may read its own constructor parameters (`p`,
`cfuParam`) and the platform members. It must not read a `val` declared in the
subclass body, which is assigned only after the platform has called the table;
allocate design state inside the table body instead.

### Declarative CFU instruction table

The scaffold borrows VexiiRiscv's **declare encodings, register implementations,
generate common decode** pattern, not its CPU scheduling machinery:

- `src/main/scala/vexiiriscv/riscv/Rvi.scala` declares `TypeR(M"...")` encodings.
- `execute/IntAluPlugin.scala` registers operations with `add(...).srcs(...).decode(...)`.
- `execute/ExecutionUnitElementSimple.scala` records the operation and decode payloads;
  CPU decode/execute plugins generate the shared controls.
- `execute/cfu/CfuPlugin.scala` already passes function_id, raw instruction, operands
  and request ID over the standard interface. None of these interfaces is modified.

`AgentCfu.scala` now overrides `protected def instructionTable: Seq[Instruction]`,
declared in the platform `AgentCfuBase`. An entry declares name, function, subop,
optional Scala `enabled`, combinational `ready`, `start`, optional multicycle `tick`,
and `resources`. The dispatcher is the single source
for decode, supported instructions and handshake. Adding/removing an entry does
not require updating separate `isCompute`, `isSupported` or central execution branches.
The default table contains the legacy compute/config no-ops and enabled load/store
handlers; replace no-ops with actual operations for an accelerated design.

#### Explicit operand and writeback resources

Like Vexii's `MicroOp.resources` / `RfResource(rf, access)`, an instruction now
carries `resources: Option[Seq[CfuResource]]`. `CfuResource` is defined in
`AgentCfu.scala` itself so the existing two-file run overlay remains sufficient.
It is an internal hardware declaration, not an extension to CfuBus.

```scala
import CfuResource._

Instruction("add", function = 1, subop = 1,
  resources = Some(Seq(CpuRead(RS1), CpuRead(RS2), CpuWriteRD)),
  start = ctx => ctx.complete(
    (ctx.read(RS1).asUInt + ctx.read(RS2).asUInt).asBits))

Instruction("vector_xor_inplace", function = 1, subop = 2,
  resources = Some(Seq(
    VectorRead(VS1, CpuValue(RS1)),
    VectorRead(VS2, CpuValue(RS2)),
    VectorWrite(VD, CpuValue(RS1)))),
  start = ctx => ctx.complete(ctx.read(VS1) ^ ctx.read(VS2)))
```

The handler computes a result; the resource declaration determines where that
result goes. `ctx.complete` does not guess from result width. `ctx.complete()`
is for acknowledgement-only commands. `ctx.fail()` completes with an error and no
vector write; `ctx.complete(result, status)` suppresses vector writeback when status
is nonzero. The unchanged CPU plugin still writes response data to a nonzero rd:
fail/invalid-index responses contain zero, not a promise to preserve the old CPU RD.
The same limitation applies to CPU-result error completions; status is not a CPU
writeback-cancel signal. First-version resources allow **at most
one data write**: CPU RD or vector VD, not an ambiguous simultaneous write.

| Resource | Value / effect in generated custom0 instructions |
|---|---|
| `CpuRead(RS1)` | CPU `X[rs1]` value, not the number of rs1 |
| `CpuRead(RS2)` | CPU `X[rs2]` value, not the number of rs2 |
| `CpuWriteRD` | result through CFU response; CPU writes this instruction's rd |
| `VectorRead(VS1, CpuValue(RS1))` | CFU `V[X[rs1]]`, vector width |
| `VectorRead(VS2, Constant(1))` | CFU `V[1]`, vector width |
| `VectorWrite(VD, CpuValue(RS1))` | write result to CFU `V[X[rs1]]` |
| `LegacyRegisterNumber(RS2)` index | instruction[24:20], legacy index-width truncation; **not** `X[rs2]` |

VS1/VS2/VD are roles, not additional bus fields. Vector index sources may be
`CpuValue(RS1/RS2)`, `Constant(n)`, or compatibility-only
`LegacyRegisterNumber(RS1/RS2)`. CPU-value indices are checked at full width
against regDepth before a handler runs; invalid indices return an error without
side effects. Constants are checked at elaboration. A vector index implicitly
consumes the selected CPU input; it does not require an extra CpuRead declaration.
A new instruction cannot invent a third CPU input or implicitly reuse CPU rd as VD.
Use explicit in-place or constant destinations; packed/configured index sources
are not supported by this first resource schema.

`regDepth = 1` is supported and is the natural choice for a single accumulator
slot. The platform addresses that one slot statically instead of through a dynamic
`Vec` index: SpinalHDL addresses a `Vec` with `log2Up(size)` bits, so a one-element
`Vec` would need a zero-width address while the generic slot index is one bit wide,
and the deferred path's latched index register carries no auto-resize tag. A
one-element `Vec` read at a static index is exactly one register, so storage stays
uniform for every `regDepth`. `CpuValue` indices must then be 0; anything else is
rejected as an out-of-range index.

An operation with vector writeback still sends a CFU completion response. Its
software intrinsic returns `void` and encodes CPU `rd=x0`; it does not write its
vector result into CPU RD. `response_id` correlates requests and is not a register
number. With CpuWriteRD, the intrinsic returns `uint32_t`. New signatures take only
CPU values needed by declared reads or index sources, in rs1/rs2 order; use the
generated header rather than assuming the old two-argument ABI.

The matching ISA fragment is:

```yaml
commands:
  - name: vector_xor_inplace
    function_id: 1
    subfunction_id: 2
    kind: compute
    semantics: bitwise XOR two resident vectors, overwriting the first
    response: status
    resources:
      - {storage: vector, access: read, role: vs1,
         index: {source: cpu_value, register: rs1}}
      - {storage: vector, access: read, role: vs2,
         index: {source: cpu_value, register: rs2}}
      - {storage: vector, access: write, role: vd,
         index: {source: cpu_value, register: rs1}}
```

CPU resources use `{storage: cpu, access: read, role: rs1}` (or rs2), and
`{storage: cpu, access: write, role: rd}`. Alternative vector indices are
`{source: constant, value: 0}` and legacy-only
`{source: legacy_register_number, register: rs1}` (or rs2).
The schema rejects unknown fields/roles, duplicate roles and multiple data writes.
Resource changes are semantic ABI changes and remain frozen in append-only rounds.

**Legacy compatibility is explicit:** omitted YAML resources / Scala `None`
preserve the old two-argument, CPU-returning interface and handler API.
`resources: []` / `Some(Seq.empty)` instead mean no operand/data-write resources,
with mandatory acknowledgement. Legacy intrinsics still bind a1/a2 and return a0.
The scaffold's old load/store index comes from raw rs2's register number (fixed
x12 for those legacy intrinsics), not their second C argument's value. New designs
should use CpuValue rather than copying that historical encoding convention.
The legacy load/store entries now declare these address/slot resources explicitly,
but their existing LSU still owns beat writes, memory completion and error status;
they do not also invoke the generic complete(result) vector-write path.

Use `ctx.defer()` with a tick callback for multicycle work. Resource reads and
vector destination indices are captured on acceptance, including vector source
snapshots; tick does not observe later RF writes or live input changes. Use
`ctx.read(role)` and `ctx.complete` rather than bypassing resource routing with
raw input/vector helpers. CPU/raw instruction/state/CFU/request metadata remain
latched for deferred contexts. Individual handlers must not drive bus ready/valid
or bypass the response buffer. Allocate shared registers/Areas inside the table
method or lazily: overriding it with a subclass `val` is unsafe during construction.
Callbacks build conditional hardware during elaboration; they are not runtime Scala.

Exact `(funct3, funct7)` pairs deliberately avoid wildcard priority ambiguity.
Elaboration rejects out-of-range encodings, duplicate names, duplicate pairs and
`defer()` without a tick callback; disabled declarations are still checked for
encoding/name mistakes. Removing/disabling an entry makes it unsupported, including
when the table is empty. A `ready` callback is combinational and must not depend on
`cmd.ready` itself (which would create a combinational loop).
No new CPU MicroOp, pipeline, CSR or CfuBus field is introduced. Matching still uses
the existing projected function/suboperation fields, not a new custom0-only mask,
so the native CPU custom1 routing is not silently changed (the agent's generated
register-register intrinsics remain custom0-only).

### Multi-instruction ISA evolution

`AgentCfu` keeps its three-bit `function_id` bus contract and decodes a seven-bit
`subfunction_id` from `raw_insn[31:25]` (funct7). The custom0 CPU decoder already
accepts these bits. `matches(function, operation = 0)` matches the complete pair;
legacy compute/config/load/store retain suboperation 0. Unknown suboperations
return the CFU error status rather than silently executing a legacy operation.
The scaffold remains algorithm-neutral: agents implement the operations they declare.

Each ISA command may declare `subfunction_id: 0..127` (default 0) and
`kind: compute|control|load|store`. Function IDs remain 0..7. The pair must be unique,
as must generated intrinsic names/macros; enable/fence/counter helper names are
reserved. Generated custom0 intrinsics encode `subfunction_id << 25`. Existing
specs without this field retain their original instruction words.

With `--loop`, the optimizer can select `extend_isa` as well as `implement_hw` or
`implement_sw`. An extension returns a complete candidate contract with at least
one new command. Old commands, encodings and all their fields remain unchanged;
shared state, memory, datapath, integration, assumptions and software ABI remain
frozen (the intrinsic name list may grow). This is deliberately conservative:
state-layout or reset-semantic migrations are not yet supported. Rejected candidates
do not replace the incumbent ISA. Accepted revisions are recorded under
`08_isa/revisions/` and extension attempts under `08_isa/extensions/`.

An accepted extension always traverses HW build, regenerated intrinsics, SW
adaptation, SW build, correctness/performance simulation and cost. Software must
call at least one new instruction as well as exercise the datapath; a control-only
kernel cannot satisfy datapath usage. Static call-site checks are preliminary
checks, not proof of dynamic execution or general correctness.

`--loop-policy explore` is the default. A model stop recommendation receives one
bounded reconsideration request covering batching, instruction extensions,
packing/transfers and area reduction. `--loop-policy conservative` retains the
one-shot stop behavior. `--loop-patience N` (default 3) stops after N consecutive
non-improving optimization results, and `--max-iters` remains the global repair/
optimization bound. A second stop, an unusable plan or provider failure ends
exploration with an explicit reason; exploration is not an unlimited loop.

Candidate selection compares exact reference/CFU cycle ratios and cell counts;
`speedup_x100` is only the legacy display metric. Only fully validated rounds can
improve the incumbent. Immutable source snapshots, ISA and gate evidence for each
new best are kept under `12_review/candidates/<round>/candidate.yaml`. The report
identifies that selected source delivery separately from the final experiment.
The mutable workspace and binaries are NOT silently restored: a failed final
experiment remains failed, while the earlier validated source candidate stays
available and can be rebuilt from its manifest.

All newly written stage artifacts retain an `immutable_path` content-addressed
copy beside the latest view, allowing earlier manifest hashes to be retrieved.
Historical runs are not rewritten. FILE and EDIT prevalidate every target and
roll back in-process I/O failures; this is not a crash-atomic filesystem transaction.
PATCH accepts text-content patches only and rejects unsupported Git metadata;
all delivery protocols confine targets to the current run workspace. Mixed
protocols are rejected. Directory copies reject symlinks and check every item.
Tool error strings are recorded as failed calls; reads and directory enumeration
are bounded, and regex search uses a conservative safe subset. With a run_root,
read/list/search also reject other runs rather than merely omitting them from prompts.
The scaffold and ISA extension regression can be run without a paid model request:

```bash
conda run -n mico_env python -m unittest discover -s agents/tests -t .
conda run -n mico_env sbt 'Test / runMain vexiiriscv.soc.mico.AgentCfuSuboperationSim'
```

Current limits: the static software call check is not a full C preprocessor or
execution trace; candidate snapshots retain editable source inputs rather than a
hermetic toolchain/build cache; simultaneous writers to the same run are unsupported.
Full context-token budgeting, structured ToolResult migration and architecture/state
ABI migrations remain separate follow-up work.

### Access and command policy

`agents/specs/agent_access_policy.yaml` is the single source of truth for what each stage may read/write and which commands it may run:

- Deny rules always take precedence; paths are resolved to repo-relative form and reject absolute paths, `..` escapes, and symlinks outside the workdir.
- Each stage has its own `read` / `write` allow lists. Secret, `.git`, build/binary (`*.elf`, `*.o`, `*.map`, `*.vcd`, `*.fst`), and non-allowlisted repo source are not exposed to the agent. Disassembly (`*.asm`) is no longer globally denied: it is text and is exposed only where a stage allowlists it (the analysis rung for the profile build, and `sw/tests/**`), while build trees stay denied. Historical runs under `agents/generated/runs/` are never auto-read.
- Visibility is a second, read-only gate on top of the stage lists: the policy's `visibility.gated` globs mark paths that require `standard`/`extended`, and `--visibility` selects the level (see [Visibility levels](#visibility-levels)).
- Commands are declared by id (e.g. `make_profile`, `soc_sim`, `cfu_generate_agent`, `yosys_cost`, `git_apply`). Shell metacharacters, absolute paths, and sensitive paths in command argv are rejected; the executable and required tokens must match the declaration.
- `--apply-patch` is required for **all three** delivery protocols. Without it, SEARCH/REPLACE
  edits, `FILE_BEGIN` blocks, and unified diffs are recorded as proposals only; the run never
  writes LLM-authored content into the workspace (the generated benchmark copy is framework
  setup, not agent output). `--run-commands` on its own only allows allowlisted commands.
- Hardware writes are confined to `workspace/design/AgentCfu.scala` and `workspace/design/AgentCfuFiber.scala`; `src/main/scala/**` is read-only for every stage.

### Run artifacts

Every run gets a unique directory `agents/generated/runs/<run_id>/` and writes:

```text
01_context/manifest.yaml          # visibility level, visible/withheld/denied files, git status
02_workload/workload_spec.yaml    # normalized workload spec
03_route/route.yaml
04_benchmark/manifest.yaml        # benchmark/project profile manifest and commands
05_profile/result.yaml
06_hotspot/hotspot.yaml
07_analysis/workload_analysis.yaml
08_isa/cfu_isa.yaml + cfu_isa.md  # structured CFU ISA spec + rendered markdown
09_implementation/hw_plan.md, hw_patch.diff, hw_changed_files.yaml
09_implementation/hw_tool_calls.yaml       # read_file/list_dir/search_text transcript
07_analysis/analysis_tool_calls.yaml       # analysis-rung read_file/list_dir/search_text transcript
09_implementation/sw_plan.md, sw_patch.diff, sw_changed_files.yaml
09_implementation/software_decision.yaml   # required/reused (software_change) + its evidence
09_implementation/hw_repair_feedback.txt   # exact error log injected on a hardware retry
09_implementation/sw_repair_feedback.txt   # exact error log injected on a software retry
02_workload/llm_raw/spec_*.txt, 07_analysis/llm_raw/analysis_*.txt
08_isa/llm_raw/isa_*.txt, 09_implementation/llm_raw/{hardware,software,software_route}_*.txt
*_<n>.reasoning.txt                   # model thinking, split out of the answer
10_validation/hw_build.yaml       # SoC generation rung
10_validation/sw_build.yaml       # software compile rung
10_validation/sim.yaml            # simulation rung
10_validation/embench.yaml        # reference/cfu cycles and speedup_x100 (embench)
11_cost/cfu_rtl/AgentCfu.v        # standalone CFU RTL (GenAgentCfu), the Yosys input
11_cost/result.yaml
12_review/decision.yaml
12_review/optimization_rounds.yaml   # --loop ledger: speedup/cells per round
12_review/optimization_decision.yaml # the round the agent chose to start
report.md
run.yaml                          # manifest with per-artifact sha256
workspace/...                     # per-run isolated benchmark/project workspace
workspace/design/...              # isolated AgentCfu.scala + AgentCfuFiber.scala
workspace/soc/...                 # MiCoSoc.v / soc.h generated from the overlay
```

Each artifact records its stage, status, inputs, visible files, SHA-256, size, and timestamp. The benchmark workspace is scoped to the run so runs do not clobber each other, and C-project inputs are always copied before instrumentation so the original source tree is never modified.

### Status semantics

- `planned`: a planning pass completed. Either it was a read-only pass, or the execution gates were skipped (execution disabled, `--skip-sim`, `--skip-yosys`). The report lists each gate and what was skipped. This is **not** a claim of correctness or execution.
- A gate that was skipped is recorded as `{"ok": false, "skipped": true, "reason": ...}`. Skipped gates never produce `failed` and can never produce `complete`.
- `complete`: every rung (workload spec, ISA contract, hardware design, SoC generation, software design, compile, simulation, cost) produced evidence and passed. Only this is a real completion.
- `failed`: a rung failed and the repair budget was exhausted. The run stops at that rung (`stopped_at` in the report) and skips the rest.
- `blocked`: an input/ISA contract could not be validated and no repair budget remained, so the loop cannot proceed honestly.
- `needs_iteration` is no longer a terminal status: repairs happen inside the ladder, before the run reaches `review`.

A run is never marked complete solely because a command was skipped or a dry-run was used.

### Live progress and token reporting

Each stage emits a live line as it runs and a second line when it finishes, e.g.:

```
[ 3/17] route input ... running
[ 3/17] route input ... OK (0 tok, 0.0s)
[ 7/17] analyze workload ... OK (375 tok, 0.2s)
[ 8/17] design CFU ISA ... OK (375 tok, 0.1s)
[11/17] check conformance ... FAIL (210 tok, 6.1s)
```

- `[i/N]` is the slot index of the stage within the 17-step workflow (every
  graph node maps to exactly one slot — including `check_conformance` at
  `[11/17]` — so a clean run prints strictly increasing indices; a repair loop
  legitimately revisits an earlier index).
- `OK`/`FAIL` is `gates.<key>.ok` **and** whether this node's own update added
  errors (`errors` is an append-reducer, so the update's list is exactly this
  node's increment). A rung that fails and routes to a repair always prints
  `FAIL`, even when the run had already seen an earlier error.
- `(N tok)` is the total LLM tokens consumed during that stage.
- Command stdout/stderr is normalized before it enters state or LLM feedback: ANSI color/CSI/OSC sequences, carriage-return progress updates, backspaces, and other terminal control bytes are removed.

**Token streaming.** Model output tokens are streamed by default (`--stream-llm`; disable with `--no-stream-llm`, env `CFU_AGENT_STREAM_LLM`, or `"stream": false` in the config). Token streaming is what makes long generations safe and visible:

- While a call is streaming, one progress line is **rewritten in place** (`\r`) so a slow generation is visibly alive without scrolling the terminal:
  ```
      [llm] analyze workload streaming ... 0 out + 1234 think chars (~308 tok)
      [llm] implement HW streaming ... 3669 chars (~917 tok)
  ```
  The line is only redrawn when the stream actually advanced, and thinking is counted
  separately, so a long reasoning phase shows growing `N think chars` instead of
  repeating `0 chars` (tool calls are not counted as characters). The line is cleared
  before any other output (tool calls, the per-call token line, stage transitions).
  When stdout is not a terminal (CI logs, `tee`) intermediate updates are suppressed
  entirely, so the log contains only the final `[llm] <label> -> N output tok` line.
- Each call then prints its exact token count and mode:
  ```
      [llm] hardware -> 1051 output tok (chat, stop)
  ```
- Because the HTTP read timeout applies per streamed chunk instead of to the whole body, a generation that would exceed `timeout` no longer fails. Measured example: a GLM-5.2 call produced 2171 output tokens over 118.6 s with the config's `timeout: 120` and completed, with 264 live progress updates and 4406 characters of thinking captured from the stream.
- Tool calls stream too: content, `reasoning_content` and `tool_calls` fragments are accumulated, and the provider's final usage chunk gives the exact counts.
- If an endpoint ignores or rejects `stream`/`stream_options`, the call retries without streaming and records `fallback_reason` ("streaming unavailable …" / "streaming ignored by the endpoint") in the raw-output metadata and the `[llm]` line.
- This is separate from LangGraph's `--stream`, which streams whole-state snapshots between nodes (`stream_mode="values"`); token streaming happens one level lower, inside the LLM client.

At the end the CLI prints a `summary:` headline block with the measured acceleration and cost, then cumulative token usage and per-stage results:

```
summary:
  cycles_before = 7853884
  cycles_after  = 4712330
  speedup       = 1.66x
  yosys_cost    = top=AgentCfu cells=3845 wire_bits=4402 memories=0 memory_bits=0
```

`cycles_before`/`cycles_after` are the reference and CFU-build cycles of the profiled embench target and `speedup` is their ratio (the same `speedup_x100` as `10_validation/embench.yaml`); they print `n/a (<reason>)` when the simulation was skipped, never ran, or the workload has no reference cycle count. `yosys_cost` is the parsed summary line of the last successful estimator run, so it prints `n/a` when cost was skipped or failed. The `report.md` includes `## Token Usage` and `## LLM Raw Outputs` sections. Each LLM call's returned text is persisted under the run's `llm_raw/` directory; retries receive a new numbered file. Inline reasoning (`<thinking>` tags, `think` blocks, or a provider reasoning field) is split out of the answer before parsing and stored beside it as `*_<n>.reasoning.txt`, so artifact parsers only ever see the final answer. Prompts and repository context are not copied into these raw-output files; the error log injected into a repair prompt is kept separately as `09_implementation/{hw,sw}_repair_feedback.txt`.

### Embench-iot workloads

Any embench-iot benchmark can be used as the workload input:

```bash
conda run -n mico_env python3 -B agents/run_cfu_agent.py \
  --embench crc32 --run-commands --skip-yosys --max-iters 1

# Same workload, but keep improving a passing design: up to two more rounds
# (--max-iters 3) chosen by the agent, each fully re-validated by the ladder.
conda run -n mico_env python3 -B agents/run_cfu_agent.py \
  --embench crc32 --run-commands --loop --max-iters 3
```

The run copies `benchmarks/embench-iot/{src/<bench>,support}` plus the tracked
VexiiRiscv board (`sw/embench/`) into `<run>/workspace/embench_<bench>/` and builds
three configurations, each with its own source tree:

| config | defines | evidence |
| --- | --- | --- |
| reference | board only | Embench verify passes + baseline cycles |
| profile | `-DEMBENCH_OBS` | per-loop inclusive cycles/calls plus containing-function context |
| cfu | `-DUSE_AGENT_CFU` | verify still passes + speedup vs reference |

By default `embench_instrument.py` instruments complete `for`/`while`/`do` loop
statements (not each iteration) and their containing functions, up to 24 total
points per workload. Stable loop IDs map to source filename, function, ordinal,
line and loop kind in `05_profile/embench.yaml`; `WorkloadSpec.hot_loop` preserves
the selected loop and `kernel_function` remains the actual C function. An explicit
`observation_plan` may select function or loop points (`kind: loop` requires a
unique `ordinal` or `line`). Ambiguous/unsupported loops are skipped rather than
instrumented at a guessed position. GCC cleanup attributes count early exits;
nested loop/function cycles are inclusive and must not be summed. This default
currently applies to the Embench path; arbitrary C-project builds retain their
existing profiler. Only the profile copy is instrumented: reference and CFU
correctness/speedup builds remain clean.

Correctness comes from Embench itself: `support/main.c` returns `!correct` and the
repo `start.S` maps that to the `pass`/`fail` symbols MiCoSocSim watches. Artifacts:
`04_benchmark/embench.yaml`, `05_profile/embench.yaml`, `10_validation/embench.yaml`
(reference/cfu cycles and `speedup_x100`).

**Verify-failure feedback:** `MiCoSocSim` only reports that the CPU reached the fail
symbol, so the harness owns the generated `embench_main.c` and adds one line after
`verify_benchmark()`:
`EMBENCH_VERIFY_FAIL benchmark=<name> verify=<code> result=<value>` (or `_PASS`).
Embench's `verify_benchmark()` returns non-zero when the result is correct and 0 on a
mismatch (main returns `!correct` and `start.S` branches to `pass` on 0), so `verify=0`
is the failure code. The
sim rung parses it (and the fail-symbol/timeout markers) into a short attribution that is
stored on the `sim` gate, in `10_validation/embench.yaml`, and injected at the top of the
hardware/software repair feedback — e.g. "Embench verify failed for depthconv
(verify_benchmark returned 0) … wrong-result failure, not a build error" or "timed out …
likely a hang". When correctness passed, no attribution is produced, so only the rung's
own gate (for example the speedup floor) is reported. The Embench prompt block also
documents these log semantics so a repair
triages a wrong result instead of rewriting the build or the benchmark oracle.
A response whose `EDIT_BEGIN` section is not closed by a line equal to `EDIT_END` is
reported as such ("was not closed"), and a trailing sampler token after `EDIT_END` is
tolerated instead of discarding an otherwise valid edit.

**Reference/profile cache:** the `reference` and `profile` configurations are scalar
builds simulated on the seeded scaffold (profiling runs before any hardware edit), so
their measurement is independent of the agent's CFU design and deterministic for fixed
inputs. A run therefore reuses the cached measurement instead of rebuilding and
re-simulating both configurations (`--profile-cache on`, the default; `off` disables,
`refresh` recomputes and overwrites; env `CFU_AGENT_PROFILE_CACHE`). The key hashes the
benchmark sources, both build commands (including their defines, with the run's own
workspace path normalized to a placeholder so the key is **stable across runs**), the
CPU/SoC flags, the seeded `workspace/design` overlay, the fixed
`soc/mico`+`soc/cfu`+`execute/cfu` sources, the toolchain versions (`gcc`, `verilator`),
the git revision/submodules, the normalized loop observation plan and the
instrumenter/board-counter source hashes, so changed instrumentation, toolchain
or scaffold invalidates it.
Entries live in the git-ignored `agents/generated/cache/embench_profile/<key>.json` and
are written only after a passing measurement. A hit runs no `make`/`sbt`, and the run
records `profile_cache: {status: hit|miss|refresh|disabled, key, saved}` in
`05_profile/embench.yaml`, `run.yaml` and `iteration_history`, so a cached baseline is
never mistaken for a fresh one.

**Closed loop (LLM):** the analysis, contract and software prompts receive the
benchmark source, the measured loop/function profile (including the loop's source
location), and the generated `cfu_intrinsics.h`. The LLM is asked to identify the
hottest loop and its containing function and to emit a patch for
`benchmarks/embench-iot/src/<bench>/<file>.c` guarded by `#ifdef USE_AGENT_CFU`; the
harness redirects that path into the run's `cfu` configuration tree, so the reference
build stays scalar and Embench's own `verify_benchmark()` still decides correctness.
The `sim` rung then reports `reference_cycles`, `cfu_cycles` and `speedup_x100`.

Note: `benchmarks/embench-iot` and `sw` are git submodules; `sw/embench/` and the
`sw/Makefile` hooks are committed inside the `sw` submodule.

### WorkloadSpec and CFU ISA spec

`agents/specs/cfu_isa_template.yaml` remains the human-readable CFU ISA contract template. The ISA LLM stage returns one JSON object and requests API-level JSON mode; the agent then writes the normalized contract as `08_isa/cfu_isa.yaml` and `08_isa/cfu_isa.md`. `agents/cfu_design_agent/isa.py` validates both the JSON protocol and legacy YAML/fenced responses, checking the supported opcode and unique `(function_id, subfunction_id)` encodings, command names, basic metadata shapes, software intrinsics/scalar fallback, integration presence, and validation requirements. Generic scaffold dimensions (`vlen`, `xlen`, `regDepth`) are optional; workload-specific choices belong in `datapath.implementation` (for example, a CRC polynomial or permutation width with its rationale, rather than universal MAC-width fields). The generic schema does not assume a MAC, accumulator, RF storage type or pipeline. Legacy flat metadata remains readable, but it is not injected into new contracts. Structural lint is not proof of state/reset or memory-ordering correctness; implementation, simulation and explicit compatibility checks provide those additional gates. The `isa` rung is the first ladder check: an invalid contract is repaired (or stops the run) before any hardware is designed.

`agents/specs/u8_weighted_average.spec` is an example normalized `WorkloadSpec` input model (the tracked reference copy may be absent in a pruned checkout). It has a versioned schema; unknown high-impact fields must be declared in `unknowns` instead of being silently guessed.

### LLM configuration

Named configs live in `agents/llm_configs/*.json`. Select one at runtime with `--llm-config <name>` (or `CFU_AGENT_LLM_CONFIG`), defaulting to `default`:

```bash
export DEEPSEEK_API_KEY="sk-..."
python3 agents/run_cfu_agent.py --task "..." --llm-config deepseek --dry-run
```

Check connectivity for a config without running the design loop:

```bash
# Resolves the selected config and makes one minimal call.
python3 agents/run_cfu_agent.py --test-api --llm-config deepseek
# Prints config/model/base_url/api_mode/api_key_set/used_llm/reply/api-status.
# Exits 0 on success, 1 on missing key, bad config, or a failed call.
```

Each config may set `model`, `base_url`, `api_mode` (`auto`/`responses`/`chat`), `temperature`, `timeout`, thinking (`thinking`, `reasoning_effort`, `reasoning_summary`, optional `thinking_body`; **off by default**), and an inline `api_key` or an `api_key_env` variable name. Model resolution order (highest wins) is: explicit `--model` > config `model` > `CFU_AGENT_MODEL` > built-in `gpt-5`. A config can leave `model`/`base_url` empty and still pick up `CFU_AGENT_MODEL`/`OPENAI_BASE_URL`. Prefer `api_key_env` to keep keys out of the tracked JSON. `default.json` is optional: if missing, the framework builds an environment-only config. See `agents/llm_configs/README.md`.

### Environment

Run the agent and its tests from `conda run -n mico_env` (or an activated `mico_env`): that environment provides `langgraph` and `sbt`. Without `langgraph` the framework falls back to a pure-Python runner with equivalent state-merge semantics, but the LangGraph path is the primary one.

- `OPENAI_API_KEY`: required only when LLM calls are enabled. Without it (or when the selected config resolves no key), the agent uses deterministic fallbacks.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible endpoint.
- `CFU_AGENT_MODEL`: optional default model name.
- `CFU_AGENT_LLM_CONFIG`: optional default config name (otherwise `default`).
- `CFU_AGENT_LLM_API`: optional `auto`/`responses`/`chat` override.
- `CFU_AGENT_VISIBILITY`: optional default visibility tier (`basic`/`standard`/`extended`); otherwise `standard`.
- `CFU_AGENT_PROFILE_CACHE`: optional default profile-cache mode (`on`/`off`/`refresh`); otherwise `on`.
- `CFU_AGENT_REPAIR_ROUTING`: optional default simulation-repair routing (`auto`/`fixed`); otherwise `auto`.
- `CFU_AGENT_SOFTWARE_CHANGE`: optional default software-rung policy (`auto`/`always`); otherwise `auto`.
- `CFU_AGENT_LOOP`: optional default for `--loop` (`on`/`off`); otherwise off.

### Tests

```bash
# Use the project environment: it provides langgraph and sbt.
conda run -n mico_env python3 -m unittest discover -s agents/tests
```

The `unittest` suite checks path traversal, deny precedence, command allowlisting (including the agent-overlay command and positional-path confinement), visibility tiers (policy gating, withheld-vs-denied context loading, tool denial, level-specific prompt text, and CLI threading), run isolation, `WorkloadSpec`/CFU ISA validation, gate honesty, ladder routing (per-rung repair, LLM-chosen simulation repair routing with its deterministic fallback and `fixed` mode, LLM-chosen software reuse behind its intrinsic/ABI guard and its `always` mode, the `--loop` optimization ledger with its Pareto improvement test and budget bound, hard stop without budget, skipped-rung honesty), dry-run flow without any external toolchain or API key, and the targeted-edit delivery protocol (SEARCH/REPLACE parsing, unique-match enforcement, atomic multi-target application, and an end-to-end hardware rung that applies an edit instead of a whole-file rewrite).

Run it inside `mico_env`: that environment has `langgraph`, so the LangGraph runner is the live path and its tests are not skipped. The suite also covers the pure-Python fallback runner, whose append-key merge semantics mirror LangGraph's.

Ready-to-run examples are under `agents/examples/`.
