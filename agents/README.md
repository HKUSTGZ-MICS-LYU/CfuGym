# CfuGym Agents

This folder contains local agent frameworks for CfuGym workflows.

## Code layout

`cfu_design_agent/` separates three responsibilities:

- `benchmark/`: workload adapters. `generic.py` prepares generated/C-project
  benchmarks; `embench.py`, `embench_instrument.py`, and `embench_loops.py`
  handle Embench workspace, observation points, and source loop discovery.
  New benchmark suites should add adapters here rather than extend the flow
  runner with benchmark-specific workspace mechanics.
- `flow/`: the flow shape and the domain control plane. `wiring.py` is the
  node/edge specification; `gates.py` holds the gate and budget vocabulary;
  `ladder.py` owns routing and repair decisions; `instrument.py` owns the node
  wrapper (progress, fingerprints, gate staleness, side-effect journaling);
  `optimize.py` owns the optimization ledger; `policies.py` declares the
  per-node LangGraph policies; `outcome.py` is the gate/rewind vocabulary every
  rung records its verdict through. LangGraph is required; the
  parallel pure-Python runner and its manual reducer simulation have been
  removed.
- `stages/`: one module per stage — `context.py` (load/ingest/route/prepare),
  `profile.py` (profile, hotspot, IR), `analysis.py`, `contract.py` (ISA),
  `feedback.py` (repair/optimization prompt blocks and attempt ledgers),
  `fallbacks.py`, and the sub-packages `implement/` (`hw`, `sw`, `unified`,
  `delivery`, `intrinsics`), `verify/` (`soc`, `conformance`, `software`,
  `cost`) and `review/` (`review_and_route`, `optimize`, `report`); the
  package root holds `conformance.py` (oracle validation, harness-owned C
  rendering, verdict parsing) and `conformance_data.py` (the fixed-seed
  boundary/random data generator). A stage
  module builds prompts, applies deliveries and reports gates; it never decides
  that a gate passed.
- `runtime/`: low-level execution services — policy, repository tools,
  artifacts, read-only agent tools, LLM transport/config, the single
  `llm_step` seam for one model interaction, `checks` for command execution,
  `commands` for state-driven argv builders, `stage_services` for the
  per-stage toolbox/artifact writers, persistence, and progress.
  `stage_executor.py` is the execution-plane seam: it obtains one stage
  deliverable through a plain or bounded tool interaction, while the stage
  modules continue to own prompts, edits, validation gates and routing.
- `state/`: the run state, declared in four named groups (`config.py` inputs,
  `evidence.py` observations and artifacts, `ledger.py` gate verdicts and
  budgets, `optimization.py` the `--loop` island) composed into the single flat
  `CfuDesignState` that LangGraph registers. `TypedDict` inheritance keeps every
  inherited key in `__annotations__` and `get_type_hints`, so the channels and
  reducers are unchanged; `state/__init__.py` stays the one import site.
- `graph.py`: the composition root only (~200 lines). It builds the graph from
  `flow/wiring.py`, exposes `build_graph` / `run_agent` / `python_nodes` /
  `export_graph_visualization`, and re-exports the node functions that
  `python_nodes()` resolves from its own globals.

Two rules keep this decomposition stable, and `agents/tests/test_graph_modules.py`
enforces both:

1. Reach a function through its owning module — `from ..runtime import checks`
   then `checks.run_check(...)`, never `from ..runtime.checks import run_check`.
   An imported name is a local binding, so a test that mocks the owning module
   would not be observed by the caller.
2. Do not re-export moved helpers from `graph.py`. Only public entry points and
   node functions belong there, so a stale
   `mock.patch.object(graph, "<helper>")` raises AttributeError instead of
   silently patching a binding nobody reads.
- `ir/`: the LLVM IR evidence exporter (P0) and per-observation evidence lookup. `--ir-features auto`
  (the default) prefers the exact reference target; because the requested ISA
  (`rv32imc_zicsr_zifencei`) is GCC-only and an installed Clang 14 rejects it, `auto` then retries the
  reduced `rv32imc` export with cross-toolchain headers and returns it **labelled**
  (`status: approximate`, `target_match: false`, the strict reason under `fallback`), so the analysis
  gets evidence instead of nothing. A precondition failure (unsafe input, missing clang) is never
  retried. `off` does nothing; `require` marks strict export failure as an independent run failure and
  never falls back; `approx` forces the approximate export. Approximate IR is **static evidence**, not
  the original target's instructions. None of these modes replace per-loop measurements, infer loop
  cycles, or alter the reference/CFU validation builds. Commands, hashes, target differences, failures,
  and any `.ll` files are recorded in the run's `07_analysis/` directory. When LLVM `opt` (14.x) is
  available, each exported TU also gets a best-effort `<stem>.features.xml` static feature report (data
  movement, typed operation counts, SSA/loop-carried/memory dependencies, SCEV trip counts when
  constant) surfaced through `inspect_observation(..., view='features')`; it is static IR evidence —
  never measured cycles or runtime byte counts — and its absence never blocks the IR or profile path.

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

A practical invocation and validation walkthrough is in
[`docs/cfu_agent_sw_hw_coding_system.md`](../docs/cfu_agent_sw_hw_coding_system.md).

These two tracked files are **read-only defaults**. Every run seeds an isolated workspace copy:

```text
agents/generated/runs/<run_id>/workspace/design/AgentCfu.scala
agents/generated/runs/<run_id>/workspace/design/AgentCfuFiber.scala
```

An agent may write **only** those two files for hardware; the access policy rejects any write
to `src/main/scala/**`. A patch that names the tracked scaffold path is redirected into the run
workspace automatically, so the repository tree is never modified by a design run.

Hardware and software implementation use the shared stage executor for bounded,
policy-checked repository discovery before producing a final delivery. By default,
each agent step allows 20 tool rounds, 4 calls per model round, and 96 calls total;
`--max-tool-rounds` and `--max-tool-calls` can lower or raise the per-step limits.
A stage can
inspect a typed failure summary and a bounded excerpt of its run-scoped complete
command log by failure ID; complete logs are never injected wholesale into prompts.
Every command the flow executes — an embench build in the profile stage, an isolated
elaboration/compile check, a SoC generation, a conformance or workload simulation, a
Yosys synthesis — persists its complete stdout/stderr under its own stage's
`logs/` directory (`05_profile/logs`, `10_validation/logs`, `11_cost/logs`) with an
immutable `versions/` copy, and the tool loop can enumerate them with
`list_command_logs(rung?, limit?)` and read any one of them with `read_file`. The
record is written for successes as well as failures, so a repairable regression can
be compared against the last passing run instead of guessed from a truncated tail.

### Benchmark verifier visibility

The Embench `verify_benchmark` interface remains visible, but its function body is
hidden from the design agent in every stage. This is an **Agent-only source view**:
original benchmark/reference files, Host hashes, compilation and actual verification
remain unchanged. Read/search tools mask the complete readable source before line
selection or matching; no-tool source/context prompts mask before bounding excerpts.
Old hash-matched memory excerpts are rebuilt from the masked file view, including
body-only reads. Tool diff/mismatch output, escaped source in JSON/YAML artifacts,
verifier IR/ASM/features and Chat/Responses request payloads are covered as well.
Hidden-body markers are not source bytes and must never be copied into an edit.

This reduces direct verifier-code answer leakage, **not** all answer disclosure:
contracts, scalar/reference outputs, failure diagnostics and constants outside the
function body are not globally erased. It is not semantic verification or a replacement
for conformance and the fixed-result hardware gate. Historical artifacts are not rewritten.

### Versioned benchmark patches

`benchmark/embench_patchers.py` defines the `BenchmarkPatcher` preparation pipeline:
scale normalization, statemate link compatibility, then benchmark-specific semantic
patches. It operates only on source copies, before all three configurations are
written/hashed. Generated support files are Host-owned and protected byte-for-byte;
patch identities/generator/asset hashes are recorded in the project, checked on resume and
included in measurement fingerprints/cache inputs. Legacy prepared runs need a new
run; historical results and vendored sources are not rewritten.

`huffbench_stages_v1` exposes separate `huffbench_encode` / `huffbench_decode` APIs.
A protected driver generates a scalar reference stream during initialization
(outside timing). Timed candidate encoding produces observable bytes/code lengths/
codes; timed decoding receives a separate copy of the reference stream and must
replace poisoned output. After timing, the oracle checks every decoded byte,
encoded bit and padding bit, prefix-free codes and optimal weighted bit cost.
Equivalent optimal codebooks are allowed, not just the original tie-breaking.
Stage reset/copy overhead is equal in reference/profile/CFU; this changes the
benchmark ABI/baseline, so old roundtrip cycle counts are not comparable.

This patch hardens the **original fixed 500-byte fixture**, rejecting identity,
no-op, malformed-stream and colluding roundtrip shortcuts. It is not the proposed
multi-case semantic engine, proof on unseen inputs, hidden reference-data isolation,
or a sandbox against arbitrary memory-corrupting C/RTL. The upstream revision is
pinned and an unfamiliar source fails closed pending patch review. The original
codec's unsupported input cases are not promoted into a new general input domain.

### Encoding, execution status, and failure attribution

Correctness means the required workload's valid input domain, not the observed answer
or a narrower Agent-authored ISA. The shared system prompt asks for reusable
algorithm-specific CFUs that handle unseen legal values and data patterns, not sample
recognizers, preset answers, approximate semantics or decorative reads. Algorithm
constants, exact lookup tables, conditional Boolean results and genuine zero success
statuses remain legal. Fixed-shape specialization needs task or scalar-interface
justification; a fixture's shape/value range alone is insufficient. Exact special fast
paths need complete behavior for remaining legal inputs, and mostly scalar fallback
must not be described as general CFU acceleration.

Analysis and contract prompts distinguish semantic invariants from fixture data;
implementation deliveries briefly describe the runtime domain, justified constants,
2-3 unseen-input counterexamples and fallback/untested limits. Optimization must
preserve that domain. This is a prompt-only generality self-review: no additional LLM
stage, simulation batch or runtime gate is added, and planned counterexamples are not
executed evidence. An incomplete implementation must disclose its limits and choose
analysis/contract repair when appropriate; this guidance is not a correctness proof.

The hardware integrity gate v2 checks literal/alias computational replies with
bounded data-guard and memory-status context. Memory-status exemptions require
explicit contract effects and a computed payload routed to a real LSU store;
merely reading a pointer or asserting a store bit is insufficient. Encoding evidence
crosschecks frozen funct3/funct7 against Scala declarations/configuration and, after
generation, the actual RTL decoder. Software build crosschecks compiled custom0
machine words against the contract and decoder and binds the artifact hashes.
Unsupported syntax/coverage is explicitly `unchecked`, never certified equivalence.
These Host checks still run with `--no-conformance`.

Fresh Embench drivers bracket initialization and benchmark execution with Host-owned
CFU sticky-status CSR diagnostics (`0x801`, `sticky_csr_801_v1`), outside the timed
region. A nonzero status invalidates correctness/performance even if the original
verifier accepts a zero result; a missing required diagnostic also fails closed.
Protected drivers and versioned metadata prevent silently resuming an older driver.

An executed reference/profile precondition failure routes directly to review/report,
without analysis/contract model calls; explicit planning/skips remain advisory.
Rejected native deliveries roll back and become typed gate failures using the
existing retry budget, not missing-report exceptions. Unexpected/provider exceptions
still fail loudly but the CLI persists a partial report, true recovered state, original
exception type and available usage. Probe bookkeeping crashes are harness faults;
commit watchdog stalls and HDL protocol assertions remain separate observations.

Current performance requires matching candidate/ISA, source/configuration fingerprint,
repair evidence epoch and known ELF/ASM/RTL product hashes. Historical valid measurements
retain their snapshot/version/hash separately and never certify the final candidate.
Experiment collection reconstructs request usage by request ID; failed requests without
usage remain unknown, with known-token lower bounds and independent warnings. Historical
artifacts are not rewritten. No new multi-case semantic adapter is implemented here.

### Run-wide memory, context budgets, and request debugging

Context reuse uses two bounded **native parent-state channels**, rather than a
second session database or one global conversation shared by every stage:

- `flow_memory` shares accepted workload/analysis/ISA facts and actual read-time
  source fragments. A file retains multiple observed line ranges, not just its
  most recent read or a file-header summary. Identical reads are idempotent and
  do not reorder existing entries. Canonical live overlay paths are retained;
  the model's `{{run-dir}}` spelling is expanded before recording provenance.
  Reference-only paths are never claimed as previously read. Rendered ranges
  distinguish historical observation from the exact lines supplied **now**;
  missing lines, modifications and exact-edit preconditions still require reads.
  Gate-bound fact replacements also require acceptance fingerprints: an inherited
  old pass cannot authorize a rejected replacement. A tracked scaffold snapshot
  shadowed by a live overlay is discarded, never relabelled as an overlay read.
- `stage_history` retains bounded, artifact-backed execution notes for same-stage
  revisits and downstream handoffs. Summaries are rebuilt from hash-matched,
  policy-readable answer artifacts; stored prose is not trusted as evidence.
  Historical FILE/EDIT/PATCH bodies and fenced code are omitted, and old tool
  calls are **not replayed**. Notes are labelled `pending` inside an unfinished
  node, then `completed` or `rejected` from the node's final outcome. Rejected
  notes stay within their own stage. A completed note is still a model claim,
  never a build/conformance/simulation verdict. ISA lint-rejection notes follow
  an explicitly stored candidate for the next ISA retry, without accepting it.
  Artifact, task, contract, policy and relevant source changes invalidate notes;
  newly accepted delivery summaries bind graph-reported workspace outputs to their
  committed versions.

Collectors make earlier reads/notes available to independent model calls inside
one node without mutating its input state. Only a normally returning node publishes
new memory. On a rejected delivery, hash-matched read-only observations can survive
rollback, while rejected domain facts do not become accepted facts. Exceptions
publish neither new read memory nor execution history.

**Cache-Hit-first layout remains unchanged:** the system prompt and the shared
user prefix precede every stage-specific section. Memory and historical notes are
appended only at the dynamic tail, with a **4000-character combined cap including
section headings** (historical notes get at most 900 characters). No timestamps,
run IDs, history counters or source bodies are inserted ahead of the shared head,
and tool schemas are not changed by this feature. All dynamic content is masked
and its canonical run paths virtualized only after assembling the complete prompt,
for plain, ordinary-tool and native-tool calls alike. The existing context-window
preflight still applies; retained/cached content is not free context capacity.

File storage is bounded to 24 entries, four ranges / 12000 characters per file,
4000 characters per retained range, and 48 ranges / 48000 characters globally.
The legacy aliases separately duplicate at most 24 × 600 characters; domain facts
retain their own existing bounds. Execution notes (including active collectors)
retain at most 12 entries globally and three per stage/label;
summary text is at most 1200 characters before prompt selection. Legacy v2
single-excerpt memory remains readable. These are storage limits, not promises
that every stored byte fits the next model input.

Both channels use LangGraph's existing state and opt-in SQLite `--checkpoint`;
there is no new Store, embedding service, summarization model call, persistent
side-effecting subgraph or automatic interrupted-tool resume. State-only summaries
are intentionally not full conversation continuation. Independent validation gates,
visibility masking, transaction rollback and conservative resume checks remain
authoritative. See the native [StateGraph state model](https://docs.langchain.com/oss/python/langgraph/graph-api#state).

Each actual model request gets a conservative context preflight, including tool
schemas and reserved output. Named LLM JSON configurations may set:

```json
{"context_window_tokens": 131072, "output_reserve_tokens": 8192}
```

An explicit capacity wins; otherwise a bounded `/models/{model}` lookup uses a
reported context field when present. Standard OpenAI model metadata does not
promise that field. Unknown capacity uses a **32768-token soft budget**, not a
claim about the actual model. The offline estimator uses serialized bytes plus
protocol overhead, so it can substantially overestimate tokens. At 80% it records
a warning; at 90% it replaces older, completed tool exchanges with bounded historical
memos, retaining current edits/failures and legal tool/result pairs. If protected
content cannot fit, the request is blocked and the current coding transaction is
rolled back. The output reservation is also sent as `max_completion_tokens` (Chat)
or `max_output_tokens` (Responses); endpoints must support that limit. Cached tokens
still occupy context space. Usage logs report provider cached-input tokens when
available and `n/a` otherwise; prefix comparison is diagnostic, not a cache-hit claim.

All graph-run requests are written **before sending** to
`agents/generated/runs/<run_id>/request_audit.jsonl`, with one JSONL event per request,
result/error, capacity discovery, compaction or blocked request. The request event
contains the complete effective application JSON: messages/input, tool definitions,
reasoning/stream options and output limits; transport options and authorization are
excluded. It includes timestamp, run/stage, request ID, attempt linkage, body hash
and context estimate. SDK implicit retries are disabled, so the framework's explicit
stream/API fallbacks appear as separate attempts. Original stage prompt/answer
artifacts remain compatible; this ledger is the exact per-round debugging source.
The JSONL is mode `0600`, rejects links/unsafe paths, and audit-write failure prevents
sending an unrecorded request. It is host-owned and excluded from model reads; because
source text can occur in prompts, treat the ledger as sensitive run-local data.
Pure API self-tests outside a graph run have no run ledger. Inspect it locally with:

```bash
python -m json.tool --json-lines agents/generated/runs/YOUR_RUN_ID/request_audit.jsonl
```

No real-provider cache-hit improvement or gateway window discovery is inferred from
offline tests; those require a separate, explicitly authorized live run.

With agent tools, patch application and command execution all enabled, both
implementation stages gain narrow transactional coding capabilities:

- software: graph-bound exact SEARCH/REPLACE on approved kernel targets plus a
  no-argument isolated compile check;
- hardware: atomic exact SEARCH/REPLACE across only the two run-workspace
  `AgentCfu.scala`/`AgentCfuFiber.scala` overlays plus a no-argument isolated
  `GenAgentCfu` elaboration check.

The model cannot choose argv, cwd, main class, output directory, simulation, or
Yosys. A repair begins read-only and must submit a structured diagnosis citing
current failure IDs before mutation unlocks. Provider/tool errors or rejected
implementation output roll all model tool edits back to the stage-entry snapshot;
a semantically accepted implementation commits them. Internal checks are evidence
only and bind to the current candidate bytes. The ordinary `hw_build` and
`sw_build` nodes independently rerun the canonical graph-owned commands and remain
the only authoritative build Gates. Conformance, simulation and cost remain fully
graph-owned.

The edit tools, `read_file`, and the EDIT/FILE/PATCH delivery agree on paths: a
tracked scaffold path (for example `src/main/scala/.../AgentCfu.scala`) maps to
the run-workspace overlay in all three, so the model edits exactly the bytes it
read; denial messages list the legal target paths. Failure evidence is
freshness-aware: every `FailureObservation` carries `status: current | resolved`
(a later passing rung resolves it), repair prompts list only current IDs as
valid, resolved IDs appear as context with a "do not re-fix" note, and
`submit_repair_diagnosis` rejects resolved IDs — accepting the failing **rung
name** as an alias for its current id(s), listing the current ids when the model
cites an unknown one, and naming only genuinely cited resolved ids as "do not
re-fix" (run `20261005-073745` lost three rounds to a rejection that blamed an
unrelated stale failure the model had never cited). Sim/conformance failures are
classified (`fail_symbol`, `assertion`, `liveness_timeout`, `sim_harness_crash`,
`timeout`, ...). A CPU commit-progress watchdog (`hasn't committed anything for
too long`) is a repairable `liveness_timeout`: conformance routes back to
`implement` using the remaining verification-submission allowance to inspect completion,
backpressure and test size/latency. It is not proof of a wrong computed value.
A simulation-harness crash that produced no case verdict stops the run as "not
attributable to the design" without spending repair budget instead of being
routed to `implement`. The category table and the 2026-09-29 case study (the
`--agent-cfu-no-store` platform bug, since fixed in `AgentCfuFiber.getM2sParameters`)
are in [`docs/cfu_agent_failure_taxonomy.md`](../docs/cfu_agent_failure_taxonomy.md).

Implement delivers the formal oracle through `write_conformance(oracle)`: the host
parses/validates the oracle (independent scalar semantics plus the *shape* of the input
space) and renders fixed C source **and its test data**. The
oracle and source are a single rollback-capable transaction, approved by state hashes
for ISA, oracle, generated C and generator version. First delivery requires HW, SW
and oracle; later rounds keep the oracle **editable**: a design repair may strengthen
(never weaken) it, because a verification contract that cannot observe the real
failure has to be repairable. A repair round must submit a `submit_repair_diagnosis`
before it can be **submitted**: the gate is enforced at the stage gate, not at the
oracle write, so the model may author the corrected oracle first and file the diagnosis
afterwards (tool ordering must not cost a round). In the no-tool delivery mode the oracle is
re-authored on every repair round, and a revision the validator rejects keeps the
approved pair instead of failing the design round. Optional
`check_conformance_syntax()` invokes only `-fsyntax-only` (no ELF/link/simulation) with `-std=gnu99`, because the framework's own generated `cfu_intrinsics.h` binds operand registers with `register uint32_t r1 asm("a1")` and `-std=c99` rejected the header before the body was ever parsed. The same preflight is **host-owned and automatic**: every accepted `write_conformance` renders the pair and compiles it, and a compiler error (positive return code) rejects the delivery — the previous pair stays on disk and the session stays open so the model can deliver a corrected oracle in the same round. A negative status (policy denial, missing toolchain, timeout) is not a rejection: the model cannot fix that, and the `hw_conformance` rung still owns the verdict.
Implement exposes no `run_conformance` or `run_simulation` capability. Verify makes
no model calls and never authors or rewrites oracle/C/header: it checks the approved
bytes and builds/simulates that delivery. Missing/stale oracle or test compile errors
return to Implement in conformance-only scope using the answer-repair budget;
healthy HW/SW need no dummy edits. A **completed differential mismatch is
unattributed**: the test ran to a verdict and disagreed with its own reference, so
either side may be wrong (the model's scalar reference is not validated anywhere
else). That round is recorded with `repair_scope: conformance` and
`oracle_unattributed: true`, entered as a design repair but **without**
forcing a hardware edit; when the delivered change revised only the oracle,
`implement` refunds the design-repair diagnostic charge and charges the answer pool instead.
The verification submission is never refunded, so any rerun still needs a remaining `--max-iters` slot; oracle churn is also bounded by `--max-answer-repairs`. A
watchdog/hang or a crash with no verdict stays design-scope, and the oracle travels
with it so a blind contract can be fixed in the same
round. Host/policy failures stop without reauthoring.
A parsed PASS does not override a nonzero simulator exit or erase its diagnostics.
Independent references must not reach CFU intrinsics through helper functions, macros
or simple aliases; setup-only helpers may still initialize CFU state. This is bounded
syntactic lint, not a complete C/preprocessor proof of semantic independence.
The validator additionally rejects a check that decides its own outcome: an operand
compared with itself (`a == a`), a `verify` built only from the case arguments (an
acknowledgement command returns nothing), a comparison against a value the same
oracle's `setup`/`helpers` assigned a literal to, and an all-zero case set (which
exercises only the zero boundary). Regression class: run
`20261003-090854-136683-d98e5be9` passed 9 conformance cases with four
self-fulfilling checks and then failed the Embench result.

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

The `analysis` stage must include a Roofline section in `07_analysis/workload_analysis.yaml`: named memory boundary, operation and byte counts, arithmetic intensity, compute/bandwidth ceilings with provenance, ridge point, bottleneck, predicted versus achieved metric, design action, and unresolved unknowns. Downstream prompts do not inline that analysis: the contract and implementation prompts point at `07_analysis/workload_analysis.yaml` and require the agent to read it, so vector width, RF residency, memory style, pipeline depth, instruction granularity, and software blocking still address the modeled bottleneck.

### Fail-fast validation ladder

The design is checked by the tool that owns each artifact **immediately after it is produced**, instead of running every stage and deciding once at the end. Each rung either continues, repairs the artifact that caused the failure, or stops the run:

| Rung | Check | Gate | Repair target |
| --- | --- | --- | --- |
| `prepare_benchmark` | workload workspace copy + instrumentation setup (policy reads must succeed) | `prepare_benchmark` | — (input/policy failure stops the run) |
| `isa` | CFU ISA/integration contract lints | `isa` | `design_contract` |
| `implementation_hw` | writes `workspace/design/AgentCfu*.scala` only | `implementation_hw` | — |
| `hw_build` | `sbt … MiCoSocGen --mico-agent-cfu` (Scala compile + SpinalHDL elaboration + Verilog) | `hw_build` | `implement` |
| `hw_conformance` | harness-generated differential micro-test of every declared instruction (`make … cfu_conformance` + `MiCoSocSim --load-elf`); must print `CFU_CONFORMANCE PASS commands=n/n …` | `hw_conformance` | `implement` |
| `implementation_sw` | writes the accelerated kernel + `cfu_intrinsics.h`; the CFU build must call a CFU datapath intrinsic (not `cfu_enable()` + scalar). The rung may be **reused** instead of re-run when the kernel already in the workspace still drives this exact ISA contract | `implementation_sw` | — |
| `sw_build` | `make … compile` | `sw_build` | `implement` |
| `sim` | `sbt … MiCoSocSim --load-elf …` plus the correctness oracle; an embench CFU build must also show >= 1.05x reference cycles (`MIN_EMBENCH_SPEEDUP_X100` in `flow/gates.py`) | `sim` | `implement` |
| `cost` | `sbt … GenAgentCfu` elaborates the CFU alone (with the **same `--agent-cfu-*` flags as the SoC build and the simulation**, derived from the ISA spec's `datapath`/`memory`), then Yosys cost report from that standalone RTL | `cost` | `implement` |

- `--max-iters N` is the **verification-submission limit**: at most N candidates enter the validation ladder, including the first; the submission is counted at `generate_soc`, even if that candidate then fails an early validation rung. `--max-iters 1` permits one submission and stops if it fails. Repairs to analysis/ISA/oracle answers use the separate `--max-answer-repairs` pool (default 2); implementation reroutes before any candidate reaches validation use `--max-preverify-repairs` (default 2). The repair itself is not a submission; any candidate rerun must still consume a verification-submission slot.
- A repair re-runs the responsible node with the exact tool output (Scala compiler / elaboration errors, `make` errors, or simulator output) embedded in its prompt. An implementation rung that produced no usable patch is likewise re-prompted with its rejection reason.
- Repair feedback is labeled by source and all available sources are injected together: a hardware repair sees `hw_build`, `sim`, `cost`, and `implementation_hw` errors, and a software repair sees `sw_build`, `sim`, and `implementation_sw` errors; every repair prompt also carries a `PRIOR ATTEMPTS` ledger (response hash, applied/rejected, rejection excerpt) so a retry cannot silently repeat a dead strategy. A repair prompt enters an explicit `*_REPAIR MODE` and must emit a `REPAIR_DIAGNOSIS` (observed errors with file:line, root cause, repair mapping, regression risks) before any file or patch output; this contract is **enforced**: a repair answer without that section is rejected before anything is applied and re-asked with the missing-section error.
- The ISA contract repair carries the previous linter problems, so a rejected contract is corrected rather than regenerated blind, and every shape problem names what was received (`assumptions must be a mapping of name: text (received a list of 2 item(s))`) — the terse `assumptions must be a mapping` cost run `20261003-130005` ten answer-repair rounds on one repeated mistake.
- The exact injected feedback is persisted as `09_implementation/hw_repair_feedback.txt` / `09_implementation/sw_repair_feedback.txt` for every repair attempt, so a run is auditable against the LLM raw output.
- A rejected SEARCH block is reported with its exact location: the error names the SEARCH line and the current file line that differ, quotes both lines, and says whether an earlier pair already changed the region. Previously an unlocalized "SEARCH block not found" plus a parenthetical about earlier pairs led the model to misdiagnose the file as stale and resend the same edit.
- Both implementation rungs detect a stuck retry: each attempt records a `(response_hash, error_hash, error_text)` (in `sw_attempts` / `hw_attempts` and the gate). When the newest attempt is byte-identical to **any** earlier attempt and failed identically (so an A-B-A oscillation is caught too), the next repair prompt carries a `REPEATED IDENTICAL ATTEMPT` directive to switch strategy (Protocol B whole-file, or a minimal SEARCH), instead of burning the remaining budget on the same rejected answer. (The hardware rung previously lacked this guard; a model that emitted one broken answer three times exhausted the whole repair budget on it.)
- Attempts also record a `search_key` (the first content line of each SEARCH block), so a **rephrased** repeat of one approach is caught: the repair prompt carries a `REPEATED APPROACH DETECTED` directive and `_apply_agent_output` rejects a delivery whose SEARCH blocks repeat an already-rejected attempt before applying anything (observed 2026-09-29: three deliveries re-fixing one already-fixed compile error with different prose). A deterministically rejected delivery (SEARCH not found, missing deliverable, omitted `REPAIR_DIAGNOSIS`) additionally gets one corrective retry via `_delivery_retry_transition` that does **not** consume `--max-iters`; a delivered-but-broken design still does.
- Delivery parsing tolerates a terminator glued to the next token: a degenerate sampler occasionally restarts its answer template inside the same completion, so the response arrives twice with the second copy beginning at `EDIT_ENDREPAIR_DIAGNOSIS` instead of a newline. The parser splits that glue (only outside `identifier` continuations like `EDIT_END_foo`) and collapses byte-identical duplicate sections to the first copy, so a correct edit delivered in a doubled answer is applied once instead of rejected as "missing EDIT_END".
- Prompts show a rung's own target files exactly once: with a tool loop they are listed as paths to read (`read_file` returns the live overlay bytes an edit targets), and without one they are inlined in full under "Exact current contents" and omitted from the truncated repo-context digest, so a truncated copy can never be mistaken for the text to copy. `DELIVERY_PROTOCOLS` also tells the model not to wrap long comment blocks in a SEARCH and to copy any included comment byte-for-byte.
- **Every stage prompt is a stable-first brief**: the sections are ordered constant → variable so a provider's prefix cache can reuse the head, which is compared from the request's first byte. In order: `Task:`, `## Where you are` (repo root + run directory), the stage-independent key-path index plus access-tier paragraph, the shared digests every design stage renders byte-identically (`Normalized WorkloadSpec`, `Measured profile`, Embench delivery notes), and only then the stage block — `## Stage`, the tools the stage may call (with the tool budget), the deliverable, this stage's artifact paths and notes — followed by the per-call material (failure evidence, optimization state, the oracle contract). Repository file bodies are not inlined while a tool loop is available; the agent reads them by path. Only a run without that loop (`--no-agent-tools`, or no tool-capable endpoint) falls back to inlining the bodies the stage needs, after the deliverable, because a SEARCH/REPLACE delivery must be built from exact bytes. Regression guard: `agents/tests/test_prompts.py::PromptCacheLayoutTest` asserts everything before the first `## Stage:` is byte-identical across stages (run `20261003-090854` shared six characters; the real md5sum prompts now share 4,961). Every path that belongs to the current run is rendered with the virtual token `{{run-dir}}` instead of `agents/generated/runs/<run-id>`, so the task text is identical across runs as well: the tools accept the token (expanded to the real root *before* any access-policy check, so a token can never widen access) and tool results render it back, which means neither a prompt nor an observation ever shows a run id. The token works for reads, searches, listings, log lookups and edit targets; a misplaced or run-less token is a usage error, not a policy denial.
- Before answering, the analysis, ISA-contract, implementation and optimization rungs may run a **bounded tool loop** (`--agent-tools`, on by default; `--no-agent-tools` disables it). The budget is `--max-tool-rounds N` (default 20, model turns that may use tools) plus `--max-tool-calls N` (default 96, cumulative calls across all rounds; `0` disables the cap) and at most 4 parallel calls per round. Common tools are policy-checked `read_file`, `list_dir`, `search_text` and `list_command_logs` (the run's persisted build/simulation/synthesis logs, newest first); analysis can additionally use `inspect_observation(id, view=summary|asm|ir)` for available Embench observation IDs, and implementation repair can use bounded `inspect_failure`. The first-attempt software capability described above is the only mutation/check exception. Observation inspection returns bounded, explicitly uncertain reference-function ASM/Clang IR snippets; a missing cached-build disassembly says so and points at `view='ir'`. Within one implementation stage the model may edit each domain up to 6 times and check twice; a stage-gate rejection re-opens those local budgets, because the gate just asked for a change and a corrective round is already bounded by the loop's turn/call caps. Every call is printed in the CLI log as one short line that reports edit/delivery payloads by size (a malformed argument shape is rendered as a placeholder, never an exception: run 20261003/20261004 crashed the CLI in that renderer when the model sent `edits` as a list of strings) (see **Live progress** below) and written to the stage tool-call artifact (`07_analysis/analysis_tool_calls.yaml`, `08_isa/isa_tool_calls.yaml`, `09_implementation/{hw,sw,stage3}_tool_calls.yaml`, `12_review/optimize_tool_calls.yaml`), so model actions are auditable; the effective budget is recorded as `tool_policy` in `run.yaml`. If the endpoint rejects the tools payload, the step degrades to a one-shot prompt instead of failing.
- When the budget is exhausted the tools are withdrawn and a reminder is injected that the next reply must be the final EDIT/FILE/PATCH deliverable. That reply is then parsed as the answer, and an answer with no delivery block fails `implementation_hw` with an explicit `no EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN deliverable` error (which the repair prompt carries) instead of failing silently and being misreported as a build failure downstream.
- If the endpoint refuses the tools payload the request is retried once; if it still fails the step degrades to a plain completion that explicitly tells the model tools are unavailable, so it does not answer with an unexecutable "Let me read …" plan. The degradation is recorded (`degraded_reason` on the raw-output metadata, a `(tools unavailable)` entry in the tool-calls artifact, `tools_degraded` on the gate). Separately, any answer that only announces reading a file is detected and reported as a `tool-intent preamble` error, which the repair prompt carries, instead of being accepted as a finished design.
- While tool budget remains, a reply that cannot be used gets **one corrective re-ask** before the step is finalized: either the model announced a read without calling a tool, or a rung that requires a delivery block returned none (plan-only answer). The re-ask states the requirement and keeps tools available, so a dropped or forgotten tool call does not burn a ladder repair.
- The stages that own a workload also allowlist its source (`benchmarks/embench-iot/{src,support}/**`), so a file listed in the prompt's key-path index is never denied to the tool loop. The analysis rung additionally reads the profile build's disassembly (`workspace/**/*.asm`), so the hot loop can be analyzed from real instructions rather than inferred from C.
- Hardware-target reads are redirected to the run-workspace overlay: `read_file`/`search_text` on `src/main/scala/vexiiriscv/soc/mico/AgentCfu*.scala` return the run's live copy (the same bytes the prompt shows and an edit targets), so a repair sees its own previous change instead of the pristine tracked scaffold. The redirect only applies where the stage may read the overlay, so the analysis rung still reads the tracked scaffold.
- The build rungs are guarded by the implementation stage that feeds them: if `implementation_hw` applied nothing, `hw_build` fails without invoking `sbt` (the seeded scaffold would otherwise compile and "pass" while validating nothing) and repairs `implement`; if `implementation_sw` applied nothing, `sw_build` fails without invoking `make` and repairs `implement`.
- A failed rung enters the single implementation stage with current failure evidence. In that existing repair interaction the model may keep fixing `implement`, or, after an accepted `submit_repair_diagnosis`, use `request_repair_route(target, reason, failure_ids)` to return to `analysis` or `contract`. This adds no separate LLM routing call. Upstream selection rolls back the entire pending HW/SW/oracle transaction, invalidates the old ISA, conformance approval, downstream gates, reuse/fingerprints and selected candidate, and preserves old failure records as stale history. A rewind reuses an already-paid design-repair slot; otherwise it is bounded by `--max-preverify-repairs`, not the contract answer-retry pool. Plain delivery supports the corresponding structured `REPAIR_ROUTE` JSON; a model cannot silently edit a frozen contract or certify a gate.
- Hardware delivery and the independent `hw_build` preflight run a host-owned **bounded fixed-result lint**, even with `--no-conformance`: a compute/value instruction that returns a recognizable literal, constant alias/helper, or init-only fixed register is rejected as `hardware_integrity`. A dummy pointer read or waiting for `lsu.done` does not validate a constant answer. Configuration/status acknowledgements are not computational results. The guard is versioned in input fingerprints, so old hardware passes cannot be reused under a new rule. This is not a Scala compiler, full taint analysis, or arbitrary anti-cheat/semantic-equivalence proof; conformance and workload differential checks remain necessary.
- Embench simulation records **current-round typed evidence** (`terminal_reason`, `correctness_ok`, `timing_ok`, `design_changed`/`changed_domains`, `performance_valid`, reference/CFU result and cycle values) in the gate and validation artifacts. Execution/assertion failures and wrong/missing results take precedence over speedup thresholds. Missing cycle counts and invalid performance are `null`, never a fictitious zero-cycle measurement or a speedup for a wrong answer; a verified but slow design still records its measured ratio. The experiment collector/classifier consumes these reasons, with a conservative legacy fallback, rather than deriving the terminal failure from the entire accumulated error history.

- The implementation stage writes both sides before the ladder runs, so `hw_build`, `hw_conformance`, `sw_build`, `sim` and `cost` always validate what the stage actually delivered. Whether the kernel was rewritten is the stage's own decision; the ladder compiles whatever is in the workspace and still enforces the correctness oracle and the speedup floor.

- After Implement delivers hardware, software and the oracle, the `hw_conformance` rung runs the **smallest possible program** that exercises each declared instruction and compares it against an independent scalar reference, so a contract/encoding/writeback mistake is found before a kernel is written on top of it. Ownership is split three ways. The **harness** owns the C test (`conformance.render_conformance`) **and the test data** (`conformance.conformance_data`): the oracle declares only the *shape* of the input space and the generator produces a fixed-seed boundary sweep — `length` gets 0, 1, the half block and the full block, `index` gets 1/half/max, `value` gets 0/1/0x7FFFFFFF/0xFFFFFFFF — followed by deterministic random draws (`data.count`, default 4). No case argument is model-authored, so a run cannot pass on a hand-picked degenerate input set (run `20261005-073745` declared a single zero-length message and never transferred a source byte) and "the expected value is whatever my table says" cannot be arranged by choosing easy inputs. The **model** owns the scalar semantics: `reference` for a command that writes CPU `rd`, or `verify` for a command with no CPU result (a vector write or an acknowledgement) — that is **weaker evidence** (the model attesting its own hardware) and is recorded as such. A command that uses the vector register file must supply `setup`; the harness never guesses how to seed it. A command with CPU-value roles must declare `data.roles` (one descriptor per argument, rs1/rs2 order) and, when a role is an `address`, a `data.buffer`; the harness defines `<NAME>_CONF_DATA_BASE`/`<NAME>_CONF_DATA_BYTES` and owns the `cfu_memory_fence()` that orders the block before the DUT reads it, so `setup`/`reference` read exactly the bytes the DUT read. A missing `reference`/`verify`/`setup`, an unusable `data` block, a legacy `cases`, an unknown command name or a role-count mismatch is a **coverage failure**, and a missing `CFU_CONFORMANCE` line is a failure too: silence is never evidence. Lazy oracles are rejected by the validator: a `reference`/`verify` that is a compile-time constant (no identifier), that calls a `cfu_*` intrinsic (self-reference), that names an undeclared case argument, that compares an operand with itself (`a == a`), that (as `verify`) is built only from the case arguments, or that compares a value the same oracle's `setup`/`helpers` assigned a literal to fails with an actionable message, as does a `helpers` symbol defined by more than one command — **function, file-scope object or macro**, since any repeated name makes the harness-owned C fail to compile (run `20261003-130005`: two entries defined `md5_oracle_mem`). `data.buffer` is capped at 1024 bytes, `data.count` at 16 and each command's generated cases at 2048 transferred bytes (`conformance_data.CASE_BYTE_BUDGET`, kept in priority order: the full-block case first, then the sweep, then the draws), so every case is still a quick functional check; the harness always binds `a`/`b` at case scope, so a stale oracle cannot fail the generated C build. The rung is purely functional — it measures no cycles and asserts no performance; correctness is proved later by the `sim` rung and the workload's own oracle. Because a model-authored reference is unvalidated ground truth, the oracle may instead `#include` the workload's own pristine scalar implementation: `commands.conformance_scalar_sources` puts `workspace/embench_<bench>/reference/{src,support}` (the harness-owned plain copy that `embench.workspace_tampering` guards — never the writable `cfu` copy) on both the syntax preflight's and the build's include path, with the benchmark's own `-D` set, so the expected value can come from code the model did not write. The include is a single translation unit: `-ffunction-sections` + `--gc-sections` drop the benchmark's `benchmark_body`/`workspace` code, so the run's `main` stays the only entry point (verified with a real `make -C sw … compile`). It is offered, not required — a config/reset/acknowledgement command has no workload counterpart and still needs its own independent expression. A **completed differential mismatch** is unattributed evidence: nothing validates the model's scalar reference, so a wrong reference makes correct hardware look broken (run `20261005-073745`: an oracle whose MD5 K table held 63 of 64 entries failed three consecutive rounds, and the last one reported a *correct* DUT). It therefore repairs `implement` in conformance scope with `oracle_unattributed: true` and no forced hardware edit, and an oracle-only delivery is charged to the separate answer-repair budget (see the repair-budget note above); a malformed/unbuildable oracle is an answer failure outright, and a watchdog/hang without a verdict stays design-scope. The generated C, the oracle and the verdict are kept under `09_implementation/conformance/`.
- The oracle declares the input space, never the values: `data.buffer` (the harness-owned byte block the DUT reads), `data.count` (random draws on top of the boundary sweep) and `data.roles` (one `kind: address|length|index|value` descriptor per CPU-value argument, rs1/rs2 order, with `max` for `length`/`index`). The validator rejects a legacy `cases` block, a model-supplied `data.seed`, an `address` role without a buffer, a `length` bound larger than the buffer, a buffer that is not a ≤1024-byte multiple of 4, an out-of-range `count`, an unknown kind (listing the vocabulary) and a role-count mismatch, each naming the fix. The oracle's C may call only the **generated intrinsics named in the prompt's command table** (plus `cfu_enable`/`cfu_memory_fence`, and the harness `<NAME>_CONF_DATA_BASE`/`<NAME>_CONF_DATA_BYTES` macros); an invented name such as `cfu_load` is caught by a static lint before the build (`unknown intrinsic cfu_load (did you mean cfu_vector_load_i32?)`) instead of surfacing as a bare linker error. Conformance build/sim failures embed the compiler/linker diagnostics in the repair feedback (previously only the `make` command line reached the model, so it repaired blind), and a **recurring** rejection — validator failure or build failure — is flagged as a `REPEATED ORACLE REJECTION` with the correct shape pinned. `run.yaml` merges artifact refs across writer instances, so every oracle attempt's raw answer stays registered and numbered (`conformance_000/001/...`) instead of being dropped and overwritten.
- With `--loop` a run does not have to stop when it succeeds (see **Multi-instruction ISA evolution** below for the current exploration and delivery rules). If every gate passes **and** a `--max-iters` verification-submission slot remains, the review node asks the agent (stage `optimize`, plus at most one stop reconsideration) whether another round is worth spending on: `{"continue": true|false, "target": "implement_hw"|"implement_sw"|"extend_isa", "focus": "performance"|"cost", "reason": ...}`. A round re-enters the normal ladder (`implement_hw` → `generate_soc` → software reuse-or-rewrite → `build_sw` → `simulate` → `estimate_cost` → review, or `implement_sw` → `build_sw` → …), so it can never skip verify, the 1.05x speedup floor, or a build rung. Each optimization candidate consumes one verification-submission slot: `--max-iters N` stays the bound on candidates sent through the ladder, and a round with no slot left, an unusable target, or a missing "concrete change" reason stops the loop deterministically; in default explore mode a `continue: false` is reconsidered once before the framework accepts it (no model answer can extend a run). The chosen rung receives an `=== OPTIMIZATION ROUND ===` block (not a `*_REPAIR MODE` block) with the current measured result, the focus, and the rule that the accepted ISA revision is frozen during implementation; only `extend_isa` may append compatible commands before re-entering hardware and software validation. Every round that produced its own measurement is recorded in `12_review/optimization_rounds.yaml` (speedup, reference/CFU cycles, Yosys cells/wires/memories), and `simulate`/`estimate_cost` stamp their gates with the round they ran in, so a round that failed before measuring is listed as `not measured` instead of repeating the previous round's numbers. The report's `## Optimization Rounds` section and the end-of-run CLI `summary:` each print one line per round with the change against the previous measured round — for example `round 1: implement_hw speedup=2.50x cells=1300 (+0.50x, +100 cells) (best)` — so the whole progression of speedup and cost is traceable; the first round is the initial ladder. A round counts as an improvement only when it is Pareto-better than the best round so far (not slower, not larger, better on at least one axis); a final measured round that regresses is reported with an explicit warning in the report and in the CLI summary. `--loop` is off by default (env `CFU_AGENT_LOOP`); without it, review stops as before.
- The cost rung synthesizes the CFU **in isolation** rather than the whole SoC: `GenAgentCfu` (a tracked, non-agent-editable entry point) elaborates only the overlaid `AgentCfu` class into `11_cost/cfu_rtl/AgentCfu.v`, and Yosys runs `synth -top AgentCfu` on it. Cost numbers therefore track the agent's CFU design instead of the CPU/SoC, and the cost rung needs no CPU build. If the standalone generation produces no RTL, the cost rung fails and repairs `implement_hw`.
- The simulation rung compares final hardware-overlay and software-source hashes with an immutable snapshot captured before the first implementation attempt. A repair may change either side or both; `no_design_change` is raised only when both still match that initial baseline. The verdict does not depend on the latest repair's touched-file list, so an earlier HW edit or a software-only repair is not lost when a later round leaves the other side untouched.
- A rung that was skipped (`--dry-run`, `--skip-sim`, execution disabled) never repairs and can never produce `complete`. A rung **disabled by an ablation flag** (`--no-conformance`, `--no-isa-contract`) is different: it is recorded as never validated but out of scope, so it neither fails the run nor blocks `complete` (see **Ablation flags**).
- A hanging simulator (`TimeoutExpired`) or a missing toolchain is a failed rung, not a crashed run.

```bash
# Planning pass (read-only): writes run artifacts, never mutates source or runs commands.
python3 agents/run_cfu_agent.py \
  --task "Accelerate a uint8 weighted average kernel with a TileLink CFU. It reads x[i] and w[i], computes sum(x[i] * w[i]) / N, arrays are 32-byte aligned, and products should stay resident inside the CFU." \
  --dry-run --skip-sim --skip-yosys
```

By default the agent is **autonomous**: it may apply an LLM patch to CFU/software
allowlisted paths and run allowlisted build/simulator/Yosys commands. To restrict
it, opt out explicitly:

```bash
# Force read-only: never mutate source or run external commands.
python3 agents/run_cfu_agent.py --task "..." --dry-run

# Keep patch application but never execute allowlisted commands.
python3 agents/run_cfu_agent.py --task "..." --no-run-commands

# Record LLM deliverables as proposals only; allowlisted commands still run.
python3 agents/run_cfu_agent.py --task "..." --no-apply-patch

# Restrict what the agent may read (default: standard).
python3 agents/run_cfu_agent.py --task "..." --visibility basic
```

`--apply-patch`, `--run-commands` and `--no-dry-run` remain accepted as no-ops so
older invocations keep working; they no longer change anything.

### Ablation flags

Three `--no-*` switches each remove exactly one capability, for ablation experiments.
All are **off by default** (the full flow). What a switch disabled is never silently
absent: it is recorded in `run.yaml`, the report's `## Ablation` section and the CLI
`summary:` block, and the affected rung gate carries a `disabled` marker instead of a
verdict. A `disabled` rung is excluded from the completeness requirement, so an
otherwise passing run still reaches `complete`; a **real** skip (`--dry-run`,
`--skip-sim`, execution disabled) keeps the honesty rule above and stays `planned`.

```bash
# Remove all LLVM IR content from the agent: forces --ir-features off (no .ll /
# feature-XML export) and narrows inspect_observation to view=summary|asm.
# The asm/summary evidence stays, so the ablated variable is the IR itself.
# Conflicts with --ir-features require|approx (explicit IR evidence is refused).
python3 agents/run_cfu_agent.py --task "..." --no-llvm-ir

# Drop the conformance requirement: Implement no longer authors/delivers the
# oracle (write_conformance/check_conformance_syntax disappear from its tools and
# its gate), and the hw_conformance rung is recorded as disabled. The sim rung
# still enforces the workload's own correctness oracle and the speedup floor.
python3 agents/run_cfu_agent.py --task "..." --no-conformance

# Skip the ISA contract stage entirely: no contract LLM step and no isa_spec.
# The agent designs the instructions itself and writes cfu_intrinsics.h from the
# tracked intrinsic template (skills/cfu-designer/assets/templates/cfu_intrinsics.h,
# inlined into the prompt so --visibility basic works) with free encodings
# (custom0 opcode 0x0B; funct3/funct7 its choice), keeping the RTL consistent.
# The header is an ordinary editable software target: seeded once from the
# template and never overwritten by the harness afterwards. Implies
# --no-conformance (the conformance oracle consumes the contract), and --loop
# loses the extend_isa target (only implementation rounds remain).
python3 agents/run_cfu_agent.py --task "..." --no-isa-contract
```

State fields `llvm_ir`, `require_conformance` and `isa_contract` (all default
`True`) carry the switches; every read defaults to enabled, so states built without
them keep the full flow.

### Visibility levels

`--visibility {basic,standard,extended}` (env `CFU_AGENT_VISIBILITY`; CLI default `extended`, policy fallback `standard`)
controls **which files the agent may read** — the key-path index it is shown and the read-only tool loop
(`read_file`/`list_dir`/`search_text`). The tier is flow-wide: every stage of a run sees the
same tree. It is a read-only gate: writes, command allowlists, and the effect flags
(`--dry-run`, `--no-apply-patch`, `--no-run-commands`) are unchanged, and the write
confinement to `workspace/design/AgentCfu*.scala` still applies at every level.

| Level | Adds | Still hidden |
| --- | --- | --- |
| `basic` | workload/benchmark sources, the CFU contract (`CfuBus`/`CfuPlugin`/`CfuLib`), `AgentCfu*`/`GenAgentCfu`, `MiCoSoc*`, the ISA template, `sw/tests`, run artifacts | the rest of `src/main/scala/vexiiriscv/**`, skills, Vpu/BitNet example CFUs and `ext/SpinalHDL/**` |
| `standard` (policy fallback) | cfu-designer skills and the full `src/main/scala/vexiiriscv/**` source tree | Vpu/BitNet example CFUs and `ext/SpinalHDL/**` |
| `extended` | `ext/SpinalHDL/**` for the whole flow, with prompt guidance to confirm the SpinalHDL API before writing hardware | Vpu/BitNet example CFUs remain denied |

- The gate is declarative: `agents/specs/agent_access_policy.yaml` has a `visibility` section
  whose `gated` globs list paths requiring `standard` or `extended`; `exemptions` preserve
  selected lower-tier subtrees, while top-level `deny` rules always win. A path outside the
  shared `read` tree stays unreadable regardless of level.
- Hiding a context file is expected at a lower level, so `load_context` records it under
  `withheld_files` in `01_context/manifest.yaml` (and does **not** fail the gate); only a real
  policy denial lands in `denied_files` and fails it. The chosen level is written to the
  manifest, `run.yaml`, the CLI summary (`visibility=`), and the report's `## Visibility`
  section.
- A read blocked by the tier returns an actionable policy error
  (`requires visibility level 'standard' (current: basic)`), and the level-specific prompt
  block tells the model not to assume the contents of a document it was not shown.
- `standard` does not pre-read the full VexiiRiscv subtree; it makes the tree available for
  on-demand source lookup. `extended` likewise does not pre-read `ext/SpinalHDL` (925 sources);
  it only makes the tree readable so the agent can look up the exact API on demand. With
  `--no-agent-tools` these trees are unreachable, and the prompt says so.

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
non-improving optimization results, and `--max-iters` remains the global verification-submission/
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

`agents/specs/agent_access_policy.yaml` is the single source of truth for what the agent may read/write and which commands it may run:

- **Reads are flow-wide.** The policy has one top-level `read` tree that every stage shares: a run's knowledge is not partitioned by stage, so `analysis`, `isa` and `implementation` see exactly the same paths. Declaring `read` inside a stage is a policy error rather than a silent narrowing.
- **Only writes and commands are stage-scoped.** Each stage declares its own `write` allow list and command ids under `stages:`.
- Deny rules always take precedence; paths are resolved to repo-relative form and reject absolute paths, `..` escapes, and symlinks outside the workdir.
- Secret, `.git` (including a submodule's `.git` pointer), build/binary (`*.elf`, `*.o`, `*.bin`, `*.map`, `*.vcd`, `*.fst`), bytecode, and non-allowlisted repo source are not exposed to the agent. Disassembly (`*.asm`) is not globally denied: it is text and is readable under the run workspace and `sw/tests/**`, while build trees stay denied. Historical runs under `agents/generated/runs/` are never auto-read: the read tree is scoped to the executing run.
- The tools traverse the same tree they may read. `list_dir` lists sub-directories (a directory is never itself a readable path, so it is listed whenever a readable file may live below it), and both it and `search_text` prune denied/gated subtrees before those can consume an enumeration budget. A partially completed search returns the hits it already found plus the reason it stopped.
- Visibility is a second, read-only gate on top of the read tree: the policy's `visibility.gated` globs mark paths that require `standard`/`extended`, and `--visibility` selects the level (see [Visibility levels](#visibility-levels)).
- Commands are declared by id (e.g. `make_profile`, `soc_sim`, `cfu_generate_agent`, `yosys_cost`, `git_apply`). Shell metacharacters, absolute paths, and sensitive paths in command argv are rejected; the executable and required tokens must match the declaration.
- Patch application (enabled by default; disabled with `--no-apply-patch` or `--dry-run`) is
  required for **all three** delivery protocols. Without it, SEARCH/REPLACE
  edits, `FILE_BEGIN` blocks, and unified diffs are recorded as proposals only; the run never
  writes LLM-authored content into the workspace (the generated benchmark copy is framework
  setup, not agent output). `--no-run-commands` alone only disables command execution and
  leaves patch application on.
- Hardware writes are confined to `workspace/design/AgentCfu.scala` and `workspace/design/AgentCfuFiber.scala`; `src/main/scala/**` is read-only for every stage.

### Run artifacts

Every run gets a unique directory `agents/generated/runs/<run_id>/` and writes:

```text
01_context/manifest.yaml          # visibility level, visible/withheld/denied files
02_workload/workload_spec.yaml    # normalized workload spec
03_route/route.yaml
04_benchmark/manifest.yaml        # benchmark/project profile manifest and commands
05_profile/result.yaml
06_hotspot/hotspot.yaml
07_analysis/workload_analysis.yaml
08_isa/cfu_isa.yaml + cfu_isa.md  # structured CFU ISA spec + rendered markdown
09_implementation/hw_plan.md, hw_patch.diff, hw_changed_files.yaml
09_implementation/hw_tool_calls.yaml       # read_file/list_dir/search_text transcript
07_analysis/analysis_tool_calls.yaml       # analysis-rung read/search/inspect_observation transcript
08_isa/isa_tool_calls.yaml                 # ISA-contract read/search transcript
09_implementation/conformance_tool_calls.yaml  # conformance-oracle read/search transcript
12_review/optimize_tool_calls.yaml         # optimization-planner read/search transcript
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

Each artifact records its stage, status, inputs, visible files, SHA-256, size, and timestamp. The benchmark workspace is scoped to the run so runs do not clobber each other, and C-project inputs are always copied before instrumentation so the original source tree is never modified. A checkpointed run also stores `checkpoint.sqlite` and `operation_journal.json` under the same run root. Current stage-input fingerprints mark affected downstream gates stale when upstream analysis, ISA or design sources change; resume refuses any previously passing gate whose source bytes no longer match. This is an evidence freshness guard, not a substitute for recomputing the independent gates.

### Stage freedom and terminal decisions

The macro stages in `docs/agent_v2_plan.md` specify deliverables and gates, **not a fixed order of model actions**. Within Stage 3 the agent may, for example, author a development conformance test before editing the hardware, or alternate reads, HW/SW edits and checks. The framework-owned `hw_build`, `hw_conformance`, `sw_build` and workload correctness gates remain independent and must pass after final submission.

### Answer contracts

Two contracts are easy to get wrong and expensive when they are, so both are stated explicitly and checked by the framework:

- **The analysis answer is raw YAML.** `workload_prompt` says the reply is one YAML mapping and nothing else, because the next node parses it directly: a prose preamble or a ` ```yaml ` fence invalidates it. `parsing.parse_yaml_mapping` still tolerates a fence and a preamble for compatibility (the same candidate list the ISA contract accepts), but a genuinely malformed mapping is never guessed at. A failed analysis is an **answer** error: it is repaired from the answer-repair pool (`--max-answer-repairs`) with the parser's complaint and the reply's own opening lines quoted back, and when that budget is spent the run stops as `blocked` instead of handing `{}` to the ISA and implementation stages.
- **The implementation deliverable is an applied edit, not a patch.** `edit_hardware`/`edit_software` write the workspace files; `submit_stage(delivery)` summarizes what changed. The gate rejects a submission that reports no applied change to a hardware and a software target, returns that specific reason, and grants one corrective retry before the stage fails — so "here is the full file" cannot end a run. The prompt names the editing tools, and no tool-loop stage carries the legacy `FILE_BEGIN`/`PATCH_BEGIN` protocol, which only applies when the tool loop is disabled.

The per-stage **evidence index** (`## Key paths`) is filtered to artifacts that already exist, so it never points at a file the stage is about to write or at a name that only exists for another project kind (for example `04_benchmark/manifest.yaml` versus `04_benchmark/embench.yaml`).

### Status semantics

- `planned`: a planning pass completed. Either it was a read-only pass, or the execution gates were skipped (execution disabled, `--skip-sim`, `--skip-yosys`). The report lists each gate and what was skipped. This is **not** a claim of correctness or execution.
- A gate that was skipped is recorded as `{"ok": false, "skipped": true, "reason": ...}`. Skipped gates never produce `failed` and can never produce `complete`.
- `complete`: every rung (workload spec, ISA contract, hardware design, SoC generation, software design, compile, simulation, cost) produced evidence and passed. Only this is a real completion.
- `failed`: a rung failed and the applicable verification-submission or repair budget was exhausted. The run stops at that rung (`stopped_at` in the report) and skips the rest.
- `blocked`: an input/ISA contract could not be validated and no applicable answer/pre-VERIFY repair budget remained, so the loop cannot proceed honestly.
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
- Each tool call prints exactly **one short line** and never its payload:
  ```
      [tool] edit_hardware(edits=[AgentCfu.scala search=2048 replace=4096]) ok -> 812 chars
      [tool] edit_software(path=crc_32.c search=138 replace=666) ok -> 940 chars
      [tool] submit_stage(delivery=<3463 chars>) ERROR -> 61 chars
      [tool] read_file(path=src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala, start_line=399, end_line=560) ok -> 6104 chars
  ```
  Edit bodies, deliveries and oracles are reported by **size** so a run's own
  progress is not buried behind file content; the full arguments and results
  stay in the stage tool-call artifact.
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
- Context compaction is announced explicitly, because it changes what the model
  can see: once the estimated budget reaches 90% of the window, completed
  historical tool batches are replaced by a memo and the run prints a durable
  line (a real line, not the in-place progress line, so it survives `tee`/CI):
  ```
      [context] context compacted: removed 3 completed tool batches, est. 121k -> 78k tokens (level ok, stage analyze_workload)
  ```
  The same count is appended to that call's token line
  (`[context compacted -3 batches]`), stored in the raw-output metadata as
  `context_compaction`, and recorded as a `compaction` event in
  `request_audit.jsonl`. Crossing 80% prints one `[context] context N% full`
  warning per climb, so an imminent compaction is visible before it happens.
- Because the HTTP read timeout applies per streamed chunk instead of to the whole body, a generation that would exceed `timeout` no longer fails. Measured example: a GLM-5.2 call produced 2171 output tokens over 118.6 s with the config's `timeout: 120` and completed, with 264 live progress updates and 4406 characters of thinking captured from the stream.
- Tool calls stream too: content, `reasoning_content` and `tool_calls` fragments are accumulated, and the provider's final usage chunk gives the exact counts.
- If an endpoint ignores or rejects `stream`/`stream_options`, the call retries without streaming and records `fallback_reason` ("streaming unavailable …" / "streaming ignored by the endpoint") in the raw-output metadata and the `[llm]` line.
- This is separate from LangGraph's `--stream`, which streams whole-state snapshots between nodes (`stream_mode="values"`); token streaming happens one level lower, inside the LLM client.

**Completion budget.** Every request is capped by `output_reserve_tokens` from the model config (shipped configs: `32768`; the framework default is `4096`). The same number is sent as `max_completion_tokens` (chat) / `max_output_tokens` (responses) *and* used as the context budget's output reservation, so it must satisfy `0 < output_reserve_tokens < context_window_tokens` and it also shifts the 80/90/100% `warning`/`compact`/`unfit` thresholds. Precedence: `--max-output-tokens N` (run-level, `0` keeps the config value) > `CFU_AGENT_OUTPUT_RESERVE` > the config file. The effective value is recorded in `run.yaml` as `llm_request.max_output_tokens`. A reasoning model shares this budget with its thinking tokens, so a small value can produce a **200 response with no content**; the tool loop re-asks **once** for an empty answer (it is a transport failure, not an answer) and fails the stage only if that retry is empty too. See `agents/llm_configs/README.md` for the field semantics.

At the end the CLI prints a `summary:` headline block with the measured acceleration and cost, then cumulative token usage and per-stage results:

```
summary:
  cycles_before = 7853884
  cycles_after  = 4712330
  speedup       = 1.66x
  yosys_cost    = top=AgentCfu cells=3845 wire_bits=4402 memories=0 memory_bits=0
```

`cycles_before`/`cycles_after` are the reference and CFU-build cycles of the profiled embench target and `speedup` is their ratio (the same `speedup_x100` as `10_validation/embench.yaml`); they print `n/a (<reason>)` when the simulation was skipped, never ran, or the workload has no reference cycle count. `yosys_cost` is the parsed summary line of the last successful estimator run, so it prints `n/a` when cost was skipped or failed. The `report.md` includes `## Token Usage` (run totals **and a per-stage table sorted by input tokens**, so an input-heavy stage is immediately visible) and `## LLM Raw Outputs` sections. Each LLM call's returned text is persisted under the run's `llm_raw/` directory; retries receive a new numbered file. Inline reasoning (`<thinking>` tags, `think` blocks, or a provider reasoning field) is split out of the answer before parsing and stored beside it as `*_<n>.reasoning.txt`, so artifact parsers only ever see the final answer. The rendered system/user prompt of **every** call — including the workload-spec call, and with this run's paths already collapsed to `{{run-dir}}` — is persisted beside the answer as `*_<n>.prompt.txt` (with `prompt_sha256` and `prompt_chars` in the artifact metadata, and a `prompt_oversize` flag above ~40k chars) and listed in the report's `## LLM Prompts` section, so a run can be replayed and inspected after the prompt code changes; the error log injected into a repair prompt is kept separately as `09_implementation/{hw,sw}_repair_feedback.txt`. Token accounting is per stage everywhere it is shown: the live `[i/N] stage ... (in=... out=... tok, ...s)` line, the live `[llm] label -> in=... out=... tok` per call, and the end-of-run CLI `summary:` block (`llm_tokens:` one line per stage sorted by input tokens, with a `TOTAL` line).

### Embench-iot workloads

Any embench-iot benchmark can be used as the workload input:

```bash
# The implementation stage mutates the run workspace and runs the toolchain;
# both effects are enabled by default (add --dry-run for a read-only planning pass).
conda run -n mico_env python3 -B agents/run_cfu_agent.py \
  --embench crc32 --skip-yosys --max-iters 1

# Same workload, but keep improving a passing design: up to two more rounds
# (--max-iters 3) chosen by the agent, each fully re-validated by the ladder.
conda run -n mico_env python3 -B agents/run_cfu_agent.py \
  --embench crc32 --loop --max-iters 3
```

The run copies `benchmarks/embench-iot/{src/<bench>,support}` plus the tracked
VexiiRiscv board (`sw/embench/`) into `<run>/workspace/embench_<bench>/` and builds
three configurations, each with its own source tree:

| config | defines | evidence |
| --- | --- | --- |
| reference | board only | Embench verify passes + baseline cycles |
| profile | `-DEMBENCH_OBS` | per-loop inclusive cycles/calls plus containing-function context |
| cfu | `-DUSE_AGENT_CFU` | verify still passes + speedup vs reference |

**Measurement protocol (future runs):** new Embench runs default to
`cold_single_invocation_v1`: one cold invocation, no warmup, and both local and
global scale factors fixed to 1. The timed invocation includes the original
preparation, memory work and CFU configuration; it is **not pure math-kernel
latency**. `benchmark_project.measurement` records `protocol`,
`local_scale_factor: 1`, `global_scale_factor: 1`, and `warmup: false`; the sim gate
and validation artifact retain this evidence. CLI/report summaries describe the
scope, and experiment records/CSV preserve `measurement_protocol` and the metadata.
Evidence without a protocol is labelled `legacy`. **New and old measurements are
not comparable**: mixed-protocol complete runs suppress only the speedup median,
with a warning; pass@k, effort and cost statistics are unchanged. Existing generated
artifacts are not migrated. Legacy Embench checkpoints cannot resume under the
new protocol: start a new run (with a new run ID).

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

**Protected benchmark scaffold:** the agent keeps the same tools and delivery
flow, with no extra model-generation stage. It may optimize kernel functions and
`benchmark_body`, but must preserve the workload's protected preparation, scaling,
benchmark wrapper and oracle scaffold. Harness-owned `board/*`, `support/*`, and
`embench_main.c` also remain protected. The host records guard metadata under
`benchmark_project.integrity` and checks both pristine/reference evidence and the
CFU copy; `benchmark_integrity: {version, ok, problems}` records the integrity gate
verdict separately from simulation success. Independently, the injected
`EMBENCH_VERIFY` line is parsed on both sides: different reference/CFU `result=`
values fail the gate without trusting `verify_benchmark()` alone.
These byte/scaffold checks and benchmark oracle results are **not proof of true
operation equivalence**, nor an arbitrary semantic anti-cheat guarantee: a passing
integrity gate does not establish that every possible input computes the same
operation. CLI/reports distinguish the gate verdict from that stronger claim.
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
the git revision/submodules, the normalized loop observation plan, the explicit
measurement protocol (cache key version 7), and the instrumenter/board-counter source
hashes, so changed protocol, instrumentation, toolchain or scaffold invalidates it.
The already-hashed pristine reference sources include the workload preparation;
old version-6 entries cannot supply a new-protocol baseline.
Entries live in the git-ignored `agents/generated/cache/embench_profile/<key>.json` and
are written only after a passing measurement. The same key also names a copy of the
reference build's disassembly (`<key>.embench_main.asm`): it is a *build output*, so a
reused measurement has none of its own, and the analysis reads it as the ASM candidate
for a measured loop. A hit restores it into `workspace/embench_<name>/reference/` and
records `profile_cache: {status: hit|miss|refresh|disabled, key, saved}` in
`05_profile/embench.yaml`, `run.yaml` and `iteration_history`, so a cached baseline is
never mistaken for a fresh one. A hit runs no `make`/`sbt`; that is also why
`inspect_observation(..., view='asm')` says the disassembly is a build output and points
at `view='ir'` when a cache entry predates the stored copy.

**Closed loop (LLM):** the analysis prompt receives the normalized spec and the
measured loop/function profile digest, and the contract and software prompts point
at `07_analysis/workload_analysis.yaml`, `08_isa/cfu_isa.yaml` and the generated
`cfu_intrinsics.h` on disk instead of inlining them; the benchmark source and the
loop's source location are read by the agent with `read_file`. The LLM is asked to identify the
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

Each config may set `model`, `base_url`, `api_mode` (`auto`/`responses`/`chat`), `temperature`, `timeout`, thinking (`thinking`, `reasoning_effort`, `reasoning_summary`, optional `thinking_body`; **off by default**), the request-size planning values (`context_window_tokens`, `output_reserve_tokens`), and an inline `api_key` or an `api_key_env` variable name. Model resolution order (highest wins) is: explicit `--model` > config `model` > `CFU_AGENT_MODEL` > built-in `gpt-5`. A config can leave `model`/`base_url` empty and still pick up `CFU_AGENT_MODEL`/`OPENAI_BASE_URL`. Prefer `api_key_env` to keep keys out of the tracked JSON. `default.json` is optional: if missing, the framework builds an environment-only config. See `agents/llm_configs/README.md`.

`api_mode` governs the tool-using rounds as well as plain completions. The agent keeps one chat-format transcript (budget compaction, source masking and the request audit all depend on it) and translates it to Responses `function_call`/`function_call_output` items only on the wire, so a stage behaves the same on either endpoint. `auto` prefers Responses and falls back to chat completions once when the route is missing or the model refuses function tools there; the reason is kept in `fallback_reason`. Models that only accept tools on chat completions (`Kimi-K3`) should pin `api_mode: chat`; pinning `responses` surfaces a rejection instead of silently degrading.

### Environment

Run the agent and its tests from `conda run -n mico_env` (or an activated `mico_env`): the environment must provide LangGraph, `langgraph-checkpoint-sqlite` (3.1.1 for persistent runs), and the configured HW/SW tools. Missing core dependencies are errors, not a switch to another runner. `--checkpoint --run-id <unique_id>` writes run-scoped SQLite checkpoints; `--resume <id>` verifies the run manifest, artifact hashes and operation journal before resuming an unfinished checkpoint. Graph nodes are journaled as replay units, and RepoToolbox workspace writes/commands also have operation records: a crash mid-node leaves an uncertain pending record and blocks automatic resume until reconciled. Other direct external effects still need explicit review before claiming crash-safe recovery. Checkpoints are not exactly-once transactions; do not use the same run ID concurrently.

- `OPENAI_API_KEY`: required for model-authored analysis, ISA and implementation in execution mode; an unavailable model must not produce a validated candidate. Dry-run may record planning hints without claiming complete execution.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible endpoint.
- `CFU_AGENT_MODEL`: optional default model name.
- `CFU_AGENT_LLM_CONFIG`: optional default config name (otherwise `default`).
- `CFU_AGENT_LLM_API`: optional `auto`/`responses`/`chat` override.
- `CFU_AGENT_VISIBILITY`: optional default visibility tier (`basic`/`standard`/`extended`); otherwise `standard`.
- `CFU_AGENT_PROFILE_CACHE`: optional default profile-cache mode (`on`/`off`/`refresh`); otherwise `on`.
- `CFU_AGENT_IR_FEATURES`: optional default LLVM IR evidence mode (`auto`/`off`/`require`/`approx`); otherwise `auto`.
- `CFU_AGENT_REPAIR_ROUTING`: optional default simulation-repair routing (`auto`/`fixed`); otherwise `auto`.
- `CFU_AGENT_SOFTWARE_CHANGE`: optional default software-rung policy (`auto`/`always`); otherwise `auto`.
- `CFU_AGENT_LOOP`: optional default for `--loop` (`on`/`off`); otherwise off.

### Tests

Regression tests never contact a real LLM. Test-package import blocks real OpenAI/Azure
SDK construction, including localhost endpoints, and outbound network connections;
there is no `CFU_AGENT_TESTS_ALLOW_NETWORK` opt-in. Inject a hand-written fake SDK/client
or the deterministic offline client. Prompt, route, gate, transport and audit tests
exercise local fixtures/mocks only; new cases must not launch live design runs, model
requests, or another full-ladder integration scenario.

```bash
# Use the project environment; tests patch model/network access and use unittest.
# The suite is kept near 15s: it is a fast contract suite, not a second
# integration harness. Inspect fixed agents/generated/runs/ test IDs before
# running it: several older tests delete their own fixed run directories.
conda run -n mico_env python3 -B -m unittest discover -s agents/tests -t . -p 'test_*.py'
```

Two modules carry the flow itself and are the only ones that run the real graph:

- `test_graph_routing.py` — the ladder decision table. A router is a pure
  `state -> node` function, so every branch (per-rung repair targets, the
  software-reuse shortcut, simulation repair escalation, the LLM repair router
  and its deterministic fallback,   direct case instead of by a full run.
- `test_graph_smoke.py` — exactly one offline end-to-end dry run, asserting the
  honest `planned` outcome, the report and `run.yaml`, the planned (unwritten)
  design overlay, IR evidence, and that every manifest artifact is
  content-addressed on disk.

Everything else is a focused unit test of one contract. Add new coverage at that
level: a full-graph test costs seconds and still only observes the route it took,
so per-scenario ladders are deliberately not maintained.

The `unittest` suite checks path traversal, deny precedence, command allowlisting (including the agent-overlay command and positional-path confinement), visibility tiers (policy gating, withheld-vs-denied context loading, tool denial, level-specific prompt text, and CLI threading), run isolation, `WorkloadSpec`/CFU ISA validation, gate honesty, ladder routing (per-rung repair, the single implementation stage's repair routing, the `--loop` optimization ledger with its Pareto improvement test and budget bound, hard stop without budget, skipped-rung honesty), dry-run flow without any external toolchain or API key, and the targeted-edit delivery protocol (SEARCH/REPLACE parsing, unique-match enforcement, atomic multi-target application, and an end-to-end hardware rung that applies an edit instead of a whole-file rewrite).

Run it inside `mico_env`: that environment has `langgraph`, so the LangGraph runner is the live path and its tests are not skipped. The suite exercises LangGraph's native state reducers and routing; there is no parallel pure-Python fallback runner.

Ready-to-run examples are under `agents/examples/`.

### Vivado FPGA evaluation and parallel validation

`--cost-backend vivado --fpga-part xcau25p-ffvb676-2-e --target-frequency-mhz 100`
selects real whole-SoC FPGA synthesis and PnR, with CFU hierarchy PPA attribution
(no second standalone CFU run). New runs default to Artix UltraScale+ AU25P,
FFVB676, speed grade -2, extended temperature (`xcau25p-ffvb676-2-e`) at 100 MHz.
An explicit `--fpga-part` overrides this choice; existing checkpoints are not
retargeted. Yosys remains the default fast cost proxy.

**Cost Wall:** both backends enforce `--max-yosys-cells N` (positive integer,
default **100000**) after the fixed generic Yosys mapping/ABC flow. Agent runs
measure standalone `AgentCfu`; this is a mapped generic-cell engineering budget,
**not FPGA LUT count or device capacity**. Equality passes. An over-budget design
fails with `cost_budget_exceeded` **before Vivado starts**, records the measured
cells/limit/report path, and returns through the existing bounded implementation
repair path with instructions to reduce lanes/unrolling and share arithmetic
across cycles without weakening correctness. Missing/invalid cell statistics fail
closed as `cost_metrics_invalid`. Analysis, ISA, hardware/unified implementation
and optimization prompts carry the actual run limit before design work begins.
Dry runs and explicit cost skips do not claim a budget pass. The limit is recorded
in run/checkpoint state and cost fingerprints; resume cannot change it or reuse
pre-wall/stale cost evidence. Start a fresh run to change a recorded budget.

New CLI runs execute simulation and EDA with `--validation-mode parallel` by
default: first command/semantic failure cancels the sibling process tree and
spends at most one repair slot. `--validation-mode serial` is available for
troubleshooting. Only bounded termination cleanup is awaited, not normal sibling
completion. `--skip-cost` skips either backend; `--skip-yosys` remains Yosys-only.

The reports distinguish whole-SoC and CFU hierarchical area/timing/dynamic power,
vectorless estimates from board measurement, and current from historical PPA.
Input hardware RTL is frozen and hashed before EDA. See
[the Vivado/parallel-validation guide](../docs/cfu_agent_vivado_ppa.md) for exact
options, artifact layout, recovery semantics and validation limits.
