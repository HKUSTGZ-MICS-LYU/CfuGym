# CFU Agent Stage Executor migration

This document records the implementation state at the start of the control-plane / execution-plane refactor. It complements `overall_revision_on_agent_flow.md`; the current source and `agents/README.md` remain authoritative when older plans disagree with the implementation.

## 1. Current architecture baseline

The agent currently has three distinct layers, although their boundaries were previously concentrated in `graph.py`:

1. **Domain control plane**
   - `flow/wiring.py` defines the node and edge layout used by both LangGraph and the pure-Python runner.
   - `graph.py` owns node implementations, repair routing, budgets, gate updates and public patch points.
   - The validation ladder remains fail-fast: ISA lint, hardware implementation, SoC build, conformance, software implementation, software build, simulation and Yosys cost.
2. **Execution services**
   - `runtime/policy.py` restricts reads, writes and commands per stage.
   - `runtime/tools.py` confines edits and generated files to the run workspace.
   - `runtime/agent_tools.py` provides bounded, read-only discovery tools.
   - `runtime/llm.py` provides plain completion and the provider-facing tool-call loop.
3. **Deterministic evidence and acceptance**
   - Compiler, elaboration, conformance, simulator and Yosys results update graph-owned gates.
   - A model can propose an implementation or choose a repair target, but cannot mark a gate as passed.
   - Verified optimization rounds retain source snapshots and measured metrics.

Important existing capabilities must not be described as absent:

- The ISA contract is structured YAML/JSON with deterministic linting, although it is not yet a complete typed semantics and acceptance IR.
- Analysis and hardware implementation already support bounded read-only tool use.
- Hardware/software edits already use policy-checked SEARCH/REPLACE, whole-file or unified-diff delivery.
- Repair prompts receive bounded diagnostics, and repeated response/error hashes are tracked.
- Candidate source snapshots exist, but they are not checkoutable git worktrees or a full Pareto archive.

## 2. Target boundary

The intended architecture is:

```text
thin domain state machine
  - stage objective
  - required artifacts
  - permissions and budget
  - deterministic acceptance gates
            |
            v
general stage coding executor
  - inspect/search
  - produce or apply edits
  - request bounded checks
  - observe failures
  - recover within a stage
            |
            v
deterministic compiler / simulator / Yosys environment
```

"Stage freedom" therefore means freedom to inspect, edit and debug **inside a capability and budget envelope**. It does not mean that the model controls the evaluation criterion, may write arbitrary repository paths, or may replace independent validation with its own claims.

The five proposed macro stages—DISCOVER, CONTRACT, IMPLEMENT, VERIFY and OPTIMIZE—are initially architectural boundaries, not an immediate requirement to collapse the physical graph into exactly five nodes. Existing fine-grained nodes remain useful as deterministic checks and compatibility seams.

## 3. First migration slice implemented
The first slice introduces `runtime/stage_executor.py`:

- `StageExecutionRequest` describes one stage-local interaction.
- `StageExecutionResult` returns the model result, raw-output artifact path and tool transcript.
- `StageExecutor` selects plain completion or a bounded tool interaction and preserves fallback/audit behavior.
- It does **not** build prompts, apply patches, execute domain validation, update graph state, or decide whether a gate passed.

`graph._run_agent_step` remains as a compatibility facade. It injects the existing graph callbacks at call time, preserving established monkeypatch points such as `graph._complete_llm`, `graph._llm`, artifact writers and progress logging.

The following stages now share this executor boundary:

- workload analysis;
- hardware implementation;
- software implementation.

Software implementation now receives the same policy-checked read-only `read_file`, `list_dir` and `search_text` capabilities as hardware when agent tools are enabled. It still emits one final EDIT/FILE/PATCH delivery; applying the delivery and running `sw_build` remain graph-controlled operations.

## 4. Compatibility and trust boundaries

During the incremental migration:

- Public graph nodes and `build_graph` / `run_agent` / `run_agent_python` remain in `graph.py`.
- `_complete_llm` keeps its positional signature and `(LlmResult, raw_path)` return value.
- `_run_agent_step` keeps its existing signature and tuple return value.
- Tool-loop failure still degrades to a plain completion and writes an auditable failure transcript.
- Full tool observations remain in run artifacts; graph state carries bounded summaries.
- `RepoToolbox`, access policy and run-scoped workspace rules remain authoritative.
- Build, conformance, simulation and cost nodes remain deterministic gates outside the model-controlled loop.

## 5. Structured failure evidence

The validation path now stores complete filtered command output as immutable,
run-scoped artifacts instead of discarding everything before the last 6000
characters. Existing `CommandResult` tails and parsed fields are preserved for
compatibility; validation results may additionally carry a full-log path,
truncation flags, and a versioned `FailureObservation`.

The observation is deterministic harness evidence, not a model verdict. It
contains a bounded summary, source locations, rung/category, and a canonical
`failure-v1:` signature that removes volatile run paths, addresses, timestamps,
and source line numbers. Repair prompts receive the structure and log reference,
not the complete log automatically. The implementation tool loop can offer
`inspect_failure` for known IDs; it is current-run policy checked, read-only, and
returns only a bounded log excerpt. Machine-readable Embench and conformance
results are parsed before output truncation.

## 6. Next slices

Implemented since this document was written: the **graph.py decomposition** (see
section 8). It was sequenced before the remaining slices because it is the
enabler for the macro-gate facade and removes the coupling that made every new
stage capability a cross-cutting change.

Implement the remaining work in this order:

1. **Controlled edit/check loop pilot (implemented for software and hardware)**
   - A dedicated coding facade binds edits to graph-owned target paths and never exposes generic `RepoToolbox` mutation.
   - It offers exact SEARCH/REPLACE and a graph-generated compile/elaboration check with no model-provided argv/cwd; the check uses an isolated build directory and no shell surface.
   - Internal checks only produce evidence. The existing `build_sw` / `hw_build` nodes independently rerun and remain the authoritative Gates; they fail closed when the implementation stage is absent or unsuccessful.
   - The coding capabilities require `!dry_run && apply_patch && run_commands`, have independent edit/check budgets, and are diagnosis-unlocked during repair: mutation stays locked until `submit_repair_diagnosis` cites current failure IDs and is accepted, so `REPAIR_DIAGNOSIS` cannot be bypassed. (An earlier draft described repair-time mutation as fully disabled; the shipped behavior is the diagnosis-unlock protocol documented in `cfu_agent_sw_hw_coding_system.md`.)
   - Implemented: candidate hashes on the implementation gates and the structured diagnosis-unlock protocol. Remaining: multi-target transactions.
2. **Candidate recovery**
   - Extend existing source snapshots with an atomic restore manifest and contract/evidence hashes.
   - Prove serial restore and replay before introducing git worktrees or parallel workers.
3. **Context management**
   - Add a compact cross-layer CFU repo map and stage summaries.
   - Reduce unconditional reference injection only after retrieval guarantees and regression tests exist.
4. **Macro-gate facade**
   - Group existing nodes behind DISCOVER / CONTRACT / IMPLEMENT / VERIFY / OPTIMIZE interfaces.
   - Collapse physical graph nodes only if it improves maintainability without weakening observability or test seams.
5. **Parallel QoR exploration**
   - After the contract is frozen, isolate candidate workspaces, build outputs and EDA budgets.
   - Maintain a true non-dominated Pareto archive with replayable evidence.

## 7. Explicit non-goals for the first slice

The first slice does not:

- give the model a shell, build, simulator or Yosys tool;
- let the model mutate evaluation criteria or gate state;
- replace the access policy;
- turn LangGraph checkpointing into source-control checkpointing;
- create git worktrees or parallel candidates;
- collapse the existing workflow into five physical nodes;
- claim that current conformance evidence is fully independent of model-authored oracle content.

These constraints keep the refactor behavior-preserving while creating the execution-plane seam needed for later stage-local autonomy.

## 8. graph.py decomposition (implemented)

The 6328-line `graph.py` is now a ~200-line composition root. Stage logic lives
in `stages/`, flow control in `flow/`, and execution services in `runtime/`; the
routing decision table is pinned by `agents/tests/test_graph_routing.py`, one
offline end-to-end run by `test_graph_smoke.py`, and the decomposition
invariants by `test_graph_modules.py`.

The migration was behaviour-preserving, and that is verified rather than
asserted:

- every one of the 190 top-level definitions in the original module exists in
  the new tree;
- the exported mermaid and ASCII graph renderings are byte-identical before and
  after the split;
- the full `unittest` suite passes.

Two rules make the split durable, and `test_graph_modules.py` enforces both:
functions are reached through their owning module (never rebound by a
`from X import f`), and `graph.py` re-exports only public entry points and node
functions. The second rule is what turns a stale
`mock.patch.object(graph, "<helper>")` into a loud `AttributeError` instead of a
patch that silently changes nothing.

Not done here, in the order they should follow: the `stage3_autonomy` dual-path
collapse, a typed `stage_result()`/`Gate`/`Rewind` vocabulary for the 35 hand-written
gate payloads, the macro-gate facade, and the LangGraph-native adoptions listed
in `agent_v2_plan.md` (`@task` for external effects, `error_handler` /
`RetryPolicy` / `TimeoutPolicy`, `ToolNode`, `get_stream_writer`, `Command` with
`destinations`).

## 9. LangGraph-native adoption

Two native mechanisms replaced hand-rolled ones, and the rest were decided
against with reasons, so a later reader does not re-litigate them blind.

**Adopted — declarative node policies (`flow/policies.py`).** `error_handler`,
`retry_policy` and (deliberately not) `timeout` are declared in one table and
applied by `flow/wiring.py::register_langgraph`. The three benchmark-preparation
rungs lost their `try/except` in favour of `ladder.prepare_failure_handler`,
which is registered through `add_node(..., error_handler=...)`. The handler keeps
the previously caught set exactly as narrow as it was — a `NodeError` carrying
anything outside `(PolicyError, ValueError, OSError)` is re-raised, so a bug is
still a loud abort rather than a "repairable" design failure.
`agents/tests/test_node_policies.py` pins the registration, the narrow caught set,
and that no model or EDA node carries a retry policy.

**Adopted — external commands as tasks (`runtime/effects.py`).** `@task` records
a command's result in the checkpoint and restores it on resume instead of
re-executing the body, which is what the operation journal was hand-rolling.
Verified by spike before wiring it in:

- a task inside a Graph-API node, interrupted mid-node and resumed, restores the
  completed call and never re-runs it (`agents/tests/test_effects.py` proves the
  same shape for the real command path);
- a task *outside* a runnable context raises `Called get_config outside of a
  runnable context`, so the dispatcher keeps a plain path — node functions are
  also called directly by tests and helpers;
- task arguments may be opaque objects, but this is not relied on: the effect
  takes only argv/stage/rung/timeout and receives the run state out of band from
  the node wrapper, so no toolbox or access policy is checkpointed.

**Not adopted, with reasons.**

- *Node-level `timeout`* — every external command already carries its own
  measured budget in `runtime/checks.py`; a second coarser cap would add a number
  nobody measured.
- *A graph-wide `error_handler`* — it would convert framework bugs into
  repairable rungs and spend model/EDA budget on them.
- *`Command(goto=...)` + `destinations`* — belongs with the typed `Rewind`
  payload (P2). The router functions stay pure and table-tested until then.
- *`get_stream_writer()`* — the next item: it replaces the `_LIVE_LINE` TTY
  rewriting and the stream callback, but it changes CLI progress rendering, which
  several tests lock.
- *`CachePolicy` + `langgraph.cache.sqlite.SqliteCache`* — the profile cache is a
  deliberate cross-run, tree-hash-keyed filesystem cache; a node cache whose key
  does not cover the same inputs would silently reuse a stale measurement. Not
  worth trading that risk for ~200 lines.
- *`create_agent`* — `langgraph.prebuilt.create_react_agent` is deprecated in
  favour of `langchain.agents.create_agent`, which is not installed, and the
  stage loop's value is in its `submit_stage` gate, budgets and audit transcript,
  none of which the prebuilt agent provides.

**The operation journal stays, for now.** `@task` protects the recorded command,
but a replayed node still re-runs the rest of its body — artifact writes, `git
apply`, and model calls. The journal's conservative "refuse to resume a node with
committed effects" is therefore still the safety net. It can go once every
non-idempotent effect in a node body is either a task or provably idempotent.

**Lesson from the decomposition.** A scripted code move silently renamed the
recorded state key to `"dependencies.input_fingerprints"`; every read then saw an
empty mapping, so gate invalidation stopped happening while the suite stayed
green. The key is fixed and `test_graph_modules.NodeWrapperContractTest` now pins
the wrapper's state-key contract, including that an upstream input change really
does mark a downstream gate stale.

## 10. Outcome vocabulary (P2, first slice)

`flow/outcome.py` now owns the shape of a rung's verdict. `gate()` merges one
rung's payload onto the state's gates and `stage_result()` adds the history line,
so the repeated

```python
"gates": {**state.get("gates", {}), rung: {"ok": ok, ...}},
"iteration_history": append_history(state, "..."),
```

is written once. Two things became checkable that were not before:

- **`next` is validated against `flow/wiring.py::NODE_NAMES`.** A router reads
  `gate["next"]`, so a typo there used to strand the run on a nonexistent node;
  it now raises where the gate is built. This is the runtime equivalent of what
  `add_node(destinations=...)` would validate at build time, without giving up
  the pure, table-tested router functions.
- **`Rewind(target, reason, evidence)`** gives a repair decision a typed home:
  the failure ids or log references it was made on are recorded as
  `rewind_evidence` instead of being flattened into a reason string.

A field is written only when the caller passes it (`None` means omit), so the
recorded gate stays byte-identical to the hand-written dict — including the sites
that deliberately record `next: ""`. The whole `verify/` stage (10 payloads
across `soc`, `conformance`, `software`, `cost`) is converted; the other stages
convert next. `agents/tests/test_outcome.py` pins the vocabulary, the exact
payload reproduction and the target validation.

**`get_stream_writer()` was investigated and deliberately not adopted.** The
live token line is driven by the LLM transport, which must also work outside a
graph run (tests, `--test-api`) and on the default `app.invoke()` path; the
framework writer exists only inside a custom-stream run. Adopting it would mean
forcing every run onto the streaming path *and* keeping the existing callback as
a fallback, i.e. more mechanism, not less. Stage timing already comes from the
node wrapper, which measures it directly.

## 11. P2 completion

**Outcome vocabulary (done).** All 27 hand-written gate payloads now go through
`flow/outcome.py` — the whole `verify/` stage first, then `context`, `profile`,
`analysis`, `contract`, `implement/*`, `review` and the prepare rungs in
`flow/ladder.py`. Nothing builds `"gates": {...}` by hand any more, so every
`next` is validated against `NODE_NAMES` and every repair can carry a typed
`Rewind`. `agents/tests/test_outcome.py` pins the vocabulary and the exact
payload reproduction.

**State groups (done).** `state.py` became a package: `config.py` (41 keys),
`evidence.py` (26), `ledger.py` (24), `optimization.py` (8), composed into
`CfuDesignState`. The split was verified against a captured baseline of
`get_type_hints(CfuDesignState, include_extras=True)`: **99 keys and the 10
reducer annotations are identical before and after**, so no channel changed.
Composition was checked as safe first — `TypedDict` inheritance keeps inherited
keys in both `__annotations__` and `get_type_hints`, and a spike graph confirmed
the channels and reducers survive compilation.

**Deferred, with reasons.**

- *`RuntimeContext` (`context_schema=RunContext`).* The value would be separating
  the ~41 input knobs from the run's outputs. It is not adopted now because it is
  the most invasive change on the list for the least correctness: every node
  signature and every `state.get("dry_run")`-style read would move, the node
  wrapper would have to forward `runtime` (it currently calls `fn(state)`), and
  nothing is currently broken by config living in the state — the fingerprint
  function already curates which config actually affects a measurement. The
  state groups above deliver most of the readability win at a fraction of the
  risk.
- *Renaming the seams (`checks._run_check` -> `checks.run_check`, ...).* Purely
  cosmetic: the names are already in their owning modules, and the tests reach
  them module-qualified, so a stale patch fails loudly. Renaming moves ~124 test
  sites without changing what is coupled.
- *Reducing 27 distinct patch points to ~12.* This is the one remaining item with
  real value, and it is a **test** change, not a production one: the seams that
  exist only so a test can stub an internal (`_write_isa_artifact`,
  `_report_review_decision`, `_software_exercises_cfu`, ...) can be removed by
  asserting on the outcome instead — the artifact on disk, the gate in the
  update, the parsed verdict. That is a separate track with its own
  justification and should be done deliberately, file by file.

## 12. P3: the Implement stage schedules its own verification

The core concept for this stage is **freedom inside a capability and budget
envelope**: the agent decides the order of editing hardware, writing software,
authoring a dev test, building, and measuring. Until now the free stage could
only edit and run two narrow post-edit checks, so "free" meant very little.

`ImplementationToolset` now carries a **verification registry**
(`runtime/implementation_tools.py`). A `Verification` declares a tool name, its
description, the rung that owns the verdict, a budget and a callback; the stage
supplies the callbacks, so command construction stays graph-side:

| tool | what it runs | authoritative rung |
|---|---|---|
| `build_soc` | the overlay `MiCoSocGen` command (`hw_build` reruns it) | `hw_build` |
| `build_software` | the software build commands | `sw_build` |
| `run_simulation` | the workload simulation, and reports the measured cycles | `sim` |
| `run_conformance` | the harness differential micro-test over agent-authored oracle content | `hw_conformance` |
| `check_hardware_elaboration` / `check_software_compile` | the existing post-edit checks | `hw_build` / `sw_build` |

Self-testing *is* the conformance tool: the agent supplies an oracle, the harness
renders and runs the C test, and the same parser/validator as the rung judges it.
The old `edit_dev_test` surface was removed — it wrote a C file that nothing
compiled or ran, so it was bookkeeping that could only mislead the model into
believing it had tested something.

Properties that keep the freedom safe, and that the tests pin:

- **No steerable arguments.** Every tool but one takes no arguments at all, so
  the model cannot supply argv, cwd, options or an output directory.
  `run_conformance` is the sole exception, and its single argument is the oracle
  *content*; the harness still renders the C test, and the oracle goes through
  the same `parse_oracle`/`validate_oracle` pair the conformance rung uses, so a
  lazy oracle is refused identically by both paths.
- **Evidence, never a verdict.** Results are rendered as "evidence only; the
  `<rung>` will rerun", and a *failing* check deliberately reports diagnostics
  without naming a gate verdict — the existing invariant that a check must never
  read as a gate.
- **Budgeted and withdrawn.** Each verification has a budget (3); when it is
  spent the tool disappears from the next round's schema set, which is the
  documented way to keep a tool list from overwhelming the model.
- **Audited.** Every call lands in the toolset's `checks` trail, so a run's
  artifacts show what the agent verified and how often.

The budget rule differs from the post-edit checks on purpose: a check answers
"did the edit I just made work", so it requires an edit in between, while a
verification answers "where do I stand", which is a legitimate thing to ask
again after an unrelated fix.

**Deferred: the collapse itself.** The free stage is still one of two
implementation paths (`--stage3-autonomy` versus the `implement_hw` /
`implement_sw` rungs). Promoting it to be the only path removes the boolean
threading, the two single-side rungs, the LLM software-reuse decision and the
simulation repair router — a much larger change than adding the capability, and
it should be done as its own verified step.

## 13. What the LangGraph tool documentation changed

Reading `oss/python/langchain/tools` (tool definition, state access, dynamic
selection, error handling) confirmed the shape chosen above and produced one
follow-up:

- **Confirmed:** framework-owned tools with no model-supplied argv; a
  budget-driven *filtering* of the offered tool schemas is the documented way to
  vary the tool set at runtime (upstream does it with `create_agent` middleware,
  which needs `langchain`, not installed here); tools must be `snake_case`;
  `config` and `runtime` are reserved parameter names.
- **Not adopted:** `@tool` schema generation — valuable, and the honest follow-up
  for the ~10 hand-written schemas in `agent_tools`/`hardware_tools`/
  `software_tools`, but it changes the tool loop's schema contract and does not
  advance the stage refactor. `ToolRuntime`/`InjectedState` would not simplify
  these tools, which need the policy-checked `RepoToolbox` rather than graph
  state. `ToolNode` + `wrap_tool_call` remain deferred for the reason recorded in
  section 9: the budget, policy-denial and audit semantics are ours, and error
  handling upstream now lives in `create_agent` middleware.

## 14. P3b in progress: one Implement stage

Planned in full, executed in verified steps so the tree is green at every
boundary. Steps 1 and 2 have landed.

**Step 1 — the stage must see every failing rung (done).** `feedback.all_failure_feedback`
now composes the hardware sources (`hardware_repair_feedback` already covers
`hw_build`, `hw_conformance`, `sim`, `cost` and the hardware rung's own
rejection) with the software ones (`sw_build`, the software rung's rejection),
and the free stage uses it. This also removed a real duplication: the stage
concatenated `sim_feedback` a second time even though the hardware aggregation
already includes it, so the same simulation evidence reached the prompt twice.
`test_failure_feedback.py` pins the attribution, the single occurrence of the
simulation block, and that a long log contributes an excerpt rather than the
whole log.

**Step 2 — the `implement` node exists (done).**

* `stages/implement/__init__.py::implement` is the single entry point. The mode
  is chosen by *capability*, not by a flag: a tool-capable model gets the free
  tool loop (`unified.implement_with_tools`), otherwise the stage falls back to
  the EDIT/FILE/PATCH delivery protocol (the hardware and software rungs' bodies,
  merged into one update). Both modes write the same two gates, so routing and
  the ladder are unaffected by which one ran. This is why
  `--no-agent-tools` keeps working, and why `--stage3-autonomy --no-agent-tools`
  — which used to raise — now works too.
* `outcome.merge_updates` folds two nodes' updates together. Concatenating the
  reducer channels is not optional: LangGraph applies a reducer once per returned
  value, so returning two updates' worth of history as one value would *replace*
  the accumulated list. The reducer keys are derived from the state annotations
  (`state.append_keys`), so a new accumulator cannot be forgotten, and a test
  iterates every one of them.
* `flow/instrument.py` learned that one node may evaluate two gates:
  `_NODE_TO_GATE["implement"]` is a tuple, both gates are re-stamped against the
  fresh fingerprint instead of being reported stale, and `_node_ok` requires both
  halves to pass.
* The node is on the graph and reachable through the autonomy path, so step 3 can
  flip the default with the node already exercised.

**Still to do (steps 3-5 of the plan):** flip every route/repair target to
`implement` and delete `implement_hw` / `implement_sw` / `implement_stage3`; then
retire `--stage3-autonomy`, `--software-change` and `--repair-routing` with their
helpers (`_choose_software_change`, `_choose_repair_target`,
`sim_repair_decision.yaml`, the `route_repair` stage) and update
`agents/experiments/run_matrix.py`; then the documentation.

### P3b step 3 — the collapse itself (done)

The graph now has **one implementation node**. `implement_hw`, `implement_sw` and
the `implement_stage3` alias are gone from the routing, `STAGE_ORDER` and the
progress slots; every route and every repair target points at `implement`, and
`check_conformance` always continues to `build_sw` (there is no
software-implementation rung left to skip). The graph rendering confirms the
intent: the five verification rungs and their static edges are untouched, and
only the implementation region changed from three nodes to one.

Retired with it, because they only existed to arbitrate between the two
implementation paths:

* the software-reuse decision (`_choose_software_change`, `_software_reuse_evidence`,
  the `software_decision` gate field, the `software_decision.yaml` artifact, and
  `generate_soc`'s whole reuse block) — the stage that writes the software
  already decided;
* the simulation repair router (`_choose_repair_target`, `_sim_repair_target`,
  `_post_conformance_default`, the `sim_repair_decision.yaml` ledger, and the
  `stage3_autonomy` branch that redirected it) — with one implementation stage
  there is nothing to arbitrate, and the stage sees every failing rung through
  `all_failure_feedback`;
* `stage3_autonomy` from every routing decision.

Their tests went with them, and the coverage was **replaced rather than dropped**:
`test_graph_routing.SingleImplementationStageTest` now asserts that the graph has
exactly one implementation node, that a stale target fails loudly
(`outcome.gate(..., next="implement_hw")` raises), that every failing rung with
budget repairs `implement`, that a spent budget stops the run instead, and that a
passing rung keeps its static successor.

One silent quality regression was caught and fixed on the way: the optimization
prompt still offered `implement_hw` / `implement_sw` as targets while the chooser
validates against `OPTIMIZATION_TARGETS = ("implement", "extend_isa")`, so a model
answering with the old names would have been judged unusable and the `--loop`
rounds would have quietly stopped instead of targeting the stage.
