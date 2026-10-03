# CFU agent failure taxonomy and evidence baseline

Recorded from run `20260928-200429-873265-70db5b1b` (embench `md5sum`, model
`gpt-5.6-terra`) and the follow-up characterization experiments of 2026-09-29.
This is the baseline for harness metrics (repair-rate, misattribution rate,
budget waste) and the reference case for the failure-evidence machinery in
`runtime/observations.py`.

## 1. Failure categories (deterministic classification)

`classify_failure` assigns each failed command exactly one category. The
category decides whether a design repair can possibly help:

| category | evidence | design-actionable | typical routing |
| --- | --- | --- | --- |
| `fail_symbol` | MiCoSocSim reported the CPU reached the fail symbol | yes | implement_hw / implement_sw |
| `case_mismatch` | conformance verdict present with failed cases | yes | implement_hw |
| `assertion` | `FAILURE <msg>` / `HDL assertion failure` in the log | yes (names the contract) | implement_hw |
| `timeout` | command timed out | yes (likely design hang) | implement_hw |
| `compile` / `elaboration` / `synthesis` | toolchain diagnostics | yes | implement_hw / implement_sw |
| `sim_harness_crash` | JVM exception whose frames are in `vexiiriscv.test.*` / `spinal.sim.*`, no assertion message | **no** | stop, no repair budget spent |
| `environment` / `policy` | launcher/policy failures | no | stop or config fix |

`HARNESS_FAILURE_CATEGORIES` and `failure_is_design_actionable()` in
`observations.py` are the single source of truth; `graph.conformance_repair_target`
returns `None` for them, which stops the ladder with
"not attributable to the design" instead of routing an unfixable failure to
`implement_hw`.

`FailureObservation.source_locations` drops runtime stack frames
(`TraversableLike.scala`, `ArrayOps.scala`, ...) and never emits partial-token
matches such as `scala.c` from `scala.collection`.

## 2. Failure ID freshness

`failure_index()` marks each observation `current` or `resolved` (a later
success of the same rung resolves it). Repair prompts list only current IDs as
"valid IDs"; resolved IDs are shown as context with an explicit
"do NOT re-apply a fix" note, `inspect_failure` prints the status, and
`submit_repair_diagnosis` rejects resolved IDs.

Case study: in the reference run the compile failure `fail_hw_build_91b6...`
was fixed by repair attempt 2, but it stayed in the "valid IDs" of attempts
3-5; all three anchored their diagnosis on the already-fixed error and re-sent
the same stale SEARCH block (`round(3 downto 0).asUInt`), which no longer
existed in the file. Two of them were rejected outright and the third applied a
semantically empty re-fix — the entire repair budget went to a resolved failure.

## 3. Approach-level loop detection

Each implementation attempt records `search_key` (the first content line of
every SEARCH block). A delivery whose key matches an already-rejected attempt
is rejected before application ("REPEATED APPROACH"), and the repair prompt
carries a `REPEATED APPROACH DETECTED` directive when the last failed attempt
repeats an earlier key — byte-level `REPEATED IDENTICAL ATTEMPT` detection
cannot see a rephrased repeat of one approach.

## 4. Delivery rejections do not spend `--max-iters`

A deterministically rejected delivery (SEARCH not found, missing deliverable,
omitted REPAIR_DIAGNOSIS) gets one corrective retry via
`_delivery_retry_transition` without consuming the design-repair budget. A
design that was delivered but fails its build is a real repair and still draws
from `--max-iters`.

## 5. Platform bug: `--agent-cfu-no-store` (load-only CfuLsu) broke simulation

**Status: FIXED (2026-09-29) in `AgentCfuFiber.getM2sParameters`.**

Characterization matrix (2026-09-29, all runs from the repo root, same
`cfu_conformance.elf` / `embench_main.elf`, `MiCoSocSim --with-rvc --with-rvm
--with-rdtime`):

| # | design overlay | ELF | flags | result |
| --- | --- | --- | --- | --- |
| a | tracked scaffold | `cfu_conformance.elf` | `--mico-agent-cfu --agent-cfu-vlen 256 --agent-cfu-bus-width 32 --agent-cfu-reg-depth 2 --agent-cfu-no-store` | crash @time=10000: `FAILURE config/load: defer requires a tick handler`, `Stream payload/valid persistence failed`, then `SimFailure: HDL assertion failure`; 0 verdicts |
| b | run MD5 design | `cfu_conformance.elf` | same explicit flags | crash @time=10000: `ArrayIndexOutOfBoundsException: Index -1 ...` at `VexiiRiscvProbe.checkCommits` (VexiiRiscvProbe.scala:595); 0 verdicts |
| c | run MD5 design | reference ELF (never calls CFU) | same explicit flags | same probe crash @time=10000 |
| d | tracked scaffold | reference ELF | same explicit flags | same HDL-assertion crash as (a) |
| e | run MD5 design | reference ELF | `--mico-agent-cfu` only (defaults) | **pass**, `EMBENCH_VERIFY_PASS`, 42 s |
| f | run MD5 design | reference ELF | `--mico-agent-cfu --agent-cfu-no-store` | probe crash @time=10000 |
| g | run MD5 design | reference ELF | `--mico-agent-cfu --agent-cfu-vlen 256 --agent-cfu-bus-width 32 --agent-cfu-reg-depth 2` (no `no-store`) | **pass** |
| h | run MD5 design | reference ELF | `--mico-agent-cfu --agent-cfu-no-load` (store-only) | **pass** |
| i | run MD5 design | reference ELF | `--agent-cfu-no-load --agent-cfu-no-store --agent-cfu-no-tilelink` | **pass** |

Conclusions:

- Defaults in `MiCoSocParam` are `vlen=256, bus-width=32, reg-depth=2`, so the
  only behavioral difference in the failing commands is `--agent-cfu-no-store`
  (`AgentCfuParameter.withStore = false`).
- The **load-enabled + store-disabled** configuration breaks the SoC simulation
  at ~time=10000 regardless of design and even when the CFU is never exercised
  (c, d, f). Store-only (h) and memory-less (i) configurations pass.
- The failure manifests as a probe bookkeeping crash (`uop.fetchId == -1` at
  commit retire) with the MD5 overlay and as HDL assertion failures with the
  scaffold; both die with zero case verdicts. The scaffold's
  `Stream payload/valid persistence failed` assertions echo the known
  width-adapter issue documented at `MiCoSoc.scala` ("Compute the actual
  CPU-side data width ... avoiding unnecessary TileLink width adapters that can
  trigger 'Stream valid persistence failed' assertions"), so the likely defect
  is the get-only TileLink master path (`putFull = SizeRange.none` in
  `AgentCfuFiber.getM2sParameters`) or its `CfuLsu` load-only FSM wiring.
- Run `20260928-200429-873265-70db5b1b` failed for exactly this reason: its ISA
  contract declared `memory.store: false`, the (deliberate, tested) flag
  forwarding added `--agent-cfu-no-store`, and the conformance ladder then died
  without a verdict. The old harness misreported this as "the CPU reached the
  fail symbol" and burned all four `implement_hw` repairs plus both
  `check_conformance` oracle repairs on it.

Follow-up (not yet done): fix the `withStore=false` platform path
(`AgentCfuFiber` / `CfuLsu` / fabric wiring), then re-run the conformance
ladder; optionally harden `VexiiRiscvProbe.checkCommits` to raise a readable
`simFailure` when a retiring uop has no fetch context instead of a raw
`ArrayIndexOutOfBoundsException`.

### 5.1 Root cause and fix (2026-09-29)

Elaboration diffing (MiCoSocGen with/without `--agent-cfu-no-store`) showed the
only structural change on the failing path: the TileLink A-channel payload
fields (`data`/`mask`/`corrupt`) vanish for a get-only master, while the fabric
keeps the aggregate fields on its internal arbiters and ties this input to `x`
literals (`.io_inputs_2_payload_mask (4'bxxxx)`). With that asymmetry present
the simulation corrupts the CPU-side streams (`LsuCacheless`/`FetchCacheless`
"Stream ... persistence failed" assertions, `AgentCfuLib` `active` assertions,
probe `fetchId == -1` crashes) even with zero CFU traffic.

Fix: `AgentCfuFiber.getM2sParameters` now declares both `get` and `putFull`
whenever the TileLink bus exists, so the A-channel layout is
configuration-independent. Over-declaring an unused transfer is safe (the
master never emits that opcode); `withLoad`/`withStore` still gate the `CfuLsu`
FSM states and the instruction table, so a disabled store builds no store
logic. `VexiiRiscvProbe.checkCommits` also reports a readable `simFailure` (with
uop/fetch/decode context) instead of a bare `ArrayIndexOutOfBoundsException`
should the bookkeeping ever desynchronize again.

Post-fix validation (same matrix):

| case | before | after |
| --- | --- | --- |
| (d) scaffold + reference ELF + explicit flags incl. `--agent-cfu-no-store` | crash @time=10000 (assertions) | **pass**, `EMBENCH_VERIFY_PASS` |
| (f) MD5 design + reference ELF + `--agent-cfu-no-store` | probe crash @time=10000 | **pass**, `EMBENCH_VERIFY_PASS` |
| (b) MD5 design + `cfu_conformance.elf` + explicit flags | probe crash, 0 verdicts | **`CFU_CONFORMANCE PASS commands=4/4 cases=12 failed=none`** |
| (h) store-only (`--agent-cfu-no-load`) | pass | pass |

The (b) result also settles the original run's verdict: its design was
correct — the conformance test passes 12/12 once the platform bug is fixed.
