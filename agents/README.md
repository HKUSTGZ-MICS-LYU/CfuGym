# CfuGym Agents

This folder contains local agent frameworks for CfuGym workflows.

## CFU Design Agent

The CFU design agent turns a workload request into a run-scoped, auditable CFU design loop. It runs through the following stages, each of which writes its own artifact under the run directory:

`load_context → ingest_input → route_input → prepare_benchmark → profile → hotspot → analysis → isa → implementation → validation → cost → review → final_report`

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
```

### Access and command policy

`agents/specs/agent_access_policy.yaml` is the single source of truth for what each stage may read/write and which commands it may run:

- Deny rules always take precedence; paths are resolved to repo-relative form and reject absolute paths, `..` escapes, and symlinks outside the workdir.
- Each stage has its own `read` / `write` allow lists. Secret, `.git`, build/binary (`*.elf`, `*.o`, `*.map`, `*.asm`, `*.vcd`, `*.fst`), and non-allowlisted repo source are not exposed to the agent. Historical runs under `agents/generated/runs/` are never auto-read.
- Commands are declared by id (e.g. `make_profile`, `soc_sim`, `yosys_cost`, `git_apply`). Shell metacharacters, absolute paths, and sensitive paths in command argv are rejected; the executable and required tokens must match the declaration.
- `use --apply-patch` without `--run-commands` is safe: it validates the patch, checks its touched paths against the policy, runs `git apply --check`, but does not apply it unless both the patch and the command allowlist permit it.

### Run artifacts

Every run gets a unique directory `agents/generated/runs/<run_id>/` and writes:

```text
01_context/manifest.yaml          # visible/denied files and filtered git status
02_workload/workload_spec.yaml    # normalized workload spec
03_route/route.yaml
04_benchmark/manifest.yaml        # benchmark/project profile manifest and commands
05_profile/result.yaml
06_hotspot/hotspot.yaml
07_analysis/workload_analysis.yaml
08_isa/cfu_isa.yaml + cfu_isa.md  # structured CFU ISA spec + rendered markdown
09_implementation/plan.md, patch.diff, changed_files.yaml
10_validation/result.yaml
11_cost/result.yaml
12_review/decision.yaml
report.md
run.yaml                          # manifest with per-artifact sha256
workspace/...                     # per-run isolated benchmark/project workspace
```

Each artifact records its stage, status, inputs, visible files, SHA-256, size, and timestamp. The benchmark workspace is scoped to the run so runs do not clobber each other, and C-project inputs are always copied before instrumentation so the original source tree is never modified.

### Status semantics

- `planned`: a read-only planning pass completed. The report lists each gate and what was skipped because it needs execution. This is **not** a claim of correctness or execution.
- `complete`: every gate (workload spec, ISA contract, validation, cost) produced evidence and passed. Only this is a real completion.
- `failed`: execution ran but a required gate failed after max iterations.
- `blocked`: an input/ISA contract could not be validated, so the loop cannot proceed honestly.
- `needs_iteration`: execution found a failing gate and will retry up to `--max-iters`.

A run is never marked complete solely because a command was skipped or a dry-run was used.

### Live progress and token reporting

Each stage emits a live line as it runs and a second line when it finishes, e.g.:

```
[ 3/13] route input ... running
[ 3/13] route input ... OK (0 tok, 0.0s)
[ 7/13] analyze workload ... OK (375 tok, 0.2s)
[ 8/13] design CFU ISA ... OK (375 tok, 0.1s)
```

- `[i/N]` is the stage index within the workflow.
- `OK`/`FAIL` comes from the stage's gate (`gates.<key>.ok`) or whether the node added errors.
- `(N tok)` is the total LLM tokens consumed during that stage.

At the end the CLI prints a summary of cumulative token usage (`calls`, `input_tokens`, `output_tokens`, `total_tokens`) and per-stage results, and the `report.md` includes a `## Token Usage` section. Token usage is read from the OpenAI Responses / Chat Completions `usage` block and accumulated per run; everything stays in-memory and is never logged as content.

### WorkloadSpec and CFU ISA spec

`agents/specs/cfu_isa_template.yaml` is the CFU ISA contract template. It is validated by `agents/cfu_design_agent/isa.py`, which enforces a single opcode/function-id space, consistent datapath (`vlen`/`xlen`/`maclen`/`accWidth`), state/reset semantics, memory/ordering rules, software intrinsics, scalar fallback, integration (component/fiber, SoC params and CLI flags), and validation requirements. `agents/specs/u8_weighted_average_isa.yaml` is a completed reference for the current `U8WeightedAvgCfu` custom0 contract (`func3=0..5`).

`agents/specs/u8_weighted_average.spec` is the normalized `WorkloadSpec` input model. It has a versioned schema; unknown high-impact fields must be declared in `unknowns` instead of being silently guessed.

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

Each config may set `model`, `base_url`, `api_mode` (`auto`/`responses`/`chat`), `temperature`, `timeout`, and an inline `api_key` or an `api_key_env` variable name. Model resolution order (highest wins) is: explicit `--model` > config `model` > `CFU_AGENT_MODEL` > built-in `gpt-5`. A config can leave `model`/`base_url` empty and still pick up `CFU_AGENT_MODEL`/`OPENAI_BASE_URL`. Prefer `api_key_env` to keep keys out of the tracked JSON. `default.json` is optional: if missing, the framework builds an environment-only config. See `agents/llm_configs/README.md`.

### Environment

- `OPENAI_API_KEY`: required only when LLM calls are enabled. Without it (or when the selected config resolves no key), the agent uses deterministic fallbacks.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible endpoint.
- `CFU_AGENT_MODEL`: optional default model name.
- `CFU_AGENT_LLM_CONFIG`: optional default config name (otherwise `default`).
- `CFU_AGENT_LLM_API`: optional `auto`/`responses`/`chat` override.

### Tests

```bash
python3 -m unittest discover -s agents/tests
```

The `unittest` suite checks path traversal, deny precedence, command allowlisting, run isolation, `WorkloadSpec`/CFU ISA validation, and dry-run flow without any external toolchain or API key.

Ready-to-run examples are under `agents/examples/`.
