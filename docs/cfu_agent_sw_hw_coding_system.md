# CFU agent SW/HW coding system

## Capability boundary

The implementation stages may run a bounded local coding loop only when all of
`agent_tools`, `apply_patch`, and `run_commands` are enabled and the run is not a
dry-run.

Software exposes `edit_software` and `check_software_compile`. Hardware exposes
`edit_hardware` and `check_hardware_elaboration`. Both retain the common read-only
tools and `inspect_failure`. On a repair attempt, `submit_repair_diagnosis` must
cite current failure IDs and be accepted before either edit tool can write.

Edits are transactional across the whole model tool interaction. The graph commits
only after its implementation acceptance checks pass; exceptions, degraded tool
calling, malformed delivery, rejected diagnosis, and semantic implementation
checks restore the exact stage-entry source bytes. This is in-process rollback,
not crash recovery after the Python process is killed. Hardware elaboration checks
use a distinct `attempt_NNN/check_NNN/cfu_rtl` output and require a newly generated,
non-empty `AgentCfu.v`, preventing an earlier successful check from masking a later
failed elaboration.

Internal compile/elaboration checks use host-generated commands, isolated outputs,
and separate budgets. They produce logs and `FailureObservation` evidence but
cannot write graph Gates. `hw_build` and `sw_build` always validate independently;
conformance, simulation and Yosys remain graph-only.

## Safe inspection run

This checks CLI/config connectivity but does not edit or build:

```bash
conda run -n mico_env python3 agents/run_cfu_agent.py \
  --llm-config gpt-5.6-luna --test-api
```

A complete dry-run creates artifacts and prompts but neither coding capability is
offered because mutation and command execution are disabled:

```bash
conda run -n mico_env python3 agents/run_cfu_agent.py \
  --task "design a CFU for the named kernel" \
  --project-root path/to/project \
  --kernel-file src/kernel.c \
  --kernel-function kernel \
  --build-cmd "make TARGET=vexii_soc ..." \
  --run-cmd "..." \
  --dry-run --run-id dry-sw-hw-001
```

## Real bounded run

Use a unique run ID and explicit effect flags. `--no-dry-run` now behaves as its
CLI help states and implies patch application plus command execution; spelling out
`--apply-patch --run-commands` as below is still recommended for an auditable command.

```bash
conda run -n mico_env python3 agents/run_cfu_agent.py \
  --llm-config gpt-5.6-luna \
  --task "accelerate <kernel> with a custom AgentCfu" \
  --project-root <PROJECT_ROOT> \
  --kernel-file <KERNEL_C> \
  --kernel-function <FUNCTION> \
  --build-cmd '<ALLOWLISTED_BUILD_COMMAND>' \
  --run-cmd '<ALLOWLISTED_RUN_COMMAND>' \
  --apply-patch --run-commands --no-dry-run \
  --agent-tools --max-tool-rounds 6 --max-tool-calls 24 \
  --max-iters 3 --run-id sw-hw-pilot-001
```

For the built-in Embench adapter, replace the project/kernel arguments with for
example `--embench crc32`. Start with `--skip-yosys` only when intentionally
testing flow mechanics rather than claiming a complete result. `--skip-sim` also
prevents an end-to-end correctness claim.

## What to inspect

All generated state stays below `agents/generated/runs/<run_id>/`:

- `09_implementation/hw_tool_calls.yaml` and `sw_tool_calls.yaml`: ordered tool audit;
- `09_implementation/hw_changed_files.yaml` and software implementation artifacts:
  committed target files and acceptance state;
- `10_validation/`: independent hardware/software build and simulation records,
  full command-log references, and typed failure observations;
- `11_cost/`: standalone RTL/Yosys evidence;
- `run.yaml` and `report.md`: final Gate summary.

A successful internal message such as “isolated hardware elaboration passed” or
“isolated software compile passed” is not completion. A usable complete result
requires the final report/Gates to show successful ISA lint, implementation,
hardware build, conformance, software build, simulation correctness/performance,
and cost (unless explicitly skipped for a mechanics-only test).

## Developer regression tests

Focused deterministic tests:

```bash
conda run -n mico_env python3 -m unittest \
  agents.tests.test_coding_sessions \
  agents.tests.test_software_tools \
  agents.tests.test_stage_executor \
  agents.tests.test_graph_gates \
  agents.tests.test_policy \
  agents.tests.test_failure_observations
```

Full agent regression:

```bash
conda run -n mico_env python3 -m unittest discover \
  -s agents/tests -t . -p 'test_*.py'
```

Do not run multiple full suites concurrently: some legacy tests use fixed generated
run paths and can race with each other. A real LLM run should only follow passing
deterministic tests, use a fresh run ID, and begin with a small disposable kernel.
