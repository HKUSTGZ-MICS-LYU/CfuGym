# CfuGym Agents

This folder contains local agent frameworks for CfuGym workflows.

## CFU Design Agent

The CFU design agent is a LangGraph workflow for the hardware/software CFU optimization loop:

```bash
python3 agents/run_cfu_agent.py \
  --task "Accelerate a uint8 weighted average kernel with a TileLink CFU. It reads x[i] and w[i], computes sum(x[i] * w[i]) / N, arrays are 32-byte aligned, and products should stay resident inside the CFU." \
  --dry-run \
  --skip-sim \
  --skip-yosys
```

The first graph stage normalizes both input modes into a `WorkloadSpec`. The next stage prepares something measurable before CFU design starts: natural-language input becomes a generated benchmark project, while C-project input is copied and instrumented so the original source tree is not modified.

```bash
# Natural-language workload. The agent creates a benchmark workspace first.
python3 agents/run_cfu_agent.py \
  --task "Accelerate a uint8 dot product over aligned arrays."

# C-project-backed workload. The agent copies and instruments the project first.
python3 agents/run_cfu_agent.py \
  --project-root sw \
  --kernel-file sw/tests/u8_wavg_cfu_test.c \
  --kernel-function main \
  --build-cmd "make TARGET=vexii_soc MAIN=tests/u8_wavg_cfu_test BUILD=build_u8_wavg_cfu MARCH=rv32imc_zicsr_zifencei OPT=cfu compile" \
  --run-cmd "sbt runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf sw/tests/u8_wavg_cfu_test.elf --with-rvc --with-rvm --with-rdtime --mico-u8-wavg-cfu"
```

For C projects, the copied profiling workspace is placed under `agents/generated/workspaces/<case>/project_profile/`. If a `make` command has no `-C`, the agent remaps it to build inside the copied project. Profile lines use `CFU_AGENT_PROFILE name=cycles`, and the largest parsed region becomes the first hot-spot candidate.

Use `--dry-run` first. Without `--dry-run`, the agent may apply LLM-generated patches, but only to allowlisted CFU/software paths.

LangGraph features used by this workflow:

- `Command` routing: input classification and iteration review return both state updates and the next node.
- State reducers: append-only fields such as iteration history and command results use reducer semantics, so loop iterations do not manually rewrite prior state.
- Checkpointing: add `--checkpoint --thread-id <id>` to compile with an in-memory checkpointer for inspectable thread-scoped execution.
- Streaming: add `--stream` to execute through LangGraph state snapshots and print the number of streamed snapshots.
- Node retry policy: repo-context loading uses a small retry policy; simulator/build/Yosys timeouts stay inside the command runner because this LangGraph version only supports node-level timeout cancellation for async nodes.

Every run exports a graph visualization:

- `agents/generated/graph/cfu_design_agent.mmd`: Mermaid diagram.
- `agents/generated/graph/cfu_design_agent.txt`: ASCII diagram.

Ready-to-run examples are under `agents/examples/`:

- `run_natural_language_llm.sh`: natural-language workload input.
- `run_c_project_llm.sh`: copied bare-metal C-project input.

Environment variables:

- `OPENAI_API_KEY`: required when LLM calls are enabled.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible endpoint.
- `CFU_AGENT_MODEL`: optional default model name.
