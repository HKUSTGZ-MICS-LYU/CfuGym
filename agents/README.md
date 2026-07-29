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

The first graph stage normalizes all input modes into a `WorkloadSpec`. Supported inputs:

```bash
# Natural-language workload
python3 agents/run_cfu_agent.py --task "..."

# Structured workload file. The checked-in example uses YAML syntax with a
# `.spec` extension because this repo globally ignores `*.yaml` and `*.json`.
python3 agents/run_cfu_agent.py --workload-spec agents/specs/u8_weighted_average.spec

# C-project-backed workload
python3 agents/run_cfu_agent.py \
  --project-root sw \
  --kernel-file sw/tests/u8_wavg_cfu_test.c \
  --kernel-function u8_wavg_scalar
```

Use `--dry-run` first. Without `--dry-run`, the agent may apply LLM-generated patches, but only to allowlisted CFU/software paths.

Every run exports a graph visualization:

- `agents/generated/graph/cfu_design_agent.mmd`: Mermaid diagram.
- `agents/generated/graph/cfu_design_agent.txt`: ASCII diagram.

Environment variables:

- `OPENAI_API_KEY`: required when LLM calls are enabled.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible endpoint.
- `CFU_AGENT_MODEL`: optional default model name.
