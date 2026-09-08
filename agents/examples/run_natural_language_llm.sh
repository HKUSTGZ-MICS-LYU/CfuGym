#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_FILE="$ROOT/agents/examples/natural_language_u8_weighted_average.txt"
TASK="$(tr '\n' ' ' < "$TASK_FILE")"

args=(
  --task "$TASK"
  --checkpoint
  --thread-id "${CFU_AGENT_THREAD_ID:-example-natural-u8-wavg}"
  --stream
  --max-iters "${CFU_AGENT_MAX_ITERS:-1}"
)

if [[ "${CFU_AGENT_APPLY_PATCH:-0}" == "1" ]]; then
  args+=(--apply-patch)
fi
if [[ "${CFU_AGENT_RUN_COMMANDS:-0}" == "1" ]]; then
  args+=(--run-commands)
fi
# Planning is the default; the framework only runs these examples read-only
# unless the capability flags above are set.
args+=(--dry-run)
if [[ "${CFU_AGENT_SKIP_SIM:-1}" == "1" ]]; then
  args+=(--skip-sim)
fi
if [[ "${CFU_AGENT_SKIP_YOSYS:-1}" == "1" ]]; then
  args+=(--skip-yosys)
fi

cd "$ROOT"
python3 -B agents/run_cfu_agent.py "${args[@]}"
