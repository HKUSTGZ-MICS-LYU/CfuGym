#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="agents/examples/c_project_u8_wavg"

args=(
  --project-root "$PROJECT"
  --kernel-file "$PROJECT/main.c"
  --kernel-function main
  --build-cmd "make TARGET=vexii_soc MARCH=rv32imc_zicsr_zifencei compile"
  --run-cmd "sbt runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf $PROJECT/main.elf --with-rvc --with-rvm --with-rdtime"
  --checkpoint
  --thread-id "${CFU_AGENT_THREAD_ID:-example-c-project-u8-wavg}"
  --stream
  --max-iters "${CFU_AGENT_MAX_ITERS:-1}"
)

# Planning is the default for these examples: they run read-only unless a
# capability env below opts in. Patch application and allowlisted commands are
# framework defaults now, so each effect that was not requested is disabled.
if [[ "${CFU_AGENT_APPLY_PATCH:-0}" == "1" || "${CFU_AGENT_RUN_COMMANDS:-0}" == "1" ]]; then
  [[ "${CFU_AGENT_APPLY_PATCH:-0}" == "1" ]] || args+=(--no-apply-patch)
  [[ "${CFU_AGENT_RUN_COMMANDS:-0}" == "1" ]] || args+=(--no-run-commands)
else
  args+=(--dry-run)
fi
if [[ "${CFU_AGENT_SKIP_SIM:-1}" == "1" ]]; then
  args+=(--skip-sim)
fi
if [[ "${CFU_AGENT_SKIP_YOSYS:-1}" == "1" ]]; then
  args+=(--skip-yosys)
fi

cd "$ROOT"
python3 -B agents/run_cfu_agent.py "${args[@]}"
