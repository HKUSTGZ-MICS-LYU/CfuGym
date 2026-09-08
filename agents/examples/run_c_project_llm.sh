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
