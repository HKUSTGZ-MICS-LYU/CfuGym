#!/usr/bin/env bash
# Two-rung ablation over the CFU design agent's ISA-contract and conformance
# capabilities, with LLVM IR and visibility held at the full default.
#
#   cell A  isa-contract ON  + conformance dev   (the plain full flow)
#   cell B  isa-contract ON  + conformance OFF   (--no-conformance)
#   cell C  isa-contract OFF + conformance OFF   (--no-isa-contract implies off)
#
# The 2x2 cell "isa-contract OFF + conformance ON" does not exist: the
# conformance oracle consumes the contract, so --no-isa-contract implies
# --conformance-mode off (see "Ablation flags" in agents/README.md). That leaves
# exactly two single-variable steps:
#   A vs B  = the conformance capability (dev tests as self-check evidence)
#   B vs C  = the ISA-contract stage (contract-authored isa_spec + intrinsics)
#
# Everything else is pinned: same commit, same working tree (checked before
# every cell), same LLM config incl. output_reserve_tokens, same seed, same
# max-iters, IR on, visibility extended, cost backend vivado.
#
# Usage:
#   agents/experiments/ablation_isa_conformance.sh            # warmup + 3 cells
#   WARMUP=0 agents/experiments/ablation_isa_conformance.sh   # cache already warm
#   REPS=5 agents/experiments/ablation_isa_conformance.sh     # more repeats
#   ONLY=isa-off-conf-off agents/experiments/ablation_isa_conformance.sh
#   DRY=1 agents/experiments/ablation_isa_conformance.sh      # print, run nothing
#
# Re-running the same TAG resumes: the harness skips a run whose terminal
# record.json already exists (pass --force manually to redo one).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BENCHMARKS="${BENCHMARKS:-matmult-int,md5sum,nettle-aes}"
MODEL="${MODEL:-gpt-5.6-terra}"
REPS="${REPS:-3}"
JOBS="${JOBS:-1}"
MAX_ITERS="${MAX_ITERS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-114514}"
WARMUP="${WARMUP:-1}"
DRY="${DRY:-0}"
ONLY="${ONLY:-}"
TAG="${TAG:-isa-conf-$(date +%Y%m%d)}"

CELL_A="isa-on-conf-on"
CELL_B="isa-on-conf-off"
CELL_C="isa-off-conf-off"
cell_flags() {
  case "$1" in
    "$CELL_A") printf '%s' "" ;;
    "$CELL_B") printf '%s' "--no-conformance" ;;
    "$CELL_C") printf '%s' "--no-isa-contract" ;;
    *) echo "unknown cell $1" >&2; exit 2 ;;
  esac
}

# A run's provenance is only as good as the tree it ran on: record HEAD plus a
# content hash of the unstaged diff and refuse to mix cells across a change.
fingerprint() {
  { git rev-parse HEAD; git diff | sha256sum; git status --porcelain | sha256sum; } \
    | sha256sum | cut -c1-16
}

budget_field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2]))
PY
}

FP0="$(fingerprint)"
BASE_CONFIG="agents/llm_configs/${MODEL}.json"
echo "tag:            $TAG"
echo "tree:           $(git rev-parse --short HEAD) dirty-tree=$(git status --porcelain | wc -l) fp=$FP0"
echo "model:          $MODEL  cfg=$BASE_CONFIG"
echo "  temperature=$TEMPERATURE  output_reserve_tokens=$(budget_field "$BASE_CONFIG" output_reserve_tokens)"
echo "benchmarks:     $BENCHMARKS"
echo "repeats:        $REPS   max-iters=$MAX_ITERS  jobs=$JOBS  seed=$SEED"
echo "cells:          $CELL_A (contract+conf) | $CELL_B (contract only) | $CELL_C (neither)"
echo

if [[ "$WARMUP" == "1" && -z "$ONLY" ]]; then
  echo "== [warmup] refresh the Embench profile cache (2 builds + 2 sims per benchmark, no LLM)"
  if [[ "$DRY" == "1" ]]; then
    echo "   conda run -n mico_env python3 -B agents/experiments/run_sweep.py --exp-id ${TAG}-warmup --raw --benchmarks $BENCHMARKS"
  else
    conda run -n mico_env python3 -B agents/experiments/run_sweep.py \
      --exp-id "${TAG}-warmup" --raw --benchmarks "$BENCHMARKS"
  fi
  echo
fi

run_cell() {
  local cell="$1"
  local exp="${TAG}-${cell}"
  local extra
  extra="$(cell_flags "$cell")"
  echo "== [$cell] exp-id=$exp"
  if [[ "$(fingerprint)" != "$FP0" ]]; then
    echo "ABORT: the working tree changed since $FP0; cells would not be comparable." >&2
    exit 3
  fi
  # shellcheck disable=SC2086  # $extra is a deliberately word-split flag list
  local cmd=(conda run -n mico_env python3 -B agents/experiments/run_sweep.py
    --exp-id "$exp"
    --benchmarks "$BENCHMARKS"
    --models "$MODEL"
    --repeats "$REPS"
    --jobs "$JOBS"
    --max-iters "$MAX_ITERS"
    --temperature "$TEMPERATURE"
    --seed "$SEED"
    --visibility extended
    --conda-env mico_env
    -- $extra --cost-backend vivado)
  if [[ "$DRY" == "1" ]]; then
    printf '   %q ' "${cmd[@]}"; echo
    return 0
  fi
  "${cmd[@]}"
  echo
}

for cell in "$CELL_A" "$CELL_B" "$CELL_C"; do
  if [[ -n "$ONLY" && "$ONLY" != "$cell" ]]; then
    continue
  fi
  run_cell "$cell"
done

if [[ "$DRY" == "1" ]]; then
  echo "(dry run: nothing was launched)"
  exit 0
fi

if [[ "$(fingerprint)" != "$FP0" ]]; then
  echo "WARNING: the working tree changed during the sweep ($FP0 -> $(fingerprint));" >&2
  echo "         cells are not directly comparable." >&2
fi

# The three cells must differ in exactly the ablation flags: compare the
# behavioural fields of the generated configs (name/description differ by design).
python3 - "$TAG" "$CELL_A" "$CELL_B" "$CELL_C" "$MODEL" <<'PY'
import json, sys
tag, cells, model = sys.argv[1], sys.argv[2:5], sys.argv[5]
keys = ("model", "base_url", "api_mode", "temperature", "thinking",
        "reasoning_effort", "context_window_tokens", "output_reserve_tokens", "timeout")
seen = {}
for cell in cells:
    path = f"agents/llm_configs/{tag}-{cell}-{model}.json"
    try:
        cfg = json.load(open(path))
    except OSError:
        continue
    seen[cell] = {k: cfg.get(k) for k in keys}
if len(seen) > 1:
    ref_cell, ref = next(iter(seen.items()))
    for cell, cfg in seen.items():
        for k in keys:
            if cfg[k] != ref[k]:
                print(f"WARNING: {cell} differs from {ref_cell} on {k}: {cfg[k]!r} != {ref[k]!r}")
print("config check:", "identical" if len(set(map(str, seen.values()))) <= 1 else "DIFFERENCES ABOVE")
PY

# Per-benchmark cells and the two ladder steps.
python3 - "$TAG" "$CELL_A" "$CELL_B" "$CELL_C" "$MODEL" <<'PY'
import csv, statistics as st, sys

tag, cells, model = sys.argv[1], sys.argv[2:5], sys.argv[5]
labels = {"isa-on-conf-on": "A contract+conf", "isa-on-conf-off": "B contract",
          "isa-off-conf-off": "C baseline"}

def load(cell):
    path = f"agents/generated/experiments/{tag}-{cell}/results/runs.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return None
    return [r for r in rows if r["model"] == model and r["status"] != "initialized"]

def fmt(r):
    ok = "ok" if r["status"] == "complete" else r["status"][:4].upper()
    att = r["attempt_index"] or "-"
    sp = r["display_speedup_x100"]
    sp = f"{int(sp)/100:.1f}x" if sp else "-"
    lut = r["vivado_cfu_lut"]
    return f"{ok} a{att} {sp}" + (f" {lut}L" if lut else "")

data = {c: load(c) for c in cells}
benches = sorted({r["benchmark"] for rows in data.values() if rows for r in rows})
head = f'{"bench":<14}' + "".join(f'{labels.get(c, c):<26}' for c in cells)
print()
if benches:
    print(head)
    print("-" * len(head))
for b in benches:
    line = f"{b:<14}"
    for c in cells:
        rows = [r for r in (data[c] or []) if r["benchmark"] == b]
        rows.sort(key=lambda r: (r["status"] != "complete", r["attempt_index"] or 99))
        line += f"{(fmt(rows[0]) if rows else '-'):<26}"
    print(line)

print()
print(f'{"cell":<22}{"runs":>5}{"pass@1":>8}{"pass@5":>8}{"med_att":>9}{"med_wall_h":>12}')
for c in cells:
    rows = data[c]
    if not rows:
        print(f"{labels.get(c, c):<22}{'0':>5}{'-':>8}{'-':>8}{'-':>9}{'-':>12}")
        continue
    def passes(k):
        n = 0
        for r in rows:
            if r["status"] != "complete":
                continue
            try:
                n += int(r["attempt_index"]) <= k
            except (TypeError, ValueError):
                pass
        return f"{n/len(rows):.2f}"
    atts = [int(r["attempt_index"]) for r in rows if r["attempt_index"]]
    walls = [float(r["wall_seconds"]) / 3600 for r in rows if r["wall_seconds"]]
    print(f'{labels.get(c, c):<22}{len(rows):>5}{passes(1):>8}{passes(5):>8}'
          f'{(st.median(atts) if atts else 0):>9.1f}{(st.median(walls) if walls else 0):>12.1f}')
PY

echo
echo "runs:   agents/generated/experiments/${TAG}-<cell>/runs/"
echo "tables: agents/generated/experiments/${TAG}-<cell>/results/"
