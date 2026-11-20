EXP_ID="abl-v3-$(date +%Y%m%d-%H%M%S)"
MODELS="gpt-6-luna"
BENCHMARKS="depthconv,matmult-int"

conda run -n mico_env --no-capture-output \
  python3 agents/experiments/run_ablation.py \
  --exp-id "$EXP_ID" \
  --models "$MODELS" \
  --benchmarks "$BENCHMARKS" \
  --layers ladder \
  --repeats 3 \
  --max-iters 5 \
  --max-conformance-runs 5 \
  --temperature 1.0 \
  --jobs 1 \
  --conda-env mico_env \
  -- --cost-backend vivado