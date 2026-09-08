# CFU Agent Examples

These examples cover the two supported input kinds for real LLM integration tests.

## Natural-Language Input

```bash
bash agents/examples/run_natural_language_llm.sh
```

This reads `natural_language_u8_weighted_average.txt`, lets the agent normalize it into a `WorkloadSpec`, and creates a generated benchmark workspace before CFU analysis.

## C-Project Input

```bash
bash agents/examples/run_c_project_llm.sh
```

This points the agent at `c_project_u8_wavg/`, copies the project into `agents/generated/workspaces/`, instruments the copied `main`, and uses the copied project for later profiling.

Both runners default to `--dry-run --skip-sim --skip-yosys`, so the agent only plans: it writes run artifacts under `agents/generated/runs/<run_id>/` but never mutates CFU source or launches long hardware flows.

Read-only behavior is the framework default. To let the same example apply a patch or run hardware commands, pass capability flags:

```bash
CFU_AGENT_APPLY_PATCH=1 bash agents/examples/run_natural_language_llm.sh
CFU_AGENT_RUN_COMMANDS=1 bash agents/examples/run_c_project_llm.sh
```

Useful overrides:

```bash
CFU_AGENT_MAX_ITERS=2 bash agents/examples/run_natural_language_llm.sh
CFU_AGENT_SKIP_SIM=0 CFU_AGENT_SKIP_YOSYS=1 bash agents/examples/run_c_project_llm.sh
```

Set `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and optional `CFU_AGENT_MODEL` before running with a real LLM endpoint.
