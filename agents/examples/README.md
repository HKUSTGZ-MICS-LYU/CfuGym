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

The framework default is autonomous (patch application and allowlisted commands are enabled), but both runners pass `--dry-run --skip-sim --skip-yosys` themselves, so the agent only plans: it writes run artifacts under `agents/generated/runs/<run_id>/` but never mutates CFU source or launches long hardware flows.

To let the same example apply a patch or run hardware commands, pass capability flags (each effect that is not requested stays disabled):

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
