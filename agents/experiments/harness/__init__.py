"""Shared experiment infrastructure for the CFU design agent's run scripts.

Layout: one module per concern -- ``layers``/``naming``/``scheduling`` shape the
matrix, ``locking``/``execution`` run it, ``parsing``/``collection``/
``classification`` read outcomes back, ``metrics``/``reporting`` aggregate
them, and ``experiment``/``cli`` hold experiment metadata and the orchestration
flow.  ``run_matrix.py`` (pass@k matrix) and ``run_ablation.py`` (capability
ladder) are thin presets over ``cli.main``.

This package re-exports the harness's public surface for convenience (the
pass@k harness historically lived in ``run_matrix`` and callers reach its
helpers by name).  When a TEST needs to patch a seam, patch the OWNING module
(e.g. ``harness.execution.run_logged``): an attribute assigned on a re-export
is a separate binding and is not observed by the owning module's code.
"""

from __future__ import annotations

__all__ = [
    "ABLATION_FLAG_NAMES",
    "ABLATION_LADDER",
    "ABLATION_LAYERS",
    "BENCHMARK_SLUGS",
    "DEFAULT_LAYERS",
    "ESTIMAND_HELP",
    "FAILURE_CLASSES",
    "FAILURES_COLUMNS",
    "GUARDED_BUILD_RUNGS",
    "INFRA_PATTERNS",
    "LAYER_ALIASES",
    "MODEL_SLUGS",
    "PASS_AT_K",
    "PASS_AT_K_COLUMNS",
    "RUNG_ORDER",
    "RUNS_COLUMNS",
    "SBT_SHIM",
    "SHIM_PATH_WRAPPER",
    "SUITE",
    "agent_script_args",
    "aggregate",
    "archive_stale_run_dir",
    "benchmark_slug",
    "bootstrap_budget_ci",
    "budget_pass_at_k",
    "build_argv",
    "build_preflight_argv",
    "cell_metrics",
    "child_env",
    "classify_failure",
    "collect_record",
    "collect_sim_attempts",
    "detect_infra",
    "effective_ablation_flags",
    "execute_one",
    "expand_matrix",
    "filter_jobs",
    "framework_artifacts_are_stale",
    "git_info",
    "job_layer_flags",
    "job_run_name",
    "launcher_argv",
    "layer_flags",
    "layer_slug",
    "load_records",
    "load_yaml",
    "make_run_name",
    "materialize_configs",
    "materialize_experiment",
    "model_slug",
    "parse_cli_stdout",
    "parse_report",
    "preflight",
    "read_experiment_yaml",
    "reconstruct_episodes",
    "reference_result_value",
    "require_flock",
    "resolve_layers",
    "run_logged",
    "sample_pass_at_k",
    "save_record",
    "sbt_lock_env",
    "schedule_jobs",
    "slugify",
    "verify_failure_in_attempt",
    "write_experiment_yaml",
    "write_failure_json",
    "write_failures_csv",
    "write_pass_at_k_csv",
    "write_runs_csv",
    "write_sbt_shim",
    "write_summary_md",
]

from agents.experiments.harness.classification import (  # noqa: F401
    FAILURE_CLASSES,
    GUARDED_BUILD_RUNGS,
    INFRA_PATTERNS,
    RUNG_ORDER,
    classify_failure,
    detect_infra,
    reconstruct_episodes,
)
from agents.experiments.harness.collection import (  # noqa: F401
    SUITE,
    collect_record,
    collect_sim_attempts,
    framework_artifacts_are_stale,
    reference_result_value,
    save_record,
    verify_failure_in_attempt,
    write_failure_json,
)
from agents.experiments.harness.configs import materialize_configs  # noqa: F401
from agents.experiments.harness.execution import (  # noqa: F401
    agent_script_args,
    archive_stale_run_dir,
    build_argv,
    build_preflight_argv,
    execute_one,
    git_info,
    preflight,
    run_logged,
)
from agents.experiments.harness.experiment import (  # noqa: F401
    materialize_experiment,
    read_experiment_yaml,
    write_experiment_yaml,
)
from agents.experiments.harness.layers import (  # noqa: F401
    ABLATION_FLAG_NAMES,
    ABLATION_LADDER,
    ABLATION_LAYERS,
    DEFAULT_LAYERS,
    LAYER_ALIASES,
    effective_ablation_flags,
    job_layer_flags,
    layer_flags,
    resolve_layers,
)
from agents.experiments.harness.locking import (  # noqa: F401
    SBT_SHIM,
    SHIM_PATH_WRAPPER,
    child_env,
    launcher_argv,
    require_flock,
    sbt_lock_env,
    write_sbt_shim,
)
from agents.experiments.harness.metrics import (  # noqa: F401
    ESTIMAND_HELP,
    PASS_AT_K,
    bootstrap_budget_ci,
    budget_pass_at_k,
    cell_metrics,
    sample_pass_at_k,
)
from agents.experiments.harness.naming import (  # noqa: F401
    BENCHMARK_SLUGS,
    MODEL_SLUGS,
    benchmark_slug,
    job_run_name,
    layer_slug,
    make_run_name,
    model_slug,
    slugify,
)
from agents.experiments.harness.parsing import (  # noqa: F401
    load_yaml,
    parse_cli_stdout,
    parse_report,
)
from agents.experiments.harness.reporting import (  # noqa: F401
    FAILURES_COLUMNS,
    PASS_AT_K_COLUMNS,
    RUNS_COLUMNS,
    aggregate,
    load_records,
    write_failures_csv,
    write_pass_at_k_csv,
    write_runs_csv,
    write_summary_md,
)
from agents.experiments.harness.scheduling import (  # noqa: F401
    expand_matrix,
    filter_jobs,
    schedule_jobs,
)
