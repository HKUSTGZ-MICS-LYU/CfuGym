"""Experiment metadata: ``experiment.yaml`` materialization and (re)loading."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agents.experiments.harness import execution, layers, metrics


def materialize_experiment(opts) -> dict:
    meta = {
        "exp_id": opts.exp_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": metrics.ESTIMAND_HELP,
        "models": opts.models,
        "benchmarks": opts.benchmarks,
        "repeats": opts.repeats,
        "max_iters": opts.max_iters,
        "attempt_budget": opts.max_iters,
        "temperature": opts.temperature,
        "conda_env": opts.conda_env,
        "python": opts.python,
        "visibility": opts.visibility,
        "profile_cache": opts.profile_cache,
        "jobs": opts.jobs,
        "timeout": opts.timeout,
        "seed": opts.seed,
        "infra_retries": opts.infra_retries,
        "infra_backoff": opts.infra_backoff,
        "rate_limit_backoff": opts.rate_limit_backoff,
        "sbt_lock": {
            "enabled": opts.sbt_lock,
            "file": str(opts.sbt_lock_file),
            "timeout": opts.sbt_lock_timeout,
        },
        "argv_template": execution.build_argv(
            opts.python, opts.conda_env, None, "<config>", "<benchmark>", "<run_name>",
            opts.max_iters, opts.visibility, opts.profile_cache,
            (), list(getattr(opts, "agent_args", ()) or ()),
        ),
        "agent_args": list(getattr(opts, "agent_args", ()) or ()),
        "layers": {
            name: {
                "flags": list(layers.ABLATION_LAYERS[name]["flags"]),
                "adds": layers.ABLATION_LAYERS[name]["adds"],
            }
            for name in opts.layers
        },
    }
    meta.update(execution.git_info(opts.workdir))
    return meta


def write_experiment_yaml(path: Path, meta: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))


def read_experiment_yaml(path: Path) -> dict:
    """Load an experiment's ``experiment.yaml`` meta; a missing file gives ``{}``."""
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}
