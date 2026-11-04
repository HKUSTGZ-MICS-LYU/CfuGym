"""Experiment metadata: ``experiment.yaml`` materialization and (re)loading."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agents.experiments.harness import execution, layers, metrics


def materialize_experiment(opts) -> dict:
    is_ladder = any(name in layers.ABLATION_LADDER for name in opts.layers)
    extra_flags = [arg for arg in getattr(opts, "agent_args", ()) if arg in layers.ABLATION_FLAG_NAMES]
    definitions = {}
    for name in opts.layers:
        entry = layers.ABLATION_LAYERS[name]
        definitions[name] = {
            "flags": list(layers.effective_ablation_flags((*layers.layer_flags(name), *extra_flags))),
            "adds": entry["adds"],
            "visibility": layers.layer_visibility(name, opts.visibility),
        }
        if name in layers.ABLATION_LADDER:
            definitions[name].update(
                conformance_mode=entry["conformance_mode"],
                conformance_promote=False,
                max_conformance_runs=getattr(opts, "max_conformance_runs", 2),
            )
    templates = {
        name: execution.build_argv(
            opts.python, opts.conda_env, None, "<config>", "<benchmark>", "<run_name>",
            opts.max_iters, entry["visibility"], opts.profile_cache,
            layers.job_run_flags({**entry, "ablation_flags": entry["flags"]}),
            [arg for arg in getattr(opts, "agent_args", ()) if arg not in layers.ABLATION_FLAG_NAMES],
        )
        for name, entry in definitions.items()
    }
    meta = {
        "ablation_scheme": layers.ABLATION_SCHEME if is_ladder else "full-flow-v1",
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
        "visibility": "per-layer" if is_ladder else opts.visibility,
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
        "argv_template": templates[opts.layers[0]],
        "argv_templates_by_layer": templates,
        "agent_args": list(getattr(opts, "agent_args", ()) or ()),
        "layers": definitions,
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
