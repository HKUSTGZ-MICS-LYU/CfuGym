"""Shared CLI + orchestration for the experiment presets.

``build_parser``/``main`` implement the whole harness flow (preflight, matrix
expansion, scheduling, per-run execution with infra retries, collection,
aggregation).  The run scripts are presets over this module:

    run_matrix.py   --layers full     (the pass@k matrix, un-ablated)
    run_ablation.py --layers ladder   (the full capability ladder)
    run_sweep.py    all embench-iot benchmarks once, in order

Every preset additionally forwards the arguments after a bare ``--``
verbatim to ``agents/run_cfu_agent.py`` (see ``split_agent_args``).
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.cfu_design_agent.runtime import processes
from agents.experiments.harness import (
    collection,
    configs,
    execution,
    experiment,
    layers,
    locking,
    naming,
    raw,
    reporting,
    scheduling,
)

def _run_jobs(jobs, worker, max_workers):
    """Cancel registered agent trees before joining interrupted pool threads."""
    token = processes.CancellationToken()

    def run_one(job):
        with processes.use_token(token):
            return worker(job)

    old_term = None
    if threading.current_thread() is threading.main_thread():
        old_term = signal.getsignal(signal.SIGTERM)
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'experiment interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupt)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            try:
                futures = [pool.submit(run_one, job) for job in jobs]
                results = []
                for future in futures:
                    results.extend(future.result())
                return results
            except BaseException:
                for future in futures:
                    future.cancel()
                token.cancel()
                raise
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODELS = ("deepseek-pro", "glm-5.2", "gpt-5.6-terra")
DEFAULT_BENCHMARKS = ("crc32", "matmult-int", "md5sum")
# Everything the harness launches runs inside the mico_env conda environment
# (sbt, the toolchain and the simulator all live there).
DEFAULT_CONDA_ENV = "mico_env"
DEFAULT_PYTHON = "python3 -B"

#: Flags the harness sets on every launched run itself, plus the artifact
#: locations its collection depends on.  None may come through the ``--``
#: passthrough: a duplicate would be last-one-wins in the agent's argparse
#: while the harness still recorded its own value (and --workdir/--out-dir
#: would silently move the artifacts out of the harness's reach).  The
#: ablation switches are NOT reserved: they pass through and are folded into
#: the run's recorded ablation flags (``layers.effective_ablation_flags``).
RESERVED_AGENT_FLAGS = frozenset({
    "--embench", "--llm-config", "--run-id", "--max-iters", "--visibility",
    "--profile-cache", "--workdir", "--out-dir", "--profile-only",
})

AGENT_ARG_HELP = (
    "Arguments after a bare '--' are forwarded verbatim to "
    "agents/run_cfu_agent.py for every run (its original parameters, e.g. "
    "--skip-sim, --skip-yosys, --task, --no-conformance).  Harness-owned "
    "flags are rejected: use --models/--max-iters/--visibility/"
    "--profile-cache for the corresponding agent options.  Ablation "
    "switches passed here are recorded in each run's ablation flags; "
    "--layers is the way to schedule the capability ladder."
)


def build_parser(
    description: str | None = None,
    default_layers: str = "full",
    default_models=DEFAULT_MODELS,
    default_benchmarks=DEFAULT_BENCHMARKS,
    default_repeats: int = 5,
    supports_raw: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=AGENT_ARG_HELP,
    )
    parser.add_argument("--exp-id", required=True, help="Experiment batch id, e.g. pk5-20260212 (run-id safe).")
    parser.add_argument("--models", default=",".join(default_models), help="Comma list of agents/llm_configs config names.")
    # Default None -> the preset default (a fixed list, or discovery over the
    # suite) is resolved in main() once --workdir is known.
    parser.add_argument("--benchmarks", default=None, help="Comma list of embench-iot benchmark names.")
    parser.add_argument("--repeats", type=int, default=default_repeats, help="Independent repeats per cell (default: 5).")
    parser.add_argument("--layers", default=default_layers,
                        help="Comma list, or 'ladder': l0-none -> l1-isa -> l2-vis -> l3-devconf -> l4-ir "
                             "(ISA Contract -> visibility -> dev conformance -> IR). "
                             f"Default: {default_layers!r}. 'full' is the separate legacy flow. "
                             "Aliases: none/l0, isa/l1, vis/l2, devconf/l3, ir/l4.")
    parser.add_argument("--max-iters", type=int, default=5, help="Maximum candidate submissions to verification; K = attempt budget for pass@K (default: 5).")
    parser.add_argument("--max-conformance-runs", type=int, default=5,
                        help="New ladder: run_conformance calls per Implement session (default: 2), not a whole-run limit.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature forced into the experiment configs (default: 0.7).")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV,
                        help=f"Conda environment every launched command runs in (default: {DEFAULT_CONDA_ENV!r}; empty disables conda).")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help=f"Python interpreter + flags launched inside the conda env (default: {DEFAULT_PYTHON!r}).")
    parser.add_argument("--workdir", default=str(REPO_ROOT), help="Repo root handed to the agent as --workdir.")
    parser.add_argument("--out-root", default="", help="Experiment output root (default: <workdir>/agents/generated/experiments).")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent runs (sbt/RTL-sim stay serialized via the lock shim; default: 1).")
    parser.add_argument("--timeout", type=int, default=10800, help="Per-run wall-clock timeout in seconds (default: 3h).")
    parser.add_argument("--seed", type=int, default=114514, help="Scheduling shuffle seed.")
    parser.add_argument("--visibility", default="extended", choices=("basic", "standard", "extended"),
                        help="Legacy full-flow visibility (default: standard). New ladder layers "
                             "have fixed basic/basic/extended/extended/extended tiers; omit this option.")
    parser.add_argument("--profile-cache", default="on", choices=("on", "off", "refresh"))
    parser.add_argument("--infra-retries", type=int, default=2, help="Harness retries for infrastructure failures only (default: 2).")
    parser.add_argument("--infra-backoff", type=float, default=60.0,
                        help="Seconds to wait (jittered 1.0-1.5x) before retrying an infrastructure failure (default: 60).")
    parser.add_argument("--rate-limit-backoff", type=float, default=300.0,
                        help="Minimum backoff for a rate-limited (HTTP 429) retry, overriding --infra-backoff (default: 300).")
    parser.add_argument("--sbt-lock", dest="sbt_lock", action="store_true", default=True,
                        help="Serialize sbt/RTL-sim calls through a flock shim (default: on).")
    parser.add_argument("--no-sbt-lock", dest="sbt_lock", action="store_false",
                        help="Disable the sbt lock shim (then run with --jobs 1).")
    parser.add_argument("--sbt-lock-file", default="", help="Lock file (default: <out-root>/.locks/sbt.lock, shared across batches).")
    parser.add_argument("--sbt-lock-timeout", type=int, default=14400, help="Seconds to wait for the sbt lock (default: 4h).")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples for the CI (default: 10000).")
    parser.add_argument("--only", default="", help="Comma list of substrings; keep only runs whose name contains ALL of them.")
    parser.add_argument("--dry-run", action="store_true", help="Write experiment.yaml and print the schedule; do not execute.")
    parser.add_argument("--collect-only", action="store_true", help="Re-collect records from existing run artifacts; do not execute.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild results/*.csv and summary.md.")
    parser.add_argument("--force", action="store_true", help="Re-run even when a terminal record.json exists.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the --test-api connectivity preflight.")
    parser.add_argument("--allow-preflight-fail", action="store_true", help="Continue even if a preflight fails.")
    if supports_raw:
        parser.add_argument(
            "--raw", action="store_true",
            help="Rebuild the Embench reference/profile measurement cache for every benchmark and "
                 "stop: no agent, no LLM, no experiment records. Use it after a repository change "
                 "invalidates the cache. Ignores --models/--repeats/--layers; flags after a bare "
                 "'--' still apply because they shape the measured build.",
        )
    return parser


def _split(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _sbt_lock(opts, out_root: Path, exp_dir: Path) -> tuple[Path | None, dict]:
    """Resolve the lock file and write the PATH shim when sbt will be launched.

    Resolving the lock file path is unconditional (an archive read still reports
    where the lock lives); writing the shim is not, so a command rejected by the
    scheduling controls leaves no ``<exp>/bin`` behind.
    """
    lock_file = Path(opts.sbt_lock_file) if opts.sbt_lock_file else out_root / ".locks" / "sbt.lock"
    if not lock_file.is_absolute():
        lock_file = opts.workdir / lock_file
    opts.sbt_lock_file = lock_file
    if not opts.sbt_lock or opts.collect_only:
        return None, {}
    locking.require_flock()
    shim_dir = exp_dir / "bin"
    locking.write_sbt_shim(shim_dir)
    return shim_dir, locking.sbt_lock_env(lock_file, opts.sbt_lock_timeout)


def _validate_raw(opts) -> None:
    """Reject combinations that would make a cache rebuild meaningless."""
    if opts.aggregate_only or opts.collect_only or opts.dry_run:
        raise SystemExit("--raw cannot be combined with --aggregate-only/--collect-only/--dry-run")
    blocked = raw.blocked_agent_args(getattr(opts, "agent_forward", ()))
    if blocked:
        raise SystemExit(
            "--raw rebuilds the measurement, so these forwarded flags are rejected: "
            + ", ".join(sorted(set(blocked)))
        )


def split_agent_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at the first bare ``--``.

    Everything after it is forwarded verbatim to ``agents/run_cfu_agent.py``
    for every run: the pass-through of that CLI's original parameters.
    """
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1:]


def validate_agent_args(agent_args: list[str]) -> None:
    """Reject passthrough flags the harness owns (see ``RESERVED_AGENT_FLAGS``)."""
    for arg in agent_args:
        flag = str(arg).split("=", 1)[0]
        if flag.startswith("--") and any(owned.startswith(flag) for owned in RESERVED_AGENT_FLAGS):
            raise SystemExit(
                f"{flag} is owned by the harness and cannot be passed after '--': "
                "use --models/--max-iters/--visibility/--profile-cache for "
                "the corresponding agent options; --workdir/--out-dir are fixed by "
                "the harness so its collection can find the run artifacts."
            )


def main(
    argv: list[str] | None = None,
    *,
    default_layers: str = "full",
    description: str | None = None,
    default_models=DEFAULT_MODELS,
    default_benchmarks=DEFAULT_BENCHMARKS,
    default_repeats: int = 5,
    ordered: bool = False,
    supports_raw: bool = False,
) -> int:
    """Run one experiment preset.

    ``default_benchmarks`` may be a list, or a callable ``(workdir) -> list[str]``
    resolved once ``--workdir`` is known (suite discovery).  ``ordered=True``
    keeps the expansion order (benchmark order) instead of the seeded
    round-robin shuffle, and is only strict with ``--jobs 1``.
    """
    cli_argv, agent_args = split_agent_args(list(sys.argv[1:] if argv is None else argv))
    validate_agent_args(agent_args)
    opts = build_parser(
        description, default_layers, default_models, default_benchmarks, default_repeats, supports_raw,
    ).parse_args(cli_argv)
    # ``agent_args`` stays verbatim for experiment.yaml; the ablation switches
    # in it are folded into each job's recorded ablation flags instead of
    # being forwarded twice (argv + job flags), and everything else is the
    # plain per-run argv tail.
    opts.agent_args = list(agent_args)
    ablation_args = [a for a in agent_args if a in layers.ABLATION_FLAG_NAMES]
    opts.agent_forward = [a for a in agent_args if a not in layers.ABLATION_FLAG_NAMES]
    if getattr(opts, "raw", False):
        _validate_raw(opts)
    opts.workdir = Path(opts.workdir).resolve()
    opts.models = _split(opts.models)
    if opts.benchmarks is not None:
        opts.benchmarks = _split(opts.benchmarks)
    else:
        opts.benchmarks = list(
            default_benchmarks(opts.workdir) if callable(default_benchmarks) else default_benchmarks
        )
        if not opts.benchmarks:
            raise SystemExit(
                f"no benchmarks resolved from {opts.workdir}; pass --benchmarks explicitly"
            )
    opts.only = _split(opts.only)
    out_root = Path(opts.out_root) if opts.out_root else opts.workdir / "agents" / "generated" / "experiments"
    if not out_root.is_absolute():
        out_root = opts.workdir / out_root
    exp_dir = out_root / opts.exp_id
    opts.exp_dir_ref = exp_dir
    prior_experiment = experiment.read_experiment_yaml(exp_dir / "experiment.yaml")
    # Archive reads must not resolve old layer names through the new vocabulary,
    # write a shim/config, or overwrite the experiment's historical controls.
    if opts.aggregate_only:
        result = reporting.aggregate(exp_dir, prior_experiment, opts.bootstrap)
        print(f"aggregated: {result['records']} runs, {result['cells']} cells -> {result['summary']}")
        return 0
    # A cache rebuild is not an experiment: dispatch before anything reads the
    # scheduling knobs, so `--layers`/`--visibility` cannot reject a command that
    # will not schedule a run, and before experiment.yaml, the LLM configs, the
    # preflight and the run records are written.
    if getattr(opts, "raw", False):
        shim_dir, lock_env = _sbt_lock(opts, out_root, exp_dir)
        return raw.warm(opts, exp_dir, shim_dir, lock_env)
    if opts.collect_only and prior_experiment.get("layers"):
        opts.layers = list(prior_experiment["layers"])
        for key in ("models", "benchmarks", "repeats", "max_iters"):
            if key in prior_experiment:
                setattr(opts, key, prior_experiment[key])
        saved_visibility = prior_experiment.get("visibility")
        if saved_visibility in ("basic", "standard", "extended"):
            opts.visibility = saved_visibility
    else:
        opts.layers = layers.resolve_layers(opts.layers)
    is_ladder = any(name in layers.ABLATION_LADDER for name in opts.layers)
    if not opts.collect_only and prior_experiment.get("ablation_scheme") in (layers.ABLATION_SCHEME, layers.V2_ABLATION_SCHEME) and not is_ladder:
        raise SystemExit("ablation scheme mismatch: legacy full cannot overwrite a new ladder batch; use a new --exp-id")
    if is_ladder and not opts.collect_only:
        if "full" in opts.layers:
            raise SystemExit("legacy full is separate from the new ladder; do not mix it with new layers")
        if opts.visibility is not None:
            raise SystemExit("new ablation layers fix --visibility to basic/basic/extended/extended/extended; omit --visibility")
        dimension_flags = ("--no-isa-contract", "--no-llvm-ir", "--ir-features",
                           "--no-conformance", "--conformance-mode", "--conformance-promote",
                           "--max-conformance-runs")
        conflicts = {
            flag for flag in (arg.split("=", 1)[0] for arg in agent_args)
            if flag.startswith("--") and any(owned.startswith(flag) for owned in dimension_flags)
        }
        if conflicts:
            raise SystemExit("agent passthrough changes fixed ablation dimensions: " + ", ".join(sorted(conflicts)))
        if prior_experiment or (exp_dir / "runs").exists():
            if prior_experiment.get("ablation_scheme") != layers.ABLATION_SCHEME:
                raise SystemExit("ablation scheme mismatch with existing experiment; use a new --exp-id (old results remain aggregate-only)")
            for name in opts.layers:
                saved = (prior_experiment.get("layers") or {}).get(name, {})
                current = layers.ABLATION_LAYERS[name]
                if (tuple(saved.get("flags", ())) != tuple(current["flags"])
                        or saved.get("visibility") != current.get("visibility", opts.visibility or "standard")
                        or saved.get("conformance_mode") != current["conformance_mode"]
                        or saved.get("conformance_promote") is not False
                        or saved.get("max_conformance_runs") != opts.max_conformance_runs):
                    raise SystemExit(f"ablation layer settings mismatch for {name}; use a new --exp-id")
    opts.visibility = opts.visibility or "standard"
    if opts.repeats < 1 or opts.max_iters < 1:
        raise SystemExit("--repeats and --max-iters must be >= 1")
    if is_ladder and not opts.collect_only and opts.max_conformance_runs < 1:
        raise SystemExit("--max-conformance-runs must be >= 1 for the new ladder")
    if opts.sbt_lock and opts.jobs > 1 and shutil.which("flock") is None:
        raise SystemExit("flock(1) unavailable: use --no-sbt-lock --jobs 1 or install util-linux")
    jobs = scheduling.expand_matrix(opts.models, opts.benchmarks, opts.repeats, opts.layers)
    if not ordered:
        jobs = scheduling.schedule_jobs(jobs, opts.seed)
    for job in jobs:
        job["exp_id"] = opts.exp_id
        job["max_iters"] = opts.max_iters
        job["ablation_scheme"] = (
            prior_experiment.get("ablation_scheme") if opts.collect_only
            else layers.ABLATION_SCHEME if is_ladder else "full-flow-v1"
        )
        if opts.collect_only and prior_experiment.get("layers"):
            entry = prior_experiment["layers"].get(job["layer"], {})
            job["ablation_flags"] = tuple(entry.get("flags", ()))
            job["visibility"] = entry.get("visibility") or opts.visibility
            for key in ("conformance_mode", "conformance_promote", "max_conformance_runs"):
                job.pop(key, None)
                if key in entry:
                    job[key] = entry[key]
        else:
            job["ablation_flags"] = layers.effective_ablation_flags(
                tuple(job.get("ablation_flags") or ()) + tuple(ablation_args))
            job["visibility"] = layers.job_visibility(job, opts.visibility)
            if job["layer"] in layers.ABLATION_LADDER:
                job["max_conformance_runs"] = opts.max_conformance_runs
    jobs = scheduling.filter_jobs(jobs, opts.only)

    shim_dir, lock_env = _sbt_lock(opts, out_root, exp_dir)

    if opts.collect_only:
        meta = experiment.read_experiment_yaml(exp_dir / "experiment.yaml")
        for job in jobs:
            run_name = naming.job_run_name(job, opts.exp_id, opts.max_iters)
            run_dir = exp_dir / "runs" / run_name
            framework_run_dir = opts.workdir / "agents" / "generated" / "runs" / run_name
            prior_meta = {}
            prior_path = run_dir / "record.json"
            if prior_path.exists():
                try:
                    prior = json.loads(prior_path.read_text())
                    prior_meta = {
                        "exit_code": prior.get("exit_code"),
                        "wall_seconds": prior.get("wall_seconds"),
                        "started_at": prior.get("started_at"),
                        "finished_at": prior.get("finished_at"),
                    }
                except (OSError, json.JSONDecodeError):
                    pass
            record = collection.collect_record(
                opts.exp_id, {**job, "infra_retry": 0}, run_name, run_dir, framework_run_dir, prior_meta
            )
            collection.save_record(record, run_dir)
            print(f"[collect] {run_name}: status={record.get('status')}")
        result = reporting.aggregate(exp_dir, meta, opts.bootstrap)
        print(f"aggregated: {result['records']} runs, {result['cells']} cells -> {result['summary']}")
        return 0

    generated_configs = configs.materialize_configs(opts.exp_id, opts.models, opts.temperature, opts.workdir)
    meta = experiment.materialize_experiment(opts)
    meta["configs"] = generated_configs
    meta["schedule_jobs"] = [dict(job) for job in jobs]
    meta["schedule"] = [
        naming.job_run_name(j, opts.exp_id, opts.max_iters) for j in jobs
    ]
    exp_dir.mkdir(parents=True, exist_ok=True)
    experiment.write_experiment_yaml(exp_dir / "experiment.yaml", meta)

    if opts.dry_run:
        for job in jobs:
            name = naming.job_run_name(job, opts.exp_id, opts.max_iters)
            argv_list = execution.build_argv(
                opts.python, opts.conda_env, shim_dir, generated_configs[job["model"]]["config"],
                job["benchmark"], name, opts.max_iters, layers.job_visibility(job, opts.visibility),
                opts.profile_cache,
                layers.job_run_flags(job),
                opts.agent_forward,
            )
            print(f"[dry ] {name}: {shlex.join(argv_list)}")
        print(f"[dry ] experiment.yaml -> {exp_dir / 'experiment.yaml'} ({len(jobs)} runs scheduled)")
        return 0

    if not opts.skip_preflight:
        meta["preflight"] = execution.preflight(opts, generated_configs, shim_dir, lock_env)
        experiment.write_experiment_yaml(exp_dir / "experiment.yaml", meta)

    results = _run_jobs(
        jobs,
        lambda job: execution.execute_one(job, opts, exp_dir, generated_configs, shim_dir, lock_env),
        max(1, opts.jobs),
    )

    result = reporting.aggregate(exp_dir, meta, opts.bootstrap)
    print(f"aggregated: {result['records']} runs, {result['cells']} cells -> {result['summary']}")
    failed = [r for r in results if r.get("status") != "complete"]
    print(f"complete: {len(results) - len(failed)}/{len(results)}")
    return 0
