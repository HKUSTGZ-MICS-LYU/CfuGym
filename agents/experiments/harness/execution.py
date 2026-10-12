"""Execution: command construction, subprocess running, one-run orchestration.

This module owns the ``run_logged`` / ``time.sleep`` seams tests patch: reach
them via ``harness.execution`` (the owning module), never via a re-export.
"""

from __future__ import annotations

import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.experiments.harness import collection, layers, locking, naming


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------


def agent_script_args(
    config_name: str,
    benchmark: str,
    run_name: str,
    max_iters: int,
    visibility: str,
    profile_cache: str,
    ablation_flags=(),
    agent_args=(),
) -> list[str]:
    return [
        "agents/run_cfu_agent.py",
        "--embench", str(benchmark),
        "--llm-config", str(config_name),
        "--run-id", str(run_name),
        "--max-iters", str(int(max_iters)),
        "--visibility", str(visibility),
        "--profile-cache", str(profile_cache),
        *list(ablation_flags or ()),
        *list(agent_args or ()),
    ]


def build_argv(
    python: str,
    conda_env: str,
    shim_dir: Path | None,
    config_name: str,
    benchmark: str,
    run_name: str,
    max_iters: int,
    visibility: str,
    profile_cache: str,
    ablation_flags=(),
    agent_args=(),
) -> list[str]:
    python_cmd = shlex.split(python) + agent_script_args(
        config_name, benchmark, run_name, max_iters,
        visibility, profile_cache, ablation_flags, agent_args,
    )
    return locking.launcher_argv(python_cmd, conda_env, shim_dir)


def build_preflight_argv(python: str, conda_env: str, shim_dir: Path | None, config_name: str) -> list[str]:
    python_cmd = shlex.split(python) + [
        "agents/run_cfu_agent.py", "--test-api", "--llm-config", str(config_name)
    ]
    return locking.launcher_argv(python_cmd, conda_env, shim_dir)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def git_info(workdir: Path) -> dict:
    info = {"git_commit": "", "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workdir, capture_output=True, text=True, timeout=30
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=workdir, capture_output=True, text=True, timeout=30
        )
        if status.returncode == 0:
            info["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def run_logged(argv: list[str], out_path: Path, err_path: Path, cwd: Path, env: dict, timeout: int) -> dict:
    """Run a command to log files; kill the whole process group on timeout."""
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    timed_out = False
    with out_path.open("wb") as out, err_path.open("wb") as err:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), env=env, stdout=out, stderr=err, start_new_session=True
        )
        try:
            proc.wait(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            exit_code = None
    return {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.monotonic() - t0, 1),
        "exit_code": exit_code,
        "timed_out": timed_out,
    }


def archive_stale_run_dir(path: Path) -> Path | None:
    """Move a pre-existing same-named framework run dir aside.

    Run ids are deterministic, so re-running an experiment reuses them. The
    framework writes a *merged* ``run.yaml`` and appends artifacts into whatever
    directory already exists, so without this a failed new attempt inherited the
    previous round's ``status``/gates while stdout came from the new process
    (2026-10-04: 29 of 30 ablation records mixed two rounds, and several runs
    whose new attempt died in analysis were collected as `complete`).
    """
    if not path.is_dir() or not any(path.iterdir()):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.superseded-{stamp}")
    index = 2
    while target.exists():
        target = path.with_name(f"{path.name}.superseded-{stamp}-{index}")
        index += 1
    shutil.move(str(path), str(target))
    return target


def execute_one(job: dict, opts, exp_dir: Path, configs: dict, shim_dir: Path | None, lock_env: dict) -> list[dict]:
    """Run one matrix entry (with its infra retries); return its records."""
    records = []
    infra_retry = int(job.get("infra_retry", 0))
    while True:
        run_name = naming.job_run_name(job, opts.exp_id, opts.max_iters, infra_retry)
        run_dir = exp_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        framework_run_dir = opts.workdir / "agents" / "generated" / "runs" / run_name
        config_name = configs[job["model"]]["config"]
        argv = build_argv(
            opts.python, opts.conda_env, shim_dir, config_name, job["benchmark"], run_name,
            opts.max_iters, opts.visibility, opts.profile_cache,
            layers.job_layer_flags(job),
            getattr(opts, "agent_forward", ()),
        )
        (run_dir / "cmd.txt").write_text(shlex.join(argv) + "\n")

        existing = run_dir / "record.json"
        if existing.exists() and not opts.force:
            try:
                prior = json.loads(existing.read_text())
            except (OSError, json.JSONDecodeError):
                prior = {}
            if prior.get("terminal") and prior.get("status") in {"complete", "failed", "blocked"}:
                if not (prior.get("infra") or {}).get("infra"):
                    print(f"[skip] {run_name} (already {prior.get('status')})")
                    records.append(prior)
                    return records

        archived = archive_stale_run_dir(framework_run_dir)
        if archived is not None:
            print(f"[warn] {run_name}: moved a pre-existing framework run dir aside "
                  f"-> {archived.name}", flush=True)
        print(f"[run ] {run_name}: {shlex.join(argv)}", flush=True)
        meta = run_logged(
            argv, run_dir / "stdout.log", run_dir / "stderr.log",
            opts.workdir, locking.child_env(shim_dir, lock_env), opts.timeout,
        )
        record = collection.collect_record(
            opts.exp_id,
            {**job, "max_iters": opts.max_iters, "infra_retry": infra_retry},
            run_name, run_dir, framework_run_dir, meta,
        )
        collection.save_record(record, run_dir)
        records.append(record)
        print(
            f"[done] {run_name}: status={record.get('status')} "
            f"stopped_at={record.get('stopped_at') or '-'} "
            f"failure={((record.get('failure') or {}).get('failure_class')) or '-'} "
            f"attempt={record.get('attempt_index')} wall={meta.get('wall_seconds')}s",
            flush=True,
        )

        if (record.get("infra") or {}).get("infra") and infra_retry < opts.infra_retries:
            infra_retry += 1
            kind = (record.get("infra") or {}).get("infra_kind") or "infra_error"
            # A rate limit needs a longer pause than a one-off transport error;
            # an immediate retry of 30 rate-limited runs only re-hit the quota
            # (19 of 24 retries in the 2026-10-04 matrix matched 429 again).
            base = max(0.0, float(getattr(opts, "infra_backoff", 0.0) or 0.0))
            if kind == "infra_rate_limit":
                base = max(base, float(getattr(opts, "rate_limit_backoff", 0.0) or 0.0))
            delay = base * random.uniform(1.0, 1.5) if base else 0.0
            if delay:
                print(f"[warn] {run_name} looks like {kind}; retry "
                      f"{infra_retry}/{opts.infra_retries} in {delay:.0f}s", flush=True)
                time.sleep(delay)
            else:
                print(f"[warn] {run_name} looks like {kind}; retry "
                      f"{infra_retry}/{opts.infra_retries}", flush=True)
            continue
        return records


def preflight(opts, configs: dict, shim_dir: Path | None, lock_env: dict) -> dict:
    results = {}
    for model, info in configs.items():
        argv = build_preflight_argv(opts.python, opts.conda_env, shim_dir, info["config"])
        run_dir = opts.exp_dir_ref / "preflight"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = run_logged(
            argv, run_dir / f"{naming.model_slug(model)}.out", run_dir / f"{naming.model_slug(model)}.err",
            opts.workdir, locking.child_env(shim_dir, lock_env), 180,
        )
        text = (run_dir / f"{naming.model_slug(model)}.out").read_text(errors="replace")
        ok = meta.get("exit_code") == 0 and "api-status=OK" in text
        results[model] = {"ok": ok, "exit_code": meta.get("exit_code"), "argv": argv}
        print(f"[preflight] {model}: {'OK' if ok else 'FAILED'}")
        if not ok and not opts.skip_preflight_ok:
            raise SystemExit(
                f"preflight failed for {model} ({info['config']}); "
                "fix the config/API key or rerun with --allow-preflight-fail."
            )
    return results
