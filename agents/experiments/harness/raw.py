"""Rebuild the Embench reference/profile measurement cache without the agent.

The measurement lives in ``agents/generated/cache/embench_profile`` and is keyed
by content: the benchmark sources, the seeded scaffold design, the CPU flags,
the toolchain versions and the git/submodule state (see
``profile_cache.build_key``).  One commit therefore invalidates every entry, and
the next sweep pays two extra builds plus two extra RTL simulations per
benchmark before any design work starts.

This module warms those entries in suite order by running
``agents/run_cfu_agent.py --profile-only``: the same interpreter, build commands
and pass-through flags a scheduled run would use, so the key written here is the
key the sweep later hits.  It deliberately does not touch ``experiment.yaml``,
the LLM config materialization or the preflight: a cache rebuild is not an
experiment run and must not enter the pass@k records.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from agents.experiments.harness import execution, locking, naming


# Warm-ups write run artifacts through the agent, whose write policy allowlists
# ``agents/generated/runs/**`` only -- so they live beside scheduled runs under a
# ``<exp>-raw-<bench>`` id that collection never selects.
FRAMEWORK_RUNS_DIR = "agents/generated/runs"
# This module owns the measurement, not the design: refuse pass-through flags
# that would skip or redirect the very commands it is supposed to run.
_BLOCKED_AGENT_FLAGS = frozenset({
    "--skip-sim", "--dry-run", "--profile-only", "--no-dry-run", "--out-dir", "--embench",
})
SUMMARY_PREFIX = "profile_only"
CACHE_DIR = "agents/generated/cache/embench_profile"


def raw_run_name(exp_id: str, benchmark: str) -> str:
    """Framework run id of one warm-up (kept distinct from scheduled runs)."""
    return f"{exp_id}-raw-{naming.benchmark_slug(benchmark)}"


def blocked_agent_args(agent_args) -> list[str]:
    """Pass-through flags that contradict a cache rebuild, as written by the caller."""
    blocked = []
    for arg in agent_args or ():
        flag = str(arg).split("=", 1)[0]
        if flag in _BLOCKED_AGENT_FLAGS:
            blocked.append(flag)
    return blocked


def build_argv(
    python: str,
    conda_env: str,
    shim_dir: Path | None,
    benchmark: str,
    run_name: str,
    agent_args=(),
) -> list[str]:
    """The one-benchmark cache rebuild command (no LLM config, no design)."""
    python_cmd = shlex.split(python) + [
        "agents/run_cfu_agent.py",
        "--embench", str(benchmark),
        "--run-id", str(run_name),
        # The point of the mode is a rebuild: a hit would leave the stale entry
        # in place and silently defeat the warm-up after a commit.
        "--profile-cache", "refresh",
        "--profile-only",
        *list(agent_args or ()),
    ]
    return locking.launcher_argv(python_cmd, conda_env, shim_dir)


def parse_summary(log_path: Path) -> dict[str, str]:
    """Read the last ``profile_only ...`` line the agent printed.

    Fields are ``key=value`` with no whitespace inside a value, so the last
    occurrence wins (a retried command may print more than one).
    """
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith(SUMMARY_PREFIX + " "):
            continue
        fields: dict[str, str] = {}
        for token in stripped.split()[1:]:
            name, sep, value = token.partition("=")
            if sep:
                fields[name] = value
        return fields
    return {}


def warm(opts, exp_dir: Path, shim_dir: Path | None, lock_env: dict) -> int:
    """Rebuild every scheduled benchmark's measurement cache; return an exit code.

    Benchmarks run sequentially (the sbt lock serializes the heavy steps anyway,
    and a warm-up has no reason to fan out).  A failure does not stop the run:
    the remaining entries are still worth rebuilding, and the summary names the
    ones the next sweep will have to pay for again.
    """
    benchmarks = [str(name) for name in (opts.benchmarks or ())]
    if not benchmarks:
        raise SystemExit("--raw needs at least one benchmark (--benchmarks or suite discovery)")
    raw_dir = exp_dir / "raw"
    print(
        f"[raw ] rebuilding the profile cache for {len(benchmarks)} benchmark(s): "
        f"{', '.join(benchmarks)}",
        flush=True,
    )
    print(f"[raw ] run evidence -> {FRAMEWORK_RUNS_DIR}, cache -> {CACHE_DIR}", flush=True)
    results: list[dict] = []
    for benchmark in benchmarks:
        run_name = raw_run_name(opts.exp_id, benchmark)
        log_dir = raw_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        argv = build_argv(
            opts.python, opts.conda_env, shim_dir, benchmark, run_name,
            getattr(opts, "agent_forward", ()),
        )
        (log_dir / "cmd.txt").write_text(shlex.join(argv) + "\n")
        framework_run_dir = opts.workdir / FRAMEWORK_RUNS_DIR / run_name
        archived = execution.archive_stale_run_dir(framework_run_dir)
        if archived is not None:
            print(f"[raw ] {benchmark}: moved a previous warm-up aside -> {archived.name}", flush=True)
        print(f"[raw ] {benchmark}: {shlex.join(argv)}", flush=True)
        meta = execution.run_logged(
            argv, log_dir / "stdout.log", log_dir / "stderr.log",
            opts.workdir, locking.child_env(shim_dir, lock_env), opts.timeout,
        )
        summary = parse_summary(log_dir / "stdout.log")
        ok = (
            meta.get("exit_code") == 0
            and not meta.get("timed_out")
            and summary.get("ok") == "true"
            and summary.get("saved") == "true"
        )
        entry = {
            "benchmark": benchmark,
            "run_name": run_name,
            "ok": ok,
            "timed_out": bool(meta.get("timed_out")),
            "exit_code": meta.get("exit_code"),
            "wall_seconds": meta.get("wall_seconds"),
            "key": summary.get("key", ""),
            "cache": summary.get("cache", ""),
            "reference_cycles": summary.get("reference_cycles", ""),
            "log": str(log_dir / "stdout.log"),
        }
        results.append(entry)
        key = entry["key"][:16] or "-"
        print(
            f"[raw ] {benchmark}: {'cached' if ok else 'FAILED'} key={key} "
            f"cache={entry['cache'] or '-'} reference_cycles={entry['reference_cycles'] or '-'} "
            f"wall={entry['wall_seconds']}s",
            flush=True,
        )
    cached = [entry for entry in results if entry["ok"]]
    failed = [entry for entry in results if not entry["ok"]]
    print(f"[raw ] cached {len(cached)}/{len(results)} benchmark measurement(s)", flush=True)
    for entry in failed:
        reason = "timeout" if entry["timed_out"] else f"exit={entry['exit_code']}"
        print(f"[raw ] FAILED {entry['benchmark']}: {reason} (log {entry['log']})", flush=True)
    return 0 if not failed else 1
