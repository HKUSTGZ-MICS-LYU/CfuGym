#!/usr/bin/env python3
"""LLM x benchmark pass@1/3/5 experiment matrix harness for the CFU design agent.

Design: docs/agent_pass_at_k_experiment_matrix.md

One harness run drives a full matrix of ``--run-cfu-agent`` runs:

    <exp_id>-<model>-<benchmark>-r<RR>-mi<K>            e.g. pk5-20260212-glm52-crc32-r01-mi5

Each cell (model x benchmark) is repeated R times with ``--max-iters K`` (= K
attempts: 1 first try + K-1 repairs).  A run that completed after ``repairs_used``
repairs has ``attempt_index = repairs_used + 1`` and therefore contributes

    pass@k = 1  iff the run is ``complete`` and attempt_index <= k        (k = 1, 3, 5)

so one ``--max-iters 5`` run yields the whole pass@1..pass@5 curve.  Aggregation
reports both this budget estimator and the i.i.d. sampling estimator
``pass@k = 1 - C(R-c, k) / C(R, k)`` over the R independent runs.

Every run is collected from stable machine-readable sources (CLI stdout,
``run.yaml``, ``report.md`` gates, ``10_validation/*.yaml``, ``11_cost/result.yaml``)
into ``record.json`` + ``runs.csv``; every failure is classified into a
``failure_class`` (which rung broke and why) and archived as ``failure.json`` +
``failures.csv``.

Concurrency / locking: all heavy commands of the agent (SoC generation,
``MiCoSocSim`` RTL simulation, ``GenAgentCfu``) are invoked as ``sbt`` and share
repo-level state (``target/`` incremental compile, ``simWorkspace/`` for the
Verilator sim, ivy/coursier caches).  The harness therefore prepends a shim
directory to the child PATH in which ``sbt`` is a wrapper that holds an
exclusive ``flock`` around the real ``sbt`` for the duration of each call.  LLM
calls, ``make`` software builds and Yosys still overlap across runs; only
sbt/RTL-simulation calls are serialized.  The lock file is shared across
harness invocations by default so two batches cannot collide either.

Usage (see --help for everything):

    python3 agents/experiments/run_matrix.py \
        --exp-id pk5-20260212 \
        --models deepseek-pro,glm-5.2,gpt-5.6-terra \
        --benchmarks crc32,matmult-int,md5sum \
        --repeats 5 --max-iters 5 --temperature 0.7 \
        --conda-env mico_env --jobs 2
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = ("deepseek-pro", "glm-5.2", "gpt-5.6-terra")
DEFAULT_BENCHMARKS = ("crc32", "matmult-int", "md5sum")
# Everything the harness launches runs inside the mico_env conda environment
# (sbt, the toolchain and the simulator all live there).
DEFAULT_CONDA_ENV = "mico_env"
DEFAULT_PYTHON = "python3 -B"
SUITE = "embench-iot"
PASS_AT_K = (1, 3, 5)

MODEL_SLUGS = {
    "deepseek-pro": "deepseekpro",
    "deepseek-flash": "deepseekflash",
    "glm-5.2": "glm52",
    "gpt-5.6-terra": "gpt56terra",
    "gpt-5.6-luna": "gpt56luna",
    "kimi-k3": "kimik3",
}
BENCHMARK_SLUGS = {
    "matmult-int": "matmultint",
    "md5sum": "md5sum",
    "crc32": "crc32",
}

# Ladder/report order; also the order used to pick the first failing rung when
# ``stopped_at`` is missing from the CLI log.
RUNG_ORDER = (
    "load_context",
    "prepare_benchmark",
    "workload_spec",
    "isa",
    "implementation_hw",
    "hw_build",
    "hw_conformance",
    "implementation_sw",
    "sw_build",
    "sim",
    "cost",
)

# Build rung guards: hw_build/sw_build fail WITHOUT running any command when the
# implementation rung that feeds them applied nothing. The guard is not the real
# failure rung -- the implementation rejection is (observed 2026-09-26: a
# missing-EDIT_END delivery failure was recorded as stopped_at=hw_build).
GUARDED_BUILD_RUNGS = {"hw_build": "implementation_hw", "sw_build": "implementation_sw"}

FAILURE_CLASSES = (
    "infra_timeout",
    "infra_error",
    "input_or_policy",
    "spec_invalid",
    "isa_contract",
    "no_deliverable",
    "hw_build",
    "hw_conformance",
    "sw_build",
    "sim_correctness",
    "sim_timeout",
    "sim_speedup_floor",
    "sim_no_change",
    "sim_error",
    "cost",
    "unknown",
)

# Harness-level (non-model) failure signatures: API transport problems and
# missing toolchain executables.  A crashed framework is detected differently
# (see detect_infra): a bare "Traceback" is too noisy because benchmark build
# logs (Python-based) legitimately contain tracebacks.
INFRA_PATTERNS = (
    re.compile(r"api-key-missing"),
    re.compile(r"api-error:"),
    re.compile(r"urlopen error", re.I),
    re.compile(r"Connection (error|refused|reset|aborted)", re.I),
    re.compile(r"Max retries exceeded"),
    re.compile(r"SSLError", re.I),
    re.compile(r"\b(?:HTTP(?: Error| status)?\s*[:=]?\s*500|upstream\b.{0,60}\b500)\b", re.I),
    re.compile(r"\b(?:502|503|504)\b.{0,40}(?:Bad Gateway|Service Unavailable|Gateway Time-out)", re.I),
    re.compile(r"conformance oracle request got no model answer", re.I),
    re.compile(r"\b(?:LLM|API|endpoint|gateway)\b.{0,80}\b(?:timed? out|timeout)\b", re.I),
    re.compile(r"No such file or directory: '?(?:sbt|make|yosys|conda)", re.I),
    # A misfiring lock shim is a harness problem, never a model failure.
    re.compile(r"sbt lock shim:"),
)

RUNS_COLUMNS = [
    "exp_id", "run_name", "run_id", "model_config", "model", "benchmark", "suite",
    "repeat", "max_iters", "status", "exit_code", "stopped_at",
    "failure_class", "failure_rung", "failure_detail",
    "repairs_used", "attempt_index", "pass_at_1", "pass_at_3", "pass_at_5",
    "sim_ok", "reference_cycles", "cfu_cycles", "speedup_x100",
    "yosys_cells", "yosys_wire_bits",
    "token_calls", "token_in", "token_out", "token_total",
    "wall_seconds", "profile_cache_status", "visibility", "infra_retries",
    "parse_warnings", "run_dir", "framework_run_dir", "report_path",
    "started_at", "finished_at",
]

FAILURES_COLUMNS = [
    "exp_id", "run_name", "model", "benchmark", "repeat", "kind", "attempt",
    "failure_class", "failure_rung", "trigger_rung", "repair_target",
    "decision_source", "budget_exhausted", "detail", "evidence",
]

PASS_AT_K_COLUMNS = [
    "model", "benchmark", "runs", "complete_runs",
    "pass_at_1_budget", "pass_at_3_budget", "pass_at_5_budget",
    "pass_at_1_sample", "pass_at_3_sample", "pass_at_5_sample",
    "pass_at_1_ci95", "pass_at_3_ci95", "pass_at_5_ci95",
    "attempt_index_median", "first_try_rate",
    "tokens_per_complete_mean", "wall_h_per_complete_mean",
    "speedup_x100_median", "yosys_cells_median", "failure_classes",
]

ESTIMAND_HELP = (
    "attempt_index = repairs_used + 1 (one --max-iters K run = K attempts). "
    "pass@k_budget = mean over R repeats of 1[complete and attempt_index <= k]; "
    "pass@k_sample = 1 - C(R-c,k)/C(R,k) with c complete runs (i.i.d. estimator)."
)

# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def slugify(value: str) -> str:
    """Run-id-safe slug ([A-Za-z0-9_-], matching artifacts._safe_run_id)."""
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return text or "x"


def model_slug(config_name: str) -> str:
    return MODEL_SLUGS.get(config_name, slugify(config_name))


def benchmark_slug(name: str) -> str:
    return BENCHMARK_SLUGS.get(name, slugify(name))


def make_run_name(
    exp_id: str,
    model_config: str,
    benchmark: str,
    repeat: int,
    max_iters: int,
    infra_retry: int = 0,
) -> str:
    name = (
        f"{exp_id}-{model_slug(model_config)}-{benchmark_slug(benchmark)}"
        f"-r{int(repeat):02d}-mi{int(max_iters)}"
    )
    if infra_retry:
        name += f"-infra{int(infra_retry)}"
    return name


# --------------------------------------------------------------------------
# Matrix + scheduling
# --------------------------------------------------------------------------


def expand_matrix(models, benchmarks, repeats: int) -> list[dict]:
    """One entry per run (cell x repeat), repeat-major order before scheduling."""
    jobs = []
    for repeat in range(1, int(repeats) + 1):
        for model in models:
            for benchmark in benchmarks:
                jobs.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "repeat": repeat,
                        "infra_retry": 0,
                    }
                )
    return jobs


def schedule_jobs(jobs: list[dict], seed: int) -> list[dict]:
    """Round-robin over repeats with a seeded shuffle of the cells each round.

    Replicates of the same cell land in different time slots, so gateway load or
    time-of-day drift is spread over all cells instead of concentrating on one.
    """
    rng = random.Random(seed)
    by_repeat: dict[int, list[dict]] = {}
    for job in jobs:
        by_repeat.setdefault(int(job["repeat"]), []).append(dict(job))
    ordered: list[dict] = []
    for repeat in sorted(by_repeat):
        round_jobs = by_repeat[repeat]
        rng.shuffle(round_jobs)
        ordered.extend(round_jobs)
    return ordered


def filter_jobs(jobs: list[dict], only: list[str]) -> list[dict]:
    """Keep jobs whose run name contains every ``--only`` substring."""
    if not only:
        return list(jobs)
    tokens = [t.strip().lower() for t in only if t.strip()]
    kept = []
    for job in jobs:
        name = make_run_name(
            job["exp_id"], job["model"], job["benchmark"], job["repeat"],
            job["max_iters"], job.get("infra_retry", 0),
        ).lower()
        if all(token in name for token in tokens):
            kept.append(job)
    return kept


# --------------------------------------------------------------------------
# sbt / RTL-simulation lock (shim)
# --------------------------------------------------------------------------

# The shim only exists to serialize ``sbt`` (SoC generation, MiCoSocSim RTL
# simulation, GenAgentCfu).  It resolves the real ``sbt`` to an absolute path at
# call time from PATH with its own directory removed first, so it works for the
# conda-env ``sbt`` the harness itself cannot see.  A recursion guard makes a
# PATH mishap loud (exit 97) instead of forkbombing.
SBT_SHIM = """#!/usr/bin/env bash
# Serialize sbt / RTL-simulation invocations across concurrent experiment runs.
# Generated by agents/experiments/run_matrix.py -- do not edit.
set -euo pipefail
if [ -n "${CFU_EXP_SBT_SHIM_ACTIVE:-}" ]; then
  echo "sbt lock shim: recursion detected (PATH=$PATH)" >&2
  exit 97
fi
export CFU_EXP_SBT_SHIM_ACTIVE=1
SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="${CFU_EXP_SBT_LOCK_FILE:?CFU_EXP_SBT_LOCK_FILE is not set}"
TIMEOUT="${CFU_EXP_SBT_LOCK_TIMEOUT:-14400}"
# Drop EVERY occurrence of the shim dir from PATH, entry by entry.  (A
# ${PATH//:$SHIM_DIR:/:} substitution is wrong here: with ADJACENT duplicates
# like <shim>:<shim>:... the first match consumes the ':' the second one needs,
# leaving a copy behind -- which resolved sbt back to this shim and tripped the
# recursion guard on every call.)  Then resolve the REAL sbt (system or
# conda-env) to an absolute path before locking.
CLEAN_PATH=""
REMAINING="$PATH"
while [ -n "$REMAINING" ]; do
  ENTRY="${REMAINING%%:*}"
  if [ "$REMAINING" = "$ENTRY" ]; then
    REMAINING=""
  else
    REMAINING="${REMAINING#*:}"
  fi
  [ -z "$ENTRY" ] && continue
  [ "$ENTRY" = "$SHIM_DIR" ] && continue
  CLEAN_PATH="${CLEAN_PATH:+$CLEAN_PATH:}$ENTRY"
done
export PATH="$CLEAN_PATH"
REAL_SBT="$(command -v sbt || true)"
if [ -z "$REAL_SBT" ]; then
  echo "sbt lock shim: real sbt not found on PATH=$CLEAN_PATH" >&2
  exit 127
fi
case "$REAL_SBT" in
  "$SHIM_DIR"/*)
    echo "sbt lock shim: resolved sbt is the shim itself ($REAL_SBT, PATH=$CLEAN_PATH)" >&2
    exit 97
    ;;
esac
exec flock -x -w "$TIMEOUT" "$LOCK" "$REAL_SBT" "$@"
"""


def write_sbt_shim(shim_dir: Path) -> Path:
    """Write the locking ``sbt`` shim into ``shim_dir`` and return its path."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "sbt"
    shim.write_text(SBT_SHIM)
    shim.chmod(0o755)
    return shim


def require_flock() -> str:
    flock = shutil.which("flock")
    if not flock:
        raise SystemExit(
            "flock(1) not found: install util-linux or rerun with --no-sbt-lock "
            "(and --jobs 1) to serialize manually."
        )
    return flock


def sbt_lock_env(lock_file: Path, timeout: int) -> dict[str, str]:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    return {
        "CFU_EXP_SBT_LOCK_FILE": str(lock_file),
        "CFU_EXP_SBT_LOCK_TIMEOUT": str(int(timeout)),
    }


def child_env(shim_dir: Path | None, lock_env: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if shim_dir is not None:
        env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    env.update(lock_env)
    return env


# --------------------------------------------------------------------------
# Experiment configs (temperature control)
# --------------------------------------------------------------------------


def materialize_configs(
    exp_id: str, models, temperature: float, workdir: Path
) -> dict[str, dict]:
    """Copy each base LLM config with an explicit ``temperature``.

    The stock configs leave temperature unset (= 0), which would collapse the R
    repeats into the same trajectory and make pass@k meaningless.  Generated
    files land beside the bases (agents/llm_configs is gitignored) and are never
    written back to the tracked/edited base config.
    """
    import yaml  # noqa: F401  (kept out of the hot path)

    cfg_dir = workdir / "agents" / "llm_configs"
    out: dict[str, dict] = {}
    for model in models:
        base_path = cfg_dir / f"{model}.json"
        if not base_path.exists():
            raise SystemExit(
                f"LLM config not found: {base_path} "
                f"(available: {', '.join(sorted(p.stem for p in cfg_dir.glob('*.json'))) or 'none'})"
            )
        data = json.loads(base_path.read_text())
        if not isinstance(data, dict):
            raise SystemExit(f"LLM config must be a JSON object: {base_path}")
        name = f"{exp_id}-{model}"
        generated = dict(data)
        generated["name"] = name
        generated["temperature"] = float(temperature)
        generated["description"] = (
            f"{data.get('description', data.get('name', model))} "
            f"[experiment {exp_id}: temperature={temperature}]"
        )
        out_path = cfg_dir / f"{name}.json"
        out_path.write_text(json.dumps(generated, indent=2, ensure_ascii=False) + "\n")
        out[model] = {
            "config": name,
            "path": str(out_path.relative_to(workdir)),
            "sha256": _sha256(out_path.read_bytes()),
            "base": str(base_path.relative_to(workdir)),
            "base_sha256": _sha256(base_path.read_bytes()),
        }
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
) -> list[str]:
    return [
        "agents/run_cfu_agent.py",
        "--embench", str(benchmark),
        "--llm-config", str(config_name),
        "--run-id", str(run_name),
        "--max-iters", str(int(max_iters)),
        "--visibility", str(visibility),
        "--profile-cache", str(profile_cache),
    ]


# Re-prepends the shim dir to PATH *inside* the conda environment.  This is the
# load-bearing piece of the sbt lock: ``conda run`` activation puts <env>/bin at
# the front of PATH, which would shadow the shim and silently disable the lock.
# $0 is a label, $1 the shim dir, $@ the command to exec.
SHIM_PATH_WRAPPER = 'export PATH="$1:$PATH"; shift; exec "$@"'


def launcher_argv(python_cmd: list[str], conda_env: str, shim_dir: Path | None) -> list[str]:
    """Wrap a command so it runs inside the conda env with the shim first on PATH.

    ``conda_env`` empty skips conda entirely (the plain-Python case); there the
    harness-level PATH prepend in ``child_env`` already puts the shim first.
    """
    if not conda_env:
        return list(python_cmd)
    prefix = ["conda", "run", "-n", str(conda_env), "--no-capture-output"]
    if shim_dir is None:
        return prefix + list(python_cmd)
    return prefix + ["bash", "-c", SHIM_PATH_WRAPPER, "cfu-exp-sbt-shim", str(shim_dir)] + list(python_cmd)


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
) -> list[str]:
    python_cmd = shlex.split(python) + agent_script_args(
        config_name, benchmark, run_name, max_iters,
        visibility, profile_cache,
    )
    return launcher_argv(python_cmd, conda_env, shim_dir)


def build_preflight_argv(python: str, conda_env: str, shim_dir: Path | None, config_name: str) -> list[str]:
    python_cmd = shlex.split(python) + [
        "agents/run_cfu_agent.py", "--test-api", "--llm-config", str(config_name)
    ]
    return launcher_argv(python_cmd, conda_env, shim_dir)


# --------------------------------------------------------------------------
# Parsing the agent's outputs
# --------------------------------------------------------------------------


def parse_cli_stdout(text: str) -> dict:
    """Parse the stable machine-readable CLI lines (status=, summary:, ...)."""
    out: dict = {}
    m = re.search(r"^status=(\S+)", text, re.M)
    if m:
        out["status"] = m.group(1)
    m = re.search(r"^stopped_at=(\S+)", text, re.M)
    if m:
        out["stopped_at"] = m.group(1)
    m = re.search(r"^report=(\S+)", text, re.M)
    if m:
        out["report_path"] = m.group(1)
    m = re.search(r"^visibility=(\S+)", text, re.M)
    if m:
        out["visibility"] = m.group(1)
    m = re.search(r"^\s*speedup\s+= (\d+(?:\.\d+)?)x", text, re.M)
    if m:
        out["speedup_x100"] = int(round(float(m.group(1)) * 100))
    m = re.search(r"^\s*yosys_cost\s+= top=(\S+) cells=(\d+) wire_bits=(\d+) memories=(\d+) memory_bits=(\d+)", text, re.M)
    if m:
        out["yosys_cost"] = {
            "top": m.group(1), "cells": int(m.group(2)), "wire_bits": int(m.group(3)),
            "memories": int(m.group(4)), "memory_bits": int(m.group(5)),
        }
    usage = {}
    for key, pattern in (
        ("token_calls", r"^\s*calls\s+= (\d+)"),
        ("token_in", r"^\s*input_tokens=\s*(\d+)"),
        ("token_out", r"^\s*output_tokens=\s*(\d+)"),
        ("token_total", r"^\s*total_tokens=\s*(\d+)"),
    ):
        m = re.search(pattern, text, re.M)
        if m:
            usage[key] = int(m.group(1))
    out.update(usage)
    return out


def parse_report(text: str, warnings: list[str]) -> dict:
    """Parse report.md: Status, the ``## Gates`` dict dump, repairs used."""
    out: dict = {}
    m = re.search(r"^Status: (\w+)", text, re.M)
    if m:
        out["status"] = m.group(1)
    m = re.search(r"^- repairs used: (\d+)/(\d+) \(iteration=(\d+)\)", text, re.M)
    if m:
        out["repairs_used"] = int(m.group(1))
        out["max_repairs"] = int(m.group(2))
        out["iteration"] = int(m.group(3))
    m = re.search(r"^- stopped at: (\S+)", text, re.M)
    if m:
        out["stopped_at"] = m.group(1)

    gates: dict = {}
    section = re.search(r"^## Gates\n(.*?)(?=^## )", text, re.M | re.S)
    if section:
        for line in section.group(1).splitlines():
            m = re.match(r"^- ([A-Za-z0-9_]+): (\{.*\})$", line)
            if not m:
                continue
            try:
                gates[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError) as exc:
                warnings.append(f"report gate {m.group(1)} unparsed: {exc}")
    else:
        warnings.append("report.md has no '## Gates' section")
    out["gates"] = gates
    return out


def load_yaml(path: Path, warnings: list[str], label: str) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - collection must never crash
        warnings.append(f"{label} unparsed: {exc}")
        return {}


# --------------------------------------------------------------------------
# Failure / infra classification
# --------------------------------------------------------------------------


def _gate_errors(gate: dict) -> str:
    parts = [str(x) for x in (gate.get("errors") or []) if x]
    for key in ("reason", "attribution", "verdict", "detail"):
        value = gate.get(key)
        if value:
            parts.append(str(value))
    return " | ".join(parts)


def classify_failure(record: dict) -> dict | None:
    """Map a non-complete run onto (failure_class, failure_rung, detail).

    The class answers "which rung broke and in what way"; ``failure_rung`` keeps
    the exact ladder position so the coarser class never loses information.
    """
    status = str(record.get("status") or "")
    if status == "complete":
        return None
    gates = record.get("gates") or {}
    stopped = str(record.get("stopped_at") or "")
    failing_rung = stopped
    failing_gate: dict = {}
    if not failing_rung:
        for rung in RUNG_ORDER:
            gate = gates.get(rung)
            if isinstance(gate, dict) and gate.get("ok") is False:
                failing_rung = rung
                failing_gate = gate
                break
    else:
        failing_gate = gates.get(failing_rung) if isinstance(gates.get(failing_rung), dict) else {}

    # Attribute a guarded build failure to the implementation rejection behind
    # it: the build rung ran zero commands, so recording it as a build failure
    # would misplace the failure stage in every experiment table.
    guard_note = ""
    impl_rung = GUARDED_BUILD_RUNGS.get(failing_rung)
    if impl_rung and not (failing_gate.get("commands") or []):
        impl_gate = gates.get(impl_rung)
        if isinstance(impl_gate, dict) and (
            impl_gate.get("ok") is False or impl_gate.get("applied") is False
        ):
            guard_note = (
                f"[{failing_rung} never ran: it refuses to build with no applied "
                f"{impl_rung} change]"
            )
            failing_rung = impl_rung
            failing_gate = impl_gate

    errors = _gate_errors(failing_gate)
    sim_errors = _gate_errors(gates.get("sim") or {})
    attribution = str(record.get("sim_attribution") or "")
    detail_text = f"{errors} {sim_errors} {attribution}".lower()

    repairs_used = record.get("repairs_used")
    max_repairs = (int(record.get("max_iters") or 1) - 1) if record.get("max_iters") else record.get("max_repairs")
    budget_exhausted = None
    if isinstance(repairs_used, int) and isinstance(max_repairs, int):
        budget_exhausted = repairs_used >= max_repairs

    failure_class, detail = _classify_rung(failing_rung, failing_gate, detail_text, record, status)
    infra = record.get("infra") or {}
    if infra.get("infra"):
        failure_class = infra.get("infra_kind") or "infra_error"
        detail = infra.get("infra_reason") or detail
        budget_exhausted = None  # An endpoint incident did not exhaust a model repair budget.
    detail = " ".join(part for part in ((detail or errors or status), guard_note) if part)
    return {
        "failure_class": failure_class,
        "failure_rung": failing_rung or "unknown",
        "failure_detail": detail,
        "budget_exhausted": budget_exhausted,
    }


def _classify_rung(rung: str, gate: dict, detail_text: str, record: dict, status: str) -> tuple[str, str]:
    if rung in ("load_context", "prepare_benchmark"):
        return "input_or_policy", _gate_errors(gate)
    if rung == "workload_spec":
        return "spec_invalid", _gate_errors(gate)
    if rung == "isa":
        return "isa_contract", _gate_errors(gate)
    if rung in ("implementation_hw", "implementation_sw"):
        return "no_deliverable", _gate_errors(gate) or "no usable EDIT/FILE/PATCH deliverable"
    if rung == "hw_build":
        return "hw_build", _gate_errors(gate)
    if rung == "hw_conformance":
        sub = "oracle" if gate.get("oracle_failure") else "design"
        return "hw_conformance", f"{sub}: {_gate_errors(gate)}"
    if rung == "sw_build":
        return "sw_build", _gate_errors(gate)
    if rung == "cost":
        return "cost", _gate_errors(gate)
    if rung == "sim":
        if re.search(r"timed\s+out|timeout|\bhang(?:s|ing)?\b|\bhung\b", detail_text):
            return "sim_timeout", _gate_errors(gate) or "simulation timed out"
        if "no design change" in detail_text:
            return "sim_no_change", _gate_errors(gate)
        if "not meaningfully faster" in detail_text:
            return "sim_speedup_floor", _gate_errors(gate)
        if record.get("speedup_x100") is not None and record.get("min_speedup_x100") is not None:
            if int(record.get("speedup_x100") or 0) < int(record.get("min_speedup_x100") or 0):
                return "sim_speedup_floor", _gate_errors(gate)
        if re.search(r"verify|fail symbol|wrong-result", detail_text):
            return "sim_correctness", _gate_errors(gate)
        return "sim_error", _gate_errors(gate)
    if status == "blocked":
        if isinstance((record.get("gates") or {}).get("isa"), dict) and not (record["gates"]["isa"] or {}).get("ok", True):
            return "isa_contract", _gate_errors(record["gates"]["isa"] or {})
        return "spec_invalid", "run blocked before the validation ladder"
    return "unknown", _gate_errors(gate) or f"no failing rung identified (status={status})"


def detect_infra(
    stdout: str, stderr: str, timed_out: bool, exit_code: int | None,
    gates: dict | None = None,
) -> dict:
    """Separate harness/endpoint incidents from model failures, including report gates."""
    if timed_out:
        return {"infra": True, "infra_kind": "infra_timeout", "infra_reason": "harness wall-clock timeout"}
    conformance = (gates or {}).get("hw_conformance") or {}
    if isinstance(conformance, dict) and conformance.get("ok") is False:
        detail = _gate_errors(conformance)
        endpoint_error = re.search(
            r"conformance oracle request got no model answer|"
            r"\b(?:LLM|API|endpoint|gateway|upstream)\b.{0,80}\b(?:timed? out|timeout|(?:HTTP\s*)?500)\b|"
            r"\bHTTP Error 500\b", detail, re.I,
        )
        if conformance.get("model_failure") is True or endpoint_error:
            kind = "infra_timeout" if re.search(r"timed? out|timeout", detail, re.I) else "infra_error"
            return {"infra": True, "infra_kind": kind,
                    "infra_reason": f"conformance oracle endpoint: {detail or 'no model answer'}"}
    text = f"{stdout}\n{stderr}"
    for pattern in INFRA_PATTERNS:
        m = pattern.search(text)
        if m:
            return {
                "infra": True,
                "infra_kind": "infra_error",
                "infra_reason": f"matched {pattern.pattern!r}: {m.group(0)[:120]}",
            }
    if exit_code is None:
        return {"infra": True, "infra_kind": "infra_error", "infra_reason": "no exit code recorded"}
    # The CLI always prints ``status=`` before exiting; its absence on a failing
    # process means the framework itself crashed before producing an outcome.
    if exit_code != 0 and not re.search(r"^status=", stdout, re.M):
        return {
            "infra": True,
            "infra_kind": "infra_error",
            "infra_reason": "agent produced no status= line (framework crash or launcher failure)",
        }
    return {"infra": False, "infra_kind": "", "infra_reason": ""}


def reconstruct_episodes(record: dict, run_root: Path) -> list[dict]:
    """Best-effort repair/episode ledger from gate evidence.

    The framework does not persist the full iteration history, so each repair
    marker that does survive (gate ``repair`` flags, the sim repair decision,
    repair-feedback files, the conformance repair target) becomes one episode.
    """
    episodes: list[dict] = []
    gates = record.get("gates") or {}

    sim_gate = gates.get("sim") or {}
    decision = sim_gate.get("repair_decision") or load_yaml(
        run_root / "10_validation/sim_repair_decision.yaml", [], "sim_repair_decision"
    )
    if decision:
        episodes.append({
            "kind": "route_repair",
            "trigger_rung": "sim",
            "repair_target": decision.get("target", ""),
            "decision_source": decision.get("source", ""),
            "detail": f"{decision.get('root_cause', '')} {decision.get('reason', '')}".strip(),
            "evidence": "10_validation/sim_repair_decision.yaml",
        })

    for rung, target, feedback in (
        ("implementation_hw", "implement_hw", "09_implementation/hw_repair_feedback.txt"),
        ("implementation_sw", "implement_sw", "09_implementation/sw_repair_feedback.txt"),
    ):
        gate = gates.get(rung) or {}
        feedback_path = run_root / feedback
        if gate.get("repair") or feedback_path.exists():
            episodes.append({
                "kind": "repair",
                "trigger_rung": rung,
                "repair_target": target,
                "decision_source": "fixed",
                "detail": str(gate.get("repair_diagnosis") or ""),
                "evidence": feedback if feedback_path.exists() else "",
            })

    conf = gates.get("hw_conformance") or {}
    if conf.get("ok") is False and conf.get("repair_target"):
        episodes.append({
            "kind": "repair",
            "trigger_rung": "hw_conformance",
            "repair_target": str(conf.get("repair_target")),
            "decision_source": "fixed",
            "detail": _gate_errors(conf),
            "evidence": "09_implementation/conformance/",
        })
    return episodes


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def collect_record(
    exp_id: str,
    job: dict,
    run_name: str,
    run_dir: Path,
    framework_run_dir: Path,
    meta: dict | None = None,
) -> dict:
    """Collect one run into a flat record (runs.csv row) + nested evidence."""
    meta = dict(meta or {})
    warnings: list[str] = []
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    if not stdout:
        warnings.append("stdout.log missing or empty")

    cli = parse_cli_stdout(stdout) if stdout else {}
    report_path = framework_run_dir / "report.md"
    report = parse_report(report_path.read_text(errors="replace"), warnings) if report_path.exists() else {}
    if not report:
        warnings.append("report.md missing")
    run_yaml = load_yaml(framework_run_dir / "run.yaml", warnings, "run.yaml")
    embench = load_yaml(framework_run_dir / "10_validation/embench.yaml", warnings, "embench.yaml")
    cost_yaml = load_yaml(framework_run_dir / "11_cost/result.yaml", warnings, "cost result.yaml")

    gates = report.get("gates") or {}
    review = load_yaml(framework_run_dir / "12_review/decision.yaml", warnings, "decision.yaml")
    if not gates and isinstance(review.get("gates"), dict):
        gates = review["gates"]

    status = cli.get("status") or run_yaml.get("status") or report.get("status") or "missing"
    stopped_at = cli.get("stopped_at") or report.get("stopped_at") or ""
    repairs_used = report.get("repairs_used")
    if repairs_used is None:
        warnings.append("repairs_used not found in report.md")
    attempt_index = (int(repairs_used) + 1) if isinstance(repairs_used, int) else None

    sim_gate = gates.get("sim") or {}
    reference_cycles = embench.get("reference_cycles", sim_gate.get("reference_cycles"))
    cfu_cycles = embench.get("cfu_cycles", sim_gate.get("cfu_cycles"))
    speedup_x100 = embench.get("speedup_x100", sim_gate.get("speedup_x100", cli.get("speedup_x100")))
    yosys = cli.get("yosys_cost") or {}

    llm_request = run_yaml.get("llm_request") or {}
    visibility = (run_yaml.get("visibility") or {}).get("level") or cli.get("visibility") or ""
    profile_cache = (run_yaml.get("profile_cache") or {}).get("status") or ""

    record: dict = {
        "exp_id": exp_id,
        "run_name": run_name,
        "run_id": run_name,
        "model_config": job.get("model", ""),
        "model": str(llm_request.get("model") or job.get("model", "")),
        "benchmark": job.get("benchmark", ""),
        "suite": SUITE,
        "repeat": job.get("repeat"),
        "max_iters": job.get("max_iters"),
        "status": status,
        "exit_code": meta.get("exit_code"),
        "stopped_at": stopped_at,
        "repairs_used": repairs_used,
        "max_repairs": report.get("max_repairs"),
        "attempt_index": attempt_index,
        "sim_ok": (sim_gate.get("ok") if sim_gate else None),
        "reference_cycles": reference_cycles,
        "cfu_cycles": cfu_cycles,
        "speedup_x100": speedup_x100,
        "min_speedup_x100": embench.get("min_speedup_x100"),
        "sim_attribution": embench.get("attribution", sim_gate.get("attribution", "")),
        "yosys_cells": yosys.get("cells"),
        "yosys_wire_bits": yosys.get("wire_bits"),
        "yosys_cost": yosys,
        "token_calls": cli.get("token_calls"),
        "token_in": cli.get("token_in"),
        "token_out": cli.get("token_out"),
        "token_total": cli.get("token_total"),
        "wall_seconds": meta.get("wall_seconds"),
        "profile_cache_status": profile_cache,
        "visibility": visibility,
        "infra_retries": job.get("infra_retry", 0),
        "run_dir": str(run_dir),
        "framework_run_dir": str(framework_run_dir),
        "report_path": cli.get("report_path") or (str(report_path) if report_path.exists() else ""),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "gates": gates,
        "cost_status": cost_yaml.get("status"),
        "evidence": {
            "run_yaml": str(framework_run_dir / "run.yaml") if run_yaml else "",
            "report": str(report_path) if report_path.exists() else "",
            "embench": str(framework_run_dir / "10_validation/embench.yaml") if embench else "",
            "hw_repair_feedback": str(framework_run_dir / "09_implementation/hw_repair_feedback.txt"),
            "sw_repair_feedback": str(framework_run_dir / "09_implementation/sw_repair_feedback.txt"),
            "conformance": str(framework_run_dir / "09_implementation/conformance"),
        },
    }

    record["infra"] = meta.get("infra") or detect_infra(
        stdout, stderr, bool(meta.get("timed_out")), meta.get("exit_code"), gates
    )
    record["failure"] = classify_failure(record)
    if record["failure"]:
        for field in ("failure_class", "failure_rung", "failure_detail"):
            record[field] = record["failure"][field]
    record["episodes"] = reconstruct_episodes(record, framework_run_dir)
    record["parse_warnings"] = "; ".join(warnings)
    record["pass_at_1"] = _pass_flag(record, 1)
    record["pass_at_3"] = _pass_flag(record, 3)
    record["pass_at_5"] = _pass_flag(record, 5)
    record["terminal"] = True
    return record


def _pass_flag(record: dict, k: int) -> int | None:
    """1/0 for a complete run, None (CSV-empty) for a failure."""
    if record.get("status") != "complete":
        return None
    attempt = record.get("attempt_index")
    if not isinstance(attempt, int):
        return None
    return 1 if attempt <= k else 0


def write_failure_json(record: dict, run_dir: Path) -> Path | None:
    failure = record.get("failure")
    if not failure:
        return None
    payload = {
        "run_name": record.get("run_name"),
        "model": record.get("model"),
        "benchmark": record.get("benchmark"),
        "repeat": record.get("repeat"),
        "terminal": {
            "status": record.get("status"),
            "stopped_at": record.get("stopped_at"),
            "budget_exhausted": failure.get("budget_exhausted"),
            "repairs_used": record.get("repairs_used"),
            "failure_class": failure.get("failure_class"),
            "failure_rung": failure.get("failure_rung"),
            "failure_detail": failure.get("failure_detail"),
            "attribution": record.get("sim_attribution"),
            "gate": (record.get("gates") or {}).get(failure.get("failure_rung", ""), {}),
        },
        "episodes": record.get("episodes") or [],
        "evidence": record.get("evidence") or {},
        "spent": {
            "token_total": record.get("token_total"),
            "wall_seconds": record.get("wall_seconds"),
        },
    }
    path = run_dir / "failure.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def budget_pass_at_k(records: list[dict], k: int) -> float | None:
    if not records:
        return None
    hits = sum(1 for r in records if _pass_flag(r, k) == 1)
    return hits / len(records)


def sample_pass_at_k(records: list[dict], k: int) -> float | None:
    """1 - C(n-c, k) / C(n, k): i.i.d. 'at least one of k samples' estimator."""
    n = len(records)
    c = sum(1 for r in records if r.get("status") == "complete")
    if n == 0 or k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_budget_ci(records: list[dict], k: int, seed: int = 20260212, iters: int = 10000) -> tuple[float | None, float | None]:
    if not records:
        return None, None
    values = [1 if _pass_flag(r, k) == 1 else 0 for r in records]
    rng = random.Random(seed + k)
    means = []
    for _ in range(int(iters)):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _median(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(nums) if nums else None


def _mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return (sum(nums) / len(nums)) if nums else None


def cell_metrics(records: list[dict], bootstrap_iters: int = 10000) -> dict:
    # Retries are audit records, not independent model samples or model failures.
    records = [r for r in records if not (r.get("infra") or {}).get("infra")]
    n = len(records)
    complete = [r for r in records if r.get("status") == "complete"]
    failure_tally: dict[str, int] = {}
    for record in records:
        failure = record.get("failure") or {}
        cls = failure.get("failure_class")
        if cls:
            failure_tally[cls] = failure_tally.get(cls, 0) + 1
    row: dict = {
        "runs": n,
        "complete_runs": len(complete),
        "attempt_index_median": _median([r.get("attempt_index") for r in complete]),
        "first_try_rate": budget_pass_at_k(records, 1),
        "tokens_per_complete_mean": _mean([r.get("token_total") for r in complete]),
        "wall_h_per_complete_mean": (
            (_mean([r.get("wall_seconds") for r in complete]) or 0) / 3600.0 if complete else None
        ),
        "speedup_x100_median": _median([r.get("speedup_x100") for r in complete]),
        "yosys_cells_median": _median([r.get("yosys_cells") for r in complete]),
        "failure_classes": ", ".join(f"{k}:{v}" for k, v in sorted(failure_tally.items())),
    }
    for k in PASS_AT_K:
        row[f"pass_at_{k}_budget"] = budget_pass_at_k(records, k)
        row[f"pass_at_{k}_sample"] = sample_pass_at_k(records, k)
        lo, hi = bootstrap_budget_ci(records, k, iters=bootstrap_iters)
        row[f"pass_at_{k}_ci95"] = f"[{_fmt(lo)}, {_fmt(hi)}]"
    return row


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def load_records(exp_dir: Path) -> list[dict]:
    records = []
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return records
    for record_path in sorted(runs_dir.glob("*/record.json")):
        try:
            data = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("terminal"):
            records.append(data)
    records.sort(key=lambda r: (str(r.get("model_config")), str(r.get("benchmark")), int(r.get("repeat") or 0)))
    return records


def write_runs_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _fmt(record.get(key)) for key in RUNS_COLUMNS})


def write_failures_csv(records: list[dict], path: Path) -> None:
    rows = []
    for record in records:
        failure = record.get("failure") or {}
        if failure:
            rows.append({
                "exp_id": record.get("exp_id"), "run_name": record.get("run_name"),
                "model": record.get("model_config"), "benchmark": record.get("benchmark"),
                "repeat": record.get("repeat"), "kind": "terminal",
                "attempt": record.get("attempt_index"),
                "failure_class": failure.get("failure_class"),
                "failure_rung": failure.get("failure_rung"),
                "trigger_rung": "", "repair_target": "", "decision_source": "",
                "budget_exhausted": failure.get("budget_exhausted"),
                "detail": failure.get("failure_detail"),
                "evidence": (record.get("evidence") or {}).get("report", ""),
            })
        for episode in record.get("episodes") or []:
            rows.append({
                "exp_id": record.get("exp_id"), "run_name": record.get("run_name"),
                "model": record.get("model_config"), "benchmark": record.get("benchmark"),
                "repeat": record.get("repeat"), "kind": episode.get("kind", "repair"),
                "attempt": record.get("attempt_index"),
                "failure_class": "", "failure_rung": "",
                "trigger_rung": episode.get("trigger_rung", ""),
                "repair_target": episode.get("repair_target", ""),
                "decision_source": episode.get("decision_source", ""),
                "budget_exhausted": "", "detail": episode.get("detail", ""),
                "evidence": episode.get("evidence", ""),
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURES_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in FAILURES_COLUMNS})


def write_pass_at_k_csv(cells: list[tuple[str, str, dict]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PASS_AT_K_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for model, benchmark, metrics in cells:
            row = {"model": model, "benchmark": benchmark}
            row.update(metrics)
            writer.writerow({key: _fmt(row.get(key)) for key in PASS_AT_K_COLUMNS})


def write_summary_md(exp_dir: Path, meta: dict, records: list[dict], cells: list[tuple[str, str, dict]]) -> Path:
    lines = [f"# Experiment {meta.get('exp_id', exp_dir.name)}", ""]
    lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- estimand: {ESTIMAND_HELP}")
    lines.append(f"- git: {meta.get('git_commit', '?')} (dirty={meta.get('git_dirty')})")
    lines.append(f"- runs: {len(records)}")
    lines.append("")

    for k in PASS_AT_K:
        lines.append(f"## pass@{k} (budget estimator)")
        lines.append("")
        lines.append("| model | benchmark | runs | complete | pass@%d | CI95 | sample | attempts(med) |" % k)
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for model, benchmark, m in cells:
            lines.append(
                f"| {model} | {benchmark} | {m['runs']} | {m['complete_runs']} "
                f"| {_fmt(m.get(f'pass_at_{k}_budget'))} | {m.get(f'pass_at_{k}_ci95', '')} "
                f"| {_fmt(m.get(f'pass_at_{k}_sample'))} | {_fmt(m.get('attempt_index_median'))} |"
            )
        lines.append("")

    lines.append("## Failure breakdown (failure_class)")
    lines.append("")
    lines.append("| model | benchmark | failure classes |")
    lines.append("| --- | --- | --- |")
    for model, benchmark, m in cells:
        lines.append(f"| {model} | {benchmark} | {m.get('failure_classes') or '—'} |")
    lines.append("")

    lines.append("## Cost of successful runs")
    lines.append("")
    lines.append("| model | benchmark | tokens/run | wall h/run | speedup_x100 | yosys cells |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for model, benchmark, m in cells:
        lines.append(
            f"| {model} | {benchmark} | {_fmt(m.get('tokens_per_complete_mean'))} "
            f"| {_fmt(m.get('wall_h_per_complete_mean'))} | {_fmt(m.get('speedup_x100_median'))} "
            f"| {_fmt(m.get('yosys_cells_median'))} |"
        )
    lines.append("")

    infra_runs = [r for r in records if (r.get("infra") or {}).get("infra")]
    lines.append("## Infrastructure incidents")
    lines.append("")
    if not infra_runs:
        lines.append("none")
    else:
        lines.append("| run | kind | reason | retry |")
        lines.append("| --- | --- | --- | --- |")
        for record in infra_runs:
            infra = record["infra"]
            lines.append(
                f"| {record.get('run_name')} | {infra.get('infra_kind')} "
                f"| {str(infra.get('infra_reason', ''))[:80]} | infra{record.get('infra_retries', 0)} |"
            )
    lines.append("")
    lines.append("Details per run: `runs/<run_name>/record.json` (+ `failure.json`); "
                 "tables: `results/runs.csv`, `results/failures.csv`, `results/pass_at_k.csv`.")
    lines.append("")

    path = exp_dir / "results" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def aggregate(exp_dir: Path, meta: dict, bootstrap_iters: int = 10000) -> dict:
    records = load_records(exp_dir)
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_cell.setdefault((str(record.get("model_config")), str(record.get("benchmark"))), []).append(record)
    cells = []
    for (model, benchmark), cell_records in sorted(by_cell.items()):
        cells.append((model, benchmark, cell_metrics(cell_records, bootstrap_iters)))

    results = exp_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_runs_csv(records, results / "runs.csv")
    write_failures_csv(records, results / "failures.csv")
    write_pass_at_k_csv(cells, results / "pass_at_k.csv")
    summary = write_summary_md(exp_dir, meta, records, cells)
    return {"records": len(records), "cells": len(cells), "summary": str(summary)}


# --------------------------------------------------------------------------
# Execution
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


def execute_one(job: dict, opts: argparse.Namespace, exp_dir: Path, configs: dict, shim_dir: Path | None, lock_env: dict) -> list[dict]:
    """Run one matrix entry (with its infra retries); return its records."""
    records = []
    infra_retry = int(job.get("infra_retry", 0))
    while True:
        run_name = make_run_name(
            opts.exp_id, job["model"], job["benchmark"], job["repeat"], opts.max_iters, infra_retry
        )
        run_dir = exp_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        framework_run_dir = opts.workdir / "agents" / "generated" / "runs" / run_name
        config_name = configs[job["model"]]["config"]
        argv = build_argv(
            opts.python, opts.conda_env, shim_dir, config_name, job["benchmark"], run_name,
            opts.max_iters, opts.visibility, opts.profile_cache,
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

        print(f"[run ] {run_name}: {shlex.join(argv)}", flush=True)
        meta = run_logged(
            argv, run_dir / "stdout.log", run_dir / "stderr.log",
            opts.workdir, child_env(shim_dir, lock_env), opts.timeout,
        )
        record = collect_record(
            opts.exp_id,
            {**job, "max_iters": opts.max_iters, "infra_retry": infra_retry},
            run_name, run_dir, framework_run_dir, meta,
        )
        (run_dir / "record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        write_failure_json(record, run_dir)
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
            print(f"[warn] {run_name} looks like an infrastructure failure; retry {infra_retry}/{opts.infra_retries}", flush=True)
            continue
        return records


def preflight(opts: argparse.Namespace, configs: dict, shim_dir: Path | None, lock_env: dict) -> dict:
    results = {}
    for model, info in configs.items():
        argv = build_preflight_argv(opts.python, opts.conda_env, shim_dir, info["config"])
        run_dir = opts.exp_dir_ref / "preflight"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = run_logged(
            argv, run_dir / f"{model_slug(model)}.out", run_dir / f"{model_slug(model)}.err",
            opts.workdir, child_env(shim_dir, lock_env), 180,
        )
        text = (run_dir / f"{model_slug(model)}.out").read_text(errors="replace")
        ok = meta.get("exit_code") == 0 and "api-status=OK" in text
        results[model] = {"ok": ok, "exit_code": meta.get("exit_code"), "argv": argv}
        print(f"[preflight] {model}: {'OK' if ok else 'FAILED'}")
        if not ok and not opts.skip_preflight_ok:
            raise SystemExit(
                f"preflight failed for {model} ({info['config']}); "
                "fix the config/API key or rerun with --allow-preflight-fail."
            )
    return results


def materialize_experiment(opts: argparse.Namespace) -> dict:
    meta = {
        "exp_id": opts.exp_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": ESTIMAND_HELP,
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
        "sbt_lock": {
            "enabled": opts.sbt_lock,
            "file": str(opts.sbt_lock_file),
            "timeout": opts.sbt_lock_timeout,
        },
        "argv_template": build_argv(
            opts.python, opts.conda_env, None, "<config>", "<benchmark>", "<run_name>",
            opts.max_iters, opts.visibility, opts.profile_cache,
        ),
    }
    meta.update(git_info(opts.workdir))
    return meta


def write_experiment_yaml(path: Path, meta: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-id", required=True, help="Experiment batch id, e.g. pk5-20260212 (run-id safe).")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma list of agents/llm_configs config names.")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS), help="Comma list of embench-iot benchmark names.")
    parser.add_argument("--repeats", type=int, default=5, help="Independent repeats per cell (default: 5).")
    parser.add_argument("--max-iters", type=int, default=5, help="Agent repair budget; K = attempt budget for pass@K (default: 5).")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature forced into the experiment configs (default: 0.7).")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV,
                        help=f"Conda environment every launched command runs in (default: {DEFAULT_CONDA_ENV!r}; empty disables conda).")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help=f"Python interpreter + flags launched inside the conda env (default: {DEFAULT_PYTHON!r}).")
    parser.add_argument("--workdir", default=str(REPO_ROOT), help="Repo root handed to the agent as --workdir.")
    parser.add_argument("--out-root", default="", help="Experiment output root (default: <workdir>/agents/generated/experiments).")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent runs (sbt/RTL-sim stay serialized via the lock shim; default: 2).")
    parser.add_argument("--timeout", type=int, default=10800, help="Per-run wall-clock timeout in seconds (default: 3h).")
    parser.add_argument("--seed", type=int, default=20260212, help="Scheduling shuffle seed.")
    parser.add_argument("--visibility", default="standard", choices=("basic", "standard", "extended"))
    parser.add_argument("--profile-cache", default="on", choices=("on", "off", "refresh"))
    parser.add_argument("--infra-retries", type=int, default=2, help="Harness retries for infrastructure failures only (default: 2).")
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
    return parser


def _split(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    opts.workdir = Path(opts.workdir).resolve()
    opts.models = _split(opts.models)
    opts.benchmarks = _split(opts.benchmarks)
    opts.only = _split(opts.only)
    if opts.repeats < 1 or opts.max_iters < 1:
        raise SystemExit("--repeats and --max-iters must be >= 1")
    if opts.sbt_lock and opts.jobs > 1 and shutil.which("flock") is None:
        raise SystemExit("flock(1) unavailable: use --no-sbt-lock --jobs 1 or install util-linux")

    out_root = Path(opts.out_root) if opts.out_root else opts.workdir / "agents" / "generated" / "experiments"
    if not out_root.is_absolute():
        out_root = opts.workdir / out_root
    exp_dir = out_root / opts.exp_id
    opts.exp_dir_ref = exp_dir
    lock_file = Path(opts.sbt_lock_file) if opts.sbt_lock_file else out_root / ".locks" / "sbt.lock"
    if not lock_file.is_absolute():
        lock_file = opts.workdir / lock_file
    opts.sbt_lock_file = lock_file

    jobs = schedule_jobs(expand_matrix(opts.models, opts.benchmarks, opts.repeats), opts.seed)
    for job in jobs:
        job["exp_id"] = opts.exp_id
        job["max_iters"] = opts.max_iters
    jobs = filter_jobs(jobs, opts.only)

    shim_dir = None
    lock_env: dict[str, str] = {}
    if opts.sbt_lock:
        require_flock()
        shim_dir = exp_dir / "bin"
        write_sbt_shim(shim_dir)
        lock_env = sbt_lock_env(opts.sbt_lock_file, opts.sbt_lock_timeout)

    if opts.aggregate_only:
        meta_path = exp_dir / "experiment.yaml"
        meta = {}
        if meta_path.exists():
            import yaml

            meta = yaml.safe_load(meta_path.read_text()) or {}
        result = aggregate(exp_dir, meta, opts.bootstrap)
        print(f"aggregated: {result['records']} runs, {result['cells']} cells -> {result['summary']}")
        return 0

    if opts.collect_only:
        meta_path = exp_dir / "experiment.yaml"
        meta = {}
        if meta_path.exists():
            import yaml

            meta = yaml.safe_load(meta_path.read_text()) or {}
        for job in jobs:
            run_name = make_run_name(opts.exp_id, job["model"], job["benchmark"], job["repeat"], opts.max_iters)
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
            record = collect_record(
                opts.exp_id, {**job, "infra_retry": 0}, run_name, run_dir, framework_run_dir, prior_meta
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "record.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n"
            )
            write_failure_json(record, run_dir)
            print(f"[collect] {run_name}: status={record.get('status')}")
        result = aggregate(exp_dir, meta, opts.bootstrap)
        print(f"aggregated: {result['records']} runs, {result['cells']} cells -> {result['summary']}")
        return 0

    configs = materialize_configs(opts.exp_id, opts.models, opts.temperature, opts.workdir)
    meta = materialize_experiment(opts)
    meta["configs"] = configs
    meta["schedule"] = [
        make_run_name(opts.exp_id, j["model"], j["benchmark"], j["repeat"], opts.max_iters) for j in jobs
    ]
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_yaml(exp_dir / "experiment.yaml", meta)

    if opts.dry_run:
        for job in jobs:
            name = make_run_name(opts.exp_id, job["model"], job["benchmark"], job["repeat"], opts.max_iters)
            argv_list = build_argv(
                opts.python, opts.conda_env, shim_dir, configs[job["model"]]["config"],
                job["benchmark"], name, opts.max_iters, opts.visibility,
                opts.profile_cache,
            )
            print(f"[dry ] {name}: {shlex.join(argv_list)}")
        print(f"[dry ] experiment.yaml -> {exp_dir / 'experiment.yaml'} ({len(jobs)} runs scheduled)")
        return 0

    if not opts.skip_preflight:
        meta["preflight"] = preflight(opts, configs, shim_dir, lock_env)
        write_experiment_yaml(exp_dir / "experiment.yaml", meta)

    from concurrent.futures import ThreadPoolExecutor

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, opts.jobs)) as pool:
        futures = [
            pool.submit(execute_one, job, opts, exp_dir, configs, shim_dir, lock_env)
            for job in jobs
        ]
        for future in futures:
            results.extend(future.result())

    result = aggregate(exp_dir, meta, opts.bootstrap)
    print(f"aggregated: {result['records']} runs, {result['cells']} cells -> {result['summary']}")
    failed = [r for r in results if r.get("status") != "complete"]
    print(f"complete: {len(results) - len(failed)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
