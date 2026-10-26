"""Parsing the agent's stable machine-readable output surfaces.

``parse_cli_stdout`` reads the CLI's ``status=``/``summary:`` lines;
``parse_report`` reads ``report.md``; ``load_yaml`` is the never-raising YAML
loader every collector uses.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

# Token-usage regex blocks: same estimand, two stable surfaces (the CLI prints
# token_usage at exit; report.md "## Token Usage" fills in what a crashed
# process never printed).
CLI_TOKEN_PATTERNS = (
    ("token_calls", r"^\s*calls\s+= (\d+)"),
    ("token_in", r"^\s*input_tokens=\s*(\d+)"),
    ("token_out", r"^\s*output_tokens=\s*(\d+)"),
    ("token_total", r"^\s*total_tokens=\s*(\d+)"),
)
REPORT_TOKEN_PATTERNS = (
    ("token_calls", r"^- llm_calls: (\d+)"),
    ("token_in", r"^- input tokens: (\d+)"),
    ("token_out", r"^- output tokens: (\d+)"),
    ("token_total", r"^- total tokens: (\d+)"),
)


def _token_usage(text: str, patterns) -> dict:
    usage: dict = {}
    for key, pattern in patterns:
        m = re.search(pattern, text, re.M)
        if m:
            usage[key] = int(m.group(1))
    return usage


def parse_cli_stdout(text: str) -> dict:
    """Parse the stable machine-readable CLI lines (status=, summary:, ...)."""
    out: dict = {}
    m = re.search(r"^status=(\S+)", text, re.M)
    if m:
        out["status"] = m.group(1)
    m = re.search(r"^termination=(\S+)", text, re.M)
    if m:
        out["termination_reason"] = m.group(1)
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
    out.update(_token_usage(text, CLI_TOKEN_PATTERNS))
    return out


def parse_report(text: str, warnings: list[str]) -> dict:
    """Parse report.md status, gate verdicts, and repair/submission counters."""
    out: dict = {}
    m = re.search(r"^Status: (\w+)", text, re.M)
    if m:
        out["status"] = m.group(1)
    m = re.search(r"^- verify submissions: (\d+)/(\d+)", text, re.M)
    if m:
        out["verify_submissions"] = int(m.group(1))
        out["max_verify_submissions"] = int(m.group(2))
    m = re.search(r"^- answer repairs used: (\d+)/(\d+)", text, re.M)
    if m:
        out["answer_repairs_used"] = int(m.group(1))
        out["max_answer_repairs"] = int(m.group(2))
    m = re.search(r"^- pre-verify repairs used: (\d+)/(\d+)", text, re.M)
    if m:
        out["preverify_repairs_used"] = int(m.group(1))
        out["max_preverify_repairs"] = int(m.group(2))
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
    for key in ("termination", "current_performance", "last_observed_valid_performance"):
        match = re.search(r"^- " + key + r": (\{.*\})$", text, re.M)
        if match:
            try:
                value = ast.literal_eval(match.group(1))
                if isinstance(value, dict):
                    out[key] = value
            except (ValueError, SyntaxError) as exc:
                warnings.append(f"report {key} unparsed: {exc}")

    # The report's "## Ablation" section is the framework's own record of which
    # --no-* switches were active; collection cross-checks it against the layer
    # the harness intended so silently ignored switches are flagged.
    ablation = re.search(r"^## Ablation\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    out["ablation_flags"] = re.findall(r"`(--no-[a-z-]+)`", ablation.group(1)) if ablation else []

    # "## Token Usage" fallback: the CLI prints token_usage at exit, but a
    # crashed/hung process never gets there while report.md already carries the
    # totals.  Same estimand as the CLI block, just another stable surface.
    out["token_usage"] = _token_usage(text, REPORT_TOKEN_PATTERNS)
    return out


def load_json(path: Path, warnings: list[str], label: str) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(errors="replace"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError) as exc:
        warnings.append(f"{label} unparsed: {exc}")
        return {}


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
