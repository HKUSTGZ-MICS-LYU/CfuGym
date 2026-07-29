"""Prompt builders for the CFU design agent."""

from __future__ import annotations

import yaml

from .state import RepoContext, WorkloadSpec


SYSTEM_PROMPT = """You are a hardware/software co-design agent for CfuGym.
Use the VexiiRiscv CFU design loop: analyze workload, extract operation, define ISA contract, design SpinalHDL CFU hardware, write accelerated C kernel, validate correctness, measure cycles, estimate Yosys cost, and iterate.
Prefer existing repo patterns: CfuPlugin/CfuBus, DirectCfuSpec or TilelinkCfuSpec, CfuLsu, MiCoSocParam CLI flags, scalar fallback tests, and generated SoC RTL for cost.
Return concise, implementable text. Do not invent performance results."""


def context_digest(context: RepoContext) -> str:
    files = context.get("files", {})
    parts: list[str] = []
    for path, text in files.items():
        excerpt = text[:6000]
        if len(text) > len(excerpt):
            excerpt += "\n...[truncated]"
        parts.append(f"## {path}\n{excerpt}")
    status = context.get("git_status", "").strip()
    if status:
        parts.append(f"## git status --short\n{status}")
    return "\n\n".join(parts)


def spec_digest(spec: WorkloadSpec) -> str:
    return yaml.safe_dump(dict(spec), sort_keys=False)


def workload_prompt(task: str, spec: WorkloadSpec, context: RepoContext) -> str:
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Repo context:
{context_digest(context)}

Analyze the workload. Classify the acceleration pattern, identify hardware/software state, data movement, expected bottlenecks, and the minimum validation needed."""


def contract_prompt(task: str, spec: WorkloadSpec, analysis: str, context: RepoContext) -> str:
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Workload analysis:
{analysis}

Repo context:
{context_digest(context)}

Design the CFU contract and integration approach. Include ISA function IDs, operand/result semantics, direct-vs-TileLink choice, SoC parameters, software intrinsic names, validation commands, and cost command."""


def implementation_prompt(task: str, spec: WorkloadSpec, analysis: str, contract: str, context: RepoContext) -> str:
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Analysis:
{analysis}

Contract:
{contract}

Repo context:
{context_digest(context)}

Produce either:
1. A unified git patch touching only CFU allowlisted paths, or
2. A precise implementation plan if the patch would be too risky.

If producing a patch, wrap it between lines PATCH_BEGIN and PATCH_END."""
