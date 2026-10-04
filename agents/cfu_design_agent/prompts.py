"""Prompt builders for the CFU design agent.

Every stage prompt is a five-part brief:

1. the task,
2. where the agent is (working directory + run directory),
3. the tools it may call and their budget,
4. the deliverable and the paths it lands on,
5. the key paths to read plus machine-generated evidence digests.

Repository file bodies are never inlined while a tool loop is available: the
agent reads what it needs with ``read_file``/``list_dir``/``search_text``. The
only fallback that still inlines bodies is a run with that loop disabled
(``--no-agent-tools``, offline client), because a SEARCH/REPLACE delivery must
be built from the exact bytes on disk.
"""

from __future__ import annotations

from typing import Iterable

import yaml

from .runtime.llm import DEFAULT_MAX_TOOL_CALLS_PER_ROUND
from .runtime.tools import RUN_DIR_TOKEN, virtualize_run_paths
from .state import RepoContext, WorkloadSpec


ROOFLINE_GUIDANCE = """Roofline analysis is mandatory. Use operations/byte arithmetic
intensity at a named memory boundary and compare it with compute and bandwidth
ceilings: attainable <= min(compute_peak, bandwidth_peak * arithmetic_intensity).
Compute the ridge point (compute_peak / bandwidth_peak) when both ceilings are
known, classify the bottleneck only from those quantities, and distinguish
algorithmic from achieved intensity. Count external loads/stores plus packing,
writeback, setup, waits, fences and residual scalar work; describe CFU RF traffic
and reuse separately. Label every quantity as measured, parameter-derived or an
explicit assumption, and use `unknown` when a value is unavailable instead of
guessing. Translate the result into a CFU choice and a verification metric.
Roofline estimates guide design but never replace cycle profiling, SoC
simulation or Yosys cost evidence."""


SYSTEM_PROMPT = """You are a hardware/software co-design agent for CfuGym (VexiiRiscv CFU). You analyze a workload, define a CFU ISA contract, design SpinalHDL hardware, write the accelerated C kernel, and verify correctness, cycles and cost.

You work inside one run-scoped workspace. Every task states your working directory, your run directory and the paths you may use; paths are repository-relative. Read what you need with the tools offered in the task instead of assuming a file's contents.

Hard rules:
- Your hardware design surface is the run overlay only: AgentCfu.scala and AgentCfuFiber.scala. Never edit MiCoSoc.scala, MiCoSocParam.scala, CfuPlugin, AgentCfuLib.scala, or any other repository source; the SoC integration is fixed and the run overlay supplies your design.
- Keep the AgentCfu contract stable (AgentCfuParameter, class AgentCfu, the AgentCfuFiber helpers, AgentCfuFunction ids) and build on CfuBus and CfuLsu.
- The accelerated (CFU) build must actually execute CFU instructions for the hot operation. Calling cfu_enable() and then running the scalar loop, or calling an intrinsic and discarding both its CPU result and its declared vector/memory effects, is not acceleration and is rejected.
- Only files your visibility tier exposes are readable; a denied reference returns a policy error. Never assume or fabricate the contents of a document you were not shown.
- Return concise, implementable text. Do not invent performance results."""


ROOFLINE_ANALYSIS_FIELDS = """The `roofline` mapping is mandatory in the YAML and must contain:
`hot_function`, `hot_loop`, `trip_count`, `memory_boundary`, `operations`,
`operation_convention`, `bytes`,
`arithmetic_intensity`, `compute_peak`, `bandwidth_peak`, `ridge_point`,
`predicted_ceiling`, `bottleneck`, `gap_to_ceiling`, `design_action`,
`verification_metric`, and `unknowns`. For each peak, include `value` and
`basis` (`measured`, `parameter-derived`, or `assumption`) when known. Use
`unknown` rather than guessing. State whether intensity is algorithmic or
achieved and include setup/data-movement costs in the end-to-end view."""


ROOFLINE_DECISION_RULE = """Preserve the Roofline conclusion in the next artifact. The CFU
architecture must address the identified bottleneck: memory-bound designs need
to justify byte reduction/reuse/transfer changes, while compute-bound designs
need to justify useful lane throughput/pipeline initiation and a feedable data
path. Include the predicted and measured metric used to verify the choice; do
not claim a bound or speedup without evidence."""


DELIVERY_PROTOCOLS = """Deliver the change with ONE of these protocols.

Protocol A (preferred): targeted edits.
Copy the existing lines exactly and emit one SEARCH/REPLACE pair per change:

EDIT_BEGIN <path>
<<<<<<< SEARCH
<exact existing lines to replace>
=======
<replacement lines>
>>>>>>> REPLACE
EDIT_END

Keep each SEARCH block as small as possible while still unique in the file: a
block that matches zero or multiple locations is rejected, so include enough
surrounding context to be unambiguous. Use several SEARCH/REPLACE pairs inside
one EDIT_BEGIN for several changes to the same file, and several EDIT_BEGIN
sections when several files change. They are applied in order, so each SEARCH
must match the file *after* the earlier sections.

Span only the code you actually change. Do NOT wrap large comment blocks, banner
comments, or long prose in a SEARCH: a single mis-transcribed character rejects
the whole response. If a comment must be part of the replaced region, copy it
byte-for-byte from the file's current content — never retype it from
memory or paraphrase it. If a SEARCH is rejected, do not resend the same block:
shrink it to a few unique code lines, or switch that file to Protocol B.

Never include a superseded draft, an alternative approach, or a second attempt
at the same region in one response: decide on the final change and emit only
that.

Protocol B: whole-file replacement, only for a new file or a large restructure.
Emit the complete new content of every file you replace, with no abbreviations:

FILE_BEGIN <path>
<complete file content>
FILE_END

Protocol C: a unified git diff wrapped between PATCH_BEGIN and PATCH_END. Every
hunk header must match the file exactly; a malformed diff is rejected wholesale."""


def delivery_protocols(tools_enabled: bool) -> str:
    """The delivery protocols, with the read-first rule when tools are available."""
    if not tools_enabled:
        return DELIVERY_PROTOCOLS
    return (
        DELIVERY_PROTOCOLS
        + "\n\nBefore emitting an EDIT_BEGIN, read every target with `read_file` and build the\n"
        "SEARCH lines from the bytes it returns (including leading whitespace/tabs). The live\n"
        "file is the only authority: an edit built from memory or from an earlier attempt will\n"
        "be rejected.\n"
    )


SENSITIVE_PATH_MARKERS = (
    ".git/",
    ".env",
    "secret",
    "token",
    "credential",
    "password",
    "agents/generated/runs/",
)

# Run artifacts are exactly what an agent must be able to list, so the run-root
# marker is dropped when rendering an explicit artifact reference.
_ARTIFACT_SENSITIVE_MARKERS = (
    ".git/",
    ".env",
    "secret",
    "token",
    "credential",
    "password",
)


# Per-tier statement of what the agent may read. Kept as plain data so the
# wording stays in one place and the prompt builders inherit it through
# paths_digest/context_digest.
VISIBILITY_GUIDANCE: dict[str, str] = {
    "basic": (
        "You may read the workload/benchmark sources, the CFU integration contract "
        "(CfuBus/CfuPlugin/CfuLib), the AgentCfu/AgentCfuFiber scaffold, "
        "MiCoSoc/MiCoSocParam, the ISA template, sw/tests, and this run's artifacts.\n"
        "You may NOT read the cfu-designer skills or any existing sibling CFU design "
        "(for example VpuCfu, BitNetCfu); read attempts on those paths are denied. "
        "Design from first principles using only the visible contract and scaffold, "
        "and do not cite or assume the contents of an unavailable document."
    ),
    "standard": (
        "You may additionally read the cfu-designer skills (skills/cfu-designer/**) and "
        "the existing sibling CFU designs (VpuCfu/BitNetCfu); use them for patterns and "
        "idioms.\n"
        "The SpinalHDL source tree (ext/SpinalHDL/**) is NOT available at this level; "
        "work from the AgentCfu scaffold and standard SpinalHDL idioms."
    ),
    "extended": (
        "You may additionally read the SpinalHDL source tree (ext/SpinalHDL/**); the "
        "tier is flow-wide, so every stage of this run sees the same tree.\n"
        "Before writing hardware, use read_file/list_dir/search_text to confirm the exact "
        "SpinalHDL API you intend to call (for example spinal.core Bits/Vec/Stream/Reg and "
        "the fiber/Area patterns) instead of guessing method names, and prefer the idioms "
        "already present in AgentCfu.scala."
    ),
}


def visibility_guidance(level: str) -> str:
    """Describe what the selected visibility tier exposes ("" when unspecified)."""
    text = VISIBILITY_GUIDANCE.get(level.strip().lower())
    if not text:
        return ""
    return f"## Visibility: {level}\n{text}"


READ_TOOL_USAGE: dict[str, str] = {
    "read_file": "`read_file(path, start_line?, end_line?)` — numbered lines of one visible file.",
    "list_dir": "`list_dir(path?, glob?)` — visible entries under a directory.",
    "search_text": "`search_text(pattern, path?, glob?, skip?)` — regex hits as path:line.",
    "list_command_logs": "`list_command_logs(rung?, limit?)` — this run's persisted build/simulation/synthesis logs, newest first.",
    "inspect_observation": "`inspect_observation(id, view=summary|asm|ir|features)` — measured observation evidence.",
    "inspect_failure": "`inspect_failure(id, view=summary|log)` — typed failure evidence.",
    "submit_repair_diagnosis": "`submit_repair_diagnosis(...)` — required before any edit in repair mode.",
    "edit_hardware": "`edit_hardware(edits=[{path, search, replace}])` — atomic SEARCH/REPLACE on the approved hardware overlay files.",
    "check_hardware_elaboration": "`check_hardware_elaboration()` — isolated host-owned elaboration; evidence only, the hw_build gate reruns it.",
    "edit_software": "`edit_software(...)` — one exact SEARCH/REPLACE pair on the approved kernel target.",
    "check_software_compile": "`check_software_compile()` — isolated host-owned compile; evidence only, the sw_build gate reruns it.",
    "build_soc": "`build_soc()` — SoC generation evidence; the hw_build gate reruns it.",
    "build_software": "`build_software()` — software build evidence; the sw_build gate reruns it.",
    "write_conformance": "`write_conformance(oracle)` — validate and deliver an independent small-case oracle; host renders C, no simulation.",
    "check_conformance_syntax": "`check_conformance_syntax()` — optional syntax-only check of delivered C; no link or simulation.",
    "submit_stage": "`submit_stage(delivery)` — finish this stage once the edits are applied; `delivery` is a summary of the change, never the change itself.",
}


def tool_list_block(
    tool_names: Iterable[str],
    tool_rounds: int = 0,
    *,
    failure_ids: list[str] | None = None,
    resolved_failure_ids: list[str] | None = None,
    tool_usage: dict[str, str] | None = None,
) -> str:
    """One-line-per-tool description plus the budget and delivery obligation."""
    usage = tool_usage or READ_TOOL_USAGE
    names = [name for name in tool_names if name in usage]
    if not names:
        return ""
    budget = (
        f"Tool budget: at most {tool_rounds} tool round(s)."
        if tool_rounds
        else "Tool budget: bounded."
    )
    budget += (
        f" At most {DEFAULT_MAX_TOOL_CALLS_PER_ROUND} tool calls per model round; "
        "split larger batches across rounds."
    )
    note = ""
    if failure_ids:
        note = (
            " Deterministic failure evidence is available through `inspect_failure`; "
            f"valid IDs: {', '.join(failure_ids)}. Use view=log only when the typed "
            "summary is insufficient."
        )
    if resolved_failure_ids:
        note += (
            " Already resolved failures (context only — do NOT re-apply a fix for "
            f"them): {', '.join(resolved_failure_ids)}."
        )
    # Never splice the deliverable text into this sentence: it is a multi-line
    # instruction for some stages, and "your next reply must be Call
    # `submit_stage` once…" told the model nothing. The `## Deliverable` section
    # that every stage_brief emits already states it verbatim.
    logs = ""
    if "list_command_logs" in names:
        logs = (
            " Every command the harness runs (build, isolated check, simulation, "
            "synthesis) leaves its complete stdout/stderr in this run; "
            "`list_command_logs` lists them and `read_file` shows one."
        )
    return (
        "## Tools (policy-checked; a denied path returns a policy error)\n"
        + "\n".join(f"- {usage[name]}" for name in names)
        + f"\n{budget}{note}{logs} Read only what you need; the key paths below are where to "
        "start. When the budget runs out the tools are withdrawn: your next reply must "
        "be exactly the deliverable under ## Deliverable. Never answer with only a "
        "promise to read or inspect something: call a tool now, or deliver from what "
        "you already have."
    )


def agent_tools_block(
    agent_tools: bool,
    tool_rounds: int = 0,
    *,
    failure_ids: list[str] | None = None,
    resolved_failure_ids: list[str] | None = None,
    tool_names: Iterable[str] = ("read_file", "list_dir", "search_text"),
) -> str:
    """Read-only tool-loop description shared by every rung that enables tools."""
    if not agent_tools:
        return ""
    return (
        "\n"
        + tool_list_block(
            tool_names,
            tool_rounds,
            failure_ids=failure_ids,
            resolved_failure_ids=resolved_failure_ids,
        )
        + "\n"
    )


def context_digest(
    context: RepoContext, skip: Iterable[str] = (), *, visibility: bool = True
) -> str:
    """Render the repo context with file bodies, omitting files shown elsewhere.

    Legacy path for calls without a tool loop: only a caller that cannot read
    files gets bodies. ``skip`` holds the rung's own targets: they are rendered
    in full in the "Exact current contents" block, and repeating a truncated
    copy here only wastes tokens and invites the model to copy the truncated
    text. ``visibility=False`` drops the access-tier paragraph for a caller
    whose shared header already states it.
    """
    files = context.get("files", {})
    skipped = {str(path) for path in skip}
    omitted = [path for path in files if path in skipped]
    parts: list[str] = []
    guidance = visibility_guidance(str(context.get("visibility", "") or "")) if visibility else ""
    if guidance:
        parts.append(guidance)
    if omitted:
        parts.append(
            "## target files\n"
            + "\n".join(f"- {path}" for path in omitted)
            + "\n(shown in full under \"Exact current contents\" below)"
        )
    for path, text in files.items():
        if path in skipped:
            continue
        if any(marker in path for marker in SENSITIVE_PATH_MARKERS):
            continue
        excerpt = text[:6000]
        if len(text) > len(excerpt):
            excerpt += "\n...[truncated]"
        parts.append(f"## {path}\n{excerpt}")
    return "\n\n".join(parts)


def _under(path: str, root: str) -> bool:
    root = (root or "").strip("/")
    return bool(root) and (path == root or path.startswith(root + "/"))


def artifact_paths_block(
    artifact_paths: Iterable[str | tuple[str, str]],
    skip: Iterable[str] = (),
) -> str:
    """This stage's own artifact paths, with the note that explains each one.

    Stage-specific by construction, so it renders *after* the shared prefix
    instead of inside the key-path index every stage shares.
    """
    skipped = {str(path) for path in skip}
    listed: list[str] = []
    for item in artifact_paths:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            path, note = str(item[0]), str(item[1])
        else:
            path, note = str(item), ""
        if not path or path in skipped:
            continue
        if any(marker in path for marker in _ARTIFACT_SENSITIVE_MARKERS):
            continue
        listed.append(f"- {path}" + (f" — {note}" if note else ""))
    if not listed:
        return ""
    return "## Stage artifacts\n" + "\n".join(dict.fromkeys(listed))


def paths_digest(
    context: RepoContext,
    skip: Iterable[str] = (),
    artifact_paths: Iterable[str | tuple[str, str]] = (),
) -> str:
    """Path-only view of what the agent may read: where to look, never bodies.

    Called without ``skip``/``artifact_paths`` this is stage-independent, which
    is what makes it part of the cached prefix ``stage_brief`` shares between
    stages.
    """
    files = context.get("files", {}) or {}
    skipped = {str(path) for path in skip}
    run_root = str(context.get("run_root", "") or "").strip("/")
    listed: list[str] = []
    for path in files:
        if path in skipped:
            continue
        if any(marker in path for marker in SENSITIVE_PATH_MARKERS):
            continue
        if path.startswith("agents/generated/runs/") and not _under(path, run_root):
            continue
        listed.append(f"- {path}")
    parts: list[str] = []
    guidance = visibility_guidance(str(context.get("visibility", "") or ""))
    if guidance:
        parts.append(guidance)
    if listed:
        parts.append(
            "## Key paths (read what you need with the tools above; skip files whose "
            "content is already covered by the run memory)\n"
            + "\n".join(dict.fromkeys(listed))
        )
    artifacts = artifact_paths_block(artifact_paths, skip=skip)
    if artifacts:
        parts.append(artifacts)
    return "\n\n".join(parts)


def _location(context: RepoContext | None, workdir: str, run_root: str) -> tuple[str, str]:
    context = context or {}
    return (
        str(workdir or context.get("workdir", "") or ""),
        str(run_root or context.get("run_root", "") or ""),
    )


def working_directory_block(
    context: RepoContext | None = None, workdir: str = "", run_root: str = ""
) -> str:
    """Where the agent is: the repo root it works from and the run directory."""
    wd, rr = _location(context, workdir, run_root)
    lines: list[str] = []
    if wd:
        lines.append(
            f"- working directory: {wd} (repository root; every path in this task is repo-relative)"
        )
    if rr:
        lines.append(
            f"- run directory: {rr.rstrip('/')}/ (this run's workspace and artifacts; the token "
            f"`{RUN_DIR_TOKEN}` in this task, in file paths and in tool calls stands for it, and "
            "the framework translates it)"
        )
    if not lines:
        return ""
    return "## Where you are\n" + "\n".join(lines)


def stage_brief(
    stage: str,
    *,
    deliverable: str,
    context: RepoContext | None = None,
    workdir: str = "",
    run_root: str = "",
    tool_names: Iterable[str] = (),
    tool_rounds: int = 0,
    agent_tools: bool = True,
    artifact_paths: Iterable[str | tuple[str, str]] = (),
    skip: Iterable[str] = (),
    failure_ids: list[str] | None = None,
    resolved_failure_ids: list[str] | None = None,
    tools_note: str = "",
    shared: str = "",
    tool_usage: dict[str, str] | None = None,
) -> str:
    """The stage brief, ordered stable-first so the provider can cache the head.

    Layout (constant -> variable):

    1. ``## Where you are`` — the repo root, the run directory and visibility;
    2. the stage-independent key-path index (``paths_digest`` without per-stage
       skips or artifacts);
    3. ``shared`` — digests every design stage renders byte-identically
       (``workload_digest_block``);
    4. the stage block proper: ``## Stage``, tools, ``## Deliverable``, this
       stage's artifact paths, notes, and (without a tool loop) the file bodies.

    Prefix caching compares the serialized request from its first byte, so a
    stage-specific section placed early discards every later shared block; run
    20261003-090854 shared only six characters between two stages' user prompts.
    Keep everything after step 4 per-call (failure evidence, optimization state,
    contracts) instead of moving it up.
    """
    context = context or {}
    parts: list[str] = []
    where = working_directory_block(context, workdir, run_root)
    if where:
        parts.append(where)
    index = paths_digest(context)
    if index:
        parts.append(index)
    if shared.strip():
        parts.append(shared.strip())
    parts.append(f"## Stage: {stage}")
    if agent_tools:
        tools = tool_list_block(
            tool_names,
            tool_rounds,
            failure_ids=failure_ids,
            resolved_failure_ids=resolved_failure_ids,
            tool_usage=tool_usage,
        )
        if tools:
            parts.append(tools)
    else:
        parts.append(
            "No tool loop is available for this call: work from the material shown below."
        )
    parts.append("## Deliverable\n" + deliverable.strip())
    if not agent_tools:
        # The shared index above already states the visibility tier.
        bodies = context_digest(context, skip=skip, visibility=False)
        if bodies:
            parts.append(bodies)
    artifacts = artifact_paths_block(artifact_paths, skip=skip)
    if artifacts:
        parts.append(artifacts)
    if tools_note:
        parts.append(tools_note.strip())
    return "\n\n".join(part for part in parts if part and part.strip())


def spec_digest(spec: WorkloadSpec) -> str:
    return yaml.safe_dump(dict(spec), sort_keys=False)


def profile_digest(profile: dict | None) -> str:
    """Render the measured per-observation-point cycle table."""
    if not profile:
        return "No profile data was measured."
    lines: list[str] = []
    for config, regions in sorted(profile.items()):
        if not isinstance(regions, dict):
            continue
        lines.append(f"[{config}]")
        ordered = sorted(
            regions.items(),
            key=lambda item: item[1].get("cycles", 0) if isinstance(item[1], dict) else 0,
            reverse=True,
        )
        for name, data in ordered:
            if not isinstance(data, dict):
                continue
            calls = data.get("calls")
            suffix = f" calls={calls}" if calls is not None else ""
            lines.append(f"  {name}: {data.get('cycles', 0)} cycles{suffix}")
    return "\n".join(lines) if lines else "No profile data was measured."


def embench_instructions(spec: WorkloadSpec) -> str:
    """How to deliver an accelerated kernel for an Embench workload."""
    if spec.get("source_kind") != "embench":
        return ""
    name = spec.get("benchmark_name", "")
    return (
        f"Embench workload: {name}\n\n"
        "The measured profile includes whole-benchmark, function and (when available)\n"
        "whole-loop inclusive cycles. Prioritize the hottest measured loop, using\n"
        "its source function and line from WorkloadSpec.hot_loop as the edit target.\n"
        "Nested loop cycles overlap; do not sum them.\n\n"
        "Deliverable: a unified git patch that rewrites that hot function so it uses the\n"
        "CFU intrinsics, keeping the scalar path for the reference build:\n\n"
        "- Guard every CFU use with #ifdef USE_AGENT_CFU / #else / #endif so the\n"
        "  reference configuration still compiles and still passes Embench verify.\n"
        f"- Patch target: benchmarks/embench-iot/src/{name}/<file>.c (the harness\n"
        "  redirects it into this run cfu workspace copy; do not invent other paths).\n"
        "- Include cfu_intrinsics.h (generated from your ISA spec) and call\n"
        "  cfu_enable() once before the first CFU instruction.\n"
        "- The kernel must produce exactly the same observable result as the scalar\n"
        "  path: Embench verify_benchmark() is the correctness oracle.\n"
        "- Do not change the benchmark data layout, scaling factors, or verify logic.\n"
        "- The scalar path belongs in the #else branch (the reference build). The\n"
        "  #ifdef USE_AGENT_CFU branch must call the generated CFU intrinsic(s) for the\n"
        "  hot operation: cfu_enable() followed by the scalar loop is not acceleration\n"
        "  and will be rejected by the CFU-usage check and the speedup gate.\n\n"
        "Simulation log semantics (how to read a failure):\n"
        "- MiCoSocSim stops when the CPU reaches the `pass` or `fail` symbol, which the\n"
        "  startup code selects from main()'s return value.\n"
        "- `Software reached the fail symbol` means main() returned non-zero, i.e.\n"
        "  verify_benchmark() rejected the result: the accelerated kernel computed a\n"
        "  WRONG VALUE. It is a correctness bug, not a build error, a crash, or a\n"
        "  harness problem, so do not rewrite the build or the benchmark oracle.\n"
        "- The build also prints `EMBENCH_VERIFY_FAIL benchmark=<name> verify=<code>\n"
        "  result=<value>` (or `..._PASS` on success) from the harness-owned main.c.\n"
        "  Embench's `verify_benchmark()` returns NON-ZERO when the result is correct\n"
        "  and 0 on a mismatch, so `verify=0` is a failure; `result` is main()'s scalar\n"
        "  return where the benchmark uses one.\n"
        "- Neither symbol appearing before the timeout means a hang: an unfinished\n"
        "  TileLink transaction, a wait on a fence, or a non-terminating loop.\n"
        "- After a verify failure, re-check the CFU branch against the scalar reference:\n"
        "  packed-lane sign/size and truncation, `input_offset` handling, any `>>`/scale\n"
        "  left in software, tail and boundary elements, accumulator width/overflow, and\n"
        "  the order in which the intrinsic writes its result.\n"
    )


def workload_digest_block(spec: WorkloadSpec, profile: dict | None) -> str:
    """The stable digests every design stage renders byte-identically.

    Passed to ``stage_brief(shared=...)`` so the analysis, ISA-contract,
    hardware and software prompts share one cached prefix: the text depends only
    on the workload spec and the measured profile, never on the stage.
    """
    parts = [
        f"Normalized WorkloadSpec:\n{spec_digest(spec)}",
        f"Measured profile:\n{profile_digest(profile)}",
    ]
    embench = embench_instructions(spec)
    if embench:
        parts.append(embench)
    return "\n\n".join(parts)


def workload_prompt(
    task: str,
    spec: WorkloadSpec,
    context: RepoContext,
    profile: dict | None = None,
    agent_tools: bool = False,
    tool_rounds: int = 0,
    ir_evidence: dict | None = None,
    observation_index: dict | None = None,
    ir_views: bool = True,
    artifact_paths: Iterable[str | tuple[str, str]] = (),
    feedback: str = "",
) -> str:
    run_root = str((context or {}).get("run_root", "") or "")
    # The next node parses this reply directly with a YAML parser, and nothing
    # else in this prompt says so. Observed runs (2026-09-29) wrapped the mapping
    # in a prose sentence and a ```yaml fence, and the whole analysis was lost.
    deliverable = (
        "One YAML analysis mapping and nothing else. Do not return prose, a "
        "preamble, Markdown fences, or the mapping inside another key. The "
        "response is parsed directly by a YAML parser, so a fence or a bad "
        "indent invalidates the entire analysis."
    )
    if run_root:
        deliverable += f" It is written to `{run_root}/07_analysis/workload_analysis.yaml`."
    tool_names = ["read_file", "list_dir", "search_text", "list_command_logs"]
    evidence_note = ""
    points = (observation_index or {}).get("observations", {})
    if agent_tools and points:
        measured = [
            (ident, point) for ident, point in points.items()
            if isinstance(point, dict) and type(point.get("cycles")) is int
            and point["cycles"] >= 0 and type(point.get("calls")) is int
            and point["calls"] > 0
        ]
        measured.sort(key=lambda item: (-item[1]["cycles"], item[0]))
        hot_loops = [(ident, point) for ident, point in measured if point.get("kind") == "loop"]
        ids = ", ".join(ident for ident, _ in (hot_loops or measured)[:5])
        if ids:
            tool_names.append("inspect_observation")
            evidence_note = (
                "\nMeasured observation IDs (descending inclusive profile cycles): " + ids + ".\n"
                "Call inspect_observation with an ID above and "
                + ("view='asm' or 'ir' "
                   "for the top measured loop before proposing the CFU shape; view='features' "
                   "returns its static feature XML (data movement, typed op counts, "
                   "dependencies); use view='summary' for provenance. "
                   if ir_views else
                   "view='asm' "
                   "for the top measured loop before proposing the CFU shape; use "
                   "view='summary' for provenance. ")
                + "If the returned "
                "evidence is unavailable, continue from the measured profile and source "
                "without guessing. "
                "ASM is at most a reference-build function candidate, not a proven loop-address match; "
                + ("LLVM IR is static Clang evidence, possibly approximate, and never measured cycles. "
                   if ir_views else "")
                + "Unavailable or ambiguous evidence must remain unknown.\n"
            )
    ir_note = ""
    if ir_views and ir_evidence and ir_evidence.get("mode") != "off":
        ir_note = (
            "\nLLVM IR export (static analysis only; do not infer loop counts, cycles, "
            "memory bytes, or source-loop matches from this P0 artifact):\n"
            f"status: {str(ir_evidence.get('status', 'unknown'))[:32]}\n"
            f"target_match: {ir_evidence.get('toolchain', {}).get('target_match', False)}\n"
            f"reason: {str(ir_evidence.get('reason', ''))[:300]}\n"
            f"limitations: {str(ir_evidence.get('limitations', ''))[:240]}\n"
            f"artifacts: {', '.join(str(path) for path in ir_evidence.get('ir_files', [])[:4])}\n"
            "Read a loop's exported IR or its static feature XML with "
            "inspect_observation(view='ir'|'features'); the ASM view needs a reference "
            "build output.\n"
        )
    brief = stage_brief(
        "workload analysis",
        deliverable=deliverable,
        context=context,
        tool_names=tool_names,
        tool_rounds=tool_rounds,
        agent_tools=agent_tools,
        artifact_paths=artifact_paths,
        shared=workload_digest_block(spec, profile),
        tool_usage=(
            None if ir_views else
            {**READ_TOOL_USAGE,
             "inspect_observation": "`inspect_observation(id, view=summary|asm)` — measured observation evidence."}
        ),
    )
    return f"""Task:
{task}

{brief}
{ir_note}{evidence_note}
{ROOFLINE_GUIDANCE}
{feedback}

Analyze the workload. Classify the acceleration pattern, identify hardware/software state, data movement, expected bottlenecks, and the minimum validation needed.
When WorkloadSpec.hot_loop is present, use its measured loop id, source function and line to identify the kernel; nested region cycles are inclusive and must not be summed. Name the exact function and loop that should become the CFU kernel.
Return exactly one YAML mapping and nothing else: no prose before or after it, no
Markdown code fence, no nested ``analysis`` key, and no trailing commentary. Keep
list items and mapping keys distinct — a value that mixes a sequence with mapping
keys on the same indent is not valid YAML and discards the whole reply.
{ROOFLINE_ANALYSIS_FIELDS}"""


def contract_prompt(
    task: str,
    spec: WorkloadSpec,
    analysis: str,
    context: RepoContext,
    profile: dict | None = None,
    feedback: str = "",
    agent_tools: bool = False,
    tool_rounds: int = 0,
    artifact_paths: Iterable[str | tuple[str, str]] = (),
) -> str:
    run_root = str((context or {}).get("run_root", "") or "").rstrip("/")
    contract_path = f"{run_root}/08_isa/cfu_isa.yaml" if run_root else "08_isa/cfu_isa.yaml"
    if agent_tools:
        analysis_block = (
            "Prior analysis (read it before deciding; carry its Roofline conclusion "
            "into `roofline`):\n"
            f"- {run_root}/07_analysis/workload_analysis.yaml"
        )
    else:
        analysis_block = f"Workload analysis:\n{analysis}"
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n=== ISA REPAIR MODE ===\n"
            "The previous ISA contract was rejected by the contract linter. The exact\n"
            "validation problems are:\n"
            "```text\n"
            + feedback
            + "\n```\n"
            "Fix each listed problem in the returned YAML instead of regenerating an\n"
            "unrelated contract. Keep every problem explicitly addressed.\n"
        )
    brief = stage_brief(
        "CFU ISA contract",
        deliverable=(
            "One YAML mapping, or exactly `ABSTAIN` when the measured workload has no "
            f"meaningful CFU opportunity. The harness writes the contract to `{contract_path}`."
        ),
        context=context,
        tool_names=("read_file", "list_dir", "search_text", "list_command_logs"),
        tool_rounds=tool_rounds,
        agent_tools=agent_tools,
        artifact_paths=artifact_paths,
        shared=workload_digest_block(spec, profile),
    )
    return f"""Task:
{task}

{brief}

{analysis_block}
{ROOFLINE_DECISION_RULE}{feedback_block}
This is the ISA-CONTRACT stage only. If the measured workload has no meaningful
CFU opportunity, return exactly `ABSTAIN` with no contract; the run will end
without a verified CFU. Otherwise the next node parses your response directly
with a YAML parser. Return exactly one YAML mapping and nothing else. Do not return analysis, prose, Markdown fences,
Scala, C, a patch, FILE_BEGIN/FILE_END blocks, or an implementation plan. Do not continue to
hardware or software implementation. The contract object must BE the top level of the response:
do not wrap it in an envelope, do not nest it under isa_contract/contract/spec, do not return it
as a string, and do not add thoughts/analysis/reasoning keys. This API may require JSON; if so,
the JSON object must have exactly the top-level keys listed below.
The YAML must contain these top-level keys:
`spec_version`, `identity`, `commands`, `operations_contract`, `state`, `memory`, `datapath`,
`software`, `integration`, `validation`, `assumptions`, `unknowns`, and `roofline`.
`assumptions` is a MAPPING of `name: text` (never a list of strings); `unknowns` is a
LIST of strings.
Copy the Roofline conclusion into `roofline`, including the selected boundary, intensity,
ceiling provenance, bottleneck, and the architecture rationale used by the contract.
Include ISA function IDs, operand/result semantics, and the datapath parameters when they
are used. Each command may declare function_id (funct3, 0..7), subfunction_id
(funct7, 0..127; default 0), and kind (compute/control/load/store). Multiple commands
may share function_id only when subfunction_id differs. Keep names and generated C
symbols unique; cfu_enable, cfu_memory_fence and counter helpers are reserved.
Use multiple compute/control operations when they reduce measured end-to-end overhead;
the hardware receives funct7 through raw_insn and must decode the complete pair.
Every NEW command must declare a `resources` list matching hardware CfuResource:
CPU reads: {{storage: cpu, access: read, role: rs1|rs2}}; CPU write:
{{storage: cpu, access: write, role: rd}}; vector reads: {{storage: vector,
access: read, role: vs1|vs2, index: {{source: cpu_value, register: rs1|rs2}}}};
vector write: same vector/index structure with access: write, role: vd.
Index may instead be {{source: constant, value: N}}. `cpu_value` means the CPU
register CONTENT is a vector slot index; it is never the raw register NUMBER.
Compatibility-only index {{source: legacy_register_number, register: rs1|rs2}}
uses the instruction's register number and legacy truncation; do not choose it
for new designs. At most one data write is supported (CPU rd OR vector vd).
An explicit empty resources list means acknowledgement only; omitted resources
means legacy ABI and must not be silently added to existing frozen commands.
Vector indices consume the selected CPU input without requiring another CPU-read
resource. Declare in-place VD explicitly, or use a constant index; do not invent
an independent third CPU input, packed index convention or config-index resource.
All operations require a completion response. No CPU write means software rd=x0
and a void intrinsic, not a missing CFU response. Keep prose semantics consistent
with these resources and use the generated intrinsic signature.
Treat generic scaffold dimensions `xlen`, `vlen`, and `regDepth` as independent optional knobs:
`0`, `null`, or `unused` means that dimension is not used (including `regDepth: 0` for
no CFU-owned vector register). Do not assume a multiply-accumulate operation,
accumulator, register-file implementation or pipeline. Put only justified algorithm-
specific parameters under `datapath.implementation` with names, units and rationale
appropriate to this workload. Omit unused parameters rather than filling in zeros
for unrelated algorithms. State implementation-specific constraints in
`assumptions`/`unknowns`. Explain how the hot function will call the intrinsics. The SoC
integration is fixed; do not invent new SoC parameters."""


HARDWARE_TARGETS = (
    "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
    "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala",
)


def _conformance_command_table(isa: dict) -> str:
    """Per-command summary of what the oracle must supply."""
    from .conformance import (
        ROLE_NAMES,
        cpu_value_roles,
        has_vector_resource,
        intrinsic_name,
        writes_cpu_rd,
    )

    lines: list[str] = []
    for command in isa.get("commands") or []:
        if not isinstance(command, dict):
            continue
        name = command.get("name", "")
        roles = cpu_value_roles(command)
        needs = "reference (command writes CPU rd)" if writes_cpu_rd(command) else "verify (no CPU result)"
        setup = "; setup required (uses the vector register file)" if has_vector_resource(command) else ""
        # Label by ROLE, not by position: an rs2-only command binds case
        # variable `b`, and a positional 'ab'[i] label would send the model
        # looking for `a`.
        arguments = ", ".join(f"{role}->{ROLE_NAMES[role]}" for role in roles) or "no arguments"
        params = ", ".join(f"uint32_t {role}" for role in roles)
        result_type = "uint32_t" if writes_cpu_rd(command) else "void"
        intrinsic = f"{intrinsic_name(command)}({params}) -> {result_type}"
        lines.append(f"- {name}: generated intrinsic `{intrinsic}`; {needs}{setup}; arguments: {arguments}")
    return "\n".join(lines) if lines else "- (no encodable command)"


def conformance_prompt(
    task: str,
    isa: dict,
    spec: WorkloadSpec,
    *,
    feedback: str = "",
    agent_tools: bool = False,
    tool_rounds: int = 0,
    workdir: str = "",
    run_root: str = "",
    artifact_paths: Iterable[str | tuple[str, str]] = (),
) -> str:
    """Prompt for the hw_conformance rung: the oracle, and nothing else."""
    run_root = run_root.rstrip("/")
    oracle_path = (
        f"{run_root}/09_implementation/conformance/oracle.yaml"
        if run_root
        else "09_implementation/conformance/oracle.yaml"
    )
    feedback_block = ""
    if feedback.strip():
        feedback_block = (
            "\n=== ORACLE REPAIR MODE ===\n"
            "The previous oracle was rejected by the conformance validator. The exact\n"
            "problems are:\n"
            "```text\n" + feedback.strip() + "\n```\n"
            "Fix every listed problem at once. You cannot change the test: the harness\n"
            "generates the C file from this oracle, so an unverifiable command is a hard\n"
            "failure. Do not fix one field while reintroducing another violation; in\n"
            "particular `cases[i].args` must be integer literals only (no casts,\n"
            "addresses, identifiers or expressions) with exactly one per CPU-value role,\n"
            "and any address/data preparation belongs in `setup`.\n"
        )
    if agent_tools:
        contract_block = (
            "CFU ISA contract under test (read it before writing the oracle; never invent "
            "an encoding or intrinsic name):\n"
            f"- {run_root}/08_isa/cfu_isa.yaml"
            if run_root
            else "CFU ISA contract under test: read `08_isa/cfu_isa.yaml`."
        )
    else:
        contract_block = f"CFU ISA contract under test:\n{spec_digest(dict(isa))}"
    brief = stage_brief(
        "hardware conformance",
        deliverable=(
            "One JSON object with a single top-level key `commands` (one oracle entry "
            f"per command above). The harness writes it to `{oracle_path}` and generates "
            "the C test itself; any file you emit would be ignored."
        ),
        workdir=workdir,
        run_root=run_root,
        tool_names=("read_file", "list_dir", "search_text", "list_command_logs"),
        tool_rounds=tool_rounds,
        agent_tools=agent_tools,
        artifact_paths=artifact_paths,
    )
    return f"""Task:
{task}

{brief}

Commands and what each one needs:
{_conformance_command_table(isa)}

{contract_block}
{feedback_block}
This is CONFORMANCE AUTHORING inside Implement, not a simulation step. The host
renders a minimal C program from your oracle; Verify later compiles and simulates
that committed test without asking a model to rewrite it. Do not author arbitrary
test files or run simulation. You MAY revise this oracle in any round: when a build
or simulation failure shows your checks could not see the real defect, strengthen
them; never weaken an oracle to hide a design failure.

Return exactly one JSON object and nothing else, with a single top-level key
`commands` holding one entry per command above (the parser also accepts the same
structure as YAML). The word JSON matters: this request sets json_object mode, and
OpenAI-compatible gateways reject a json_object request whose messages never say
"json" — with an opaque HTTP 400 that has nothing to do with your answer.

commands:
  - name: <exact command name>
    helpers: |            # optional C definitions inserted at the top of the file
      static uint32_t ref_<name>(uint32_t a, uint32_t b) {{ ... }}
    setup: |              # required when the command uses the vector register file
      // C statements that seed the vector slot(s) this command reads
    reference: "ref_<name>(a, b)"   # required when the command writes CPU rd
    verify: "..."                   # required instead when it has no CPU result
    cases:
      - args: [0, 1]      # one value per CPU-value argument, in rs1/rs2 order

Rules:
- `reference` must be an INDEPENDENT scalar implementation of the command's
  documented semantics, transcribed from the workload's own scalar code. Never
  copy a constant table or an expected value out of your hardware design, and never
  make the reference call the CFU intrinsic: that would only re-state the hardware
  and prove nothing.
- `verify` is only for commands with no CPU result (a vector write or an
  acknowledgement), which cannot be read back generically. It is a C expression
  that must be non-zero after the call, evaluated over state your `setup`/`helpers`
  established. It is weaker evidence than `reference`; use `reference` whenever the
  command returns a CPU value. Because the command returns nothing, a `verify`
  built only from the case arguments (`a`, `b`) is rejected: it would restate the
  inputs. Observe real state your `setup`/a helper reads back (memory or a slot the
  command acts on), and prefer declaring a readback command with a CPU `rd` so the
  harness can compare it against `reference`.
- `args` bind to `a` (rs1 value) and `b` (rs2 value) in the declared order; the
  same names are available to `reference` and `verify`.
- Every `args` entry must be an INTEGER LITERAL (decimal, or a C-style integer
  string such as `0xFFFFFFFF`). Casts like `(uintptr_t)ptr`, addresses of your
  `helpers` arrays, identifiers and expressions are rejected: the harness embeds
  each arg as a compile-time constant into the generated intrinsic call. If a
  command loads or stores data, put that address/data preparation in `setup` and
  give the argument the literal address/value it uses (e.g. `args: [0x80000000]`
  for a load whose data your `setup` seeded at that address); if the command
  declares no CPU-value role, use `args: []`.
- At least one case per command must bind a NON-ZERO argument. An all-zero case
  set only exercises the zero boundary and is rejected, so a real address, slot
  index or count is never touched.
- The oracle's C (`helpers`/`setup`/`reference`/`verify`) may call ONLY the
  generated intrinsics named in the command table above (exact spellings!) plus
  `cfu_enable` and `cfu_memory_fence`. Never invent a plausible name like
  `cfu_load`: an unknown name is an `undefined reference` at link time. Use the
  exact generated name, e.g. `cfu_vector_load_i32(...)` rather than `cfu_load(...)`.
- `reference`/`verify` must remain independent of the DUT: neither the expression
  nor any helper/macro/alias it reaches may invoke a CFU intrinsic. Intrinsic-using
  helpers are allowed only for `setup`, not for computing the expected result.
- A check must be able to FAIL: `a == a`, an operand compared with itself, and a
  comparison against a value your own `setup`/`helpers` just assigned a literal to
  are all rejected as self-decided. Compute the expectation from the workload's
  scalar code or from memory the DUT wrote, never from a constant you wrote next to
  the check.
- You choose ALL test inputs and data sizes; the harness executes exactly your
  declared `cases` and never appends hidden edge or random inputs. Use SMALL DATA
  VOLUMES for a quick functional smoke test, not a full benchmark or stress test.
  Choose sizes/counts/loop bounds that finish in well under 32000 cycles per
  blocking CFU call, and keep scalar reference/setup work small. For length/count
  operands start with 0/1/2 (if legal), then a few short representative values;
  do NOT use 0xFFFFFFFF or unconstrained 32-bit random values as iteration counts,
  lengths, repetitions or allocations. Full-width seeds/data values are fine only
  when they do not increase work. Cover useful boundaries with affordable inputs.
- Give each command at least one case. One representative case per command is
  enough; add more only if they are genuinely useful.
- Do not measure or print anything: no cycles, counters, timers or throughput.
  The harness owns all reporting.
- `cases`, `reference`/`verify` and vector `setup` are mandatory for every command;
  a missing one fails the rung instead of being skipped."""


def hardware_prompt(
    task: str,
    spec: WorkloadSpec,
    analysis: str,
    contract: str,
    context: RepoContext,
    profile: dict | None = None,
    feedback: str = "",
    optimization: str = "",
    agent_tools: bool = False,
    tool_rounds: int = 0,
    failure_ids: list[str] | None = None,
    resolved_failure_ids: list[str] | None = None,
    coding_tools: bool = False,
    diagnosis_required: bool = False,
    contract_free: bool = False,
    artifact_paths: Iterable[str | tuple[str, str]] = (),
) -> str:
    """Prompt for the hardware rung: AgentCfu.scala / AgentCfuFiber.scala only."""
    run_root = str((context or {}).get("run_root", "") or "").rstrip("/")
    tool_names = ["read_file", "list_dir", "search_text", "list_command_logs"]
    if coding_tools:
        if failure_ids:
            tool_names.append("inspect_failure")
        if diagnosis_required:
            tool_names.append("submit_repair_diagnosis")
        tool_names.extend(("edit_hardware", "check_hardware_elaboration"))
    deliverable = (
        "ONE EDIT_BEGIN/SEARCH/REPLACE, FILE_BEGIN/FILE_END or PATCH_BEGIN/PATCH_END "
        "block for the allowed target paths below; the harness applies it to this run's "
        "overlay."
    )
    tools_note = ""
    if coding_tools:
        tools_note = (
            "`edit_hardware` atomically applies exact SEARCH/REPLACE operations only to the "
            "approved AgentCfu overlay files; `check_hardware_elaboration` runs an isolated "
            "host-owned GenAgentCfu elaboration and is evidence only (the final hw_build gate "
            "reruns independently)."
        )
        if diagnosis_required:
            tools_note += " In repair mode, submit the repair diagnosis before the first edit."
    brief = stage_brief(
        "hardware design",
        deliverable=deliverable,
        context=context,
        tool_names=tool_names,
        tool_rounds=tool_rounds,
        agent_tools=agent_tools,
        artifact_paths=artifact_paths,
        skip=HARDWARE_TARGETS,
        failure_ids=failure_ids,
        resolved_failure_ids=resolved_failure_ids,
        tools_note=tools_note,
        shared=workload_digest_block(spec, profile),
    )
    if agent_tools:
        design_block = (
            "Design surface: `AgentCfu.scala` is YOUR design file. It holds only "
            "AgentCfuParameter, AgentCfuFunction and the\n"
            "`override protected def instructionTable: Seq[Instruction]` design. The bus shell, "
            "vector RF, response buffer, CfuLsu/FSM, the CfuResource vocabulary and the dispatcher "
            "live in the read-only platform\n"
            "`src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala` (`AgentCfuBase`). Never edit, "
            "rename or re-declare that platform file: read it with `read_file` for the exact "
            "`Instruction(...)`, `CfuResource` and `ctx.read/index/complete/defer` signatures, and "
            "keep the two-file split.\n"
            "Declare each instruction once in `instructionTable` (name, function/subop pair, "
            "optional `enabled`, combinational `ready`, `start`, optional multicycle `tick`, and "
            "its `resources`); decode, unsupported commands, acceptance, response backpressure and "
            "writeback routing are derived from the table. Register only instructions in the "
            "accepted ISA. New instructions must declare resources; at most one data write (CPU RD "
            "or vector VD) is allowed, `cpu_value` indices are the CPU VALUE (never the raw "
            "register number), and a vector index consumes the selected CPU input without an extra "
            "CpuRead. Use `ctx.read`/`ctx.complete` so the platform resolves reads and routes the "
            "single write; do not drive the standard bus handshake from individual handlers, and "
            "allocate design state inside `instructionTable` (or lazy vals).\n"
            "Keep CfuBus, AgentCfuFiber and the CPU decoder interface unchanged. SpinalHDL width "
            "pitfall: `>>` and `<<` on `UInt`/`Bits` CHANGE the result width, so a shift inside a "
            "fixed-width expression silently narrows and fails with `WIDTH MISMATCH`; use `|>>`/`|<<` "
            "or `resize` whenever the width must stay fixed."
        )
    else:
        design_block = (
            "Design surface: AgentCfu.scala is YOUR design file; AgentCfuFiber.scala is "
            "agent-owned and changed only when the fiber contract genuinely must change.\n"
            "The scaffold is split. AgentCfu.scala holds only\n"
            "AgentCfuParameter, AgentCfuFunction and the\n"
            "`override protected def instructionTable: Seq[Instruction]` design. The bus shell, "
            "vector RF, response buffer, CfuLsu/FSM, the CfuResource vocabulary and the\n"
            "dispatcher live in the read-only platform\n"
            "`src/main/scala/vexiiriscv/soc/mico/AgentCfuLib.scala` (`AgentCfuBase`). Never edit,\n"
            "rename or re-declare that platform file; read it with your tools only when you need\n"
            "an exact signature. A whole-file rewrite of AgentCfu.scala cannot delete the\n"
            "platform, so keep this file limited to the design itself.\n"
            "Declare each instruction once in `instructionTable`: name, function/subop pair,\n"
            "optional `enabled`, combinational `ready`, `start`, optional multicycle `tick`, and\n"
            "its `resources`. Decode, unsupported commands, acceptance, response backpressure and\n"
            "writeback routing are derived from the table; do not add separate supported/decode/\n"
            "FSM branches. Register only instructions in the accepted ISA, and remove or disable\n"
            "unused entries. Existing ISA encodings must remain compatible in an\n"
            "optimization/repair round.\n"
            "Platform API (defined in AgentCfuLib.scala):\n"
            "- `Instruction(name, function, subop = 0, enabled = true, ready, start, tick = None,\n"
            "  resources = None)`.\n"
            "- `import AgentCfuBase.CfuResource._` then `CpuRead(RS1)`, `CpuRead(RS2)`,\n"
            "  `CpuWriteRD`, `VectorRead(VS1|VS2, index)`, `VectorWrite(VD, index)`; a vector\n"
            "  index is `CpuValue(RS1|RS2)` the CPU VALUE, `Constant(n)`, or the\n"
            "  compatibility-only `LegacyRegisterNumber(RS1|RS2)` register NUMBER.\n"
            "- Context: `ctx.read(role)`, `ctx.index(vectorRole)`, `ctx.complete(result, status)`,\n"
            "  `ctx.complete()` for acknowledgement, `ctx.fail()`, `ctx.defer()`, and the latched\n"
            "  `rawInstruction`/`stateIndex`/`cfuIndex`/`requestId`.\n"
            "New instructions must declare resources: CPU RS1/RS2 are register VALUES; VS1/VS2 are\n"
            "vector reads with explicit index sources; RD is CPU response writeback and VD is\n"
            "vector writeback. At most one data write (CPU RD or vector VD) is allowed, and a\n"
            "vector index consumes the selected CPU input without an extra CpuRead. Use\n"
            "`ctx.read`/`ctx.complete` so the platform resolves reads, full-width checks indices\n"
            "and routes the single write; do not bypass this with `ctx.input`/`ctx.reply` or the\n"
            "raw vectorRead/vectorWrite helpers in a resource-aware instruction. Use `ctx.defer`\n"
            "plus `tick` for multicycle work; request operands, vector sources and the destination\n"
            "index are captured on acceptance, including vector source snapshots. CPU-value\n"
            "indices are never silently truncated. `LegacyRegisterNumber` is a compatibility\n"
            "exception for the historical load/store encodings, not a new default. A vector write\n"
            "still needs a completion response but has no CPU result. Do not drive the standard bus\n"
            "handshake from individual handlers. Allocate design state inside `instructionTable`\n"
            "(or use lazy vals); do not read an uninitialized subclass `val`.\n"
            "SpinalHDL width pitfall: `>>` and `<<` on `UInt`/`Bits` CHANGE the result width\n"
            "(`UInt.>>` is `ShiftRightByInt`), so a shift inside a fixed-width expression\n"
            "silently becomes narrower and fails with `WIDTH MISMATCH (32 bits <- 31 bits)`.\n"
            "Use `|>>`/`|<<`, or `resize` the shifted value, whenever the width must stay fixed.\n"
            "Keep CfuBus,\n"
            "AgentCfuFiber and the CPU decoder interface unchanged."
        )
    visibility = str(context.get("visibility", "") or "")
    if visibility == "extended":
        if agent_tools:
            brief += (
                "\nThis run has the extended visibility tier: the SpinalHDL source tree "
                "under ext/SpinalHDL/** is readable here. Before writing hardware, confirm "
                "the exact API you call with read_file/search_text (for example "
                "spinal.core Bits/Vec/Stream/Reg and the fiber/Area patterns) instead of "
                "guessing method names.\n"
            )
        else:
            brief += (
                "\nThis run has the extended visibility tier, but the tool loop is disabled, "
                "so ext/SpinalHDL/** cannot be read here; rely on the visible AgentCfu "
                "scaffold and standard SpinalHDL idioms.\n"
            )
    if contract_free:
        if agent_tools:
            analysis_ref = (
                "Prior analysis (read it). There is NO frozen ISA contract in this run: "
                "you design the instruction encodings yourself (custom0 opcode 0x0B; "
                "funct3 0..7; funct7 0..127) and the software side calls them through the "
                "`cfu_intrinsics.h` it writes — keep the RTL decodable for exactly those "
                "encodings:\n"
                f"- {run_root}/07_analysis/workload_analysis.yaml"
            )
        else:
            analysis_ref = "Analysis:\n" + analysis
    elif agent_tools:
        analysis_ref = (
            "Prior analysis and contract (read them; the ISA contract is the frozen interface):\n"
            f"- {run_root}/07_analysis/workload_analysis.yaml\n"
            f"- {run_root}/08_isa/cfu_isa.yaml"
        )
    else:
        analysis_ref = "Analysis:\n" + analysis + "\n\nContract:\n" + contract
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n=== HARDWARE REPAIR MODE ===\n"
            "This is a repair of the previous hardware attempt, not a fresh redesign. "
            "The following is the authoritative tool output from the failing rung. "
            "Do not replace it with a guessed or unrelated diagnosis.\n"
            "\nSTRUCTURED FAILURE EVIDENCE (with full-log reference when available):\n"
            "```text\n"
            + feedback
            + "\n```\n"
            "Before emitting any EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN content, emit a concise "
            "REPAIR_DIAGNOSIS section with exactly these headings:\n"
            "- Observed errors: quote the relevant file:line and compiler/elaboration message.\n"
            "- Root cause: explain why the current code triggers each observed error.\n"
            "- Repair mapping: state the exact code change that addresses each error.\n"
            "- Regression risks: list behavior that must remain unchanged.\n"
            "Then emit the repair as targeted SEARCH/REPLACE edits (preferred), or the "
            "complete hardware files or patch as a fallback. Do not silently rewrite the "
            "design without this diagnosis, and do not claim success before the next build.\n"
        )
    else:
        feedback_block = (
            "\nThis is the first hardware attempt. Do not present a repair diagnosis; design "
            + ("from the workload and current scaffold.\n" if contract_free
               else "from the workload, contract, and current scaffold.\n")
        )
    optimization_block = _optimization_block(optimization)
    targets = "\n".join(f"   - {path}" for path in HARDWARE_TARGETS)
    if agent_tools:
        target_note = (
            "Exact current contents of the hardware targets are NOT inlined: read them with "
            "`read_file` (repo-relative paths are redirected into this run's overlay) before "
            "emitting a SEARCH block."
        )
    else:
        target_blocks = []
        for path in HARDWARE_TARGETS:
            text = context.get("files", {}).get(path, "")
            if text:
                target_blocks.append(f"### {path}\n```scala\n{text}\n```")
        target_note = (
            "Exact current contents of the hardware targets (base your change on these):\n\n"
            + "\n\n".join(target_blocks)
        )
    return f"""Task:
{task}

{brief}

{analysis_ref}
This is the HARDWARE rung only. Design the SpinalHDL CFU datapath; the
accelerated software is written in a later step and must not appear here.
{design_block}
{ROOFLINE_DECISION_RULE}
Use the Roofline result to justify direct versus TileLink, vector width, RF
residency, lane count, pipeline initiation interval, and command granularity.
The implementation must state which modeled quantity it is intended to improve.

{target_note}

{delivery_protocols(agent_tools)}

Allowed target paths (repo-relative paths are redirected into the run workspace
automatically):
{targets}

Do not edit MiCoSoc.scala, MiCoSocParam.scala, CfuPlugin, or any other
repository source. Only fall back to a plan if a required target file is
genuinely absent.
{optimization_block}{feedback_block}"""


def optimization_prompt(
    task: str,
    spec: WorkloadSpec,
    score: str,
    rounds: str,
    isa_spec: dict | None,
    hardware_summary: str,
    software_summary: str,
    budget_left: int,
    focus: str,
    *,
    agent_tools: bool = False,
    tool_rounds: int = 0,
    workdir: str = "",
    run_root: str = "",
    contract_free: bool = False,
    artifact_paths: Iterable[str | tuple[str, str]] = (),
) -> str:
    """Ask whether another round should improve speed or hardware cost (PPA).

    Only called when the run already passes every gate and the iteration budget
    still has room, so both answers are legitimate: keep optimizing, or stop with
    the current validated design.
    """
    run_root = run_root.rstrip("/")
    if agent_tools:
        design_block = (
            "Current design (read what you need before deciding):\n"
            + ("" if contract_free else f"- ISA contract: {run_root}/08_isa/cfu_isa.yaml\n")
            + f"- Hardware plan: {run_root}/09_implementation/hw_plan.md\n"
            f"- Software plan: {run_root}/09_implementation/sw_plan.md"
        )
    else:
        design_block = (
            "ISA spec:\n"
            + (yaml.safe_dump(isa_spec, sort_keys=False).strip() if isa_spec else "n/a")
            + "\n\nHardware design summary:\n"
            + (hardware_summary or "n/a")
            + "\n\nSoftware design summary:\n"
            + (software_summary or "n/a")
        )
    brief = stage_brief(
        "optimization decision",
        deliverable="One JSON object with the continue/target/focus/reason decision.",
        workdir=workdir,
        run_root=run_root,
        tool_names=("read_file", "list_dir", "search_text", "list_command_logs"),
        tool_rounds=tool_rounds,
        agent_tools=agent_tools,
        artifact_paths=artifact_paths,
    )
    extend_bullet = (
        ""
        if contract_free else
        "- `extend_isa` appends new compute/control instructions, including different\n"
        "  subfunction_id values under the same function_id. Keep every old encoding,\n"
        "  operand/result semantic, shared state/memory/datapath/assumptions contract and\n"
        "  integration/software ABI unchanged (intrinsic_names may be appended).\n"
        "  It re-enters ISA validation, hardware build and software adaptation before any\n"
        "  measurement. Name the new operation and how software will use it; adding an\n"
        "  unused instruction is not an optimization. Do not assume the ISA is frozen\n"
        "  when considering this target.\n"
    )
    target_enum = '"implement"' if contract_free else '"implement" or "extend_isa"'
    return f"""Task:
{task}

{brief}

The run already passes every gate. Measured result of the current design:
{score}

Optimization rounds so far (newest last):
{rounds or "none yet: this would be the first optimization round"}

Optimization budget left: {budget_left} round(s).

Workload specification (existing command encodings/semantics must remain compatible):
{spec_digest(spec)}

{design_block}

Decide whether another round is worth spending budget on, and where.

- `implement` hands the round to the implementation stage, where you change
  whichever side the evidence points at — the AgentCfu datapath (lane count,
  width, pipelining, register/stream handling, command granularity, TileLink
  usage) and/or the accelerated kernel (packing, blocking, loop structure,
  intrinsic granularity, residual scalar work, fences). The whole ladder then
  reruns, so name the concrete change rather than the side.
{extend_bullet}Prefer `{focus}` unless the evidence clearly points at the other side.

Answer `continue: true` only when you can name a concrete change that should beat
the current numbers and keep every gate passing (the correctness oracle, the
minimum speedup floor enforced by the simulation gate, and a successful
synthesis). Do not repeat a change that
already ran and did not improve; a round that makes the result worse is a real
regression in the delivered design, not a free experiment.
A reasonable speedup alone is not a reason to stop exploring. Consider instruction
batching, retained intermediates within the existing state contract, control/result
operations, packing/transfer overhead and area reduction before proposing a stop.
Answer `continue: false` only when you have no distinct, concrete candidate left;
the framework owns the budget and may ask you to reconsider this recommendation.

Return exactly one JSON object and nothing else:
{{"continue": true or false,
  "target": {target_enum},
  "focus": "performance" or "cost",
  "reason": "<the concrete change you would make, or why you stop>"}}"""


def _optimization_block(optimization: str) -> str:
    """A passing design is being improved on purpose, not repaired."""
    if not optimization:
        return ""
    return (
        "\n=== OPTIMIZATION ROUND ===\n"
        "Every gate already passes, so this is not a repair: keep the parts that work and\n"
        "make one focused change that improves the measured result below.\n"
        + optimization
        + "\n"
    )


def software_prompt(
    task: str,
    spec: WorkloadSpec,
    analysis: str,
    contract: str,
    context: RepoContext,
    profile: dict | None = None,
    intrinsics: str = "",
    targets: list[tuple[str, str]] | None = None,
    feedback: str = "",
    optimization: str = "",
    agent_tools: bool = False,
    tool_rounds: int = 0,
    failure_ids: list[str] | None = None,
    resolved_failure_ids: list[str] | None = None,
    coding_tools: bool = False,
    diagnosis_required: bool = False,
    intrinsics_path: str = "",
    contract_free: bool = False,
    artifact_paths: Iterable[str | tuple[str, str]] = (),
) -> str:
    """Prompt for the software rung: the accelerated kernel using the intrinsics.

    ``targets`` is [(emit_path, current_text)]; ``emit_path`` is the exact
    run-workspace path the model must use in FILE_BEGIN / the diff header.
    Bodies are inlined only when no tool loop can read them.
    """
    targets = targets or []
    run_root = str((context or {}).get("run_root", "") or "").rstrip("/")
    if agent_tools and intrinsics_path:
        intrinsics_block = (
            "\n"
            + ("Agent-authored CFU intrinsics header (yours to edit with the software "
               "target edits; keep its signatures and the RTL encodings consistent):\n"
               if contract_free else
               "Generated CFU intrinsics (read the exact signatures before calling them):\n")
            + f"- {intrinsics_path}\n"
        )
    elif intrinsics:
        intrinsics_block = (
            "\n"
            + ("Intrinsic template — write `cfu_intrinsics.h` from it (adapt it freely) and "
               "call your own intrinsics from the accelerated kernel:\n"
               if contract_free else
               "Generated CFU intrinsics (call these from the accelerated kernel):\n")
            + "```c\n"
            + intrinsics
            + "\n```\n"
        )
    else:
        intrinsics_block = ""
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n=== SOFTWARE REPAIR MODE ===\n"
            "This is a repair of the previous software attempt, not a fresh rewrite. "
            "The following is the authoritative tool output from the failing rung. "
            "Do not replace it with a guessed or unrelated diagnosis.\n"
            "\nSTRUCTURED FAILURE EVIDENCE (with full-log reference when available):\n"
            "```text\n"
            + feedback
            + "\n```\n"
            "Before emitting any EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN content, emit a concise "
            "REPAIR_DIAGNOSIS section with exactly these headings:\n"
            "- Observed errors: quote the relevant file:line and compiler/simulator message.\n"
            "- Root cause: explain why the current code triggers each observed error.\n"
            "- Repair mapping: state the exact code change that addresses each error.\n"
            "- Regression risks: list behavior that must remain unchanged.\n"
            "Then emit the repair as targeted SEARCH/REPLACE edits (preferred), or the "
            "complete software files or patch as a fallback, and keep the scalar fallback.\n"
        )
    else:
        feedback_block = (
            "\nThis is the first software attempt. Do not present a repair diagnosis; "
            + ("design the instructions (`cfu_intrinsics.h`) and the kernel together — "
               "there is no ISA contract in this run.\n"
               if contract_free else
               "implement the kernel from the contract and intrinsics.\n")
            if not optimization
            else "\nThe kernel is already written and passing; improve it as described above.\n"
        )
    tool_names = ["read_file", "list_dir", "search_text", "list_command_logs"]
    if coding_tools:
        if failure_ids:
            tool_names.append("inspect_failure")
        if diagnosis_required:
            tool_names.append("submit_repair_diagnosis")
        tool_names.extend(("edit_software", "check_software_compile"))
    deliverable = (
        "ONE EDIT_BEGIN/SEARCH/REPLACE, FILE_BEGIN/FILE_END or PATCH_BEGIN/PATCH_END "
        "block for the allowed target paths below."
    )
    tools_note = ""
    if coding_tools:
        tools_note = (
            "`edit_software` accepts exactly one approved target and one exact "
            "SEARCH/REPLACE pair; `check_software_compile` runs a host-owned isolated "
            "compile and is evidence only (the final sw_build gate reruns independently). "
            "After tool edits, still emit a final EDIT/FILE/PATCH delivery describing the "
            "resulting change for audit."
        )
        if diagnosis_required:
            tools_note += " In repair mode, submit the repair diagnosis before the first edit."
    target_paths = [path for path, _ in targets]
    brief = stage_brief(
        "software implementation",
        deliverable=deliverable,
        context=context,
        tool_names=tool_names,
        tool_rounds=tool_rounds,
        agent_tools=agent_tools,
        artifact_paths=artifact_paths,
        skip=target_paths,
        failure_ids=failure_ids,
        resolved_failure_ids=resolved_failure_ids,
        tools_note=tools_note,
        shared=workload_digest_block(spec, profile),
    )
    if contract_free:
        if agent_tools:
            context_ref = (
                "Prior analysis (read it). There is NO ISA contract in this run: the "
                "`cfu_intrinsics.h` you write IS the instruction interface — design it from "
                "the intrinsic template and keep it consistent with the hardware:\n"
                f"- {run_root}/07_analysis/workload_analysis.yaml"
            )
        else:
            context_ref = "Analysis:\n" + analysis
    elif agent_tools:
        context_ref = (
            "Prior analysis and contract (read them; the ISA contract is the frozen interface):\n"
            f"- {run_root}/07_analysis/workload_analysis.yaml\n"
            f"- {run_root}/08_isa/cfu_isa.yaml"
        )
    else:
        context_ref = "Analysis:\n" + analysis + "\n\nContract:\n" + contract
    if agent_tools:
        target_note = (
            "Exact current contents of the software targets are NOT inlined: read them with "
            "`read_file` before emitting a SEARCH block."
        )
    else:
        target_note = (
            "Exact current contents of the software targets (base your change on these):\n\n"
            + "\n\n".join(f"### {path}\n```c\n{text}\n```" for path, text in targets if text)
        )
    target_list = "\n".join(f"   - {path}" for path in target_paths)
    optimization_block = _optimization_block(optimization)
    if contract_free:
        abi_block = (
            "There is NO ISA contract in this run: you define the instruction interface "
            "yourself in `cfu_intrinsics.h` (custom0 opcode 0x0B; funct3 0..7; funct7 0..127). "
            "Whatever encodings and operand conventions you pick, the header and the hardware "
            "must agree on them exactly: CPU rs1/rs2 arguments are values (or vector slot "
            "indices), CPU rd writes return uint32_t, and a void-returning operation must not "
            "fabricate a C return value."
        )
        intrinsic_rule = (
            "The CFU build (#ifdef USE_AGENT_CFU) must call your cfu_* intrinsic(s)"
        )
    else:
        abi_block = (
            "Follow each command's CfuResource/resources declarations and the exact generated\n"
            "intrinsic signature. CPU rs1/rs2 arguments are values, or vector slot indices when\n"
            "selected by a cpu_value index resource; never pass a physical register number in\n"
            "place of a slot value. CPU rd writes return uint32_t; vector writes and ack-only\n"
            "commands return void and encode rd=x0. Vector-only operations must feed a later\n"
            "vector read/store or another useful operation; their lack of a C return value is\n"
            "not grounds for fabricating one. Omitted resources denote the legacy two-argument\n"
            "ABI; do not silently change legacy encodings or the frozen command contract."
        )
        intrinsic_rule = (
            "The CFU build (#ifdef USE_AGENT_CFU) must call the generated CFU intrinsic(s)"
        )
    return f"""Task:
{task}

{brief}

{context_ref}
{intrinsics_block}
This is the SOFTWARE rung only. The hardware design is already fixed and
already passed SoC generation; do not propose hardware changes here.
{abi_block}
{ROOFLINE_DECISION_RULE}
{intrinsic_rule}
for the hot operation and derive its result from them. A branch that only calls
cfu_enable() and then computes with the scalar loop, or that calls an intrinsic
and ignores both its CPU result and any declared vector/memory effects, is
rejected: it contributes no CFU datapath result to the workload. Keep the scalar loop only in the #else (reference) branch.
Use the Roofline result to guide packing, blocking, data reuse, load/store
traffic, fences, intrinsic granularity, and residual scalar work. Preserve the
scalar fallback and report how the kernel should change achieved operations/cycle
or external bytes/cycle relative to the predicted ceiling.

{target_note}

{delivery_protocols(agent_tools)}

Allowed target paths (use these paths verbatim):
{target_list}

Only fall back to a plan if a required target file is genuinely absent.
{optimization_block}{feedback_block}"""
