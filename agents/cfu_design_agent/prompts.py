"""Prompt builders for the CFU design agent."""

from __future__ import annotations

import yaml

from .state import RepoContext, WorkloadSpec


ROOFLINE_GUIDANCE = """Roofline analysis is mandatory for workload and CFU decisions. Use
operations/byte arithmetic intensity at a named memory boundary and compare it
with compute and bandwidth ceilings using
attainable <= min(compute_peak, bandwidth_peak * arithmetic_intensity).
Compute the ridge point (compute_peak / bandwidth_peak) when both ceilings are
known, classify the bottleneck only from those quantities, and distinguish
algorithmic from achieved intensity. Count external loads/stores plus packing,
writeback, setup, waits, fences, and residual scalar work; describe CFU RF
traffic and reuse separately. Label every quantity as measured,
parameter-derived, or an explicit assumption. If a value is unavailable, keep it
unknown and state how to measure it. Translate the result into a CFU choice and
a verification metric. Roofline estimates guide design but never replace cycle
profiling, SoC simulation, or Yosys cost evidence."""


SYSTEM_PROMPT = f"""You are a hardware/software co-design agent for CfuGym.
Use the VexiiRiscv CFU design loop: analyze workload, extract operation, define ISA contract, design SpinalHDL CFU hardware, write accelerated C kernel, validate correctness, measure cycles, estimate Yosys cost, and iterate.
{ROOFLINE_GUIDANCE}
Your hardware design surface is the AgentCfu scaffold only: AgentCfu.scala and AgentCfuFiber.scala in the run's isolated design workspace. Do not edit MiCoSoc.scala, MiCoSocParam.scala, CfuPlugin, or any other repository source; the SoC integration is fixed and the run overlay supplies your design.
Keep the AgentCfu contract stable (AgentCfuParameter, class AgentCfu, the AgentCfuFiber helpers, AgentCfuFunction ids) and build on CfuBus and CfuLsu.
The accelerated (CFU) build must actually execute CFU instructions for the hot operation. Calling cfu_enable() and then running the scalar loop, or calling an intrinsic and discarding its result, is not acceleration and is rejected.
Only files your visibility tier exposes are readable; a denied reference returns a policy error. Never assume or fabricate the contents of a document you were not shown.
Return concise, implementable text. Do not invent performance results."""


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

Keep each SEARCH block as small as possible while still unique in the file. A
block that matches zero or multiple locations is rejected, so include enough
surrounding context to be unambiguous. Use several SEARCH/REPLACE pairs inside
one EDIT_BEGIN for several changes to the same file, and several EDIT_BEGIN
sections when several files change. You may also open several EDIT_BEGIN
sections for the same file: they are applied in order, and each SEARCH must
match the file *after* the earlier sections. Prefer one section per file unless
the change genuinely needs to be split.

Never include a superseded draft, an alternative approach, or a second attempt
at the same region in one response: decide on the final change and emit only
that. Consecutive sections that re-search the same original lines will fail,
because the earlier section already replaced them.

Before emitting an EDIT_BEGIN, re-read the current content of each target shown
above and confirm the SEARCH lines match it byte-for-byte (including leading
whitespace/tabs). If you cannot match a region exactly, use Protocol B for that
file instead of guessing.

Protocol B: whole-file replacement, only for a new file or a large restructure.
Emit the complete new content of every file you replace, with no abbreviations:

FILE_BEGIN <path>
<complete file content>
FILE_END

Protocol C: a unified git diff wrapped between PATCH_BEGIN and PATCH_END. Every
hunk header must match the file exactly; a malformed diff is rejected wholesale."""


SENSITIVE_PATH_MARKERS = (
    ".git/",
    ".env",
    "secret",
    "token",
    "credential",
    "password",
    "agents/generated/runs/",
)


# Per-tier statement of what the agent may read. Kept as plain data so the
# wording stays in one place and the four prompt builders inherit it through
# context_digest.
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
        "You may additionally read the SpinalHDL source tree (ext/SpinalHDL/**) in the "
        "isa and implementation stages.\n"
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


def context_digest(context: RepoContext) -> str:
    files = context.get("files", {})
    parts: list[str] = []
    guidance = visibility_guidance(str(context.get("visibility", "") or ""))
    if guidance:
        parts.append(guidance)
    for path, text in files.items():
        if any(marker in path for marker in SENSITIVE_PATH_MARKERS):
            continue
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
        "The measured profile above lists cycle counts per observation point. Use the\n"
        "hottest entry (excluding the whole-benchmark entry) as your target.\n\n"
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
        "  and will be rejected by the CFU-usage check and the speedup gate.\n"
    )


def workload_prompt(
    task: str,
    spec: WorkloadSpec,
    context: RepoContext,
    profile: dict | None = None,
) -> str:
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Measured profile:
{profile_digest(profile)}

Repo context:
{context_digest(context)}

{embench_instructions(spec)}
{ROOFLINE_GUIDANCE}
Analyze the workload. Classify the acceleration pattern, identify hardware/software state, data movement, expected bottlenecks, and the minimum validation needed.
Name the exact function and loop that should become the CFU kernel.
Return exactly one YAML mapping. {ROOFLINE_ANALYSIS_FIELDS}"""


def contract_prompt(
    task: str,
    spec: WorkloadSpec,
    analysis: str,
    context: RepoContext,
    profile: dict | None = None,
    feedback: str = "",
) -> str:
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
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Measured profile:
{profile_digest(profile)}

Workload analysis:
{analysis}

Repo context:
{context_digest(context)}

{embench_instructions(spec)}
{ROOFLINE_DECISION_RULE}
{feedback_block}
This is the ISA-CONTRACT stage only. The next node parses your response directly with a YAML parser.
Return exactly one YAML mapping and nothing else. Do not return analysis, prose, Markdown fences,
Scala, C, a patch, FILE_BEGIN/FILE_END blocks, or an implementation plan. Do not continue to
hardware or software implementation. The YAML must contain these top-level keys:
`spec_version`, `identity`, `commands`, `operations_contract`, `state`, `memory`, `datapath`,
`software`, `integration`, `validation`, `assumptions`, `unknowns`, and `roofline`.
Copy the Roofline conclusion into `roofline`, including the selected boundary, intensity,
ceiling provenance, bottleneck, and the architecture rationale used by the contract.
Include ISA function IDs, operand/result semantics, and the datapath parameters when they
are used. Treat `xlen`, `vlen`, `maclen`, and `regDepth` as independent optional knobs:
`0`, `null`, or `unused` means that dimension is not used (including `regDepth: 0` for
no CFU-owned vector register). Do not require divisibility, positivity, or a MAC slice
when the contract does not use it; state any implementation-specific constraints in
`assumptions`/`unknowns`. Explain how the hot function will call the intrinsics. The SoC
integration is fixed; do not invent new SoC parameters."""


HARDWARE_TARGETS = (
    "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
    "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala",
)


def hardware_prompt(
    task: str,
    spec: WorkloadSpec,
    analysis: str,
    contract: str,
    context: RepoContext,
    profile: dict | None = None,
    feedback: str = "",
    agent_tools: bool = False,
) -> str:
    """Prompt for the hardware rung: AgentCfu.scala / AgentCfuFiber.scala only."""
    tools_block = ""
    if agent_tools:
        tools_block = (
            "\nYou have read-only tools: `read_file`, `list_dir`, `search_text`. "
            "Use them to inspect the current workspace, headers, and related sources "
            "before you answer. Reading is subject to the stage access policy; a denied "
            "path returns a policy error rather than content. When you have enough "
            "information, reply with the final deliverable and no tool call.\n"
        )
    visibility = str(context.get("visibility", "") or "")
    if visibility == "extended":
        if agent_tools:
            tools_block += (
                "\nThis run has the extended visibility tier: the SpinalHDL source tree "
                "under ext/SpinalHDL/** is readable here. Before writing hardware, confirm "
                "the exact API you call with read_file/search_text (for example "
                "spinal.core Bits/Vec/Stream/Reg and the fiber/Area patterns) instead of "
                "guessing method names.\n"
            )
        else:
            tools_block += (
                "\nThis run has the extended visibility tier, but the tool loop is disabled, "
                "so ext/SpinalHDL/** cannot be read here; rely on the visible AgentCfu "
                "scaffold and standard SpinalHDL idioms.\n"
            )
    target_blocks = []
    for path in HARDWARE_TARGETS:
        text = context.get("files", {}).get(path, "")
        if text:
            target_blocks.append(f"### {path}\n```scala\n{text}\n```")
    target_contents = "\n\n".join(target_blocks)
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n=== HARDWARE REPAIR MODE ===\n"
            "This is a repair of the previous hardware attempt, not a fresh redesign. "
            "The following is the authoritative tool output from the failing rung. "
            "Do not replace it with a guessed or unrelated diagnosis.\n"
            "\nEXACT ERROR LOG:\n"
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
            "from the workload, contract, and current scaffold.\n"
        )
    targets = "\n".join(f"   - {path}" for path in HARDWARE_TARGETS)
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Measured profile:
{profile_digest(profile)}

Analysis:
{analysis}

Contract:
{contract}

Repo context:
{context_digest(context)}
{feedback_block}
{embench_instructions(spec)}{tools_block}
This is the HARDWARE rung only. Design the SpinalHDL CFU datapath; the
accelerated software is written in a later step and must not appear here.
{ROOFLINE_DECISION_RULE}
Use the Roofline result to justify direct versus TileLink, vector width, RF
residency, lane count, pipeline initiation interval, and command granularity.
The implementation must state which modeled quantity it is intended to improve.

Exact current contents of the hardware targets (base your change on these):

{target_contents}

{DELIVERY_PROTOCOLS}

Allowed target paths (repo-relative paths are redirected into the run workspace
automatically):
{targets}

Do not edit MiCoSoc.scala, MiCoSocParam.scala, CfuPlugin, or any other
repository source. Only fall back to a plan if a required target file is
genuinely absent."""


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
) -> str:
    """Prompt for the software rung: the accelerated kernel using the intrinsics.

    ``targets`` is [(emit_path, current_text)]; ``emit_path`` is the exact
    run-workspace path the model must use in FILE_BEGIN / the diff header.
    """
    targets = targets or []
    intrinsics_block = ""
    if intrinsics:
        intrinsics_block = (
            "\nGenerated CFU intrinsics (call these from the accelerated kernel):\n"
            "```c\n"
            + intrinsics
            + "\n```\n"
        )
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n=== SOFTWARE REPAIR MODE ===\n"
            "This is a repair of the previous software attempt, not a fresh rewrite. "
            "The following is the authoritative tool output from the failing rung. "
            "Do not replace it with a guessed or unrelated diagnosis.\n"
            "\nEXACT ERROR LOG:\n"
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
            "implement the kernel from the contract and intrinsics.\n"
        )
    target_contents = "\n\n".join(
        f"### {path}\n```c\n{text}\n```" for path, text in targets if text
    )
    target_list = "\n".join(f"   - {path}" for path, _ in targets)
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Measured profile:
{profile_digest(profile)}

Analysis:
{analysis}

Contract:
{contract}

Repo context:
{context_digest(context)}
{intrinsics_block}
{feedback_block}
{embench_instructions(spec)}
This is the SOFTWARE rung only. The hardware design is already fixed and
already passed SoC generation; do not propose hardware changes here.
{ROOFLINE_DECISION_RULE}
The CFU build (#ifdef USE_AGENT_CFU) must call the generated CFU intrinsic(s)
for the hot operation and derive its result from them. A branch that only calls
cfu_enable() and then computes with the scalar loop, or that calls an intrinsic
and ignores its result, is rejected: it exercises no CFU datapath and produces
no acceleration. Keep the scalar loop only in the #else (reference) branch.
Use the Roofline result to guide packing, blocking, data reuse, load/store
traffic, fences, intrinsic granularity, and residual scalar work. Preserve the
scalar fallback and report how the kernel should change achieved operations/cycle
or external bytes/cycle relative to the predicted ceiling.

Exact current contents of the software targets (base your change on these):

{target_contents}

{DELIVERY_PROTOCOLS}

Allowed target paths (use these paths verbatim):
{target_list}

Only fall back to a plan if a required target file is genuinely absent."""
