"""Prompt builders for the CFU design agent."""

from __future__ import annotations

from typing import Iterable

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

Span only the code you actually change. Do NOT wrap large comment blocks, banner
comments, or long prose in a SEARCH: a single mis-transcribed character rejects
the whole response. If a comment must be part of the replaced region, copy it
byte-for-byte from the current file content shown above — never retype it from
memory or paraphrase it. If a SEARCH is rejected, do not resend the same block:
shrink it to a few unique code lines, or switch that file to Protocol B.

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


def context_digest(context: RepoContext, skip: Iterable[str] = ()) -> str:
    """Render the repo context, omitting files shown verbatim elsewhere.

    ``skip`` holds the rung's own targets: they are rendered in full in the
    "Exact current contents" block, and repeating a truncated copy here only
    wastes tokens and invites the model to copy the truncated text.
    """
    files = context.get("files", {})
    skipped = {str(path) for path in skip}
    omitted = [path for path in files if path in skipped]
    parts: list[str] = []
    guidance = visibility_guidance(str(context.get("visibility", "") or ""))
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


def software_need_prompt(
    task: str,
    spec: WorkloadSpec,
    isa_spec: dict | None,
    hardware_summary: str,
    software_summary: str,
    calls: list[dict],
    evidence: str,
) -> str:
    """Ask whether the software rung must run after the hardware was built.

    The deterministic guard has already established that the current kernel
    bytes call a CFU datapath intrinsic and that the ISA contract is unchanged,
    so both answers are defensible; the model decides whether the hardware
    change still needs a matching software change.
    """
    call_lines = "\n".join(
        f"- {item.get('file', '')}:{item.get('line', '')}: {str(item.get('text', '')).strip()}"
        for item in calls
    )
    return f"""Task:
{task}

Decide whether the software rung must run again before this hardware is compiled
and simulated. The CFU hardware has been designed and the SoC generates from it.

CFU ISA contract (the intrinsics the hardware promises):
{spec_digest(spec)}

ISA spec:
{yaml.safe_dump(isa_spec, sort_keys=False).strip() if isa_spec else "n/a"}

Hardware design summary:
{hardware_summary or "n/a"}

Software design summary so far:
{software_summary or "n/a"}

Deterministic evidence about the current software:
{evidence}
{call_lines}

Answer `required` when the software must change for this hardware to work at all:
the ISA contract added, removed or renamed a command, changed an operand/result
layout, offset, granularity or response value, and the current kernel does not
already drive the new contract.
Answer `not_required` when the existing kernel already exercises every command
this design needs with the current calling convention, so re-writing it would
only risk breaking working software.

Return exactly one JSON object and nothing else:
{{"software_change": "required" or "not_required",
  "reason": "<why the existing kernel is or is not sufficient>"}}"""


def agent_tools_block(agent_tools: bool, tool_rounds: int = 0, deliverable: str = "") -> str:
    """Read-only tool-loop description shared by every rung that enables tools."""
    if not agent_tools:
        return ""
    budget = (
        f"You have a bounded tool budget of at most {tool_rounds} tool rounds"
        if tool_rounds
        else "You have a bounded tool budget"
    )
    what = deliverable or "the final deliverable required by this rung"
    return (
        "\nYou have read-only tools: `read_file`, `list_dir`, `search_text`. "
        f"{budget}. The repo context above already contains the design targets and "
        "the workload source; read additional files only when it is insufficient. "
        "Reading is subject to the stage access policy; a denied path returns a policy "
        "error rather than content. "
        f"When the budget runs out the tools are withdrawn and your next reply must be {what}, "
        "so reserve time for it: do not spend every round reading. Never answer with only a "
        "promise to read or inspect something: either call a tool now, or emit the final "
        "deliverable from the context already shown.\n"
    )


def repair_routing_prompt(
    task: str,
    attribution: str,
    sim_log: str,
    spec: WorkloadSpec,
    profile: dict | None = None,
    isa_spec: dict | None = None,
    hardware_summary: str = "",
    software_summary: str = "",
    sim_repairs: int = 0,
) -> str:
    """Ask which side (software or hardware) should repair a failed simulation.

    Both branches are real, and the sim log alone rarely names the side, so the
    model decides from the attribution, the design summaries and the measured
    numbers instead of a fixed software-first escalation.
    """
    return f"""Task:
{task}

The SoC simulation of the CFU build failed. Choose which rung repairs it:
- `implement_sw`: the accelerated kernel that drives the CFU (packing, loop
  structure, offsets, tails, scaling, intrinsic granularity). The hardware is
  left untouched and only the software changes.
- `implement_hw`: the AgentCfu datapath/protocol (lane handling, sign/width,
  response value, TileLink/command handshake). The SoC is regenerated and the
  software is re-checked against the new hardware: it is kept when it still
  drives it, and updated as well when the new hardware needs it. Choose this when
  the defect is (or may be) in the hardware — a purely software bug is cheaper to
  fix through `implement_sw`, which does not re-elaborate the SoC.

Authoritative failure attribution:
{attribution or "n/a"}

Measured reference/CFU cycles:
{profile_digest(profile)}

CFU ISA contract (what the hardware promises):
{spec_digest(spec)}

ISA spec:
{yaml.safe_dump(isa_spec, sort_keys=False).strip() if isa_spec else "n/a"}

Hardware design summary:
{hardware_summary or "n/a"}

Software design summary:
{software_summary or "n/a"}

Simulation log tail:
```text
{sim_log[-4000:]}
```

Decide by where the defect actually is:
- Wrong result (Embench verify failed): if the kernel extracts/packs operands,
  applies the offset, handles tails, or shifts/scales wrongly, choose software;
  if the CFU datapath's signedness, width, lane order, or returned value is wrong,
  choose hardware.
- Not meaningfully faster: usually software (packing, blocking, intrinsic
  granularity, residual scalar work) unless the CFU design itself cannot reduce
  the operation count or stall time, in which case choose hardware.
- No pass/fail symbol before the timeout (hang): usually hardware (unfinished
  TileLink transaction, missing response, waiting on a fence), unless the kernel
  never issues or never terminates.
- {sim_repairs} simulation repair(s) already ran. Do not pick a side that was
  already tried unless the new evidence points there with a concrete reason.

Return exactly one JSON object and nothing else:
{{"repair_target": "implement_sw" or "implement_hw",
  "root_cause": "<one sentence naming the defect>",
  "reason": "<why this side is the one that must change>"}}"""


def workload_prompt(
    task: str,
    spec: WorkloadSpec,
    context: RepoContext,
    profile: dict | None = None,
    agent_tools: bool = False,
    tool_rounds: int = 0,
) -> str:
    tools_block = agent_tools_block(
        agent_tools, tool_rounds, "the final YAML analysis mapping"
    )
    return f"""Task:
{task}

Normalized WorkloadSpec:
{spec_digest(spec)}

Measured profile:
{profile_digest(profile)}

Repo context:
{context_digest(context)}
{tools_block}
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
    optimization: str = "",
    agent_tools: bool = False,
    tool_rounds: int = 0,
) -> str:
    """Prompt for the hardware rung: AgentCfu.scala / AgentCfuFiber.scala only."""
    tools_block = agent_tools_block(
        agent_tools, tool_rounds, "the final EDIT/FILE/PATCH deliverable"
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
    optimization_block = _optimization_block(optimization)
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
{context_digest(context, skip=HARDWARE_TARGETS)}
{optimization_block}{feedback_block}
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
) -> str:
    """Ask whether another round should improve speed or hardware cost (PPA).

    Only called when the run already passes every gate and the iteration budget
    still has room, so both answers are legitimate: keep optimizing, or stop with
    the current validated design.
    """
    return f"""Task:
{task}

The run already passes every gate. Measured result of the current design:
{score}

Optimization rounds so far (newest last):
{rounds or "none yet: this would be the first optimization round"}

Optimization budget left: {budget_left} round(s).

CFU ISA contract (frozen for this run — do not add, remove or rename commands):
{spec_digest(spec)}

ISA spec:
{yaml.safe_dump(isa_spec, sort_keys=False).strip() if isa_spec else "n/a"}

Hardware design summary:
{hardware_summary or "n/a"}

Software design summary:
{software_summary or "n/a"}

Decide whether another round is worth spending budget on, and where.

- `implement_hw` improves the AgentCfu datapath (lane count, width, pipelining,
  register/stream handling, command granularity, TileLink usage). It regenerates
  the SoC; the software is re-checked and only rewritten if the new hardware
  needs it.
- `implement_sw` improves the accelerated kernel (packing, blocking, loop
  structure, intrinsic granularity, residual scalar work, fences). The hardware
  is untouched and the design is recompiled and re-simulated.
Prefer `{focus}` unless the evidence clearly points at the other side.

Answer `continue: true` only when you can name a concrete change that should beat
the current numbers and keep every gate passing (the correctness oracle, the
minimum speedup floor enforced by the simulation gate, and a successful
synthesis). Do not repeat a change that
already ran and did not improve; a round that makes the result worse is a real
regression in the delivered design, not a free experiment.
Answer `continue: false` when the design is already at a reasonable point for
this budget, or when you have no concrete improvement left.

Return exactly one JSON object and nothing else:
{{"continue": true or false,
  "target": "implement_hw" or "implement_sw",
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
            if not optimization
            else "\nThe kernel is already written and passing; improve it as described above.\n"
        )
    target_contents = "\n\n".join(
        f"### {path}\n```c\n{text}\n```" for path, text in targets if text
    )
    target_list = "\n".join(f"   - {path}" for path, _ in targets)
    target_paths = [path for path, _ in targets]
    optimization_block = _optimization_block(optimization)
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
{context_digest(context, skip=target_paths)}
{intrinsics_block}
{optimization_block}{feedback_block}
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
