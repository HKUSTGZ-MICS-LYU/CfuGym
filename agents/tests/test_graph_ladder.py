"""Fail-fast validation ladder: per-rung repair, escalation, and hard stop.

Every scenario runs the real graph (node logic, policy, artifacts, file writes)
with only two things faked: the LLM response and the external commands. That is
enough to prove the routing contract without an API key or an sbt toolchain.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.tools import RepoToolbox, parse_command_output
from agents.tests.common import REPO_ROOT

VALID_ISA = """spec_version: "1"
identity:
  name: u8_wavg_agent
  status: draft
  opcode: custom0 (0x0B)
  function_id_width: 3
  reference_workload: uint8 weighted average
commands:
  - function_id: 1
    name: compute
    operand: rs1, rs2
    result: sum
    semantics: multiply and accumulate uint8 lanes
    response: status
datapath:
  vlen: 256
  xlen: 32
  implementation:
    result_width_bits: 32
  regDepth: 2
software:
  intrinsics_header: cfu_intrinsics.h
  scalar_fallback: true
integration:
  hardware_component: AgentCfu
  hardware_fiber: AgentCfuFiber
  bus_style: CfuBus
  single_cpu_bus_owner: true
validation:
  status: draft
  requirements: [scalar reference equals CFU output]
assumptions: {}
unknowns: []"""

HARDWARE_BLOCKS = (
    "FILE_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
    "// fake hardware\n"
    "FILE_END\n"
    "FILE_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala\n"
    "// fake fiber\n"
    "FILE_END\n"
)

TIMEOUT = -1
_REAL_RUN_COMMAND = RepoToolbox.run_command


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.used_llm = False


CONFORMANCE_STDOUT = "CFU_CONFORMANCE PASS commands=1/1 cases=1 failed=none\n"


def _soc_sim_stdout(args: list[str]) -> str:
    """Embench profile lines: reference is slower than profile/cfu."""
    joined = " ".join(args)
    if "cfu_conformance" in joined:
        # The harness-owned differential micro-test reports its own verdict line.
        return "CFU_CONFORMANCE PASS commands=1/1 cases=1 failed=none\n"
    match = re.search(r"/workspace/embench_([^/]+)/(reference|profile|cfu)/", joined)
    if match:
        name, config = match.group(1), match.group(2)
        cycles = 100 if config == "reference" else 50
        return (
            f"CFU_AGENT_PROFILE {name}={cycles}\n"
            f"CFU_AGENT_PROFILE scalar_kernel={cycles}\n"
            "CFU_AGENT_PASS\n"
        )
    return "CFU_AGENT_PROFILE scalar_kernel=100\nCFU_AGENT_PASS\n"


def _scripted_oracle(user: str) -> str:
    """Oracle covering whatever commands the conformance prompt listed.

    The prompt prints one table row per command, so the fake oracle follows the ISA
    through extension rounds instead of hard-coding command names.
    """
    lines = ["commands:"]
    section = user.split("Commands and what each one needs:", 1)[-1].split("\n\n", 1)[0]
    for name, rest in re.findall(r"^- ([^\s:]+): (.*)$", section, re.MULTILINE):
        arguments = re.search(r"arguments: (.*)$", rest)
        count = (
            0
            if not arguments or arguments.group(1).strip() == "no arguments"
            else len(arguments.group(1).split(","))
        )
        lines.append(f"  - name: {name}")
        if "setup required" in rest:
            lines.append("    setup: |")
            lines.append("      /* the scripted micro-test does not seed vector slots */")
        lines.append('    verify: "1"' if "no CPU result" in rest else '    reference: "0"')
        lines.append("    cases:")
        lines.append(f"      - args: [{', '.join(['0'] * count)}]")
    return "\n".join(lines) + "\n"


class _ScriptedLlm:
    """Deterministic responses; records every prompt for assertions."""
    available = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        self.prompts.append(user)
        if "Design the CFU contract" in user:
            return _FakeResult(VALID_ISA)
        if "HARDWARE CONFORMANCE rung only" in user:
            return _FakeResult(_scripted_oracle(user))
        if "This is the HARDWARE rung only" in user:
            if "HARDWARE REPAIR MODE" in user:
                return _FakeResult("REPAIR_DIAGNOSIS\n- Observed errors: FAKE\n" + HARDWARE_BLOCKS)
            return _FakeResult(HARDWARE_BLOCKS)
        if "This is the SOFTWARE rung only" in user:
            paths = re.findall(r"^   - (agents/generated/\S+)$", user, re.MULTILINE)
            body = "\n".join(
                f"FILE_BEGIN {path}\n// fake software\n"
                "cfu_enable();\n"
                "acc += cfu_compute(rs1, rs2);\n"
                "FILE_END"
                for path in paths
            )
            if "SOFTWARE REPAIR MODE" in user:
                body = "REPAIR_DIAGNOSIS\n- Observed errors: FAKE\n" + body
            return _FakeResult(body)
        return _FakeResult(fallback)

    def hardware_prompts(self) -> list[str]:
        return [p for p in self.prompts if "This is the HARDWARE rung only" in p]

    def software_prompts(self) -> list[str]:
        return [p for p in self.prompts if "This is the SOFTWARE rung only" in p]


class _NoHardwarePatchLlm(_ScriptedLlm):
    """Contract and software work, but the hardware rung never yields a patch."""

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        if "This is the HARDWARE rung only" in user:
            self.prompts.append(user)
            return _FakeResult("I could not produce a patch for this design.")
        return super().complete(system, user, fallback=fallback, response_format=response_format)


class _ReuseSoftwareLlm(_ScriptedLlm):
    """The software router keeps the kernel the first pass already wrote."""

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        if "Decide whether the software rung must run again" in user:
            from agents.cfu_design_agent.llm import LlmResult

            self.prompts.append(user)
            return LlmResult(
                '{"software_change": "not_required", '
                '"reason": "the current kernel already calls cfu_compute"}',
                used_llm=True,
            )
        return super().complete(system, user, fallback=fallback, response_format=response_format)


class _LoopingLlm(_ScriptedLlm):
    """One extra hardware round after a passing run, then stop."""

    def __init__(self) -> None:
        super().__init__()
        self.optimize_prompts: list[str] = []
        self.software_need_prompts: list[str] = []

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        from agents.cfu_design_agent.llm import LlmResult

        if "Decide whether another round is worth spending budget on" in user:
            self.optimize_prompts.append(user)
            if len(self.optimize_prompts) == 1:
                return LlmResult(
                    '{"continue": true, "target": "implement_hw", "focus": "cost", '
                    '"reason": "shrink the vector register file"}',
                    used_llm=True,
                )
            return LlmResult(
                '{"continue": false, "reason": "the datapath is small enough now"}',
                used_llm=True,
            )
        if "Decide whether the software rung must run again" in user:
            self.software_need_prompts.append(user)
            return LlmResult(
                '{"software_change": "not_required", "reason": "the kernel still drives the CFU"}',
                used_llm=True,
            )
        return super().complete(system, user, fallback=fallback, response_format=response_format)


class _ScalarFallbackLlm(_ScriptedLlm):
    """Software reverts to scalar inside the CFU build to pass verification."""

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        if "This is the SOFTWARE rung only" in user:
            self.prompts.append(user)
            paths = re.findall(r"^   - (agents/generated/\S+)$", user, re.MULTILINE)
            body = "\n".join(
                f"FILE_BEGIN {path}\n"
                "// USE_AGENT_CFU branch: enable then run scalar to pass verify\n"
                "// cfu_compute(rs1, rs2);  /* disabled on purpose */\n"
                "cfu_enable();\n"
                "for (int i = 0; i < n; ++i) acc += x[i] * w[i];\n"
                "FILE_END"
                for path in paths
            )
            if "SOFTWARE REPAIR MODE" in user:
                body = "REPAIR_DIAGNOSIS\n- Observed errors: FAKE\n" + body
            return _FakeResult(body)
        return super().complete(system, user, fallback=fallback, response_format=response_format)


# A targeted edit against a line that exists in the tracked AgentCfu.scala
# scaffold. The run workspace is seeded from that file, so this SEARCH matches.
EDIT_HARDWARE = (
    "EDIT_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
    "<<<<<<< SEARCH\n"
    "  val compute = 1\n"
    "=======\n"
    "  val compute = 1\n"
    "  val edited_by_test = true\n"
    ">>>>>>> REPLACE\n"
    "EDIT_END\n"
)


class _EditHardwareLlm(_ScriptedLlm):
    """Hardware rung delivers a targeted edit instead of a whole-file rewrite."""

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        if "This is the HARDWARE rung only" in user:
            self.prompts.append(user)
            return _FakeResult(EDIT_HARDWARE)
        return super().complete(system, user, fallback=fallback, response_format=response_format)


class _EnvelopeIsaLlm(_ScriptedLlm):
    """The ISA rung answers with the contract wrapped in a JSON envelope.

    Observed with GLM: {"analysis": ..., "roofline": {...}, "isa_contract": "<yaml>"}.
    """

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        if "This is the ISA-CONTRACT stage only" in user:
            import json

            import yaml

            self.prompts.append(user)
            return _FakeResult(json.dumps({
                "analysis": "hot loop analysis",
                "roofline": {"bottleneck": "unknown"},
                "isa_contract": yaml.safe_dump(yaml.safe_load(VALID_ISA)),
            }))
        return super().complete(system, user, fallback=fallback, response_format=response_format)


class _EnvelopeWithoutContractLlm(_ScriptedLlm):
    """The ISA rung wraps analysis but never returns a contract."""

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        if "This is the ISA-CONTRACT stage only" in user:
            import json

            self.prompts.append(user)
            return _FakeResult(json.dumps({"analysis": "x", "roofline": {"bottleneck": "u"}}))
        return super().complete(system, user, fallback=fallback, response_format=response_format)


class _ScriptedCommands:
    """Stand-in for ``RepoToolbox.run_command``.

    ``plan`` maps a command id to the return codes to return in order; a missing
    id succeeds. ``git_apply*`` always delegates to the real implementation so
    the patch path stays real.
    """

    def __init__(self, plan: dict[str, list[int]] | None = None, sim_stdout=None) -> None:
        self.plan = {key: list(value) for key, value in (plan or {}).items()}
        self.calls: list[tuple[str, str]] = []
        self.sim_stdout = sim_stdout or _soc_sim_stdout

    @property
    def ladder(self) -> list[str]:
        """Command ids run after the profiling setup, i.e. the ladder itself."""
        return [command_id for stage, command_id in self.calls if stage != "profile"]

    def handle(self, toolbox: RepoToolbox, args: list[str], **kwargs):
        command_id = str(kwargs.get("command_id") or "")
        self.calls.append((toolbox.stage, command_id))
        if command_id.startswith("git_apply"):
            return _REAL_RUN_COMMAND(toolbox, args, **kwargs)
        if command_id == "soc_sim" and any("cfu_conformance" in str(arg) for arg in args):
            # The conformance micro-test is a separate program: it always passes and it
            # must not consume the workload simulation's scripted return codes or its
            # custom stdout, which models a different binary entirely.
            stdout = CONFORMANCE_STDOUT
            return {
                "command": " ".join(args),
                "cwd": str(toolbox.workdir),
                "returncode": 0,
                "stdout_tail": stdout,
                "stderr_tail": "",
                "parsed": parse_command_output(stdout),
            }
        if toolbox.stage == "profile":
            # Profiling is setup, not a ladder rung: it always succeeds and does
            # not consume the scripted ladder plan. It still reports the profile
            # lines the real board prints, so the reference cycle count exists.
            stdout = self.sim_stdout(args) if command_id == "soc_sim" else ""
            return {
                "command": " ".join(args),
                "cwd": str(toolbox.workdir),
                "returncode": 0,
                "stdout_tail": stdout,
                "stderr_tail": "",
                "parsed": parse_command_output(stdout),
            }
        queue = self.plan.setdefault(command_id, [])
        code = queue.pop(0) if queue else 0
        if code == TIMEOUT:
            raise subprocess.TimeoutExpired(cmd=args, timeout=1)
        result = {
            "command": " ".join(args),
            "cwd": str(toolbox.workdir),
            "returncode": code,
            "stdout_tail": "",
            "stderr_tail": "" if code == 0 else "FAKE ERROR: the tool rejected the design",
            "parsed": {},
        }
        if code == 0 and command_id == "soc_generate_agent":
            soc = toolbox.workdir / toolbox.run_root / "workspace/soc"
            soc.mkdir(parents=True, exist_ok=True)
            (soc / "MiCoSoc.v").write_text("// generated\n")
        if code == 0 and command_id == "cfu_generate_agent":
            # The standalone cost rung elaborates only the CFU, so the mock must
            # drop AgentCfu.v where Yosys expects it.
            cfu = toolbox.workdir / toolbox.run_root / "11_cost/cfu_rtl"
            cfu.mkdir(parents=True, exist_ok=True)
            (cfu / "AgentCfu.v").write_text("// generated standalone cfu\n")
        if code == 0:
            for token in args:
                if token.startswith("MAIN="):
                    elf = (toolbox.workdir / "sw" / f"{token.split('=', 1)[1]}.elf").resolve()
                    elf.parent.mkdir(parents=True, exist_ok=True)
                    elf.write_text("elf")
        if command_id == "soc_sim":
            # A simulator prints its log whether or not the software passed.
            result["stdout_tail"] = self.sim_stdout(args)
            result["parsed"] = parse_command_output(result["stdout_tail"])
        return result


class LadderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"ladder-{self._testMethodName}"[:60]
        self.run_dir = REPO_ROOT / "agents/generated/runs" / self.run_id
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def _run(
        self,
        plan: dict[str, list[int]] | None = None,
        *,
        max_iters: int = 2,
        dry_run: bool = False,
        run_commands: bool = True,
        skip_yosys: bool = False,
        input_mode: str = "natural_language",
        benchmark_name: str = "",
        runner=None,
        llm=None,
        sim_stdout=None,
        **overrides,
    ) -> tuple[dict, _ScriptedCommands, _ScriptedLlm]:
        scripted = _ScriptedCommands(plan, sim_stdout=sim_stdout)
        llm = llm or _ScriptedLlm()
        state = {
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": input_mode,
            "benchmark_name": benchmark_name,
            "dry_run": dry_run,
            "run_commands": run_commands and not dry_run,
            "apply_patch": not dry_run,
            "skip_sim": False,
            "skip_yosys": skip_yosys,
            "max_iters": max_iters,
            "run_id": self.run_id,
            # Tests use scripted builds; never read or write the shared profile cache.
            "profile_cache": "off",
        }
        state.update(overrides)
        def fake_run_command(toolbox, args, **kwargs):
            return scripted.handle(toolbox, args, **kwargs)

        with mock.patch.object(RepoToolbox, "run_command", fake_run_command), mock.patch.object(
            graph, "_llm", lambda state: llm
        ):
            final = (runner or graph.run_agent_python)(state)
        return final, scripted, llm

    def test_hardware_generation_failure_repairs_hardware_then_completes(self) -> None:
        final, scripted, llm = self._run(
            {"soc_generate_agent": [1, 0], "yosys_cost": [0]},
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["repairs_used"], 1)
        self.assertEqual(final["attempts"].get("hw_build"), 1)
        self.assertEqual(
            scripted.ladder,
            ["soc_generate_agent", "soc_generate_agent", "conformance_build", "soc_sim",
             "make_profile", "soc_sim", "cfu_generate_agent", "yosys_cost"],
        )
        # The repair prompt must carry the exact tool error.
        repairs = llm.hardware_prompts()
        self.assertEqual(len(repairs), 2)
        self.assertIn("FAKE ERROR", repairs[1])
        self.assertNotIn("FAKE ERROR", repairs[0])
        # The repair must be labeled as a repair, not a fresh design.
        self.assertIn("HARDWARE REPAIR MODE", repairs[1])
        self.assertIn("REPAIR_DIAGNOSIS", repairs[1])
        # The injected feedback is persisted so the run is auditable.
        feedback_path = self.run_dir / "09_implementation/hw_repair_feedback.txt"
        self.assertTrue(feedback_path.is_file())
        self.assertIn("FAKE ERROR", feedback_path.read_text())
        # The repair artifact records that the model reasoned about the log.
        import yaml as _yaml

        changed = _yaml.safe_load(
            (self.run_dir / "09_implementation/hw_changed_files.yaml").read_text()
        )
        self.assertTrue(changed["repair"])
        self.assertTrue(changed["repair_diagnosis"])

    def test_hardware_generation_failure_without_budget_stops_immediately(self) -> None:
        final, scripted, _llm = self._run({"soc_generate_agent": [1]}, max_iters=1)
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["stopped_at"], "hw_build")
        self.assertEqual(scripted.ladder, ["soc_generate_agent"])
        self.assertNotIn("sw_build", final["gates"])
        self.assertNotIn("sim", final["gates"])
        self.assertNotIn("cost", final["gates"])
        self.assertTrue(any("no repair budget" in err for err in final["errors"]))

    def test_missing_hardware_patch_stops_before_running_the_soc_toolchain(self) -> None:
        # An implementation rung that applies nothing must not be papered over by
        # compiling the seeded scaffold: the SoC rung fails immediately instead.
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0]}, max_iters=1, llm=_NoHardwarePatchLlm()
        )
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["stopped_at"], "hw_build")
        self.assertFalse(final["gates"]["implementation_hw"]["ok"])
        self.assertNotIn("soc_generate_agent", scripted.ladder)
        self.assertTrue(
            any("did not apply a design change" in err for err in final["errors"]),
            final["errors"],
        )

    def test_wrapped_isa_contract_is_unwrapped(self) -> None:
        # GLM returned {"analysis", "roofline", "isa_contract": "<yaml>"}: the contract
        # was complete but nested. The rung must use it instead of spending every
        # repair attempt on the envelope's missing top-level keys.
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [0]}, llm=_EnvelopeIsaLlm()
        )
        self.assertTrue(final["gates"]["isa"]["ok"], final["gates"]["isa"])
        self.assertEqual(final["isa_spec"]["identity"]["name"], "u8_wavg_agent")
        # Leniency must not be silent: the history records that it happened.
        self.assertTrue(
            any("unwrapped from a model envelope" in line for line in final["iteration_history"]),
            final["iteration_history"][-3:],
        )

    def test_envelope_without_a_contract_reports_what_the_parser_saw(self) -> None:
        final, scripted, llm = self._run({"soc_generate_agent": [0]}, llm=_EnvelopeWithoutContractLlm())
        gate = final["gates"]["isa"]
        self.assertFalse(gate["ok"], gate)
        self.assertTrue(any("identity.name is required" in p for p in gate["problems"]), gate)
        # The repair prompt must say what the parser actually extracted, otherwise the
        # model only sees symptoms and re-emits the same envelope.
        self.assertTrue(any(p.startswith("note: the parser read") for p in gate["problems"]), gate)

    def test_hardware_targeted_edit_is_applied(self) -> None:
        # Protocol A (preferred): a targeted SEARCH/REPLACE edit is cheaper than
        # a whole-file rewrite and must be applied to the seeded design file.
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [0]},
            llm=_EditHardwareLlm(),
        )
        self.assertEqual(final["status"], "complete", final.get("errors"))
        design_files = final["changed_design_files"]
        self.assertTrue(
            any(path.endswith("workspace/design/AgentCfu.scala") for path in design_files),
            design_files,
        )
        edited = (self.run_dir / "workspace/design/AgentCfu.scala").read_text()
        self.assertIn("edited_by_test", edited)
        # The edit protocol is requested, and no FILE_BEGIN whole-file rewrite
        # was emitted for this rung.
        hardware_prompt_text = llm.hardware_prompts()[0]
        self.assertIn("EDIT_BEGIN", hardware_prompt_text)
        self.assertIn("Protocol A (preferred): targeted edits", hardware_prompt_text)

    def test_missing_hardware_patch_is_repaired_before_the_soc_toolchain(self) -> None:
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0]}, max_iters=2, llm=_NoHardwarePatchLlm()
        )
        # One repair attempt is allowed, but the rung still cannot produce a
        # patch, so the run stops without ever invoking sbt.
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["repairs_used"], 1)
        self.assertEqual(final["attempts"].get("hw_build"), 1)
        self.assertEqual(len(llm.hardware_prompts()), 2)
        self.assertNotIn("soc_generate_agent", scripted.ladder)

    def test_compile_failure_repairs_software_then_completes(self) -> None:
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "make_profile": [2, 0], "yosys_cost": [0]},
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["repairs_used"], 1)
        self.assertEqual(final["attempts"].get("sw_build"), 1)
        self.assertEqual(
            scripted.ladder,
            ["soc_generate_agent", "conformance_build", "soc_sim", "make_profile",
             "make_profile", "soc_sim", "cfu_generate_agent", "yosys_cost"],
        )
        repairs = llm.software_prompts()
        self.assertEqual(len(repairs), 2)
        self.assertIn("FAKE ERROR", repairs[1])

    def test_simulation_failure_repairs_software_then_hardware(self) -> None:
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "soc_sim": [1, 1, 0], "yosys_cost": [0]},
            max_iters=3,
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["repairs_used"], 2)
        self.assertEqual(final["attempts"].get("sim"), 2)
        self.assertEqual(
            scripted.ladder,
            [
                "soc_generate_agent",
                "conformance_build",
                "soc_sim",
                "make_profile",
                "soc_sim",
                "make_profile",
                "soc_sim",
                "soc_generate_agent",
                "conformance_build",
                "soc_sim",
                "make_profile",
                "soc_sim",
                "cfu_generate_agent",
                "yosys_cost",
            ],
        )
        # The escalated hardware repair must carry the simulation error log,
        # not just a generic "the build failed" instruction.
        hardware_repairs = llm.hardware_prompts()
        self.assertTrue(any("sim (simulation) errors" in p for p in hardware_repairs))
        self.assertTrue(any("FAKE ERROR" in p for p in hardware_repairs))

    def test_earlier_sw_repair_does_not_skip_the_sim_software_retry(self) -> None:
        # A compile failure already repaired software. The *first* simulation
        # failure must still retry software (not jump straight to hardware),
        # otherwise the promised sw-then-hw escalation is short-circuited.
        final, scripted, _llm = self._run(
            {
                "soc_generate_agent": [0],
                "make_profile": [2, 0, 0],
                "soc_sim": [1, 0],
                "yosys_cost": [0],
            },
            max_iters=3,
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["attempts"].get("sw_build"), 1)
        self.assertEqual(final["attempts"].get("sim"), 1)
        self.assertEqual(
            scripted.ladder,
            [
                "soc_generate_agent",
                "conformance_build",
                "soc_sim",
                "make_profile",
                "make_profile",
                "soc_sim",
                "make_profile",
                "soc_sim",
                "cfu_generate_agent",
                "yosys_cost",
            ],
        )

    def test_compile_failure_without_budget_stops_before_simulation(self) -> None:
        final, scripted, _llm = self._run(
            {"soc_generate_agent": [0], "make_profile": [2]}, max_iters=1
        )
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["stopped_at"], "sw_build")
        # Only the conformance micro-test ran; the workload software never reached a
        # simulator.
        self.assertEqual(scripted.ladder.count("soc_sim"), 1)
        self.assertNotIn("sim", final["gates"])

    def test_timed_out_check_is_a_failed_rung_not_a_crash(self) -> None:
        final, _scripted, _llm = self._run(
            {"soc_generate_agent": [TIMEOUT]}, max_iters=1
        )
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["stopped_at"], "hw_build")
        self.assertTrue(final["gates"]["hw_build"]["ok"] is False)
        timed_out = [
            r for r in final["validation_results"] if r.get("rung") == "hw_build" and r.get("timed_out")
        ]
        self.assertTrue(timed_out, final["validation_results"])
        self.assertIn("timed out", timed_out[0]["stderr_tail"])

    def test_a_hardware_repair_may_reuse_the_software(self) -> None:
        # Pass 1 writes the kernel and simulates. The cost rung then fails and
        # repairs the hardware, so the second hardware build runs with the kernel
        # already in the workspace: the agent may keep it instead of paying for
        # another software rewrite.
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [1, 0]},
            llm=_ReuseSoftwareLlm(),
        )
        self.assertEqual(final["status"], "complete")
        # The software rung ran once (the first pass) and was reused afterwards.
        self.assertEqual(len(llm.software_prompts()), 1)
        self.assertTrue(final["gates"]["implementation_sw"]["reused"])
        self.assertTrue(final["gates"]["implementation_sw"]["ok"])
        self.assertEqual(final["gates"]["hw_build"]["next"], "check_conformance")
        self.assertEqual(
            graph._post_conformance_default(final), "build_sw",
            "the reuse decision must survive the conformance rung",
        )
        self.assertEqual(final["gates"]["hw_build"]["software_decision"]["source"], "llm")
        self.assertTrue(final["gates"]["sw_build"]["software_reused"])
        self.assertTrue(final["gates"]["sw_build"]["ok"])
        # The reused kernel is still compiled and simulated twice: reuse skips the
        # rewrite, never the validation. Each round also runs one conformance sim.
        self.assertEqual(scripted.ladder.count("soc_sim"), 4)
        self.assertIn("Software reused unchanged", "\n".join(final["iteration_history"]))
        decision_path = self.run_dir / "09_implementation/software_decision.yaml"
        self.assertTrue(decision_path.is_file())
        self.assertIn("not_required", decision_path.read_text())
        router_prompts = [p for p in llm.prompts if "Decide whether the software rung must run again" in p]
        self.assertEqual(len(router_prompts), 1)
        self.assertIn("cfu_compute", router_prompts[0])

    def test_a_pristine_harness_does_not_count_as_driving_the_cfu(self) -> None:
        # The generated benchmark's own wrapper calls cfu_agent_counter_read(),
        # which reports cycles and is not datapath evidence. A build that only
        # calls it must not be treated as an accelerated kernel.
        final, _scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [0]},
            llm=_ReuseSoftwareLlm(),
        )
        self.assertEqual(final["status"], "complete")
        self.assertFalse(final["gates"]["implementation_sw"].get("reused", False))
        self.assertEqual(len(llm.software_prompts()), 1)
        # No hardware repair, so the software router was never consulted.
        self.assertEqual(
            [p for p in llm.prompts if "Decide whether the software rung must run again" in p],
            [],
        )

    def test_loop_extends_isa_and_revalidates_both_implementations(self) -> None:
        import json
        import yaml
        from agents.cfu_design_agent.llm import LlmResult

        class ExtendingLlm(_LoopingLlm):
            def complete(self, system, user, *, fallback, response_format=None):
                if "APPEND-ONLY ISA EXTENSION" in user:
                    self.prompts.append(user)
                    candidate = yaml.safe_load(VALID_ISA)
                    candidate["commands"].append({"function_id": 1, "subfunction_id": 1,
                                                   "name": "batch", "semantics": "batch updates"})
                    return LlmResult(json.dumps(candidate), used_llm=True)
                if "This is the ISA-CONTRACT stage only" in user:
                    self.prompts.append(user)
                    return LlmResult(VALID_ISA, used_llm=True)
                if "Decide whether another round is worth spending budget on" in user:
                    self.optimize_prompts.append(user)
                    return LlmResult('{"continue":true,"target":"extend_isa","reason":"batch updates"}', used_llm=True)
                result = super().complete(system, user, fallback=fallback, response_format=response_format)
                if "This is the SOFTWARE rung only" in user and "cfu_batch" in user:
                    result.text = result.text.replace("cfu_compute(rs1, rs2)", "cfu_batch(rs1, rs2)")
                return result

        for runner in ([graph.run_agent_python, graph.run_agent] if graph._LANGRAPH_AVAILABLE else [graph.run_agent_python]):
            with self.subTest(runner=runner.__name__):
                final, commands, llm = self._run(
                    {"soc_generate_agent": [0], "yosys_cost": [0]}, max_iters=2,
                    benchmark_name="crc32", input_mode="embench", llm=ExtendingLlm(),
                    loop_optimize=True, runner=runner,
                )
                self.assertEqual(final["status"], "complete", repr(final.get("gates")))
                self.assertEqual(final["isa_revision"], 1)
                self.assertEqual(len(final["isa_spec"]["commands"]), 2)
                self.assertEqual(commands.ladder.count("soc_generate_agent"), 2)
                # Each round runs one conformance sim and one workload sim.
                self.assertEqual(commands.ladder.count("soc_sim"), 4)
                self.assertEqual(len(llm.software_prompts()), 2)
                self.assertFalse(final["gates"]["sw_build"]["software_reused"])
                self.assertTrue(final["best_candidate"]["manifest"])

    def test_loop_keeps_optimizing_a_passing_run(self) -> None:
        # --loop: the run already passes, and the agent uses the leftover budget
        # for one more hardware round that improves the measured speedup.
        cfu_runs = {"n": 0}

        def sim_stdout(args: list[str]) -> str:
            joined = " ".join(args)
            if "/reference/" in joined:
                return (
                    "CFU_AGENT_PROFILE crc32=100\n"
                    "CFU_AGENT_PROFILE scalar_kernel=100\n"
                    "CFU_AGENT_PASS\n"
                )
            if "/cfu/" in joined:
                cfu_runs["n"] += 1
                cycles = 50 if cfu_runs["n"] == 1 else 40
                return (
                    f"CFU_AGENT_PROFILE crc32={cycles}\n"
                    f"CFU_AGENT_PROFILE scalar_kernel={cycles}\n"
                    "CFU_AGENT_PASS\n"
                )
            return "CFU_AGENT_PROFILE scalar_kernel=100\nCFU_AGENT_PASS\n"

        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [0]},
            benchmark_name="crc32",
            input_mode="embench",
            max_iters=3,
            sim_stdout=sim_stdout,
            llm=_LoopingLlm(),
            loop_optimize=True,
        )
        self.assertEqual(final["status"], "complete")
        # One initial ladder pass plus exactly one optimization round.
        self.assertEqual(scripted.ladder.count("soc_generate_agent"), 2)
        # Each round runs one conformance sim and one workload sim.
        self.assertEqual(scripted.ladder.count("soc_sim"), 4)
        self.assertEqual(final["repairs_used"], 1)
        self.assertEqual(final["attempts"]["optimize"], 1)
        # An optimization round is not a repair and not a software rewrite.
        # The stop recommendation gets one bounded reconsideration in explore mode.
        self.assertEqual(len(llm.optimize_prompts), 3)
        self.assertEqual(len(llm.software_prompts()), 1)
        self.assertEqual(len(llm.software_need_prompts), 1)
        hardware = llm.hardware_prompts()
        self.assertEqual(len(hardware), 2)
        self.assertIn("=== OPTIMIZATION ROUND ===", hardware[1])
        self.assertIn("reduce the synthesized CFU cost", hardware[1])
        self.assertIn("shrink the vector register file", hardware[1])
        self.assertNotIn("HARDWARE REPAIR MODE", hardware[1])
        rounds = final["optimization_rounds"]
        self.assertEqual([entry["round"] for entry in rounds], [0, 1])
        self.assertEqual(rounds[0]["target"], "baseline")
        self.assertEqual(rounds[0]["speedup_x100"], 200)
        self.assertEqual(rounds[1]["target"], "implement_hw")
        self.assertEqual(rounds[1]["speedup_x100"], 250)
        self.assertTrue(rounds[1]["improved"])
        self.assertFalse(final["optimization_plan"])
        ledger = self.run_dir / "12_review/optimization_rounds.yaml"
        self.assertTrue(ledger.is_file())
        self.assertIn("implement_hw", ledger.read_text())

    def test_loop_without_budget_stops_after_the_first_pass(self) -> None:
        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [0]},
            max_iters=1,
            llm=_LoopingLlm(),
            loop_optimize=True,
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(scripted.ladder.count("soc_generate_agent"), 1)
        self.assertEqual(final["gates"]["review"]["optimization"]["source"], "budget")
        self.assertEqual(final["gates"]["review"]["next"], "final_report")
        self.assertEqual(len(final["optimization_rounds"]), 1)
        self.assertNotIn("repairs_used", final)

    def test_skipped_ladder_is_planned_without_repairs(self) -> None:
        final, scripted, _llm = self._run(dry_run=True)
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final.get("repairs_used", 0), 0)
        for command_id in ("soc_generate_agent", "make_profile", "soc_sim"):
            self.assertNotIn(command_id, scripted.ladder)
        for rung in ("hw_build", "sw_build", "sim"):
            self.assertTrue(final["gates"][rung]["skipped"])

    def test_passing_ladder_with_skipped_cost_is_planned(self) -> None:
        final, _scripted, _llm = self._run({"soc_generate_agent": [0]}, skip_yosys=True)
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final["gates"]["sim"]["ok"], True)
        self.assertTrue(final["gates"]["cost"]["skipped"])

    def test_ladder_artifacts_are_written_per_rung(self) -> None:
        final, _scripted, _llm = self._run({"soc_generate_agent": [1]}, max_iters=1)
        self.assertEqual(final["status"], "failed")
        self.assertTrue((self.run_dir / "10_validation/hw_build.yaml").is_file())
        self.assertTrue((self.run_dir / "09_implementation/hw_plan.md").is_file())
        self.assertTrue((self.run_dir / "report.md").is_file())
        report = (self.run_dir / "report.md").read_text()
        self.assertIn("## Validation Ladder", report)
        self.assertIn("stopped at: hw_build", report)

    def test_embench_ladder_compiles_and_simulates_the_cfu_config(self) -> None:
        final, scripted, _llm = self._run(
            {"soc_generate_agent": [0], "embench_build": [0], "soc_sim": [0], "yosys_cost": [0]},
            input_mode="embench",
            benchmark_name="crc32",
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(
            scripted.ladder,
            ["soc_generate_agent", "conformance_build", "soc_sim", "embench_build", "soc_sim",
             "cfu_generate_agent", "yosys_cost"],
        )
        self.assertTrue(final["gates"]["sim"]["ok"])
        self.assertTrue(final["gates"]["sw_build"]["elf"].endswith("workspace/embench_crc32/cfu/embench_main.elf"))
        self.assertTrue(
            (self.run_dir / "workspace/embench_crc32/cfu/include/cfu_intrinsics.h").is_file()
        )

    def test_verify_failure_is_attributed_and_fed_back_to_the_repair(self) -> None:
        def sim_stdout(args):
            return (
                "CFU_AGENT_PROFILE crc32=120 calls=1\n"
                "EMBENCH_VERIFY_FAIL benchmark=crc32 verify=0 result=0\n"
                "SimFailure: Software reached the fail symbol :(\n"
            )

        final, _scripted, llm = self._run(
            {"soc_generate_agent": [0], "embench_build": [0], "soc_sim": [1]},
            max_iters=2,
            input_mode="embench",
            benchmark_name="crc32",
            sim_stdout=sim_stdout,
        )
        gate = final["gates"]["sim"]
        self.assertFalse(gate["ok"])
        self.assertIn("verify failed for crc32", gate["attribution"])
        # The attribution reaches the software repair prompt, not just the report.
        self.assertTrue(
            any("verify failed for crc32" in prompt for prompt in llm.software_prompts())
        )
        artifact = (self.run_dir / "10_validation/embench.yaml").read_text()
        self.assertIn("attribution:", artifact)

    def test_c_project_ladder_compiles_the_copied_kernel(self) -> None:
        project = "agents/examples/c_project_u8_wavg"
        final, scripted, _llm = self._run(
            {"soc_generate_agent": [0], "project_test": [0], "soc_sim": [0], "yosys_cost": [0]},
            input_mode="c_project",
            project_root=project,
            kernel_file=f"{project}/main.c",
            kernel_function="main",
            build_cmd="make TARGET=vexii_soc MARCH=rv32imc_zicsr_zifencei compile",
            run_cmd=(
                "sbt runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf "
                f"{project}/main.elf --with-rvc --with-rvm --with-rdtime"
            ),
        )
        self.assertEqual(final["status"], "complete")
        self.assertIn("soc_generate_agent", scripted.ladder)
        self.assertTrue(final["changed_sw_files"], final.get("implementation_sw_summary"))
        self.assertTrue(
            final["changed_sw_files"][0].endswith("_project_profile/project/main.c"),
            final["changed_sw_files"],
        )

    def test_unreadable_c_project_fails_cleanly_instead_of_crashing(self) -> None:
        # A kernel outside the access-policy read roots raises PolicyError.
        # Preparation must convert that into a failed run, not a traceback.
        final, _scripted, _llm = self._run(
            {},
            input_mode="c_project",
            project_root="docs",
            kernel_file="README.md",
            kernel_function="main",
        )
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final.get("stopped_at"), "prepare_benchmark")
        self.assertTrue(
            any("prepare_c_project_profile failed" in error for error in final.get("errors", [])),
            final.get("errors"),
        )

    def test_unallowlisted_project_command_is_a_failed_rung(self) -> None:
        # A c-project run_cmd the policy does not declare must become a failed
        # simulation rung (with feedback), not a PolicyError escaping the graph.
        project = "agents/examples/c_project_u8_wavg"
        final, scripted, _llm = self._run(
            {"soc_generate_agent": [0], "project_test": [0]},
            input_mode="c_project",
            project_root=project,
            kernel_file=f"{project}/main.c",
            kernel_function="main",
            build_cmd="make TARGET=vexii_soc MARCH=rv32imc_zicsr_zifencei compile",
            run_cmd="cmake --build .",
            max_iters=1,
        )
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final.get("stopped_at"), "sim")
        # Only the conformance micro-test ran; the workload software never reached a
        # simulator.
        self.assertEqual(scripted.ladder.count("soc_sim"), 1)
        sim_gate = final["gates"]["sim"]
        self.assertFalse(sim_gate["ok"])

    def test_hardware_repair_prompt_shows_the_live_workspace(self) -> None:
        # The repair prompt must show the overlay bytes the edit is applied to,
        # not the tracked scaffold captured at load_context. Otherwise a repair
        # SEARCH can never match and every retry burns the budget.
        final, _scripted, llm = self._run(
            {"soc_generate_agent": [1, 0], "yosys_cost": [0]},
        )
        self.assertEqual(final["status"], "complete")
        repair_prompts = [
            prompt for prompt in llm.hardware_prompts() if "HARDWARE REPAIR MODE" in prompt
        ]
        self.assertTrue(repair_prompts, "expected a hardware repair prompt")
        repair_prompt = repair_prompts[0]
        # Attempt 1 wrote these markers into the overlays via FILE_BEGIN.
        self.assertIn("// fake hardware", repair_prompt)
        self.assertIn("// fake fiber", repair_prompt)

    def test_scalar_fallback_in_the_cfu_build_is_rejected(self) -> None:
        final, _scripted, _llm = self._run(
            {"soc_generate_agent": [0]},
            llm=_ScalarFallbackLlm(),
            max_iters=1,
        )
        self.assertFalse(final["gates"]["implementation_sw"]["ok"])
        self.assertTrue(
            any(
                "does not call any CFU datapath intrinsic" in error
                for error in final.get("errors", [])
            ),
            final.get("errors"),
        )

    def test_cfu_build_without_meaningful_speedup_is_rejected(self) -> None:
        flat_profile = (
            "CFU_AGENT_PROFILE crc32=100\n"
            "CFU_AGENT_PROFILE scalar_kernel=100\n"
            "CFU_AGENT_PASS\n"
        )
        final, _scripted, _llm = self._run(
            {"soc_generate_agent": [0], "embench_build": [0], "soc_sim": [0], "yosys_cost": [0]},
            benchmark_name="crc32",
            input_mode="embench",
            max_iters=1,
            sim_stdout=lambda _args: flat_profile,
        )
        self.assertEqual(final.get("stopped_at"), "sim")
        self.assertFalse(final["gates"]["sim"]["ok"])
        self.assertTrue(
            any("not meaningfully faster" in error for error in final.get("errors", [])),
            final.get("errors"),
        )

    @unittest.skipUnless(graph._LANGRAPH_AVAILABLE, "langgraph is not installed")
    def test_langgraph_loop_keeps_optimizing_a_passing_run(self) -> None:
        cfu_runs = {"n": 0}

        def sim_stdout(args: list[str]) -> str:
            joined = " ".join(args)
            if "/reference/" in joined:
                return (
                    "CFU_AGENT_PROFILE crc32=100\n"
                    "CFU_AGENT_PROFILE scalar_kernel=100\n"
                    "CFU_AGENT_PASS\n"
                )
            if "/cfu/" in joined:
                cfu_runs["n"] += 1
                cycles = 50 if cfu_runs["n"] == 1 else 40
                return (
                    f"CFU_AGENT_PROFILE crc32={cycles}\n"
                    f"CFU_AGENT_PROFILE scalar_kernel={cycles}\n"
                    "CFU_AGENT_PASS\n"
                )
            return "CFU_AGENT_PROFILE scalar_kernel=100\nCFU_AGENT_PASS\n"

        final, scripted, llm = self._run(
            {"soc_generate_agent": [0], "yosys_cost": [0]},
            benchmark_name="crc32",
            input_mode="embench",
            max_iters=3,
            sim_stdout=sim_stdout,
            llm=_LoopingLlm(),
            loop_optimize=True,
            runner=graph.run_agent,
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(scripted.ladder.count("soc_generate_agent"), 2)
        self.assertEqual([entry["round"] for entry in final["optimization_rounds"]], [0, 1])
        self.assertTrue(final["optimization_rounds"][1]["improved"])

    @unittest.skipUnless(graph._LANGRAPH_AVAILABLE, "langgraph is not installed")
    def test_langgraph_ladder_repairs_and_stops(self) -> None:
        final, scripted, _llm = self._run(
            {"soc_generate_agent": [1, 0], "yosys_cost": [0]},
            runner=graph.run_agent,
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["repairs_used"], 1)
        self.assertEqual(scripted.ladder.count("soc_generate_agent"), 2)

    @unittest.skipUnless(graph._LANGRAPH_AVAILABLE, "langgraph is not installed")
    def test_langgraph_ladder_stops_without_budget(self) -> None:
        final, scripted, _llm = self._run(
            {"soc_generate_agent": [1]},
            max_iters=1,
            runner=graph.run_agent,
        )
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["stopped_at"], "hw_build")
        self.assertEqual(scripted.ladder, ["soc_generate_agent"])


class RouterTest(unittest.TestCase):
    def _state(self, **overrides) -> dict:
        state = {"dry_run": False, "repairs_used": 0, "max_iters": 2, "gates": {}}
        state.update(overrides)
        return state

    def test_routers_read_the_node_decision(self) -> None:
        state = self._state(gates={"hw_build": {"ok": False, "next": "implement_hw"}})
        self.assertEqual(graph._route_after_hw_build(state), "implement_hw")
        state = self._state(gates={"hw_build": {"ok": True, "next": "implement_sw"}})
        self.assertEqual(graph._route_after_hw_build(state), "implement_sw")
        state = self._state(gates={"sw_build": {"ok": False, "next": "review_and_route"}})
        self.assertEqual(graph._route_after_sw_build(state), "review_and_route")

    def test_routers_default_to_the_next_rung_when_skipped(self) -> None:
        state = self._state(gates={"hw_build": {"ok": False, "skipped": True}})
        self.assertEqual(graph._route_after_hw_build(state), "check_conformance")
        state = self._state(gates={"hw_conformance": {"ok": False, "skipped": True}})
        self.assertEqual(graph._route_after_hw_conformance(state), "implement_sw")
        state = self._state(gates={"sim": {"ok": False, "skipped": True}})
        self.assertEqual(graph._route_after_sim(state), "estimate_cost")

    def test_max_repairs_follows_max_iters(self) -> None:
        self.assertEqual(graph.max_repairs({"max_iters": 1}), 0)
        self.assertEqual(graph.max_repairs({"max_iters": 2}), 1)
        self.assertEqual(graph.max_repairs({"max_iters": 4}), 3)

    def test_sim_repair_escalates_to_hardware(self) -> None:
        # The first simulation failure retries software; only a second one (or
        # later) escalates to hardware. The escalation must not be fooled by an
        # unrelated repair target from an earlier rung.
        state = self._state(attempts={}, last_repair_target="implement_sw")
        self.assertEqual(graph._sim_repair_target(state), "implement_sw")
        state = self._state(attempts={"sim": 1}, last_repair_target="implement_hw")
        self.assertEqual(graph._sim_repair_target(state), "implement_hw")


if __name__ == "__main__":
    unittest.main()
