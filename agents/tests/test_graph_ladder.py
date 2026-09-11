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
from agents.cfu_design_agent.tools import RepoToolbox
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
  maclen: 32
  accWidth: 32
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


def _soc_sim_stdout(args: list[str]) -> str:
    """Embench profile lines: reference is slower than profile/cfu."""
    joined = " ".join(args)
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


class _ScriptedLlm:
    """Deterministic responses; records every prompt for assertions."""

    available = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        self.prompts.append(user)
        if "Design the CFU contract" in user:
            return _FakeResult(VALID_ISA)
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
        if toolbox.stage == "profile":
            # Profiling is setup, not a ladder rung: it always succeeds and does
            # not consume the scripted ladder plan. It still reports the profile
            # lines the real board prints, so the reference cycle count exists.
            return {
                "command": " ".join(args),
                "cwd": str(toolbox.workdir),
                "returncode": 0,
                "stdout_tail": self.sim_stdout(args) if command_id == "soc_sim" else "",
                "stderr_tail": "",
                "parsed": {},
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
        if code == 0 and command_id == "soc_sim":
            result["stdout_tail"] = self.sim_stdout(args)
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
            ["soc_generate_agent", "soc_generate_agent", "make_profile", "soc_sim", "cfu_generate_agent", "yosys_cost"],
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
            ["soc_generate_agent", "make_profile", "make_profile", "soc_sim", "cfu_generate_agent", "yosys_cost"],
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
                "make_profile",
                "soc_sim",
                "make_profile",
                "soc_sim",
                "soc_generate_agent",
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
        self.assertNotIn("soc_sim", scripted.ladder)
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
            ["soc_generate_agent", "embench_build", "soc_sim", "cfu_generate_agent", "yosys_cost"],
        )
        self.assertTrue(final["gates"]["sim"]["ok"])
        self.assertTrue(final["gates"]["sw_build"]["elf"].endswith("workspace/embench_crc32/cfu/embench_main.elf"))
        self.assertTrue(
            (self.run_dir / "workspace/embench_crc32/cfu/include/cfu_intrinsics.h").is_file()
        )

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
        self.assertNotIn("soc_sim", scripted.ladder)
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
        self.assertEqual(graph._route_after_hw_build(state), "implement_sw")
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
