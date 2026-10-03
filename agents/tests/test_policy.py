"""Path, command, and deny-rule enforcement for the access policy."""

from __future__ import annotations

import unittest
from pathlib import Path

from agents.cfu_design_agent.runtime.policy import (
    PolicyError,
    load_policy,
    normalize_repo_path,
)
from agents.tests.common import REPO_ROOT, repo_files


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(REPO_ROOT)

    def test_loads_stages_and_commands(self) -> None:
        self.assertIn("load_context", self.policy.stages)
        self.assertIn("soc_sim", self.policy.commands)
        self.assertEqual(self.policy.commands["make_profile"].required_tokens, ("TARGET=vexii_soc",))

    def test_reads_are_one_shared_tree(self) -> None:
        # The contract: a run's knowledge is shared, so no read depends on the
        # stage. A per-stage `read` declaration is a policy error, not a silent
        # narrowing (see test_per_stage_read_list_is_rejected).
        self.assertTrue(self.policy.read)
        for stage in ("load_context", "analysis", "isa", "implementation", "validation"):
            self.assertFalse(hasattr(self.policy.stage(stage), "read"), stage)

    def test_per_stage_read_list_is_rejected(self) -> None:
        import tempfile

        text = (
            "version: '1'\n"
            "read:\n  - \"src/**\"\n"
            "stages:\n"
            "  analysis:\n"
            "    read: [\"src/**\"]\n"
            "    write: [\"agents/generated/runs/**\"]\n"
            "    commands: []\n"
            "commands: {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agents/specs/agent_access_policy.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(text)
            with self.assertRaises(PolicyError) as ctx:
                load_policy(Path(tmp))
            self.assertIn("reads are declared once at the top level", str(ctx.exception))

    def test_deny_takes_precedence_over_allow(self) -> None:
        # A path can pass the read tree but still be denied.
        self.assertTrue(self.policy.can_read("src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala"))
        self.assertFalse(self.policy.can_read(".git/config"))
        self.assertTrue(self.policy.is_denied(".git/config"))
        self.assertFalse(self.policy.can_read("README.md"))

    def test_deny_reaches_a_submodule_git_pointer(self) -> None:
        # `sw/**` is readable, and a submodule's `.git` is a file, not a tree.
        self.assertTrue(self.policy.is_denied("sw/.git"))
        self.assertFalse(self.policy.can_read("sw/.git"))
        self.assertTrue(self.policy.is_denied("skills/cfu-designer/scripts/__pycache__/x.pyc"))

    def test_unknown_stage_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            self.policy.check_write("not_a_stage", "sw/tests/u8_wavg_cfu_test.c")
        with self.assertRaises(PolicyError):
            self.policy.check_command("not_a_stage", "make_profile", [], REPO_ROOT)

    def test_normalize_rejects_absolute_and_escape(self) -> None:
        with self.assertRaises(PolicyError):
            normalize_repo_path(REPO_ROOT, "/etc/passwd")
        with self.assertRaises(PolicyError):
            normalize_repo_path(REPO_ROOT, "../../etc/passwd")

    def test_command_allowlist(self) -> None:
        good = ["make", "-C", "sw", "TARGET=vexii_soc", "compile"]
        self.policy.check_command("profile", "make_profile", good, REPO_ROOT)
        bad_shell = ["make", "-C", "sw", "TARGET=vexii_soc", "compile", ";", "rm", "-rf", "/"]
        with self.assertRaises(PolicyError):
            self.policy.check_command("profile", "make_profile", bad_shell, REPO_ROOT)

    def test_command_rejects_absolute_path_option(self) -> None:
        args = ["sbt", "runMain vexiiriscv.soc.mico.MiCoSocSim --load-elf /tmp/x.elf --with-rvc"]
        with self.assertRaises(PolicyError):
            self.policy.check_command("validation", "soc_sim", args, REPO_ROOT)

    def test_command_sensitive_path_denied(self) -> None:
        args = ["python3", "skills/cfu-designer/scripts/yosys_cost_report.py", "MiCoSoc.v", "--out-dir", ".env.production"]
        with self.assertRaises(PolicyError):
            self.policy.check_command("cost", "yosys_cost", args, REPO_ROOT)

    def test_run_artifacts_are_part_of_the_shared_read_tree(self) -> None:
        # The policy exposes the runs area to every stage; which *run* those
        # paths may come from is a toolbox property (RepoToolbox.run_root),
        # tested in test_tools/test_tool_hardening.
        self.assertTrue(self.policy.can_read("agents/generated/runs/x/02_workload/workload_spec.yaml"))
        self.assertTrue(self.policy.can_read("agents/generated/runs/x/07_analysis/workload_analysis.yaml"))

    def test_hardware_source_is_never_writable(self) -> None:
        for stage in ("implementation", "validation"):
            for rel in (
                "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
                "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala",
                "src/main/scala/vexiiriscv/soc/mico/MiCoSocParam.scala",
                "src/main/scala/vexiiriscv/soc/cfu/CfuLib.scala",
            ):
                self.assertFalse(self.policy.can_write(stage, rel), f"{stage}:{rel}")

    def test_design_workspace_is_the_only_hardware_write_target(self) -> None:
        self.assertTrue(
            self.policy.can_write(
                "implementation",
                "agents/generated/runs/x/workspace/design/AgentCfu.scala",
            )
        )
        self.assertTrue(
            self.policy.can_write(
                "implementation",
                "agents/generated/runs/x/workspace/design/AgentCfuFiber.scala",
            )
        )
        self.assertFalse(
            self.policy.can_write(
                "implementation",
                "agents/generated/runs/x/workspace/design/Other.scala",
            )
        )

    def test_agent_generation_command_requires_the_agent_flag(self) -> None:
        argv = [
            "sbt",
            "set Compile/unmanagedSourceDirectories += (Compile/baseDirectory).value / "
            '"agents/generated/runs/x/workspace/design"',
            "set Compile/unmanagedSources := AgentCfuSourceFilter.sources("
            "(Compile/unmanagedSourceDirectories).value)",
            'set run / baseDirectory := (Compile/baseDirectory).value / '
            '"agents/generated/runs/x/workspace/soc"',
            "runMain vexiiriscv.soc.mico.MiCoSocGen --with-rvc --mico-agent-cfu",
        ]
        self.policy.check_command("validation", "soc_generate_agent", argv, REPO_ROOT)
        without_flag = [argv[0], *argv[1:-1], "runMain vexiiriscv.soc.mico.MiCoSocGen"]
        with self.assertRaises(PolicyError):
            self.policy.check_command("validation", "soc_generate_agent", without_flag, REPO_ROOT)

    def test_agent_generation_id_is_selected_over_the_loose_one(self) -> None:
        from agents.cfu_design_agent.runtime.tools import infer_command_id

        overlay = [
            "sbt",
            "set Compile/unmanagedSourceDirectories += (Compile/baseDirectory).value / "
            '"agents/generated/runs/x/workspace/design"',
            "set run / baseDirectory := (Compile/baseDirectory).value / "
            '"agents/generated/runs/x/workspace/soc"',
            "runMain vexiiriscv.soc.mico.MiCoSocGen --with-rvc --mico-agent-cfu",
        ]
        self.assertEqual(infer_command_id(overlay, self.policy), "soc_generate_agent")
        loose = [overlay[0], *overlay[1:-1], "runMain vexiiriscv.soc.mico.MiCoSocGen"]
        self.assertEqual(infer_command_id(loose, self.policy), "soc_generate")

    def test_positional_paths_are_checked(self) -> None:
        base = ["python3", "skills/cfu-designer/scripts/yosys_cost_report.py"]
        for bad in ("/etc/passwd", ".env.production", "../../etc/passwd"):
            with self.assertRaises(PolicyError, msg=bad):
                self.policy.check_command("cost", "yosys_cost", [*base, bad], REPO_ROOT)
        self.policy.check_command(
            "cost",
            "yosys_cost",
            [*base, "agents/generated/runs/x/workspace/soc/MiCoSoc.v", "--top", "AgentCfu"],
            REPO_ROOT,
        )

    def test_agent_generation_is_the_only_generator_allowed(self) -> None:
        with self.assertRaises(PolicyError):
            self.policy.check_command(
                "validation", "soc_generate", ["sbt", "runMain vexiiriscv.soc.mico.MiCoSocGen"], REPO_ROOT
            )

    def test_standalone_cfu_cost_generation_is_declared_and_stage_scoped(self) -> None:
        # The cost rung synthesizes the CFU alone, so its generator must be a
        # declared command, allowed in `cost`, require --out-dir, and stay out of
        # the SoC/validation stages.
        self.assertIn("cfu_generate_agent", self.policy.commands)
        self.assertIn("cfu_generate_agent", self.policy.stage("cost").commands)
        for stage in ("validation", "implementation", "isa"):
            self.assertNotIn("cfu_generate_agent", self.policy.stage(stage).commands)

        argv = [
            "sbt",
            "set Compile/unmanagedSourceDirectories += (Compile/baseDirectory).value / "
            '"agents/generated/runs/x/workspace/design"',
            "set Compile/unmanagedSources := "
            "AgentCfuSourceFilter.sources((Compile/unmanagedSourceDirectories).value)",
            "runMain vexiiriscv.soc.mico.GenAgentCfu --out-dir "
            "agents/generated/runs/x/11_cost/cfu_rtl --agent-cfu-vlen 64",
        ]
        self.policy.check_command("cost", "cfu_generate_agent", argv, REPO_ROOT)

        # --out-dir is mandatory so RTL cannot land in an arbitrary directory...
        without_out = [argv[0], *argv[1:-1], "runMain vexiiriscv.soc.mico.GenAgentCfu"]
        with self.assertRaises(PolicyError):
            self.policy.check_command("cost", "cfu_generate_agent", without_out, REPO_ROOT)
        # ...and the command is not reachable from another stage.
        with self.assertRaises(PolicyError):
            self.policy.check_command("validation", "cfu_generate_agent", argv, REPO_ROOT)

    def test_hardware_fast_check_has_distinct_narrow_authority(self) -> None:
        self.assertIn("cfu_check_agent", self.policy.stage("implementation").commands)
        for stage in ("validation", "cost", "isa"):
            self.assertNotIn("cfu_check_agent", self.policy.stage(stage).commands)
        argv = [
            "sbt",
            "set Compile/unmanagedSourceDirectories += (Compile/baseDirectory).value / "
            '"agents/generated/runs/x/workspace/design"',
            "set Compile/unmanagedSources := "
            "AgentCfuSourceFilter.sources((Compile/unmanagedSourceDirectories).value)",
            "runMain vexiiriscv.soc.mico.GenAgentCfu --out-dir "
            "agents/generated/runs/x/workspace/checks/hw/attempt_000/check_001/cfu_rtl",
        ]
        self.policy.check_command("implementation", "cfu_check_agent", argv, REPO_ROOT)
        wrong = [*argv[:-1], argv[-1].replace("workspace/checks/hw/attempt_000", "11_cost")]
        with self.assertRaises(PolicyError):
            self.policy.check_command("implementation", "cfu_check_agent", wrong, REPO_ROOT)


class VisibilityPolicyTest(unittest.TestCase):
    """The optional visibility tiers gate reads only."""

    SKILL = "skills/cfu-designer/SKILL.md"
    VPU = "src/main/scala/vexiiriscv/soc/mico/VpuCfu.scala"
    AGENT_CFU = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
    SPINAL = "ext/SpinalHDL/core/src/main/scala/spinal/core/Bits.scala"

    def test_default_level_is_standard(self) -> None:
        self.assertEqual(load_policy(REPO_ROOT).visibility, "standard")

    def test_basic_hides_skills_siblings_and_spinalhdl(self) -> None:
        policy = load_policy(REPO_ROOT, visibility="basic")
        for path in (
            self.SKILL,
            self.VPU,
            "src/main/scala/vexiiriscv/soc/mico/BitNetCfu.scala",
            self.SPINAL,
        ):
            self.assertFalse(policy.can_read(path), path)
        for path in (
            self.AGENT_CFU,
            "src/main/scala/vexiiriscv/soc/cfu/CfuLib.scala",
            "agents/specs/cfu_isa_template.yaml",
            "benchmarks/embench-iot/src/matmult-int/matmult-int.c",
        ):
            self.assertTrue(policy.can_read(path), path)

    def test_standard_adds_skills_and_siblings_but_not_spinalhdl(self) -> None:
        policy = load_policy(REPO_ROOT, visibility="standard")
        self.assertTrue(policy.can_read(self.SKILL))
        self.assertTrue(policy.can_read(self.VPU))
        self.assertFalse(policy.can_read(self.SPINAL))

    def test_extended_exposes_spinalhdl_to_the_whole_flow(self) -> None:
        policy = load_policy(REPO_ROOT, visibility="extended")
        self.assertTrue(policy.can_read(self.SPINAL))
        # Tiers are flow-wide, so the ancestor directory becomes navigable too.
        self.assertTrue(policy.may_contain_readable("ext"))
        self.assertTrue(policy.may_contain_readable("ext/SpinalHDL"))

    def test_gated_subtree_is_not_walked_at_a_lower_level(self) -> None:
        # A directory that roots a gated subtree is itself invisible, so the
        # tools prune it instead of descending into it.
        standard = load_policy(REPO_ROOT, visibility="standard")
        self.assertFalse(standard.may_contain_readable("ext/SpinalHDL"))
        self.assertEqual(
            standard.subtree_exclusion("ext/SpinalHDL"),
            "requires visibility level 'extended' (current: standard)",
        )
        self.assertFalse(standard.may_contain_readable("ext"))
        extended = load_policy(REPO_ROOT, visibility="extended")
        self.assertTrue(extended.may_contain_readable("ext/SpinalHDL"))
        self.assertEqual(extended.subtree_exclusion("ext/SpinalHDL"), "")

    def test_a_directory_is_visible_when_a_readable_file_may_live_below_it(self) -> None:
        policy = load_policy(REPO_ROOT)
        # Directories are never themselves in the read tree...
        self.assertFalse(policy.can_read("src/main/scala/vexiiriscv/soc/mico"))
        # ...but navigation must see them.
        for path in ("src", "src/main/scala/vexiiriscv/soc/mico", "agents", "skills"):
            self.assertTrue(policy.may_contain_readable(path), path)
            self.assertEqual(policy.subtree_exclusion(path), "", path)
        for path in (".git", "target", "docs", "benchmark_results"):
            self.assertFalse(policy.may_contain_readable(path), path)
        self.assertTrue(policy.subtree_exclusion("target"))

    def test_check_read_names_the_blocking_level(self) -> None:
        policy = load_policy(REPO_ROOT, visibility="basic")
        with self.assertRaises(PolicyError) as ctx:
            policy.check_read(self.SKILL)
        self.assertIn("requires visibility level 'standard'", str(ctx.exception))
        self.assertIn("current: basic", str(ctx.exception))

    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            load_policy(REPO_ROOT, visibility="nope")

    def test_writes_are_never_gated(self) -> None:
        for level in ("basic", "standard", "extended"):
            policy = load_policy(REPO_ROOT, visibility=level)
            self.assertTrue(
                policy.can_write(
                    "implementation",
                    "agents/generated/runs/x/workspace/design/AgentCfu.scala",
                ),
                level,
            )

    def test_workload_source_shown_in_context_stays_tool_readable(self) -> None:
        # The workload source is loaded into the prompt at load_context, so the
        # shared tree must expose the same file to every later stage.
        workload = "benchmarks/embench-iot/src/depthconv/depthconv.c"
        policy = load_policy(REPO_ROOT, visibility="extended")
        self.assertTrue(policy.can_read(workload))

    def test_asm_is_readable_but_build_noise_stays_denied(self) -> None:
        policy = load_policy(REPO_ROOT, visibility="extended")
        # The profile build's disassembly is useful analysis input.
        for path in (
            "agents/generated/runs/x/workspace/embench_depthconv/reference/embench_main.asm",
            "agents/generated/runs/x/workspace/embench_depthconv/profile/embench_main.asm",
            "sw/tests/latency_lane_cfu_test.asm",
        ):
            self.assertTrue(policy.can_read(path), path)
        # .asm is not globally denied, but build noise still is.
        self.assertFalse(policy.is_denied("agents/generated/runs/x/workspace/a.asm"))
        self.assertTrue(policy.is_denied("sw/build/foo.asm"))
        self.assertFalse(policy.can_read("sw/build/foo.asm"))
        self.assertTrue(policy.can_read("agents/generated/runs/x/07_analysis/ir/kernel.ll"))


if __name__ == "__main__":
    unittest.main()
