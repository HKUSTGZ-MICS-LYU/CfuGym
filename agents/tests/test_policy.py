"""Path, command, and deny-rule enforcement for the access policy."""

from __future__ import annotations

import unittest

from agents.cfu_design_agent.policy import (
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

    def test_deny_takes_precedence_over_allow(self) -> None:
        # A path can pass the stage allow list but still be denied.
        self.assertTrue(self.policy.can_read("load_context", "src/main/scala/vexiiriscv/soc/mico/MiCoSoc.scala"))
        self.assertFalse(self.policy.can_read("load_context", ".git/config"))
        self.assertTrue(self.policy.is_denied(".git/config"))
        self.assertFalse(self.policy.can_read("load_context", "agents/generated/runs/20260729-162009/report.md"))

    def test_unknown_stage_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            self.policy.check_read("not_a_stage", "sw/tests/u8_wavg_cfu_test.c")

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

    def test_generated_context_is_not_readable_by_load_context(self) -> None:
        self.assertFalse(self.policy.can_read("load_context", "agents/generated/runs/x/02_workload/workload_spec.yaml"))

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
        from agents.cfu_design_agent.tools import infer_command_id

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


if __name__ == "__main__":
    unittest.main()
