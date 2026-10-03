"""Fixed implementation-only syntax preflight and formal test-input boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent.runtime import commands, tools
from agents.cfu_design_agent.runtime.artifacts import RunArtifacts
from agents.cfu_design_agent.runtime.policy import CommandPolicy, PolicyError, load_policy
from agents.tests.common import REPO_ROOT


class ConformanceSyntaxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name)
        self.run_root = "agents/generated/runs/syntax-test"
        self.state = {
            "workdir": str(self.workdir),
            "run_root": self.run_root,
            # User/workload options cannot alter the preflight target or command.
            "benchmark_project": {"build_cmd": "make compile", "source": "wrong.c"},
            "isa_spec": {"march": "rv64gc", "mabi": "lp64"},
        }
        self.header = f"{self.run_root}/workspace/demo_benchmark/cfu_intrinsics.h"
        self.source = f"{self.run_root}/{commands.CONFORMANCE_SUBDIR}/{commands.CONFORMANCE_SOURCE}"
        self.policy = load_policy(REPO_ROOT)
        self.toolbox = tools.RepoToolbox(
            self.workdir, policy=self.policy, stage="implementation", run_root=self.run_root,
        )

    def argv(self) -> list[str]:
        return commands._conformance_syntax_command(self.state, self.header)

    def test_exact_syntax_only_command_uses_repository_relative_paths(self) -> None:
        argv = self.argv()
        self.assertEqual(argv, [
            "riscv64-unknown-elf-gcc",
            "-fsyntax-only",
            "-std=c99",
            "-Werror=implicit-function-declaration",
            "-march=rv32imc_zicsr_zifencei",
            "-mabi=ilp32",
            "-I", f"{self.run_root}/workspace/demo_benchmark",
            "-I", "sw/MiCo-Lib/targets/vexii_soc",
            self.source,
        ])
        for forbidden in ("-o", "-c", "-S", "make", "sbt", "compile", ";", "&&", "sh"):
            self.assertNotIn(forbidden, argv)
        self.assertEqual([token for token in argv if token.endswith(".c")], [self.source])
        self.assertTrue(all(not Path(token).is_absolute() for token in argv))

    def test_header_at_repository_root_and_empty_header(self) -> None:
        argv = commands._conformance_syntax_command(self.state, "cfu_intrinsics.h")
        self.assertEqual(argv[argv.index("-I") + 1], ".")
        for empty in ("", "   ", None):
            with self.subTest(header=empty), self.assertRaises(ValueError):
                commands._conformance_syntax_command(self.state, empty)

    def test_syntax_permission_is_implementation_only(self) -> None:
        self.assertEqual(tools.infer_command_id(self.argv(), self.policy), "conformance_syntax")
        self.policy.check_command("implementation", "conformance_syntax", self.argv(), self.workdir)
        self.assertTrue(self.toolbox.command_allowed(self.argv()))
        for stage in self.policy.stages:
            if stage != "implementation":
                with self.subTest(stage=stage), self.assertRaises(PolicyError):
                    self.policy.check_command(stage, "conformance_syntax", self.argv(), self.workdir)
        self.assertIn("soc_sim", self.policy.stage("conformance").commands)
        self.assertNotIn("soc_sim", self.policy.stage("implementation").commands)
        self.assertNotIn("conformance_build", self.policy.stage("implementation").commands)

    def test_strict_syntax_descriptor_wins_over_loose_compiler_descriptor(self) -> None:
        policy = replace(self.policy, commands={
            "loose_compiler": CommandPolicy(executable="riscv64-unknown-elf-gcc"),
            **self.policy.commands,
        })
        self.assertEqual(tools.infer_command_id(self.argv(), policy), "conformance_syntax")

    def test_policy_rejects_arbitrary_compilers_and_compile_link_commands(self) -> None:
        invalid = [
            ["gcc", "-fsyntax-only", "-std=c99", self.source],
            ["clang", "-fsyntax-only", "-std=c99", self.source],
            ["riscv64-unknown-elf-gcc", "-c", self.source, "-o", "test.o"],
            ["riscv64-unknown-elf-gcc", self.source, "-o", "test.elf"],
            ["sh", "-c", "riscv64-unknown-elf-gcc -fsyntax-only cfu_conformance.c"],
            ["make", "-C", "sw", "TARGET=vexii_soc", "compile"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                self.assertFalse(self.toolbox.command_allowed(argv))
                with self.assertRaises(PolicyError):
                    self.policy.check_command("implementation", "conformance_syntax", argv, self.workdir)
        for flag in self.policy.commands["conformance_syntax"].required_tokens:
            with self.subTest(missing=flag):
                self.assertFalse(self.toolbox.command_allowed([arg for arg in self.argv() if arg != flag]))
        wrong_source = ["other.c" if arg == self.source else arg for arg in self.argv()]
        self.assertFalse(self.toolbox.command_allowed(wrong_source))
        self.assertFalse(self.toolbox.command_allowed([*self.argv(), ";", "make", "compile"]))

    def test_mocked_execution_uses_root_workdir_without_compiler_or_output_files(self) -> None:
        completed = subprocess.CompletedProcess(self.argv(), 0, stdout="", stderr="")
        with mock.patch.object(tools.subprocess, "run", return_value=completed) as run:
            result = self.toolbox.run_command(self.argv(), command_id="conformance_syntax")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], self.argv())
        self.assertEqual(run.call_args.kwargs["cwd"], self.workdir)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_formal_stage_cannot_write_test_inputs_but_implementation_can(self) -> None:
        cfg = f"{self.run_root}/{commands.CONFORMANCE_SUBDIR}"
        for path in (
            self.source,
            f"{cfg}/oracle.yaml",
            f"{cfg}/versions/oracle.yaml.digest",
            f"{cfg}/versions/cfu_conformance.c.digest",
            self.header,
            f"{self.run_root}/workspace/embench_demo/cfu/include/cfu_intrinsics.h",
            f"{self.run_root}/workspace/demo_project_profile/cfu_intrinsics.h",
        ):
            with self.subTest(path=path):
                self.policy.check_write("implementation", path)
                with self.assertRaises(PolicyError):
                    self.policy.check_write("conformance", path)
        with self.assertRaises(PolicyError):
            self.policy.check_write("conformance", self.header.replace("cfu_intrinsics.h", "kernel.c"))

    def test_formal_result_artifact_versions_and_manifest_remain_writable(self) -> None:
        artifacts = RunArtifacts.create(self.workdir, self.policy, run_id="syntax-test")
        ref = artifacts.write_yaml(
            "conformance", f"{commands.CONFORMANCE_SUBDIR}/result.yaml", {"ok": True},
        )
        self.assertTrue((self.workdir / ref["path"]).is_file())
        self.assertTrue((self.workdir / ref["immutable_path"]).is_file())
        self.assertTrue((self.workdir / self.run_root / "run.yaml").is_file())
        self.assertIn("/conformance/versions/result.yaml.", ref["immutable_path"])
        self.assertFalse((self.workdir / self.source).exists())


if __name__ == "__main__":
    unittest.main()
