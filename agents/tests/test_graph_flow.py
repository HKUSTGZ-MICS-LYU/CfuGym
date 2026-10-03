"""Prompt visibility, overlay command construction, and test hermeticity."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.prompts import context_digest
from agents.cfu_design_agent import ir
from agents.cfu_design_agent.runtime import llm_step
from agents.tests.common import REPO_ROOT, offline_llm
from agents.cfu_design_agent import ir as ir_mod
from agents.cfu_design_agent.runtime import commands as commands_mod
from agents.cfu_design_agent.stages import feedback as feedback_mod
from agents.cfu_design_agent.stages import profile


class ContextDigestTest(unittest.TestCase):
    def test_context_digest_omits_sensitive_and_other_runs(self) -> None:
        context = {
            "files": {
                "skills/cfu-designer/SKILL.md": "allowed",
                ".git/config": "secret",
                "agents/generated/runs/20260729-162009/report.md": "other run",
            },
            "git_status": "",
            "cfu_entrypoints": [],
            "software_tests": [],
        }
        digest = context_digest(context)
        self.assertIn("allowed", digest)
        self.assertNotIn(".git/config", digest)
        self.assertNotIn("other run", digest)


class OverlayCommandTest(unittest.TestCase):
    """The overlay command swaps sources without touching the repository."""

    def test_overlay_command_swaps_scaffolds_and_scopes_output(self) -> None:
        from agents.cfu_design_agent.runtime.commands import agent_cfu_overlay_commands
        from agents.cfu_design_agent.runtime.policy import load_policy

        run_root = "agents/generated/runs/overlay-test"
        state = {"run_root": run_root, "workdir": str(REPO_ROOT)}
        commands = agent_cfu_overlay_commands(
            state, {"datapath": {"vlen": 128, "xlen": 32, "regDepth": 4}}
        )
        self.assertEqual(len(commands), 1)
        argv = commands[0]
        self.assertEqual(argv[0], "sbt")
        self.assertTrue(any(f"{run_root}/workspace/design" in token for token in argv))
        self.assertTrue(any("AgentCfuSourceFilter.sources" in token for token in argv))
        self.assertTrue(any(f"{run_root}/workspace/soc" in token for token in argv))
        run_main = argv[-1]
        self.assertIn("runMain vexiiriscv.soc.mico.MiCoSocGen", run_main)
        self.assertIn("--mico-agent-cfu", run_main)
        self.assertIn("--agent-cfu-vlen 128", run_main)
        self.assertIn("--agent-cfu-reg-depth 4", run_main)
        # No absolute path or shell metacharacter may appear in the argv.
        for token in argv:
            self.assertFalse(token.startswith("/"), token)
            for bad in (";", "&", "|", "`", "$("):
                self.assertNotIn(bad, token, token)
        load_policy(REPO_ROOT).check_command(
            "validation", "soc_generate_agent", argv, REPO_ROOT
        )

class TestHermeticityTest(unittest.TestCase):
    """Tests must never dial a real model endpoint (operator tokens)."""

    def test_offline_client_is_unavailable_and_patches_the_graph_factory(self) -> None:
        from agents.cfu_design_agent import graph
        from agents.tests.common import OfflineLlm

        self.assertFalse(OfflineLlm().available)
        client = OfflineLlm()
        result = client.complete("s", "u", fallback="FALLBACK")
        self.assertFalse(result.used_llm)
        self.assertEqual(result.text, "FALLBACK")
        with offline_llm() as patched:
            self.assertIs(llm_step._llm({}), patched)
            self.assertFalse(llm_step._llm({}).available)

    @unittest.skipIf(
        __import__("os").environ.get("CFU_AGENT_TESTS_ALLOW_NETWORK", "").lower()
        in {"1", "true", "yes", "on"},
        "network explicitly allowed for this run",
    )
    def test_outbound_network_is_blocked(self) -> None:
        import socket

        with self.assertRaises(RuntimeError):
            socket.create_connection(("example.com", 80), timeout=1)


class RepairFeedbackTest(unittest.TestCase):
    """A hardware repair must see every failing rung's exact error log."""

    def test_hardware_repair_feedback_labels_each_source(self) -> None:
        from agents.cfu_design_agent.stages.feedback import hardware_repair_feedback

        state = {
            "validation_results": [
                {
                    "rung": "hw_build",
                    "returncode": 1,
                    "command": "sbt ... MiCoSocGen",
                    "stdout_tail": "BUILD_ERROR_MARKER",
                    "stderr_tail": "",
                },
                {
                    "rung": "sim",
                    "returncode": 1,
                    "command": "sbt ... MiCoSocSim",
                    "stdout_tail": "SIM_ERROR_MARKER",
                    "stderr_tail": "",
                },
                {
                    "rung": "hw_build",
                    "returncode": 0,
                    "command": "sbt ... MiCoSocGen",
                    "stdout_tail": "IGNORED_SUCCESS",
                    "stderr_tail": "",
                },
            ],
            "cost_results": [
                {
                    "returncode": 1,
                    "command": "python3 yosys_cost_report.py",
                    "stdout_tail": "COST_ERROR_MARKER",
                    "stderr_tail": "",
                }
            ],
            "gates": {"implementation_hw": {"errors": ["no usable patch"]}},
        }
        feedback = hardware_repair_feedback(state)
        self.assertNotIn("hw_build (SoC generation) errors", feedback)
        self.assertIn("sim (simulation) errors", feedback)
        self.assertIn("cost (Yosys) errors", feedback)
        self.assertIn("implementation_hw rejection", feedback)
        self.assertNotIn("BUILD_ERROR_MARKER", feedback)
        self.assertIn("SIM_ERROR_MARKER", feedback)
        self.assertIn("COST_ERROR_MARKER", feedback)
        self.assertIn("no usable patch", feedback)
        self.assertNotIn("IGNORED_SUCCESS", feedback)

    def test_hardware_repair_feedback_is_empty_without_failures(self) -> None:
        from agents.cfu_design_agent.stages.feedback import hardware_repair_feedback

        self.assertEqual(hardware_repair_feedback({}), "")


class IrFeatureXmlTest(unittest.TestCase):
    """collect_ir_evidence writes feature XML best-effort beside the .ll files."""

    def setUp(self) -> None:
        import sys
        import types
        from unittest import mock

        self.mock = mock
        self.run_id = "flow-irfeat-000000-00"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        self.fake_module = types.ModuleType("agents.cfu_design_agent.ir.features")
        self.extract_calls: list[dict] = []
        self.extract_error: Exception | None = None

        def fake_extract(text, **kwargs):
            self.extract_calls.append({**kwargs, "text_len": len(text)})
            if self.extract_error is not None:
                raise self.extract_error
            return {"schema_version": "1", "source_file": kwargs.get("source_file", "")}, {}

        self.fake_module.extract_ir_features = fake_extract
        self.fake_module.render_features_xml = (
            lambda features: '<ir_features schema_version="1"/>\n'
        )
        self.module_patch = self.mock.patch.dict(
            sys.modules, {"agents.cfu_design_agent.ir.features": self.fake_module}
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def _state(self, **overrides) -> dict:
        state = {
            "workdir": str(REPO_ROOT),
            "run_id": self.run_id,
            "run_root": f"agents/generated/runs/{self.run_id}",
            "benchmark_project": {"kind": "embench_benchmark", "benchmark": "crc32"},
            "ir_features": "auto",
            "run_commands": True,
            "dry_run": False,
            "workload_spec": {
                "hot_loop": {"file": "crc_32.c", "function": "crc32pseudo", "line": 158},
            },
            "gates": {},
        }
        state.update(overrides)
        return state

    def test_feature_xml_written_with_hotspot_focus(self) -> None:
        from agents.cfu_design_agent.stages.profile import collect_ir_evidence

        summary = {"schema_version": "1", "status": "ok", "reason": "",
                   "toolchain": {"target_match": True}}
        files = {"07_analysis/ir/crc_32.ll": "define void @f() {\n  ret void\n}\n",
                 "07_analysis/ir/other.ll": "define void @g() {\n  ret void\n}\n"}
        with self.mock.patch(
            "agents.cfu_design_agent.ir.export_embench_ir",
            return_value=(summary, files),
        ):
            result = collect_ir_evidence(self._state())
        ir = result["ir_result"]
        self.assertEqual(ir["feature_extraction"], {"status": "ok"})
        self.assertEqual(len(ir["feature_files"]), 2)
        for path in ir["feature_files"]:
            self.assertTrue((REPO_ROOT / path).exists(), path)
        self.assertTrue((self.run_dir / "07_analysis/ir/crc_32.features.xml").exists())
        self.assertEqual(len(self.extract_calls), 2)
        focused = [c for c in self.extract_calls if c["function"] == "crc32pseudo"]
        self.assertEqual(len(focused), 1)
        self.assertEqual(focused[0]["hotspot_line"], 158)
        self.assertEqual(focused[0]["source_file"], "crc_32.c")
        unfocused = [c for c in self.extract_calls if c["function"] == ""]
        self.assertEqual(len(unfocused), 1)
        self.assertIsNone(unfocused[0]["hotspot_line"])
        self.assertEqual(unfocused[0]["source_file"], "other.c")

    def test_feature_extraction_failure_does_not_block_ir_evidence(self) -> None:
        from agents.cfu_design_agent.stages.profile import collect_ir_evidence

        self.extract_error = RuntimeError("opt exploded")
        summary = {"schema_version": "1", "status": "ok", "reason": "",
                   "toolchain": {"target_match": True}}
        files = {"07_analysis/ir/crc_32.ll": "define void @f() {\n  ret void\n}\n"}
        with self.mock.patch(
            "agents.cfu_design_agent.ir.export_embench_ir",
            return_value=(summary, files),
        ):
            result = collect_ir_evidence(self._state())
        ir = result["ir_result"]
        self.assertEqual(ir["status"], "ok")
        self.assertEqual(len(ir["ir_files"]), 1)
        self.assertEqual(ir["feature_extraction"]["status"], "unavailable")
        self.assertIn("opt exploded", ir["feature_extraction"]["errors"][0])
        self.assertTrue(result["gates"]["ir"]["ok"])


if __name__ == "__main__":
    unittest.main()
