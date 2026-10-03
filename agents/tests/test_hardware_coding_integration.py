from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.runtime.tools import DESIGN_SOURCE_FILES, design_workspace_paths
from agents.cfu_design_agent.runtime import checks
from agents.cfu_design_agent.runtime import commands
from agents.cfu_design_agent.runtime import llm_step
from agents.cfu_design_agent.stages.implement import hw as impl_hw
from agents.tests.test_graph_gates import REPO_ROOT, _base_state


class HardwareCodingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.run_id = "gate-hw-coding-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        self.state = _base_state(
            self.run_id,
            workload_spec={"name": "x"},
            isa_spec={"datapath": {"xlen": 32, "vlen": 256, "regDepth": 8}},
            agent_tools=True, apply_patch=True, run_commands=True, dry_run=False,
        )
        for tracked, overlay in design_workspace_paths(self.state["run_root"]).items():
            target = REPO_ROOT / overlay
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((REPO_ROOT / tracked).read_text())

    def test_hardware_tool_edit_and_check_commit_without_setting_hw_build(self):
        target = design_workspace_paths(self.state["run_root"])[DESIGN_SOURCE_FILES[0]]
        original = (REPO_ROOT / target).read_text()
        search = original.splitlines()[0]
        replacement = search + "\n// coding integration marker"

        class HardwareLlm:
            available = True
            config_name = "fake-tools"
            model = "fake-model"
            base_url = ""
            api_mode = "chat"
            thinking = False
            reasoning_effort = None
            thinking_body = None

            def complete_with_tools(self, system, user, *, tools, tool_runner, **kwargs):
                names = {item["function"]["name"] for item in tools}
                assert {"edit_hardware", "check_hardware_elaboration"} <= names
                edit = tool_runner("edit_hardware", {"edits": [{
                    "path": target, "search": search, "replace": replacement,
                }]})
                check = tool_runner("check_hardware_elaboration", {})
                return LlmResult(
                    "EDIT_BEGIN\nPATH: " + target + "\nSEARCH\n" + search +
                    "REPLACE\n" + replacement + "EDIT_END",
                    used_llm=True,
                    tool_calls=(
                        {"round": 1, "name": "edit_hardware", "arguments": {}, "result": edit, "ok": True},
                        {"round": 2, "name": "check_hardware_elaboration", "arguments": {}, "result": check, "ok": True},
                    ),
                )

        check_rtl = REPO_ROOT / commands._agent_cfu_check_out_dir(self.state, 1) / "AgentCfu.v"
        check_rtl.parent.mkdir(parents=True, exist_ok=True)
        check_rtl.write_text("module AgentCfu; endmodule\n")
        with mock.patch.object(llm_step, "_llm", lambda state: HardwareLlm()), \
                mock.patch.object(checks, "_run_check", return_value={"returncode": 0, "full_log_path": "hw-check.log"}) as check:
            result = impl_hw.implement_hw(self.state)
        check.assert_called_once()
        self.assertTrue(result["gates"]["implementation_hw"]["ok"])
        self.assertTrue(result["gates"]["implementation_hw"]["candidate_hash"])
        self.assertNotIn("hw_build", result["gates"])
        self.assertIn("coding integration marker", (REPO_ROOT / target).read_text())
        first = commands.agent_cfu_check_commands(self.state, check_index=1)[0][-1]
        second = commands.agent_cfu_check_commands(self.state, check_index=2)[0][-1]
        self.assertIn("check_001/cfu_rtl", first)
        self.assertIn("check_002/cfu_rtl", second)

    def test_rejected_hardware_attempt_rolls_back_tool_edits(self):
        target = design_workspace_paths(self.state["run_root"])[DESIGN_SOURCE_FILES[0]]
        original = (REPO_ROOT / target).read_text()
        search = original.splitlines()[0]

        class HardwareLlm:
            available = True
            config_name = "fake-tools"
            model = "fake-model"
            base_url = ""
            api_mode = "chat"
            thinking = False
            reasoning_effort = None
            thinking_body = None

            def complete_with_tools(self, system, user, *, tools, tool_runner, **kwargs):
                tool_runner("edit_hardware", {"edits": [{
                    "path": target, "search": search, "replace": search + "\n// leaked",
                }]})
                # A tool-intent-only final answer has no auditable delivery.
                return LlmResult("Let me inspect another file.", used_llm=True)

        with mock.patch.object(llm_step, "_llm", lambda state: HardwareLlm()), \
                mock.patch.object(checks, "_run_check", return_value={"returncode": 0}):
            result = impl_hw.implement_hw(self.state)
        self.assertFalse(result["gates"]["implementation_hw"]["ok"])
        self.assertFalse(result["gates"]["implementation_hw"]["applied"])
        self.assertEqual((REPO_ROOT / target).read_text(), original)


if __name__ == "__main__":
    unittest.main()
