"""End-to-end visibility tiers: context loading, tools, prompts, CLI."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import unittest
from unittest import mock

from agents.cfu_design_agent import cli, graph
from agents.cfu_design_agent.runtime.agent_tools import ReadOnlyToolset
from agents.cfu_design_agent.cli import build_parser
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.prompts import context_digest, hardware_prompt
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT
from agents.cfu_design_agent.stages import context
from agents.cfu_design_agent.stages.review import report as report_mod

SKILL = "skills/cfu-designer/SKILL.md"
AGENT_CFU = "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"
SPINAL = "ext/SpinalHDL/core/src/main/scala/spinal/core/Bits.scala"


class LoadContextVisibilityTest(unittest.TestCase):
    def _load(self, level: str) -> tuple[dict, object]:
        run_id = f"vis-{level}"
        run_dir = REPO_ROOT / "agents/generated/runs" / run_id
        shutil.rmtree(run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        state = {
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": True,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": False,
            "skip_yosys": False,
            "max_iters": 1,
            "run_id": run_id,
            "visibility": level,
        }
        return context.load_context(state), run_dir

    def test_basic_withholds_skills_without_failing_the_gate(self) -> None:
        final, run_dir = self._load("basic")
        self.assertNotIn(SKILL, final["context"]["files"])
        self.assertEqual(final["context"]["visibility"], "basic")
        self.assertEqual(final["visibility"], "basic")
        # Withholding is expected at this tier, so the gate must still pass.
        self.assertTrue(final["gates"]["load_context"]["ok"])
        withheld = final["context"]["withheld"]
        self.assertTrue(any(SKILL in item for item in withheld), withheld)
        manifest = (run_dir / "01_context/manifest.yaml").read_text()
        self.assertIn("visibility: basic", manifest)
        self.assertIn("withheld_files", manifest)
        # The tool budget is part of run.yaml so a failed attempt is reproducible.
        run_yaml = (run_dir / "run.yaml").read_text()
        self.assertIn("tool_policy:", run_yaml)
        self.assertIn("max_calls_per_round:", run_yaml)

    def test_standard_loads_skills_and_withholds_nothing(self) -> None:
        final, _ = self._load("standard")
        self.assertIn(SKILL, final["context"]["files"])
        self.assertEqual(final["context"]["visibility"], "standard")
        self.assertEqual(final["context"]["withheld"], [])


class VisibilityToolTest(unittest.TestCase):
    def _toolset(self, level: str, stage: str = "implementation") -> ReadOnlyToolset:
        policy = load_policy(REPO_ROOT, visibility=level)
        return ReadOnlyToolset(RepoToolbox(REPO_ROOT, dry_run=True, policy=policy, stage=stage))

    def test_basic_denies_skill_read_but_allows_the_scaffold(self) -> None:
        # SKILL.md is inside the shared read tree, so the only reason to deny it
        # is the visibility tier.
        tools = self._toolset("basic", stage="load_context")
        denied = tools.run("read_file", {"path": SKILL})
        self.assertIn("denied by access policy", denied)
        self.assertIn("requires visibility level 'standard'", denied)
        self.assertIn("class AgentCfu", tools.run("read_file", {"path": AGENT_CFU}))

    def test_standard_allows_skill_for_any_stage(self) -> None:
        # Visibility is flow-wide: the tier does not depend on the stage.
        for stage in ("load_context", "analysis", "isa", "implementation"):
            tools = self._toolset("standard", stage=stage)
            self.assertNotIn(
                "denied by access policy", tools.run("read_file", {"path": SKILL}), stage
            )

    def test_standard_denies_spinalhdl(self) -> None:
        tools = self._toolset("standard")
        self.assertIn("denied by access policy", tools.run("read_file", {"path": SPINAL}))

    def test_standard_opens_vexii_sources_but_hides_example_cfus(self) -> None:
        standard = self._toolset("standard")
        readable = standard.run(
            "read_file", {"path": "src/main/scala/vexiiriscv/execute/MiCoPlugin.scala"}
        )
        self.assertNotIn("denied by access policy", readable)
        for path in (
            "src/main/scala/vexiiriscv/soc/mico/VpuCfu.scala",
            "src/main/scala/vexiiriscv/soc/mico/BitNetCfu.scala",
        ):
            self.assertIn("denied by access policy", standard.run("read_file", {"path": path}))
        listing = standard.run("list_dir", {"path": "src/main/scala/vexiiriscv/soc/mico"})
        self.assertNotIn("VpuCfu.scala", listing)
        self.assertNotIn("BitNetCfu.scala", listing)

    def test_list_dir_hides_a_gated_subtree_but_says_why(self) -> None:
        tools = self._toolset("basic", stage="load_context")
        listing = tools.run("list_dir", {"path": "skills/cfu-designer"})
        self.assertIn("requires visibility level 'standard'", listing)
        self.assertNotIn("SKILL.md", listing)
        # The only readable content under `skills` at this tier is the gated
        # subtree, so navigating there explains the tier instead of saying the
        # path is unknown.
        self.assertIn(
            "requires visibility level 'standard'", tools.run("list_dir", {"path": "skills"})
        )
        # ...and at `standard` the subtree opens.
        self.assertIn("SKILL.md", self._toolset("standard").run("list_dir", {"path": "skills/cfu-designer"}))

    def test_extended_exposes_spinalhdl_to_every_stage(self) -> None:
        for stage in ("analysis", "isa", "implementation"):
            tools = self._toolset("extended", stage=stage)
            text = tools.run("read_file", {"path": SPINAL})
            if (REPO_ROOT / SPINAL).is_file():
                self.assertNotIn("denied by access policy", text, stage)
            self.assertIn("ext/SpinalHDL/", tools.run("list_dir", {"path": "ext"}), stage)


class VisibilityPromptTest(unittest.TestCase):
    def test_context_digest_states_the_level(self) -> None:
        basic = context_digest({"files": {}, "visibility": "basic"})
        self.assertIn("## Visibility: basic", basic)
        self.assertIn("may NOT read the cfu-designer skills", basic)
        standard = context_digest({"files": {}, "visibility": "standard"})
        self.assertIn("src/main/scala/vexiiriscv/**", standard)
        self.assertIn("denied at every tier", standard)
        extended = context_digest({"files": {}, "visibility": "extended"})
        self.assertIn("## Visibility: extended", extended)
        self.assertIn("ext/SpinalHDL", extended)
        self.assertIn("example CFUs remain denied", extended)

    def test_unknown_level_adds_no_guidance(self) -> None:
        self.assertNotIn("## Visibility", context_digest({"files": {}}))

    def test_hardware_prompt_tells_extended_agents_to_consult_spinalhdl(self) -> None:
        prompt = hardware_prompt(
            "task", {}, "", "", {"files": {}, "visibility": "extended"}, agent_tools=True
        )
        self.assertIn("ext/SpinalHDL", prompt)
        self.assertIn("Before writing hardware", prompt)

    def test_hardware_prompt_notes_disabled_tools_at_extended(self) -> None:
        prompt = hardware_prompt(
            "task", {}, "", "", {"files": {}, "visibility": "extended"}, agent_tools=False
        )
        self.assertIn("tool loop is disabled", prompt)


class VisibilityCliTest(unittest.TestCase):
    def test_visibility_flag_is_parsed(self) -> None:
        args = build_parser().parse_args(["--task", "x", "--visibility", "basic"])
        self.assertEqual(args.visibility, "basic")

    def test_default_visibility_is_standard(self) -> None:
        with mock.patch.dict(os.environ, {"CFU_AGENT_VISIBILITY": "standard"}):
            args = build_parser().parse_args(["--task", "x"])
        self.assertEqual(args.visibility, "standard")

    def test_invalid_visibility_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--task", "x", "--visibility", "nope"])

    def test_main_threads_the_level_into_state(self) -> None:
        captured: dict = {}

        def fake_run_agent(state, **kwargs):
            captured.update(state)
            return {"status": "planned", "final_report_path": "report.md", "visibility": "basic"}

        with (
            mock.patch.object(cli, "run_agent", fake_run_agent),
            mock.patch.object(cli, "export_graph_visualization", lambda *a, **k: {}),
            mock.patch.object(cli, "enable_progress", lambda *a, **k: None),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc = cli.main(["--task", "t", "--visibility", "basic", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured.get("visibility"), "basic")


class VisibilityReportTest(unittest.TestCase):
    def test_report_lists_the_level_and_what_was_withheld(self) -> None:
        report = report_mod.render_report(
            {
                "task": "t",
                "visibility": "basic",
                "context": {
                    "visibility": "basic",
                    "withheld": ["skills/cfu-designer/SKILL.md: requires visibility level 'standard'"],
                },
            }
        )
        self.assertIn("## Visibility", report)
        self.assertIn("- level: basic", report)
        self.assertIn("SKILL.md", report)
        self.assertIn("writes and command allowlists are unaffected", report)


if __name__ == "__main__":
    unittest.main()
