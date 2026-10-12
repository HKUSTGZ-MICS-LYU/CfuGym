"""Fairness transport/gate contracts; tempfile workspaces, no compiler or model."""
from __future__ import annotations

import difflib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent import prompts
from agents.cfu_design_agent.benchmark import cfu_execution, embench, embench_integrity
from agents.cfu_design_agent.runtime import dependencies, stage_services
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.cfu_design_agent.stages import implement, profile
from agents.cfu_design_agent.stages.verify import software


class FairnessTransportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        policy = Path(__file__).resolve().parents[1] / "specs/agent_access_policy.yaml"
        dest = self.root / "agents/specs/agent_access_policy.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(policy.read_bytes())
        self.run = "agents/generated/runs/fairness-test"
        self.sw = f"{self.run}/workspace/embench_demo/cfu/src/demo.c"
        self.hw = f"{self.run}/workspace/design/AgentCfu.scala"
        for path, text in ((self.sw, "int kernel(void) { return 1; }\n"),
                           (self.hw, "class AgentCfu {}\n")):
            full = self.root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(text)
        self.box = RepoToolbox(self.root, stage="implementation", policy=load_policy(self.root),
            run_root=self.run, embench_benchmark="demo", apply_patch=True,
            enforce_embench_integrity=True)
        self.state = {
            "workdir": str(self.root), "run_root": self.run,
            "benchmark_project": {"kind": "embench_benchmark", "benchmark": "demo",
                "verification": "bench_verify", "measurement": {
                    "protocol": embench.MEASUREMENT_PROTOCOL, "local_scale_factor": 1,
                    "global_scale_factor": 1, "warmup": False,
                     "cfu_status_protocol": cfu_execution.PROTOCOL}},
            "workload_spec": {"source_kind": "embench", "benchmark_name": "demo"},
            "gates": {"implementation_sw": {"ok": True}},
            "embench_profiles": {"reference": {"demo": {"cycles": 100, "result": 0}}},
            "changed_design_files": [self.hw], "run_commands": True, "apply_patch": True,
            "max_iters": 5, "iteration": 0,
        }

    def validator(self, problems):
        return mock.patch.object(embench_integrity, "validate_candidate", return_value=problems)

    def test_edit_rejected_before_any_write(self):
        with self.validator(["input changed"]) as check:
            result = self.box.apply_edit_blocks([
                (self.hw, [("class AgentCfu {}", "class AgentCfu { val x = 1 }")]),
                (self.sw, [("int kernel(void) { return 1; }", "int kernel(void) { return 2; }")])])
        self.assertEqual(result["returncode"], 1)
        self.assertIn("benchmark integrity", result["stderr_tail"])
        self.assertEqual((self.root / self.sw).read_text(), "int kernel(void) { return 1; }\n")
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")
        self.assertEqual(set(check.call_args.args[3]), {self.sw, self.hw})

    def test_file_rejected_atomically(self):
        with self.validator(["scale changed"]):
            result = self.box.apply_file_blocks([(self.hw, "class Other {}\n"),
                                                (self.sw, "int kernel(void) { return 0; }\n")])
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["parsed"]["touched_paths"], [])
        self.assertEqual((self.root / self.hw).read_text(), "class AgentCfu {}\n")

    def test_patch_rejected_and_all_targets_restored(self):
        originals = {path: (self.root / path).read_text() for path in (self.hw, self.sw)}
        changed = {self.hw: "class Other {}\n", self.sw: "int kernel(void) { return 0; }\n"}
        patch = "".join("".join(difflib.unified_diff(originals[path].splitlines(True),
            changed[path].splitlines(True), fromfile=f"a/{path}", tofile=f"b/{path}"))
            for path in originals)
        with self.validator(["driver changed"]):
            result = self.box.apply_patch_text(patch)
        self.assertEqual(result["returncode"], 1, result)
        self.assertIn("benchmark integrity", result["stderr_tail"])
        self.assertEqual(result["parsed"]["touched_paths"], [])
        for path, text in originals.items():
            self.assertEqual((self.root / path).read_text(), text)

    def test_stage_clone_keeps_host_enforcement(self):
        clone = self.box.stage_bucket("validation")
        self.assertTrue(clone.enforce_embench_integrity)
        with self.validator(["missing metadata"]):
            self.assertEqual(clone.benchmark_integrity_problems(), ["missing metadata"])

    def test_factory_enables_real_and_legacy_embench_states(self):
        with mock.patch.object(stage_services, "_policy", return_value=load_policy(self.root)):
            box = stage_services._toolbox_for_stage(self.state, "validation")
        self.assertTrue(box.enforce_embench_integrity)
        self.assertEqual(box.embench_benchmark, "demo")
        with self.validator(["legacy run requires new run"]):
            self.assertTrue(stage_services.benchmark_integrity_problems(box))

    def test_build_gate_rejects_without_running_commands(self):
        with self.validator(["scale changed"]), \
             mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box), \
             mock.patch.object(software.commands_mod, "_software_build_commands", return_value=[["make"]]), \
             mock.patch.object(software.commands_mod, "_software_elf", return_value=""), \
             mock.patch.object(stage_services, "_software_targets", return_value=[]), \
             mock.patch.object(stage_services, "_write_ladder_artifact"), \
             mock.patch.object(software.effects, "run_command") as run:
            result = software.build_sw(self.state)
        run.assert_not_called()
        gate = result["gates"]["sw_build"]
        self.assertFalse(gate["ok"])
        self.assertFalse(gate["benchmark_integrity"]["ok"])
        self.assertEqual(gate["measurement_protocol"], embench.MEASUREMENT_PROTOCOL)

    def test_sim_gate_rejects_without_running_commands(self):
        artifacts = mock.Mock()
        with self.validator(["input changed"]), \
             mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box), \
             mock.patch.object(software.commands_mod, "_software_sim_commands", return_value=[["sbt"]]), \
             mock.patch.object(stage_services, "_write_ladder_artifact"), \
             mock.patch.object(stage_services, "_artifacts_for_stage", return_value=artifacts), \
             mock.patch.object(software.effects, "run_command") as run:
            result = software.simulate(self.state)
        run.assert_not_called()
        gate = result["gates"]["sim"]
        self.assertFalse(gate["ok"])
        self.assertFalse(gate["benchmark_integrity"]["ok"])
        self.assertIsNone(gate["speedup_x100"])
        evidence = artifacts.write_yaml.call_args.args[2]
        self.assertEqual(evidence["measurement_protocol"], embench.MEASUREMENT_PROTOCOL)

    def test_corrupt_or_legacy_run_rejected_before_model_request(self):
        with self.validator(["legacy run requires a new run"]), \
             mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box), \
             mock.patch.object(implement.llm_step, "_llm") as llm:
            with self.assertRaisesRegex(ValueError, "new run"):
                implement.implement(self.state)
        llm.assert_not_called()

    def test_non_embench_validation_has_no_embench_protocol_fields(self):
        self.box.enforce_embench_integrity = False
        state = {**self.state, "benchmark_project": {"kind": "c_benchmark"}, "workload_spec": {}}
        with mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box), \
             mock.patch.object(software.commands_mod, "_software_build_commands", return_value=[["make"]]), \
             mock.patch.object(software.commands_mod, "_software_sim_commands", return_value=[["sbt"]]), \
             mock.patch.object(software.commands_mod, "_software_elf", return_value=""), \
             mock.patch.object(stage_services, "_software_targets", return_value=[]), \
             mock.patch.object(stage_services, "_write_ladder_artifact") as artifact, \
             mock.patch.object(software.effects, "run_command", return_value={"returncode": 0}):
            built = software.build_sw(state)
            simulated = software.simulate(state)
        for result, rung in ((built, "sw_build"), (simulated, "sim")):
            self.assertTrue(result["gates"][rung]["ok"])
            self.assertNotIn("measurement_protocol", result["gates"][rung])
            self.assertNotIn("benchmark_integrity", result["gates"][rung])
        for call in artifact.call_args_list:
            self.assertEqual(call.kwargs["evidence"], {})

    def test_profile_rejects_before_cache_or_commands(self):
        with self.validator(["legacy metadata; start a new run"]), \
             mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.box), \
             mock.patch.object(profile, "_write_embench_profile_artifact"), \
             mock.patch.object(profile.profile_cache, "build_key") as key, \
             mock.patch.object(profile.effects, "run_command") as run:
            result = profile.profile_embench(self.state)
        key.assert_not_called()
        run.assert_not_called()
        self.assertEqual(result["embench_profiles"], {})
        self.assertFalse(result["gates"]["profile"]["benchmark_integrity"]["ok"])

    def test_resume_refuses_legacy_protocol_even_without_completed_gates(self):
        state = {**self.state, "gates": {}}
        state["benchmark_project"] = {"kind": "embench_benchmark", "benchmark": "demo"}
        with self.assertRaisesRegex(ValueError, "start a new run"):
            dependencies.completed_gate_inputs(state, resume=True)

    def test_resume_rechecks_prepared_sources(self):
        state = {**self.state, "gates": {}, "benchmark_project": {
            **self.state["benchmark_project"], "integrity": {"version": 1}}}
        with self.validator(["missing reference"]):
            with self.assertRaisesRegex(ValueError, "missing reference"):
                dependencies.completed_gate_inputs(state, resume=True)
        with self.validator([]):
            dependencies.completed_gate_inputs(state, resume=True)

    def test_measurement_change_invalidates_sim_cost_fingerprints(self):
        before = dependencies.input_fingerprints(self.state)
        changed = {**self.state, "benchmark_project": {
            **self.state["benchmark_project"], "measurement": {"protocol": "other"}}}
        after = dependencies.input_fingerprints(changed)
        self.assertNotEqual(before["sim"], after["sim"])
        self.assertNotEqual(before["cost"], after["cost"])
        self.assertEqual(before["implementation_hw"], after["implementation_hw"])

    def test_shared_prompt_adds_rules_not_deliverables(self):
        prompt = prompts.embench_instructions(self.state["workload_spec"])
        self.assertIn("one cold workload invocation", prompt)
        self.assertIn("host-owned", prompt)
        self.assertIn("precompute expected", prompt)
        self.assertNotIn("operation counter", prompt)


if __name__ == "__main__":
    unittest.main()
