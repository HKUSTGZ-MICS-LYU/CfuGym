"""Model source views hide verifier bodies without modifying host source/evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.cfu_design_agent import prompts, spec
from agents.cfu_design_agent.runtime import flow_memory, source_visibility, stage_services
from agents.cfu_design_agent.runtime.policy import AccessPolicy, StagePolicy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.cfu_design_agent.stages import context as context_stage


SOURCE = """static const int algorithm_constant = 7654321;
int kernel(int input) { return input + algorithm_constant; }
int verify_benchmark(int result)
{
  static const int ORACLE_BODY_ONLY = 7654321;
  return result == ORACLE_BODY_ONLY;
}
int neighbour(void) { return 99; }
"""


class PromptSourceVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = "agents/generated/runs/source-view-unit"
        self.policy = AccessPolicy(
            version="unit", deny=(), write_roots=(), commands={},
            stages={"analysis": StagePolicy(write=(), commands=())},
            read=("src/**", "agents/generated/runs/**"), visibility="standard",
        )
        self.toolbox = RepoToolbox(self.root, policy=self.policy, run_root=self.run_root)
        self.path = "src/kernel.c"
        self.put(self.path, SOURCE)
        self.state = {
            "workdir": str(self.root), "run_id": "source-view-unit", "run_root": self.run_root,
            "visibility": "standard", "isa_spec": {},
        }

    def put(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def assert_masked(self, text):
        self.assertNotIn("ORACLE_BODY_ONLY", text)
        self.assertIn("verify_benchmark(int result)", text)
        self.assertIn("int kernel(int input) { return input + algorithm_constant; }", text)
        self.assertIn("int neighbour(void) { return 99; }", text)
        # Identical values outside the verifier remain available, not scrubbed globally.
        self.assertIn("algorithm_constant = 7654321", text)

    def test_no_tools_stage_prompts_mask_context_and_software_target(self):
        context = {"files": {self.path: SOURCE}}
        workload = {"name": "unit", "source_kind": "embench", "benchmark_name": "unit"}
        before = deepcopy(context)
        builders = {
            "analysis": lambda: prompts.workload_prompt("accelerate", workload, context),
            "contract": lambda: prompts.contract_prompt("accelerate", workload, "analysis", context),
            "hardware": lambda: prompts.hardware_prompt("accelerate", workload, "analysis", "contract", context),
            "software": lambda: prompts.software_prompt(
                "accelerate", workload, "analysis", "contract", context,
                targets=[(self.path, SOURCE)],
            ),
        }
        for stage, build in builders.items():
            with self.subTest(stage=stage):
                self.assert_masked(build())
        self.assertEqual(context, before)
        self.assertEqual(self.toolbox.read_text(self.path), SOURCE)
        self.assertIn("7654321", prompts.profile_digest({"reference": {"verify_benchmark": {"cycles": 7654321}}}))

    def test_context_digest_masks_before_6000_character_summary(self):
        source = SOURCE.replace("ORACLE_BODY_ONLY;", "ORACLE_BODY_ONLY;\n" + "x" * 7000)
        with mock.patch.object(source_visibility, "mask_verifier_bodies", return_value="safe view") as masking:
            text = prompts.context_digest({"files": {self.path: source}})
        masking.assert_called_once_with(source)
        self.assertIn("safe view", text)
        self.assertNotIn("ORACLE_BODY_ONLY", text)

    def test_model_source_reader_masks_full_file_before_limit(self):
        source = "int verify_benchmark(int result) {\n" + "ORACLE_BODY_ONLY\n" * 3000 + "}\n"
        self.put(self.path, source)
        with mock.patch.object(source_visibility, "mask_verifier_bodies", return_value="visible data") as masking:
            text = stage_services.model_source_text(self.toolbox, self.path, limit=7)
        masking.assert_called_once_with(source)
        self.assertEqual(text, "visible")
        self.assertEqual((self.root / self.path).read_text(), source)

    def test_model_source_reader_caps_large_file_prefix(self):
        toolbox = mock.Mock()
        toolbox.resolve.return_value = self.root / self.path
        toolbox.read_text.return_value = SOURCE
        with mock.patch.object(Path, "stat", return_value=mock.Mock(st_size=8_000_000)):
            text = stage_services.model_source_text(toolbox, self.path, limit=20_000)
        toolbox.read_text.assert_called_once_with(self.path, limit=2_000_000)
        self.assert_masked(text)

    def test_context_stage_stores_only_masked_excerpt(self):
        artifacts = mock.Mock(root_rel=self.run_root, run_id="source-view-unit")
        design_toolbox = mock.Mock()
        design_toolbox.seed_design_workspace.return_value = []
        state = {**self.state, "kernel_file": self.path}
        with mock.patch.object(stage_services, "_policy", return_value=self.policy), \
                mock.patch.object(stage_services, "_artifacts", return_value=artifacts), \
                mock.patch.object(stage_services, "_toolbox_for_stage", return_value=self.toolbox), \
                mock.patch.object(context_stage, "DEFAULT_CONTEXT_FILES", []), \
                mock.patch.object(context_stage, "RepoToolbox", return_value=design_toolbox):
            result = context_stage.load_context(state)
        self.assert_masked(result["context"]["files"][self.path])
        artifacts.write_text.assert_called_once()
        self.assert_masked(artifacts.write_text.call_args.args[2])
        self.assertEqual((self.root / self.path).read_text(), SOURCE)

    def test_host_software_targets_are_not_masked(self):
        state = {**self.state, "benchmark_project": {"kind": "generated_benchmark", "source_path": self.path}}
        targets = stage_services._software_targets(state, self.toolbox)
        self.assertEqual(targets, [(self.path, SOURCE)])
        self.assert_masked(prompts.software_prompt("accelerate", {}, "", "", {}, targets=targets))

    def test_c_project_optional_source_masks_before_20000_characters(self):
        source = "int verify_benchmark(int result) {\n" + "ORACLE_BODY_ONLY\n" * 3000 + "}\n"
        self.put(self.path, source)
        with mock.patch.object(source_visibility, "mask_verifier_bodies", return_value="safe") as masking:
            text = spec.read_optional(self.root / self.path)
        masking.assert_called_once_with(source)
        self.assertEqual(text, "safe")
        self.assertEqual((self.root / self.path).read_text(), source)

    def memory_with_excerpt(self, text, first, last):
        memory = flow_memory.derive_memory(self.state, {"kernel_file": self.path}, (), self.toolbox)
        entry = memory["entries"][0]
        lines = text.split("\n")
        entry.update(
            origin="tool", coverage={"first_line": first, "last_line": last, "full": False},
            excerpt="\n".join(f"{number}: {lines[number - 1]}" for number in range(first, last + 1))[:600],
        )
        return memory

    def test_restored_old_body_only_raw_excerpt_rebuilds_masked_observed_lines(self):
        memory = self.memory_with_excerpt(SOURCE, 5, 6)
        before = deepcopy(memory)
        self.assertIn("ORACLE_BODY_ONLY", memory["entries"][0]["excerpt"])
        clean = flow_memory.validate_memory(memory, self.state, self.toolbox)
        entry = clean["entries"][0]
        expected = source_visibility.mask_verifier_bodies(SOURCE).split("\n")
        self.assertEqual(entry["excerpt"], "5: " + expected[4] + "\n6: " + expected[5])
        self.assertEqual(entry["coverage"], memory["entries"][0]["coverage"])
        self.assertEqual(entry["sha256"], hashlib.sha256(SOURCE.encode()).hexdigest())
        for stage in ("analysis", "isa", "implement", "review"):
            rendered = flow_memory.render_memory(memory, self.state, self.toolbox, stage)
            self.assertIn("observed lines 5-6", rendered)
            self.assertNotIn("ORACLE_BODY_ONLY", rendered)
            self.assertNotIn("algorithm_constant", rendered)
            self.assertNotIn("neighbour", rendered)
        self.assertEqual(memory, before)

    def test_old_truncated_body_excerpt_is_masked_from_complete_source(self):
        source = SOURCE.replace("  static const int ORACLE_BODY_ONLY = 7654321;", "  ORACLE_BODY_ONLY" + "x" * 1000)
        self.put(self.path, source)
        memory = self.memory_with_excerpt(source, 5, 6)
        self.assertEqual(len(memory["entries"][0]["excerpt"]), 600)
        rendered = flow_memory.render_memory(memory, self.state, self.toolbox, "analysis", max_chars=3000)
        self.assertIn("observed lines 5-6", rendered)
        self.assertNotIn("ORACLE_BODY_ONLY", rendered)

    def test_masked_read_capture_keeps_raw_hash_and_precise_coverage(self):
        visible = source_visibility.mask_verifier_bodies(SOURCE).split("\n")
        with flow_memory.collect_tool_transcripts() as calls:
            flow_memory.capture_read(self.toolbox, self.path, {}, "5\t" + visible[4] + "\n6\t" + visible[5])
        self.assertEqual(len(calls), 1)
        entry = calls[0]["entry"]
        self.assertEqual(entry["sha256"], hashlib.sha256(SOURCE.encode()).hexdigest())
        self.assertEqual(entry["coverage"], {"first_line": 5, "last_line": 6, "full": False})
        self.assertNotIn("ORACLE_BODY_ONLY", entry["excerpt"])
        memory = flow_memory.derive_memory(self.state, {}, calls, self.toolbox)
        rendered = flow_memory.render_memory(memory, self.state, self.toolbox, "analysis")
        self.assertIn("observed lines 5-6", rendered)
        self.assertNotIn("ORACLE_BODY_ONLY", rendered)
        self.assertEqual((self.root / self.path).read_text(), SOURCE)

    def test_forged_cached_fragment_never_becomes_observed_text(self):
        memory = self.memory_with_excerpt(SOURCE, 5, 6)
        memory["entries"][0]["excerpt"] = "5: ORACLE_BODY_ONLY fabricated source"
        rendered = flow_memory.render_memory(memory, self.state, self.toolbox, "analysis")
        self.assertNotIn("ORACLE_BODY_ONLY", rendered)
        self.assertIn("reference only; not observed", rendered)

    def test_changed_source_invalidates_old_raw_cache(self):
        memory = self.memory_with_excerpt(SOURCE, 5, 6)
        self.put(self.path, SOURCE + "// changed\n")
        self.assertEqual(flow_memory.validate_memory(memory, self.state, self.toolbox)["entries"], [])

    def test_structured_memory_fact_masks_before_json_encoding(self):
        value = {"summary": SOURCE, "cycles": 7654321}
        update = {"workload_analysis": value, "gates": {"analysis": {"ok": True}}}
        before = deepcopy(value)
        memory = flow_memory.derive_memory(self.state, update, (), self.toolbox)
        self.assertNotIn("ORACLE_BODY_ONLY", memory["facts"][0]["excerpt"])
        self.assertIn('"cycles": 7654321', memory["facts"][0]["excerpt"])
        self.assertIn("algorithm_constant", memory["facts"][0]["excerpt"])
        self.assertEqual(memory["facts"][0]["sha256"], flow_memory._digest(value))
        self.assertEqual(value, before)

    def test_prompt_analysis_and_optimization_summaries_mask_json_encoded_source(self):
        encoded = json.dumps({"summary": SOURCE})
        builders = (
            prompts.contract_prompt("accelerate", {}, encoded, {}),
            prompts.hardware_prompt("accelerate", {}, encoded, "contract", {}),
            prompts.software_prompt("accelerate", {}, encoded, "contract", {}),
            prompts.optimization_prompt("accelerate", {}, "cycles=7654321", "", {}, encoded, encoded, 1, "performance"),
        )
        for text in builders:
            self.assertNotIn("ORACLE_BODY_ONLY", text)
            self.assertIn("algorithm_constant", text)
            self.assertIn("7654321", text)

    def test_spec_digest_masks_structured_source_before_yaml(self):
        value = {"description": SOURCE, "cycles": 7654321}
        before = deepcopy(value)
        text = prompts.spec_digest(value)
        self.assertNotIn("ORACLE_BODY_ONLY", text)
        self.assertIn("algorithm_constant", text)
        self.assertIn("cycles: 7654321", text)
        self.assertEqual(value, before)

    def test_context_artifact_encoded_source_masks_before_summary(self):
        path = "src/legacy.yaml"
        raw = json.dumps({"source": SOURCE, "result": 7654321})
        self.put(path, raw)
        self.assertNotIn("ORACLE_BODY_ONLY", prompts.context_digest({"files": {path: raw}}))
        self.assertNotIn("ORACLE_BODY_ONLY", stage_services.model_source_text(self.toolbox, path, limit=20000))
        self.assertEqual((self.root / path).read_text(), raw)

    def test_restored_json_artifact_fragment_masks_file_view_before_range_selection(self):
        self.path = f"{self.run_root}/logs/legacy.yaml"
        raw = json.dumps({"source": SOURCE, "result": 7654321})
        self.put(self.path, raw)
        memory = self.memory_with_excerpt(raw, 1, 1)
        clean = flow_memory.validate_memory(memory, self.state, self.toolbox)
        entry = clean["entries"][0]
        self.assertNotIn("ORACLE_BODY_ONLY", entry["excerpt"])
        self.assertIn("algorithm_constant", entry["excerpt"])
        self.assertIn('"result": 7654321', entry["excerpt"])
        self.assertEqual(entry["sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(entry["coverage"], memory["entries"][0]["coverage"])
        self.assertEqual((self.root / self.path).read_text(), raw)

    def test_memory_source_fact_masks_before_bounded_summary(self):
        update = {
            "software_plan": SOURCE,
            "gates": {"implementation_sw": {"ok": True}},
        }
        memory = flow_memory.derive_memory(self.state, update, (), self.toolbox)
        self.assertNotIn("ORACLE_BODY_ONLY", memory["facts"][0]["excerpt"])
        self.assertEqual(memory["facts"][0]["sha256"], flow_memory._digest(SOURCE))
        self.assertEqual(update["software_plan"], SOURCE)


if __name__ == "__main__":
    unittest.main()
