"""Shared answer contracts: tolerant YAML parsing and an honest evidence index.

Observed 2026-09-29 (crc32-run-0001/0002): a model answer wrapped in a prose
sentence plus a ```yaml fence was dropped, the workload analysis silently became
``{}``, and the hand-written evidence index pointed at artifacts that did not
exist yet (so the model burned tool calls on them).
"""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.runtime import parsing, stage_services
from agents.tests.common import REPO_ROOT


class AnswerParsingTest(unittest.TestCase):
    def test_a_plain_mapping_is_returned_unchanged(self) -> None:
        self.assertEqual(parsing.parse_yaml_mapping("a: 1\nb: 2\n"), {"a": 1, "b": 2})

    def test_a_preamble_and_a_yaml_fence_are_tolerated(self) -> None:
        text = "Based on the profile, here is the analysis.\n\n```yaml\na: 1\nb: 2\n```\n"
        self.assertEqual(parsing.parse_yaml_mapping(text), {"a": 1, "b": 2})

    def test_a_fenced_json_body_is_tolerated(self) -> None:
        self.assertEqual(parsing.parse_yaml_mapping('```json\n{"a": 1}\n```'), {"a": 1})

    def test_invalid_yaml_is_not_silently_reinterpreted(self) -> None:
        # A sequence/mapping indentation clash is the model's error to fix; the
        # parser must not guess a mapping out of it.
        self.assertIsNone(parsing.parse_yaml_mapping("hardware_state:\n  - name: x\n  rf: y\n"))

    def test_a_scalar_or_sequence_answer_is_not_a_mapping(self) -> None:
        self.assertIsNone(parsing.parse_yaml_mapping("- a\n- b\n"))
        self.assertIsNone(parsing.parse_yaml_mapping("just prose"))

    def test_candidates_try_the_response_before_the_fence(self) -> None:
        candidates = list(parsing.document_candidates("head\n```yaml\nbody\n```\ntail"))
        self.assertEqual(candidates[0], "head\n```yaml\nbody\n```\ntail")
        self.assertEqual(candidates[1], "body\n")


class EvidenceIndexTest(unittest.TestCase):
    """The index must describe artifacts that exist right now."""

    run_id = "answer-contracts-00000000"
    run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"

    def setUp(self) -> None:
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def state(self) -> dict:
        return {"workdir": str(REPO_ROOT), "run_root": f"agents/generated/runs/{self.run_id}"}

    def test_missing_paths_are_dropped_and_existing_ones_kept(self) -> None:
        (self.run_dir / "02_workload").mkdir(parents=True)
        (self.run_dir / "02_workload/workload_spec.yaml").write_text("name: x\n")
        run_root = f"agents/generated/runs/{self.run_id}"
        kept = stage_services.existing_evidence(self.state(), [
            (f"{run_root}/02_workload/workload_spec.yaml", "spec"),
            # For an embench run only one of these shapes exists.
            (f"{run_root}/04_benchmark/manifest.yaml", "plan"),
            (f"{run_root}/04_benchmark/embench.yaml", "plan"),
            # A stage must not be pointed at the artifact it is about to write.
            (f"{run_root}/08_isa/cfu_isa.yaml", "contract"),
        ])
        self.assertEqual(kept, [(f"{run_root}/02_workload/workload_spec.yaml", "spec")])

    def test_a_directory_target_is_kept(self) -> None:
        (self.run_dir / "workspace/design").mkdir(parents=True)
        run_root = f"agents/generated/runs/{self.run_id}"
        self.assertEqual(
            stage_services.existing_evidence(self.state(), [
                (f"{run_root}/workspace/design/", "overlay"),
            ]),
            [(f"{run_root}/workspace/design/", "overlay")],
        )


if __name__ == "__main__":
    unittest.main()
