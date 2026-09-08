"""Run-scoped artifact storage and manifest persistence."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.artifacts import RunArtifacts
from agents.cfu_design_agent.policy import PolicyError, load_policy
from agents.tests.common import REPO_ROOT, repo_files


class ArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(REPO_ROOT)
        self.run = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")

    def tearDown(self) -> None:
        shutil.rmtree(REPO_ROOT / "agents/generated/runs/test-run-00000000-0000", ignore_errors=True)

    def test_writes_manifest_with_hash(self) -> None:
        ref = self.run.write_yaml("ingest_input", "02_workload/workload_spec.yaml", {"operation": "weighted_average"})
        self.assertTrue(ref["sha256"])
        self.assertTrue((REPO_ROOT / ref["path"]).exists())
        self.assertTrue(any(entry["path"] == ref["path"] for entry in self.run.manifest["artifacts"]))

    def test_manifest_persists_across_writes(self) -> None:
        self.run.write_yaml("ingest_input", "02_workload/workload_spec.yaml", {"operation": "weighted_average"})
        self.run.write_yaml("ingest_input", "02_workload/validation.yaml", {"status": "valid"})
        self.assertEqual(len(self.run.manifest["artifacts"]), 2)

    def test_second_open_reuses_same_run(self) -> None:
        self.run.write_yaml("ingest_input", "02_workload/workload_spec.yaml", {"operation": "weighted_average"})
        reopened = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        self.assertEqual(len(reopened.manifest["artifacts"]), 1)

    def test_write_outside_stage_policy_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            self.run.write_text("load_context", "02_workload/should_not_be_written.yaml", "x", kind="text")

    def test_run_id_sanitized(self) -> None:
        with self.assertRaises(PolicyError):
            RunArtifacts.create(REPO_ROOT, self.policy, run_id="../../evil")


if __name__ == "__main__":
    unittest.main()
