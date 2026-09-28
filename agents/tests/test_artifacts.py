"""Run-scoped artifact storage and manifest persistence."""

from __future__ import annotations

import shutil
import unittest

from agents.cfu_design_agent.runtime.artifacts import RunArtifacts
from agents.cfu_design_agent.runtime.policy import PolicyError, load_policy
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

    def test_stale_instance_cannot_drop_newer_refs(self) -> None:
        # Regression (2026-09-26): check_conformance held one instance while
        # the LLM call registered its raw answer through another, and the later
        # write with the stale snapshot dropped that ref from run.yaml.
        stale = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        fresh = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        fresh.write_text(
            "implementation",
            "09_implementation/llm_raw/conformance_000.txt",
            "answer",
            kind="llm_raw_output",
        )
        stale.write_yaml(
            "implementation", "09_implementation/conformance/oracle.yaml", {"ok": True}
        )
        reopened = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        paths = [entry["path"] for entry in reopened.manifest["artifacts"]]
        self.assertTrue(any(p.endswith("llm_raw/conformance_000.txt") for p in paths), paths)
        self.assertTrue(any(p.endswith("conformance/oracle.yaml") for p in paths), paths)

    def test_rewritten_path_keeps_one_ref_with_latest_hash(self) -> None:
        first = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        first.write_text(
            "implementation", "09_implementation/llm_raw/conformance_000.txt", "v1",
            kind="llm_raw_output",
        )
        second = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        ref = second.write_text(
            "implementation", "09_implementation/llm_raw/conformance_000.txt", "v2",
            kind="llm_raw_output",
        )
        reopened = RunArtifacts.create(REPO_ROOT, self.policy, run_id="test-run-00000000-0000")
        entries = [
            entry for entry in reopened.manifest["artifacts"]
            if entry["path"].endswith("llm_raw/conformance_000.txt")
        ]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["sha256"], ref["sha256"])

    def test_write_outside_stage_policy_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            self.run.write_text("load_context", "02_workload/should_not_be_written.yaml", "x", kind="text")

    def test_run_id_sanitized(self) -> None:
        with self.assertRaises(PolicyError):
            RunArtifacts.create(REPO_ROOT, self.policy, run_id="../../evil")


if __name__ == "__main__":
    unittest.main()
