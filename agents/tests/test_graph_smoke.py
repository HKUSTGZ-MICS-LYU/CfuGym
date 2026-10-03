"""One end-to-end offline dry run of the real graph.

Everything the graph needs is real (node logic, policy, artifacts, file writes,
run manifest); only the model and the external commands are absent. This is the
single test that proves the flow is wired, that a dry run terminates honestly,
and that every recorded artifact is content-addressed on disk.

Scenario coverage lives in the fast table tests: ``test_graph_routing`` for the
ladder decisions, ``test_graph_gates`` for per-rung gates, ``test_artifacts``
for manifest merge semantics.
"""

from __future__ import annotations

import hashlib
import shutil
import unittest
from pathlib import Path

import yaml

from agents.cfu_design_agent.graph import run_agent
from agents.tests.common import REPO_ROOT, offline_llm

RUN_ID = "smoke-dry-run-00000000"


class DryRunSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_dir = REPO_ROOT / f"agents/generated/runs/{RUN_ID}"
        shutil.rmtree(cls.run_dir, ignore_errors=True)
        state = {
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": True,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": True,
            "skip_yosys": True,
            "ir_features": "auto",
            "max_iters": 1,
            "errors": ["seed-error"],
            "run_id": RUN_ID,
        }
        with offline_llm():
            cls.final = run_agent(state)
        cls.manifest = yaml.safe_load((cls.run_dir / "run.yaml").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.run_dir, ignore_errors=True)

    def test_a_dry_run_ends_planned_and_writes_its_report(self) -> None:
        self.assertEqual(self.final["status"], "planned")
        report = REPO_ROOT / self.final["final_report_path"]
        self.assertTrue(report.is_file(), report)
        self.assertIn("planned", report.read_text().lower())

    def test_seeded_errors_and_history_survive_the_whole_flow(self) -> None:
        self.assertEqual(self.final.get("errors"), ["seed-error"])
        self.assertGreaterEqual(len(self.final.get("iteration_history", [])), 10)

    def test_the_design_overlay_is_planned_but_not_written_in_a_dry_run(self) -> None:
        self.assertTrue(self.final.get("design_workspace", {}).get("root", "").endswith("workspace/design"))
        expected = {
            f"agents/generated/runs/{RUN_ID}/workspace/design/AgentCfu.scala",
            f"agents/generated/runs/{RUN_ID}/workspace/design/AgentCfuFiber.scala",
        }
        self.assertEqual(set(self.final.get("design_files", [])), expected)
        # A dry run must never write agent-owned source, even the seeded overlay.
        for path in expected:
            self.assertFalse((REPO_ROOT / path).exists(), path)

    def test_ir_evidence_is_recorded_without_blocking_a_dry_run(self) -> None:
        self.assertEqual(self.final["ir_result"]["status"], "unavailable")
        self.assertTrue(self.final["gates"]["ir"]["ok"])
        self.assertTrue((self.run_dir / "07_analysis/ir_features.yaml").is_file())

    def test_every_manifest_artifact_is_content_addressed_on_disk(self) -> None:
        refs = self.manifest["artifacts"]
        self.assertGreater(len(refs), 5)
        registered = {ref["path"] for ref in refs}
        report_rel = Path(self.final["final_report_path"]).relative_to(REPO_ROOT).as_posix()
        self.assertIn(report_rel, registered)
        for ref in refs:
            with self.subTest(path=ref["path"]):
                self.assertRegex(ref["sha256"], r"[0-9a-f]{64}")
                immutable = REPO_ROOT / ref["immutable_path"]
                self.assertTrue(immutable.is_file(), immutable)
                self.assertEqual(hashlib.sha256(immutable.read_bytes()).hexdigest(), ref["sha256"])


if __name__ == "__main__":
    unittest.main()
