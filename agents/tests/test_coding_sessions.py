from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.cfu_design_agent.runtime.coding_session import CodingSession
from agents.cfu_design_agent.runtime.hardware_tools import HardwareCodingToolset
from agents.cfu_design_agent.runtime.software_tools import SoftwareCodingToolset


class FakeToolbox:
    def __init__(self, root: Path):
        self.workdir = root

    def read_text(self, path, limit=0):
        return (self.workdir / path).read_text()

    def write_text(self, path, text):
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return path

    def apply_edit_blocks(self, edits):
        updates = {}
        for path, pairs in edits:
            text = self.read_text(path)
            for search, replace in pairs:
                if text.count(search) != 1:
                    return {"returncode": 1, "stderr_tail": "not unique", "parsed": {"touched_paths": []}}
                text = text.replace(search, replace)
            updates[path] = text
        for path, text in updates.items():
            self.write_text(path, text)
        return {"returncode": 0, "stdout_tail": "applied", "parsed": {"touched_paths": list(updates)}}


class CodingSessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "a.c").write_text("int a = 1;\n")
        self.toolbox = FakeToolbox(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_rollback_restores_and_commit_preserves(self):
        session = CodingSession(self.toolbox, frozenset({"a.c"}))
        self.assertEqual(session.apply_edits([("a.c", [("1", "2")])])["returncode"], 0)
        session.rollback()
        self.assertEqual((self.root / "a.c").read_text(), "int a = 1;\n")
        session.rollback()

        session = CodingSession(self.toolbox, frozenset({"a.c"}))
        session.apply_edits([("a.c", [("1", "3")])])
        session.commit()
        session.rollback()
        self.assertEqual((self.root / "a.c").read_text(), "int a = 3;\n")

    def test_diagnosis_unlock_requires_current_failure(self):
        session = CodingSession(
            self.toolbox,
            frozenset({"a.c"}),
            failure_index={"failures": {"fail_current": {}}},
            diagnosis_required=True,
        )
        denied = session.apply_edits([("a.c", [("1", "2")])])
        self.assertNotEqual(denied["returncode"], 0)
        bad = session.submit_diagnosis({
            "failure_ids": ["fail_old"], "observed_errors": ["compile error"],
            "root_cause": "bad constant", "repair_mapping": ["replace it"],
            "regression_risks": ["none"],
        })
        self.assertIn("rejected", bad)
        empty = session.submit_diagnosis({
            "failure_ids": ["fail_current"], "observed_errors": [""],
            "root_cause": "bad constant", "repair_mapping": ["replace it"],
            "regression_risks": [],
        })
        self.assertIn("rejected", empty)
        good = session.submit_diagnosis({
            "failure_ids": ["fail_current"], "observed_errors": ["compile error"],
            "root_cause": "bad constant", "repair_mapping": ["replace it"],
            "regression_risks": ["none"],
        })
        self.assertIn("accepted", good)

    def test_diagnosis_rejects_resolved_failure_ids(self):
        session = CodingSession(
            self.toolbox,
            frozenset({"a.c"}),
            failure_index={"failures": {
                "fail_fixed": {"status": "resolved"},
                "fail_live": {"status": "current"},
            }},
            diagnosis_required=True,
        )
        stale = session.submit_diagnosis({
            "failure_ids": ["fail_fixed"], "observed_errors": ["old error"],
            "root_cause": "already fixed", "repair_mapping": ["re-apply the old fix"],
            "regression_risks": ["none"],
        })
        self.assertIn("rejected", stale)
        self.assertIn("fail_fixed", stale)
        self.assertIn("resolved", stale)
        good = session.submit_diagnosis({
            "failure_ids": ["fail_live"], "observed_errors": ["current error"],
            "root_cause": "real defect", "repair_mapping": ["fix it"],
            "regression_risks": ["none"],
        })
        self.assertIn("accepted", good)

    def test_edit_tools_map_tracked_scaffold_paths_onto_the_overlay(self):
        overlay = "agents/generated/runs/r-1/workspace/design/AgentCfu.scala"
        target = self.root / overlay
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("class A\n")

        class OverlayToolbox(FakeToolbox):
            run_root = "agents/generated/runs/r-1"
            embench_benchmark = ""

            def normalize(self, path):
                return path.lstrip("./")

        tool = HardwareCodingToolset(
            OverlayToolbox(self.root),
            frozenset({overlay}),
            check_elaboration=lambda: {"returncode": 0, "full_log_path": "log"},
        )
        applied = tool.run("edit_hardware", {"edits": [{
            "path": "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
            "search": "class A", "replace": "class B",
        }]})
        self.assertIn("applied", applied)
        self.assertEqual(target.read_text(), "class B\n")

        denied = tool.run("edit_hardware", {"edits": [{
            "path": "src/main/scala/other/File.scala",
            "search": "class B", "replace": "class C",
        }]})
        self.assertIn("denied", denied)
        self.assertIn(overlay, denied)
        self.assertEqual(target.read_text(), "class B\n")

    def test_hardware_multi_file_edit_is_atomic_and_check_has_no_args(self):
        (self.root / "A.scala").write_text("class A\n")
        (self.root / "B.scala").write_text("class B\n")
        tool = HardwareCodingToolset(
            self.toolbox,
            frozenset({"A.scala", "B.scala"}),
            check_elaboration=lambda: {"returncode": 0, "full_log_path": "log"},
        )
        rejected = tool.run("edit_hardware", {"edits": [
            {"path": "A.scala", "search": "class A", "replace": "class A2"},
            {"path": "B.scala", "search": "missing", "replace": "x"},
        ]})
        self.assertIn("rejected", rejected)
        self.assertEqual((self.root / "A.scala").read_text(), "class A\n")
        self.assertIn("no arguments", tool.run("check_hardware_elaboration", {"argv": []}))

    def test_software_tool_proxies_session_touched_paths(self):
        tool = SoftwareCodingToolset(
            self.toolbox, frozenset({"a.c"}),
            check_compile=lambda: {"returncode": 0, "full_log_path": "log"},
        )
        tool.run("edit_software", {"path": "a.c", "search": "int a = 1;", "replace": "int a = 2;"})
        self.assertEqual(tool.touched_paths, ["a.c"])
        tool.session.rollback()
        self.assertEqual(tool.touched_paths, [])


if __name__ == "__main__":
    unittest.main()
