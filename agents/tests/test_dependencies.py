"""Pure, disposable dependency invalidation checks."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from agents.cfu_design_agent.runtime.dependencies import (
    completed_gate_inputs, input_fingerprints, invalidate_changed_gates,
)


class DependencyTest(unittest.TestCase):
    def test_isa_change_stales_downstream_not_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = {"workdir": tmp, "workload_analysis": "measured evidence", "isa_spec": {"commands": [1]}}
            before = input_fingerprints(state)
            state["isa_spec"] = {"commands": [1, 2]}
            after = input_fingerprints(state)
            gates = {name: {"ok": True} for name in before}
            updated = invalidate_changed_gates(gates, before, after)
            self.assertTrue(updated["analysis"]["ok"])
            for name in ("isa", "implementation_hw", "hw_build", "sim", "cost"):
                self.assertFalse(updated[name]["ok"])
                self.assertTrue(updated[name]["stale"])

    def test_unlisted_workspace_source_change_invalidates_software(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = "agents/generated/runs/disposable"
            extra = root / run_root / "workspace/project_profile/project/helper.c"
            extra.parent.mkdir(parents=True)
            extra.write_text("old")
            state = {"workdir": tmp, "run_root": run_root, "isa_spec": {"commands": [1]},
                     "changed_sw_files": []}
            before = input_fingerprints(state)
            extra.write_text("new")
            after = input_fingerprints(state)
            updated = invalidate_changed_gates({name: {"ok": True} for name in before}, before, after)
            self.assertTrue(updated["isa"]["ok"])
            self.assertFalse(updated["implementation_sw"]["ok"])
            self.assertFalse(updated["sw_build"]["ok"])
            self.assertFalse(updated["sim"]["ok"])

    def test_conformance_source_or_header_change_stales_formal_check(self) -> None:
        for relative in ("09_implementation/conformance/oracle.yaml",
                         "09_implementation/conformance/cfu_conformance.c",
                         "workspace/demo_benchmark/cfu_intrinsics.h"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                run_root = "agents/generated/runs/disposable"
                path = Path(tmp) / run_root / relative
                path.parent.mkdir(parents=True)
                path.write_text("old")
                state = {"workdir": tmp, "run_root": run_root, "isa_spec": {"commands": [1]}}
                before = input_fingerprints(state)
                path.write_text("new")
                after = input_fingerprints(state)
                updated = invalidate_changed_gates({key: {"ok": True} for key in before}, before, after)
                self.assertFalse(updated["hw_conformance"]["ok"])
                self.assertTrue(updated["hw_conformance"]["stale"])
                self.assertTrue(updated["hw_build"]["ok"])

    def test_missing_verified_hardware_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = "agents/generated/runs/disposable"
            state = {"workdir": tmp, "run_root": run_root,
                     "gates": {"hw_build": {"ok": True}}}
            state["input_fingerprints"] = input_fingerprints(state)
            with self.assertRaisesRegex(ValueError, "lacks hardware source"):
                completed_gate_inputs(state)

    def test_hw_source_change_invalidates_build_and_sim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = "agents/generated/runs/disposable"
            design = root / run_root / "workspace/design/AgentCfu.scala"
            design.parent.mkdir(parents=True)
            design.write_text("old")
            state = {"workdir": tmp, "run_root": run_root, "isa_spec": {"commands": [1]}}
            before = input_fingerprints(state)
            design.write_text("new")
            after = input_fingerprints(state)
            updated = invalidate_changed_gates({name: {"ok": True} for name in before}, before, after)
            self.assertTrue(updated["isa"]["ok"])
            self.assertFalse(updated["hw_build"]["ok"])
            self.assertFalse(updated["hw_conformance"]["ok"])
            self.assertFalse(updated["sim"]["ok"])


if __name__ == "__main__":
    unittest.main()
