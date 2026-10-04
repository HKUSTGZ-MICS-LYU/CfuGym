"""Ablation-switch contracts: --no-llvm-ir / --no-conformance / --no-isa-contract.

Router/unit level only (per the fast contract-suite rule): these pin the
switching semantics — disabled gates, the contract short-circuit, the
agent-authored intrinsics header — without a full-ladder run. The default
(no flag) behavior is pinned by the rest of the suite.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent import isa as isa_mod
from agents.cfu_design_agent.flow import gates as gates_mod
from agents.cfu_design_agent.stages import contract as contract_mod
from agents.cfu_design_agent.stages.implement import intrinsics
from agents.cfu_design_agent.stages.verify import conformance as conformance_mod
from agents.cfu_design_agent.runtime import stage_services
from agents.tests.common import REPO_ROOT  # noqa: F401  (installs the network guard)


def _complete_gates(disabled: dict[str, str] | None = None) -> dict:
    """Every gate ok, with the named rungs recorded as ablation-disabled."""
    disabled = disabled or {}
    gates = {}
    for rung in gates_mod.REQUIRED_GATES:
        if rung in disabled:
            gates[rung] = {"ok": False, "skipped": True, "disabled": disabled[rung],
                           "reason": f"disabled by {disabled[rung]} (ablation)"}
        else:
            gates[rung] = {"ok": True}
    return gates


class DisabledGateTest(unittest.TestCase):
    """A disabled rung is out of scope; a real skip still degrades to planned."""

    def test_disabled_rungs_are_excluded_from_the_requirement(self) -> None:
        state = {
            "isa_contract": False, "require_conformance": False,
            "gates": _complete_gates({"isa": "--no-isa-contract",
                                      "hw_conformance": "--no-conformance"}),
        }
        self.assertNotIn("isa", gates_mod.required_gates(state))
        self.assertNotIn("hw_conformance", gates_mod.required_gates(state))
        self.assertTrue(gates_mod._is_complete(state))
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "complete"))

    def test_disabled_rung_with_ok_false_does_not_fail_the_run(self) -> None:
        state = {
            "require_conformance": False,
            "gates": _complete_gates({"hw_conformance": "--no-conformance"}),
        }
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "complete"))

    def test_a_real_skip_still_reports_planned(self) -> None:
        gates = _complete_gates()
        gates["sim"] = {"ok": False, "skipped": True, "reason": "--skip-sim"}
        state = {"gates": gates}
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "planned"))
        self.assertFalse(gates_mod._is_complete(state))

    def test_defaults_keep_every_rung_required(self) -> None:
        state = {"gates": _complete_gates()}
        self.assertEqual(gates_mod.required_gates(state), gates_mod.REQUIRED_GATES)
        self.assertTrue(gates_mod._is_complete(state))


class ContractShortCircuitTest(unittest.TestCase):
    """--no-isa-contract records a disabled isa gate and authors no contract."""

    def _run(self, node):
        state = {
            "isa_contract": False, "gates": {}, "task": "t",
            "workdir": str(REPO_ROOT), "run_root": "",
        }
        with mock.patch.object(stage_services, "_artifacts_for_stage") as artifacts:
            update = node(state)
        return update, artifacts

    def test_design_contract_is_skipped_not_authored(self) -> None:
        update, artifacts = self._run(contract_mod.design_contract)
        self.assertEqual(update["isa_spec"], {})
        self.assertEqual(update["design_contract"], "")
        gate = update["gates"]["isa"]
        self.assertEqual(gate.get("disabled"), "no-isa-contract")
        self.assertFalse(gate.get("ok"))
        self.assertTrue(gate.get("skipped"))
        self.assertEqual(gate.get("next"), "implement")
        self.assertNotIn("llm_raw_outputs", update)
        artifacts.return_value.write_yaml.assert_called()

    def test_extend_isa_is_skipped_too(self) -> None:
        update, _artifacts = self._run(contract_mod.extend_isa)
        self.assertEqual(update["gates"]["isa"].get("disabled"), "no-isa-contract")
        self.assertEqual(update["gates"]["isa"].get("next"), "implement")

    def test_default_still_runs_the_contract_stage(self) -> None:
        state = {"isa_contract": True, "gates": {}, "task": "t"}
        with mock.patch.object(contract_mod, "_isa_step") as step, \
                mock.patch.object(contract_mod.llm_step, "_llm", lambda state: None), \
                mock.patch.object(contract_mod.llm_step, "_tools_enabled", lambda state, client: False), \
                mock.patch.object(stage_services, "_write_isa_artifact"):
            step.return_value = (mock.Mock(text="ABSTAIN", used_llm=True), "raw")
            update = contract_mod.design_contract(state)
        step.assert_called_once()
        self.assertIsNone(update["gates"]["isa"].get("disabled"))
        self.assertTrue(update["gates"]["isa"].get("abstain"))


class ConformanceDisabledTest(unittest.TestCase):
    """--no-conformance records a disabled hw_conformance rung and moves on."""

    def test_check_conformance_short_circuits(self) -> None:
        state = {"require_conformance": False, "gates": {}, "workdir": str(REPO_ROOT),
                 "run_root": "", "task": "t"}
        with mock.patch.object(stage_services, "_write_ladder_artifact") as write_ladder:
            update = conformance_mod.check_conformance(state)
        gate = update["gates"]["hw_conformance"]
        self.assertEqual(gate.get("disabled"), "no-conformance")
        self.assertFalse(gate.get("ok"))
        self.assertTrue(gate.get("skipped"))
        self.assertIn("disabled", str(write_ladder.call_args))
        self.assertTrue(update["validation_results"][0].get("skipped"))

    def test_default_still_validates(self) -> None:
        state = {"require_conformance": True, "gates": {}, "workdir": str(REPO_ROOT),
                 "run_root": "", "task": "t", "dry_run": False, "run_commands": False}
        with mock.patch.object(stage_services, "_write_ladder_artifact"), \
                mock.patch.object(stage_services, "_profile_reason", return_value="execution-disabled"):
            update = conformance_mod.check_conformance(state)
        self.assertIsNone(update["gates"]["hw_conformance"].get("disabled"))


class _FakeToolbox:
    """Just enough RepoToolbox for header ownership unit tests."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.run_root = ""
        self.stage = "implementation"
        self.writes: list[tuple[str, str]] = []

    def write_text(self, path: str, text: str) -> None:
        self.writes.append((path, text))
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def read_text(self, path: str, *, limit: int = 1_000_000) -> str:
        return (self.workdir / path).read_text()[:limit]


class HeaderOwnershipTest(unittest.TestCase):
    """--no-isa-contract: the header is agent-authored, seeded once, never clobbered."""

    def _state(self, workdir: Path, **overrides) -> dict:
        state = {
            "workdir": str(workdir), "run_root": "",
            "benchmark_project": {"kind": "generated_benchmark", "workspace": "ws/demo_benchmark"},
            "apply_patch": True, "dry_run": False,
            "isa_contract": False,
        }
        state.update(overrides)
        return state

    def test_template_is_seeded_once_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            toolbox = _FakeToolbox(workdir)
            state = self._state(workdir)
            path = intrinsics._write_intrinsics_header(state, toolbox)
            self.assertEqual(path, "ws/demo_benchmark/cfu_intrinsics.h")
            self.assertEqual(len(toolbox.writes), 1)
            self.assertEqual(toolbox.writes[0][1], intrinsics.intrinsics_template_text(str(workdir)))
            # The agent edits the seeded header; a later round must keep its bytes.
            toolbox.write_text(path, "// my own encodings\n")
            again = intrinsics._write_intrinsics_header(state, toolbox)
            self.assertEqual(again, path)
            self.assertEqual(len(toolbox.writes), 2)  # only the agent edit
            self.assertEqual(toolbox.read_text(path, limit=1000), "// my own encodings\n")

    def test_default_contract_still_renders_the_host_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            toolbox = _FakeToolbox(workdir)
            state = self._state(workdir, isa_contract=True,
                                isa_spec={"commands": [{"function_id": 1, "name": "op"}]})
            path = intrinsics._write_intrinsics_header(state, toolbox)
            self.assertTrue(path)
            self.assertEqual(toolbox.writes[-1][1], isa_mod.render_cfu_intrinsics(state["isa_spec"]))

    def test_call_site_scan_ignores_the_header_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            toolbox = _FakeToolbox(workdir)
            header = "ws/demo_benchmark/cfu_intrinsics.h"
            kernel = "ws/demo_benchmark/main.c"
            toolbox.write_text(header, "#define cfu_foo(a, b) /* macro body */\n")
            toolbox.write_text(kernel, "int main(void) { return 0; }\n")
            state = self._state(workdir)
            exercises, why = intrinsics._software_exercises_cfu(
                state, toolbox, [(header, ""), (kernel, "")])
            self.assertFalse(exercises)
            self.assertIn("cfu_intrinsics.h", why)
            toolbox.write_text(kernel, "int main(void) { return cfu_foo(1, 2); }\n")
            exercises, _why = intrinsics._software_exercises_cfu(
                state, toolbox, [(header, ""), (kernel, "")])
            self.assertTrue(exercises)


if __name__ == "__main__":
    unittest.main()
