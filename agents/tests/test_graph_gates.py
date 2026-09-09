"""Gate honesty, state-merge semantics, and iteration behavior."""

from __future__ import annotations

import shutil
import unittest
from unittest import mock

from agents.cfu_design_agent import graph
from agents.tests.common import REPO_ROOT

try:
    from agents.cfu_design_agent.graph import _LANGRAPH_AVAILABLE
except ImportError:  # pragma: no cover
    _LANGRAPH_AVAILABLE = False


GOOD_PATCH = """PATCH_BEGIN
--- a/src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala
+++ b/src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala
@@ -36,7 +36,7 @@
 }
 
 object AgentCfuFunction {
-  val compute = 1
+  val compute = 6
   val config = 2
   val load = 4
   val store = 5
 }
PATCH_END"""

BAD_PATCH = """PATCH_BEGIN
--- a/src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala
+++ b/src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala
@@ -36,7 +36,7 @@
 this context does not exist in the file
-  val compute = 1
+  val compute = 6
   val config = 2
   val load = 4
PATCH_END"""


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.used_llm = False


class _FakeLlm:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, system: str, user: str, *, fallback: str):
        return _FakeResult(self.text)


def _base_state(run_id: str, **overrides) -> dict:
    state = {
        "workdir": str(REPO_ROOT),
        "run_root": f"agents/generated/runs/{run_id}",
        "run_id": run_id,
        "task": "accelerate a kernel",
        "agent_cfu": True,
        "dry_run": False,
        "apply_patch": True,
        "run_commands": False,
        "iteration": 0,
        "max_iters": 2,
        "gates": {},
    }
    state.update(overrides)
    return state


class MergeSemanticsTest(unittest.TestCase):
    def _run(self, run_id: str) -> dict:
        run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"
        shutil.rmtree(run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        return graph.run_agent_python({
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": True,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": True,
            "skip_yosys": True,
            "max_iters": 1,
            "run_id": run_id,
            "errors": ["seed-error"],
        })

    def test_python_runner_appends_history_and_keeps_errors_once(self) -> None:
        final = self._run("merge-python-00000000")
        self.assertEqual(final.get("errors"), ["seed-error"])
        self.assertGreaterEqual(len(final.get("iteration_history", [])), 10)

    @unittest.skipUnless(_LANGRAPH_AVAILABLE, "langgraph is not installed")
    def test_langgraph_runner_does_not_duplicate_errors(self) -> None:
        run_id = "merge-langgraph-00000000"
        run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"
        shutil.rmtree(run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        final = graph.run_agent({
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": True,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": True,
            "skip_yosys": True,
            "max_iters": 1,
            "run_id": run_id,
            "errors": ["seed-error"],
        })
        self.assertEqual(final.get("errors"), ["seed-error"])
        self.assertGreaterEqual(len(final.get("iteration_history", [])), 10)


class ImplementationGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "gate-impl-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def _seed(self) -> None:
        from agents.cfu_design_agent.tools import RepoToolbox

        RepoToolbox(
            REPO_ROOT,
            dry_run=False,
            stage="implementation",
            run_root=f"agents/generated/runs/{self.run_id}",
        ).seed_design_workspace()

    def _implement(self, patch: str, **overrides):
        self._seed()
        state = _base_state(self.run_id, **overrides)
        with mock.patch.object(graph, "_llm", lambda state: _FakeLlm(patch)):
            return graph.implement_hw_sw(state)

    def test_gate_ok_only_when_patch_applies(self) -> None:
        result = self._implement(GOOD_PATCH)
        gate = result["gates"]["implementation"]
        self.assertTrue(gate["ok"])
        self.assertTrue(gate["applied"])
        self.assertEqual(len(result["changed_design_files"]), 1)

    def test_gate_fails_when_patch_does_not_apply(self) -> None:
        result = self._implement(BAD_PATCH)
        gate = result["gates"]["implementation"]
        self.assertFalse(gate["ok"])
        self.assertFalse(gate["applied"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["changed_design_files"], [])

    def test_iteration_reseeds_workspace_before_patch(self) -> None:
        first = self._implement(GOOD_PATCH)
        self.assertTrue(first["gates"]["implementation"]["applied"])
        second = self._implement(GOOD_PATCH, iteration=1)
        self.assertTrue(second["gates"]["implementation"]["applied"])
        self.assertIn("Re-seeded", second["implementation_summary"])


class ReviewDecisionTest(unittest.TestCase):
    def test_skipped_gates_report_planned_not_failed(self) -> None:
        state = _base_state("review-skip", gates={
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "validation": {"ok": False, "skipped": True, "reason": "execution-disabled"},
            "cost": {"ok": False, "skipped": True, "reason": "execution-disabled"},
        })
        self.assertEqual(graph._review_decision(state), ("final_report", "planned"))

    def test_executed_failure_still_iterates_then_fails(self) -> None:
        gates = {
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "validation": {"ok": False, "skipped": False},
            "cost": {"ok": True, "skipped": False},
        }
        state = _base_state("review-fail", gates=gates, iteration=0, max_iters=2)
        self.assertEqual(graph._review_decision(state), ("implement_hw_sw", "needs_iteration"))
        state["iteration"] = 2
        self.assertEqual(graph._review_decision(state), ("final_report", "failed"))

VALID_ISA = """spec_version: \"1\"
identity:
  name: u8_wavg_agent
  status: draft
  opcode: custom0 (0x0B)
  function_id_width: 3
  reference_workload: uint8 weighted average
commands:
  - function_id: 1
    name: compute
    operand: rs1, rs2
    result: sum
    semantics: multiply and accumulate uint8 lanes
    response: status
datapath:
  vlen: 256
  xlen: 32
  maclen: 32
  accWidth: 32
  regDepth: 2
software:
  intrinsics_header: sw/tests/u8_wavg_cfu_test.c
  scalar_fallback: true
integration:
  hardware_component: AgentCfu
  hardware_fiber: AgentCfuFiber
  bus_style: CfuBus
  single_cpu_bus_owner: true
validation:
  status: draft
  requirements: [scalar reference equals CFU output]
assumptions: {}
unknowns: []"""


class _ContractLlm:
    """Returns a valid ISA contract; falls back for every other prompt."""

    available = False

    def complete(self, system: str, user: str, *, fallback: str):
        if "Design the CFU contract" in user:
            return _FakeResult(VALID_ISA)
        return _FakeResult(fallback)


class NoExecutionStatusTest(unittest.TestCase):
    """--no-dry-run without --run-commands must not report failed/blocked."""

    def _run(self, runner) -> dict:
        run_id = "status-noexec-00000000"
        run_dir = REPO_ROOT / f"agents/generated/runs/{run_id}"
        shutil.rmtree(run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        state = {
            "workdir": str(REPO_ROOT),
            "task": "Accelerate a uint8 weighted average kernel with a TileLink CFU.",
            "input_mode": "natural_language",
            "dry_run": False,
            "run_commands": False,
            "apply_patch": False,
            "skip_sim": False,
            "skip_yosys": False,
            "max_iters": 2,
            "run_id": run_id,
        }
        with mock.patch.object(graph, "_llm", lambda state: _ContractLlm()):
            return runner(state)

    def test_python_runner_reports_planned_when_nothing_runs(self) -> None:
        final = self._run(graph.run_agent_python)
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final["iteration"], 1)

    @unittest.skipUnless(_LANGRAPH_AVAILABLE, "langgraph is not installed")
    def test_langgraph_runner_reports_planned_when_nothing_runs(self) -> None:
        final = self._run(graph.run_agent)
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final["iteration"], 1)


if __name__ == "__main__":
    unittest.main()
