"""Gate honesty, state-merge semantics, and iteration behavior."""

from __future__ import annotations

import shutil
import unittest
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.llm import LlmResult
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

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        return _FakeResult(self.text)


class _ToolUsingLlm:
    """Fake client that performs one read_file tool call, then returns a patch."""

    available = True
    config_name = "fake-tools"
    model = "fake-model"
    base_url = ""
    api_mode = "chat"
    thinking = False
    reasoning_effort = None
    thinking_body = None

    def __init__(self, final_text: str, read_path: str) -> None:
        self.final_text = final_text
        self.read_path = read_path

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        return _FakeResult(self.final_text)

    def complete_with_tools(
        self, system, user, *, tools, tool_runner, max_rounds=6,
        fallback="", response_format=None, on_tool_call=None,
    ):
        observation = tool_runner("read_file", {"path": self.read_path})
        record = {
            "round": 1,
            "name": "read_file",
            "arguments": {"path": self.read_path},
            "result": observation,
            "ok": True,
        }
        if on_tool_call is not None:
            on_tool_call(record)
        return LlmResult(
            self.final_text,
            used_llm=True,
            model="fake-model",
            api_mode_used="chat+tools",
            tool_calls=(record,),
            tool_rounds=1,
        )


_THINK_OPEN = chr(60) + "thinking" + chr(62)
_THINK_CLOSE = chr(60) + "/thinking" + chr(62)
THINKING_PATCH = f"{_THINK_OPEN}I will change compute to 6{_THINK_CLOSE}\n{GOOD_PATCH}"


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
            return graph.implement_hw(state)

    def test_gate_ok_only_when_patch_applies(self) -> None:
        result = self._implement(GOOD_PATCH)
        gate = result["gates"]["implementation_hw"]
        self.assertTrue(gate["ok"])
        self.assertTrue(gate["applied"])
        self.assertEqual(len(result["changed_design_files"]), 1)

    def test_gate_fails_when_patch_does_not_apply(self) -> None:
        result = self._implement(BAD_PATCH)
        gate = result["gates"]["implementation_hw"]
        self.assertFalse(gate["ok"])
        self.assertFalse(gate["applied"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["changed_design_files"], [])

    def test_repair_edits_compose_on_the_live_workspace(self) -> None:
        # The prompt shows, and the protocol edits, the live run-workspace copy.
        # A repair must therefore apply on top of the earlier attempt instead of
        # being reset against the tracked original (which the prompt never shows).
        self._seed()
        state = _base_state(self.run_id, iteration=0)
        with mock.patch.object(graph, "_llm", lambda state: _FakeLlm(GOOD_PATCH)):
            first = graph.implement_hw(state)
        self.assertTrue(first["gates"]["implementation_hw"]["applied"], first["implementation_hw_summary"])

        target = REPO_ROOT / f"agents/generated/runs/{self.run_id}/workspace/design/AgentCfu.scala"
        self.assertIn("val compute = 6", target.read_text())

        # Second repair searches for the value written by the first attempt.
        second_patch = GOOD_PATCH.replace(
            "-  val compute = 1", "-  val compute = 6"
        ).replace("+  val compute = 6", "+  val compute = 7")
        state2 = dict(state)
        state2.update(first)
        state2["iteration"] = 1
        with mock.patch.object(graph, "_llm", lambda state: _FakeLlm(second_patch)):
            second = graph.implement_hw(state2)
        self.assertTrue(second["gates"]["implementation_hw"]["applied"], second["implementation_hw_summary"])
        text = target.read_text()
        self.assertIn("val compute = 7", text)
        self.assertNotIn("val compute = 1", text)
        self.assertNotIn("Re-seeded", second["implementation_hw_summary"])
        self.assertNotIn("Could not re-seed", second["implementation_hw_summary"])

    def test_llm_raw_output_is_persisted_verbatim(self) -> None:
        result = self._implement(GOOD_PATCH)
        paths = result.get("llm_raw_outputs", [])
        self.assertEqual(len(paths), 1)
        raw_path = REPO_ROOT / paths[0]
        self.assertTrue(raw_path.is_file())
        self.assertEqual(raw_path.read_text(), GOOD_PATCH)

    def test_thinking_text_is_split_out_before_patch_parsing(self) -> None:
        result = self._implement(THINKING_PATCH)
        self.assertTrue(
            result["gates"]["implementation_hw"]["ok"], result["implementation_hw_summary"]
        )
        paths = result.get("llm_raw_outputs", [])
        self.assertEqual(len(paths), 1)
        raw_path = REPO_ROOT / paths[0]
        raw = raw_path.read_text()
        self.assertIn("PATCH_BEGIN", raw)
        self.assertNotIn("I will change compute", raw)
        reasoning_path = raw_path.with_suffix(".reasoning.txt")
        self.assertTrue(reasoning_path.is_file())
        self.assertIn("I will change compute", reasoning_path.read_text())

    def test_reasoning_sidecar_does_not_shift_raw_output_numbering(self) -> None:
        self._seed()
        state = _base_state(self.run_id, iteration=0)
        with mock.patch.object(graph, "_llm", lambda state: _FakeLlm(THINKING_PATCH)):
            first = graph.implement_hw(state)
        state2 = dict(state)
        state2.update(first)
        state2["iteration"] = 1
        with mock.patch.object(graph, "_llm", lambda state: _FakeLlm(THINKING_PATCH)):
            second = graph.implement_hw(state2)
        self.assertTrue(first["llm_raw_outputs"][0].endswith("hardware_000.txt"))
        self.assertTrue(second["llm_raw_outputs"][0].endswith("hardware_001.txt"))

    def test_agent_tools_are_logged_and_audited(self) -> None:
        self._seed()
        state = _base_state(self.run_id)
        read_path = f"agents/generated/runs/{self.run_id}/workspace/design/AgentCfu.scala"
        llm = _ToolUsingLlm(GOOD_PATCH, read_path)
        with mock.patch.object(graph, "_llm", lambda state: llm):
            result = graph.implement_hw(state)
        gate = result["gates"]["implementation_hw"]
        self.assertTrue(gate["ok"], result["implementation_hw_summary"])
        self.assertEqual(gate["tool_calls"], 1)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["tool"], "read_file")
        self.assertGreater(result["tool_calls"][0]["result_chars"], 0)
        import yaml

        log_path = (
            REPO_ROOT
            / f"agents/generated/runs/{self.run_id}/09_implementation/hw_tool_calls.yaml"
        )
        self.assertTrue(log_path.is_file())
        data = yaml.safe_load(log_path.read_text())
        self.assertEqual(data["call_count"], 1)
        self.assertEqual(data["calls"][0]["tool"], "read_file")
        self.assertGreater(data["calls"][0]["result_chars"], 0)

    def test_report_lists_tool_calls(self) -> None:
        state = _base_state(
            "report-tools",
            tool_calls=[
                {
                    "round": 1,
                    "tool": "read_file",
                    "arguments": {"path": "bench/x.c"},
                    "ok": True,
                    "result_chars": 42,
                }
            ],
        )
        report = graph.render_report(state)
        self.assertIn("## Tool Calls", report)
        self.assertIn("read_file", report)

    def test_agent_content_is_not_written_without_apply_patch(self) -> None:
        # --run-commands (apply_patch=False) must not let Protocols A/B/C mutate
        # the run workspace; the response has to stay a proposal.
        edit_text = (
            "EDIT_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n"
            "  val compute = 1\n"
            "=======\n"
            "  val compute = 6\n"
            ">>>>>>> REPLACE\n"
            "EDIT_END\n"
        )
        file_text = (
            "FILE_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
            "// agent rewrite that must not be written\n"
            "FILE_END\n"
        )
        for label, text in (("edit", edit_text), ("file", file_text), ("patch", GOOD_PATCH)):
            with self.subTest(protocol=label):
                self._seed()
                target = (
                    REPO_ROOT
                    / f"agents/generated/runs/{self.run_id}/workspace/design/AgentCfu.scala"
                )
                before = target.read_text()
                state = _base_state(self.run_id, apply_patch=False)
                with mock.patch.object(graph, "_llm", lambda state: _FakeLlm(text)):
                    result = graph.implement_hw(state)
                gate = result["gates"]["implementation_hw"]
                self.assertFalse(gate["ok"], label)
                self.assertFalse(gate["applied"], label)
                self.assertEqual(result["changed_design_files"], [], label)
                self.assertTrue(
                    any("--apply-patch" in error for error in result["errors"]),
                    result["errors"],
                )
                self.assertEqual(target.read_text(), before, label)


class ReviewDecisionTest(unittest.TestCase):
    def _gates(self, **overrides) -> dict:
        gates = {
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "implementation_hw": {"ok": True},
            "hw_build": {"ok": True},
            "implementation_sw": {"ok": True},
            "sw_build": {"ok": True},
            "sim": {"ok": True},
            "cost": {"ok": True},
        }
        gates.update(overrides)
        return gates

    def test_all_executed_and_passing_is_complete(self) -> None:
        state = _base_state("review-complete", gates=self._gates())
        self.assertEqual(graph._review_decision(state), ("final_report", "complete"))

    def test_skipped_rung_reports_planned(self) -> None:
        gates = self._gates(cost={"ok": False, "skipped": True, "reason": "execution-disabled"})
        state = _base_state("review-skip", gates=gates)
        self.assertEqual(graph._review_decision(state), ("final_report", "planned"))

    def test_nothing_executed_reports_planned(self) -> None:
        gates = self._gates(
            hw_build={"ok": False, "skipped": True, "reason": "dry-run"},
            sw_build={"ok": False, "skipped": True, "reason": "dry-run"},
            sim={"ok": False, "skipped": True, "reason": "dry-run"},
            cost={"ok": False, "skipped": True, "reason": "dry-run"},
        )
        state = _base_state("review-noexec", gates=gates)
        self.assertEqual(graph._review_decision(state), ("final_report", "planned"))

    def test_executed_failure_is_failed(self) -> None:
        gates = self._gates(sim={"ok": False, "skipped": False})
        state = _base_state("review-fail", gates=gates)
        self.assertEqual(graph._review_decision(state), ("final_report", "failed"))

    def test_invalid_workload_is_blocked(self) -> None:
        gates = self._gates(workload_spec={"ok": False, "problems": ["missing"]})
        state = _base_state("review-blocked", gates=gates)
        self.assertEqual(graph._review_decision(state), ("final_report", "blocked"))

    def test_missing_rung_cannot_be_complete(self) -> None:
        gates = self._gates()
        gates.pop("sim")
        state = _base_state("review-missing", gates=gates)
        self.assertEqual(graph._review_decision(state), ("final_report", "blocked"))
        self.assertFalse(graph._is_complete(state))

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

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
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
        self.assertEqual(final.get("iteration", 0), 0)
        self.assertEqual(final["gates"]["hw_build"]["skipped"], True)

    @unittest.skipUnless(_LANGRAPH_AVAILABLE, "langgraph is not installed")
    def test_langgraph_runner_reports_planned_when_nothing_runs(self) -> None:
        final = self._run(graph.run_agent)
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final.get("iteration", 0), 0)

    def test_failed_rung_with_budget_repairs_instead_of_reviewing(self) -> None:
        gates = {
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "implementation_hw": {"ok": True},
            "hw_build": {"ok": False, "skipped": False, "next": "implement_hw"},
        }
        state = _base_state("review-repair", gates=gates, repairs_used=0, max_iters=3)
        self.assertEqual(graph._route_after_hw_build(state), "implement_hw")

    def test_failed_rung_without_budget_stops(self) -> None:
        gates = {
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "hw_build": {"ok": False, "skipped": False, "next": "review_and_route"},
        }
        state = _base_state("review-stop", gates=gates, repairs_used=1, max_iters=2)
        self.assertEqual(graph._route_after_hw_build(state), "review_and_route")


if __name__ == "__main__":
    unittest.main()
