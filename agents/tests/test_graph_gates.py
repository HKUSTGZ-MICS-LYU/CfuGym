"""Gate honesty, state-merge semantics, and iteration behavior."""

from __future__ import annotations

import io
import shutil
import sys
import unittest
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime.llm import LlmResult
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


class _UsedLlm(_FakeLlm):
    """Fake client whose completion claims the model really answered."""

    def complete(self, system: str, user: str, *, fallback: str, response_format=None):
        return LlmResult(self.text, used_llm=True)


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
        max_calls_per_round=4, max_total_calls=24,
        fallback="", response_format=None, deliverable_hint="",
        nudge_on_tool_intent=True, is_deliverable=None, on_tool_call=None,
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
        from agents.cfu_design_agent.runtime.tools import RepoToolbox

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

    def test_answer_without_a_delivery_block_is_a_reported_error(self) -> None:
        # Regression: an agent that only narrates (for example it exhausted its
        # tool budget while reading) must fail the rung with an explicit format
        # error instead of an empty error list that the downstream build
        # misattributes.
        narration = (
            "Now I understand the design pattern. Let me look at the next file "
            "to see how it initializes the accumulator."
        )
        result = self._implement(narration)
        gate = result["gates"]["implementation_hw"]
        self.assertFalse(gate["ok"])
        self.assertFalse(gate["applied"])
        self.assertTrue(
            any("no EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN" in err for err in result["errors"]),
            result["errors"],
        )
        self.assertTrue(
            any("tool-intent preamble" in err for err in result["errors"]),
            result["errors"],
        )
        feedback = graph.implementation_feedback({"gates": result["gates"]}, "implementation_hw")
        self.assertIn("no EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN", feedback)

    def test_unterminated_edit_block_is_reported(self) -> None:
        # A valid EDIT_BEGIN whose terminator is malformed must be reported as
        # such, not as a generic "no deliverable".
        text = (
            "EDIT_BEGIN src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala\n"
            "<<<<<<< SEARCH\n  a\n=======\n  b\n>>>>>>> REPLACE\n"
        )
        result = self._implement(text)
        self.assertFalse(result["gates"]["implementation_hw"]["ok"])
        self.assertTrue(
            any("was not closed" in err for err in result["errors"]), result["errors"]
        )

    def test_tool_intent_detection_matches_the_observed_failure(self) -> None:
        from agents.cfu_design_agent.runtime.llm import looks_like_tool_intent

        narration = (
            "I'll analyze the depthconv workload and design the CFU hardware. Let me "
            "first examine the full benchmark source to understand the exact computation "
            "pattern.\n\nLet me read the complete depthconv.c file to understand the full "
            "kernel structure."
        )
        self.assertTrue(looks_like_tool_intent(narration))
        self.assertFalse(looks_like_tool_intent("PATCH_BEGIN\n--- a/x\n+++ b/x\n"))
        self.assertFalse(looks_like_tool_intent(""))

        reason = graph.tool_intent_error(narration, "RuntimeError: tools rejected")
        self.assertIn("tool-intent preamble", reason)
        self.assertIn("RuntimeError: tools rejected", reason)
        self.assertEqual(graph.tool_intent_error("a finished design", ""), "")

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
            "hw_conformance": {"ok": True},
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
  implementation:
    result_width_bits: 32
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


class AnalysisToolLoopTest(unittest.TestCase):
    """The analysis rung can consult the workload source and disassembly."""

    ANALYSIS_YAML = (
        "operation: depthconv\n"
        "pattern: convolution\n"
        "hot_spot: depthconv\n"
        "element_type: int8\n"
        "roofline:\n"
        "  hot_function: depthconv\n"
        "  bottleneck: compute\n"
        "unknowns: []\n"
    )

    def setUp(self) -> None:
        self.run_id = "gate-analysis-tools-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def test_analysis_records_tool_calls_and_parses_the_answer(self) -> None:
        state = _base_state(
            self.run_id,
            workload_spec={"name": "depthconv", "operation": "embench_workload"},
            benchmark_name="depthconv",
        )
        llm = _ToolUsingLlm(
            self.ANALYSIS_YAML, "benchmarks/embench-iot/src/depthconv/depthconv.c"
        )
        with mock.patch.object(graph, "_llm", lambda state: llm):
            result = graph.analyze_workload(state)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["tool"], "read_file")
        # The answer is parsed as analysis YAML, not treated as a patch.
        self.assertIn("hot_function: depthconv", result["workload_analysis"])
        import yaml

        log = self.run_dir / "07_analysis/analysis_tool_calls.yaml"
        self.assertTrue(log.is_file())
        data = yaml.safe_load(log.read_text())
        self.assertEqual(data["call_count"], 1)
        self.assertEqual(data["calls"][0]["tool"], "read_file")

    def test_analysis_tool_loop_degrades_when_the_endpoint_rejects_tools(self) -> None:
        class _NoToolsLlm:
            available = True
            config_name = "no-tools"
            model = "m"
            base_url = ""
            api_mode = "chat"
            thinking = False
            reasoning_effort = None
            thinking_body = None

            def __init__(self, text: str) -> None:
                self.text = text

            def complete(self, system, user, *, fallback, response_format=None):
                return _FakeResult(self.text)

            def complete_with_tools(self, *args, **kwargs):
                raise RuntimeError("tools not supported")

        state = _base_state(self.run_id, workload_spec={"name": "x"})
        with mock.patch.object(graph, "_llm", lambda state: _NoToolsLlm(self.ANALYSIS_YAML)):
            result = graph.analyze_workload(state)
        # The step degrades to a plain completion and the failure is auditable.
        self.assertEqual(result["tool_calls"], [])
        self.assertIn("hot_function: depthconv", result["workload_analysis"])
        log = self.run_dir / "07_analysis/analysis_tool_calls.yaml"
        self.assertTrue(log.is_file())
        self.assertIn("tool loop failed", log.read_text())


class SoftwareRetryBreakerTest(unittest.TestCase):
    """A rejected response leaves the workspace unchanged; the retry must not
    be allowed to repeat the identical answer until the budget runs out."""

    def setUp(self) -> None:
        self.run_id = "gate-sw-retry-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def _state(self) -> dict:
        return _base_state(
            self.run_id,
            workload_spec={"name": "depthconv", "benchmark_name": "depthconv"},
            benchmark_project={"benchmark": "depthconv", "kind": "embench_benchmark"},
            isa_spec={},
        )

    def test_directive_fires_only_for_identical_consecutive_attempts(self) -> None:
        identical = {"sw_attempts": [{"response_hash": "a", "error_hash": "b"}] * 2}
        self.assertIn("REPEATED IDENTICAL ATTEMPT", graph.software_stuck_directive(identical))
        self.assertEqual(graph.software_stuck_directive({"sw_attempts": []}), "")
        self.assertEqual(
            graph.software_stuck_directive(
                {"sw_attempts": [{"response_hash": "a", "error_hash": "b"}]}
            ),
            "",
        )
        changed = {
            "sw_attempts": [
                {"response_hash": "a", "error_hash": "b"},
                {"response_hash": "c", "error_hash": "b"},
            ]
        }
        self.assertEqual(graph.software_stuck_directive(changed), "")

    def test_repeated_identical_answers_get_a_strategy_directive(self) -> None:
        class _RecordingLlm(_FakeLlm):
            def __init__(self, text: str) -> None:
                super().__init__(text)
                self.prompts: list[str] = []

            def complete(self, system, user, *, fallback, response_format=None):
                self.prompts.append(user)
                return _FakeResult(self.text)

        llm = _RecordingLlm("just a plan; no delivery block")
        state = self._state()
        with mock.patch.object(graph, "_llm", lambda s: llm):
            first = graph.implement_sw(state)
        self.assertEqual(len(first["sw_attempts"]), 1)
        gate = first["gates"]["implementation_sw"]
        self.assertTrue(gate["response_hash"])
        self.assertTrue(gate["error_hash"])
        self.assertNotIn("REPEATED IDENTICAL ATTEMPT", llm.prompts[0])

        state2 = {**state, **first}
        with mock.patch.object(graph, "_llm", lambda s: llm):
            second = graph.implement_sw(state2)
        # Only one prior attempt exists, so the second try is not yet flagged.
        self.assertNotIn("REPEATED IDENTICAL ATTEMPT", llm.prompts[1])

        state3 = {
            **state2,
            **second,
            "sw_attempts": [*first["sw_attempts"], *second["sw_attempts"]],
        }
        with mock.patch.object(graph, "_llm", lambda s: llm):
            graph.implement_sw(state3)
        self.assertIn("REPEATED IDENTICAL ATTEMPT", llm.prompts[2])


class SimAttributionTest(unittest.TestCase):
    """A failed simulation must say why, not just "reached the fail symbol"."""

    def test_verify_failure_is_attributed(self) -> None:
        from agents.cfu_design_agent.runtime.tools import parse_command_output

        log = (
            "CFU_AGENT_PROFILE depthconv=120 calls=1\n"
            "EMBENCH_VERIFY_FAIL benchmark=depthconv verify=0 result=0\n"
            "SimFailure: Software reached the fail symbol :(\n"
        )
        results = [
            {"returncode": 1, "parsed": parse_command_output(log), "stdout_tail": log, "stderr_tail": ""}
        ]
        text = graph.sim_attribution(results, log)
        self.assertIn("verify failed for depthconv", text)
        self.assertIn("verify_benchmark returned 0", text)
        self.assertIn("wrong-result", text)

    def test_fail_symbol_without_the_diagnostic_is_still_attributed(self) -> None:
        log = "SimFailure: Software reached the fail symbol :("
        results = [{"returncode": 1, "stdout_tail": log, "stderr_tail": ""}]
        self.assertIn("fail symbol", graph.sim_attribution(results, log))

    def test_timeout_is_attributed_to_a_hang(self) -> None:
        results = [{"returncode": -9, "timed_out": True, "stdout_tail": "", "stderr_tail": ""}]
        self.assertIn("timed out", graph.sim_attribution(results, ""))

    def test_passing_sim_with_a_speedup_failure_has_no_attribution(self) -> None:
        # Regression: verify PASS + rc 0 means correctness held; the remaining
        # failure is the speedup gate, which already reports itself. The old code
        # claimed "ended without reaching the pass or fail symbol".
        from agents.cfu_design_agent.runtime.tools import parse_command_output

        log = (
            "[info] CFU_AGENT_PROFILE depthconv=28912051\n"
            "[info] EMBENCH_VERIFY_PASS benchmark=depthconv verify=1 result=0\n"
            "[success] Total time: 173 s\n"
        )
        results = [
            {"returncode": 0, "parsed": parse_command_output(log), "stdout_tail": log, "stderr_tail": ""}
        ]
        self.assertEqual(graph.sim_attribution(results, log), "")

    def test_sim_feedback_puts_the_attribution_with_the_log(self) -> None:
        state = {
            "gates": {"sim": {"attribution": "ATTRIBUTION_TEXT"}},
            "validation_results": [],
        }
        text = graph.sim_feedback(state)
        self.assertIn("ATTRIBUTION_TEXT", text)
        self.assertIn("sim (simulation) errors", text)

    def test_sim_feedback_carries_the_agent_routing_diagnosis(self) -> None:
        state = {
            "gates": {
                "sim": {
                    "attribution": "ATTRIBUTION_TEXT",
                    "repair_decision": {
                        "target": "implement_hw",
                        "root_cause": "the CFU never returns a response",
                        "source": "llm",
                    },
                }
            },
            "validation_results": [],
        }
        text = graph.sim_feedback(state)
        self.assertIn("assigned this failure to implement_hw", text)
        self.assertIn("never returns a response", text)
        # A deterministic decision is bookkeeping, not a diagnosis to inject.
        fallback = {
            "gates": {
                "sim": {
                    "repair_decision": {"target": "implement_sw", "source": "fallback", "reason": "x"}
                }
            },
            "validation_results": [],
        }
        self.assertNotIn("assigned this failure", graph.sim_feedback(fallback))


class RepairRoutingTest(unittest.TestCase):
    """A failed simulation is routed by the agent, not by a fixed software first."""

    def setUp(self) -> None:
        self.run_id = "gate-repair-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def _state(self, **overrides) -> dict:
        fields = {
            "workload_spec": {"name": "depthconv", "benchmark_name": "depthconv"},
            "benchmark_project": {"benchmark": "depthconv", "kind": "embench_benchmark"},
            "isa_spec": {},
            "repair_routing": "auto",
        }
        fields.update(overrides)
        return _base_state(self.run_id, **fields)

    def _choose(self, state: dict, text: str) -> tuple[str, dict, str]:
        with mock.patch.object(graph, "_llm", lambda s: _UsedLlm(text)):
            return graph._choose_repair_target(
                state, attribution="verify failed for depthconv", sim_text="log", results=[]
            )

    def test_the_llm_may_route_the_repair_to_hardware(self) -> None:
        target, decision, raw = self._choose(
            self._state(),
            '{"repair_target": "implement_hw", "root_cause": "CFU returns the wrong lane", '
            '"reason": "the datapath signedness is wrong"}',
        )
        self.assertEqual(target, "implement_hw")
        self.assertEqual(decision["source"], "llm")
        self.assertIn("wrong lane", decision["root_cause"])
        self.assertTrue(raw)
        self.assertTrue((REPO_ROOT / raw).exists())

    def test_the_llm_may_route_the_repair_to_software(self) -> None:
        target, decision, raw = self._choose(
            self._state(),
            '{"repair_target": "implement_sw", "root_cause": "packing is wrong", '
            '"reason": "the kernel offsets are wrong"}',
        )
        self.assertEqual(target, "implement_sw")
        self.assertEqual(decision["source"], "llm")
        self.assertTrue(raw)

    def test_an_unusable_answer_falls_back_to_the_escalation(self) -> None:
        target, decision, _ = self._choose(self._state(), '{"repair_target": "implement_ql"}')
        self.assertEqual(target, "implement_sw")
        self.assertEqual(decision["source"], "fallback")
        # With one simulation repair already spent the escalation moves on to the
        # hardware, and the router must record that as its advisory.
        target, decision, _ = self._choose(
            self._state(attempts={"sim": 1}), "not even json"
        )
        self.assertEqual(target, "implement_hw")
        self.assertEqual(decision["advisory"], "implement_hw")

    def test_a_broken_router_never_breaks_the_ladder(self) -> None:
        def _boom(*args, **kwargs):
            raise RuntimeError("gateway exploded")

        with mock.patch.object(graph, "_complete_llm", _boom):
            target, decision, raw = graph._choose_repair_target(
                self._state(attempts={"sim": 1}), attribution="", sim_text="", results=[]
            )
        self.assertEqual(target, "implement_hw")
        self.assertEqual(decision["source"], "fallback")
        self.assertIn("gateway exploded", decision["reason"])
        self.assertEqual(raw, "")

    def test_fixed_mode_never_calls_the_llm(self) -> None:
        def _boom(_state):  # pragma: no cover - must not run
            raise AssertionError("fixed mode must not consult the model")

        with mock.patch.object(graph, "_llm", _boom):
            target, decision, raw = graph._choose_repair_target(
                self._state(repair_routing="fixed"), attribution="", sim_text="", results=[]
            )
        self.assertEqual(target, "implement_sw")
        self.assertEqual(decision["source"], "fixed")
        self.assertEqual(raw, "")

    def test_simulate_records_and_follows_the_agent_decision(self) -> None:
        from agents.cfu_design_agent.runtime.tools import parse_command_output

        log = (
            "EMBENCH_VERIFY_FAIL benchmark=depthconv verify=0 result=0\n"
            "SimFailure: Software reached the fail symbol :(\n"
        )
        state = self._state(
            run_commands=True,
            dry_run=False,
            max_iters=2,
            embench_profiles={"reference": {"depthconv": {"cycles": 1000}}},
            changed_design_files=["src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"],
            gates={},
        )
        failure = {
            "command": "fake-sim",
            "returncode": 1,
            "parsed": parse_command_output(log),
            "stdout_tail": log,
            "stderr_tail": "",
        }
        with mock.patch.object(graph, "_toolbox_for_stage", lambda s, stage: object()), \
                mock.patch.object(graph, "_software_sim_commands", lambda s, tb: [["fake-sim"]]), \
                mock.patch.object(graph, "_run_check", lambda *a, **k: failure), \
                mock.patch.object(graph, "_llm", lambda s: _UsedLlm(
                    '{"repair_target": "implement_hw", "root_cause": "missing response", '
                    '"reason": "the transaction never completes"}'
                )):
            update = graph.simulate(state)
        gate = update["gates"]["sim"]
        self.assertFalse(gate["ok"])
        self.assertEqual(gate["repair_decision"]["target"], "implement_hw")
        self.assertEqual(gate["next"], "implement_hw")
        self.assertEqual(update["stage_jobs"], "implement_hw")
        self.assertTrue(
            (self.run_dir / "10_validation/sim_repair_decision.yaml").exists()
        )
        self.assertTrue(
            any("Repair routing chose implement_hw" in line for line in update["iteration_history"])
        )

    def test_simulate_stays_deterministic_in_fixed_mode(self) -> None:
        from agents.cfu_design_agent.runtime.tools import parse_command_output

        log = "EMBENCH_VERIFY_FAIL benchmark=depthconv verify=0 result=0\n"
        state = self._state(
            run_commands=True,
            dry_run=False,
            max_iters=2,
            repair_routing="fixed",
            embench_profiles={"reference": {"depthconv": {"cycles": 1000}}},
            changed_design_files=["src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala"],
            gates={},
        )
        failure = {
            "command": "fake-sim",
            "returncode": 1,
            "parsed": parse_command_output(log),
            "stdout_tail": log,
            "stderr_tail": "",
        }

        def _boom(_state):  # pragma: no cover - must not run
            raise AssertionError("fixed mode must not consult the model")

        with mock.patch.object(graph, "_toolbox_for_stage", lambda s, stage: object()), \
                mock.patch.object(graph, "_software_sim_commands", lambda s, tb: [["fake-sim"]]), \
                mock.patch.object(graph, "_run_check", lambda *a, **k: failure), \
                mock.patch.object(graph, "_llm", _boom):
            update = graph.simulate(state)
        gate = update["gates"]["sim"]
        self.assertEqual(gate["repair_decision"]["source"], "fixed")
        self.assertEqual(gate["next"], "implement_sw")


class SoftwareChangeRoutingTest(unittest.TestCase):
    """After the hardware is built the agent may keep a kernel that already drives it."""

    ISA = {
        "spec_version": "1",
        "identity": {"name": "depthconv_cfu", "opcode": "custom0 (0x0B)", "function_id_width": 3},
        "commands": [
            {
                "function_id": 1,
                "name": "compute",
                "operand": "a0,a1",
                "result": "a0",
                "semantics": "depthwise convolution step",
                "response": "value",
            }
        ],
    }
    SCALAR_KERNEL = "int kernel(void) { return 7; }\n"
    CFU_KERNEL = "int kernel(void) { return cfu_compute(1, 2); }\n"

    def setUp(self) -> None:
        self.run_id = "gate-swroute-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def _kernel_path(self) -> "Path":
        from pathlib import Path

        return (
            REPO_ROOT
            / f"agents/generated/runs/{self.run_id}/workspace/embench_depthconv/cfu/src/depthconv.c"
        )

    def _write_kernel(self, body: str) -> None:
        path = self._kernel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def _state(self, **overrides) -> dict:
        fields = {
            "workload_spec": {"name": "depthconv", "benchmark_name": "depthconv"},
            "benchmark_project": {"benchmark": "depthconv", "kind": "embench_benchmark"},
            "isa_spec": self.ISA,
            "software_change": "auto",
            "run_commands": True,
            "dry_run": False,
            "gates": {},
        }
        fields.update(overrides)
        return _base_state(self.run_id, **fields)

    def _choose(self, state: dict, text: str) -> tuple[bool, dict, str]:
        with mock.patch.object(graph, "_llm", lambda s: _UsedLlm(text)):
            return graph._choose_software_change(
                state, graph._toolbox_for_stage(state, "implementation")
            )

    def test_a_kernel_without_a_cfu_call_forces_the_software_rung(self) -> None:
        self._write_kernel(self.SCALAR_KERNEL)

        def _boom(_state):  # pragma: no cover - the guard must answer first
            raise AssertionError("a kernel that does not drive the CFU needs no router call")

        with mock.patch.object(graph, "_llm", _boom):
            reuse, decision, raw = graph._choose_software_change(
                self._state(), graph._toolbox_for_stage(self._state(), "implementation")
            )
        self.assertFalse(reuse)
        self.assertEqual(decision["source"], "guard")
        self.assertIn("does not call any CFU datapath intrinsic", decision["reason"])
        self.assertEqual(raw, "")

    def test_the_agent_may_reuse_a_kernel_that_already_drives_the_cfu(self) -> None:
        self._write_kernel(self.CFU_KERNEL)
        reuse, decision, raw = self._choose(
            self._state(), '{"software_change": "not_required", "reason": "the ABI is unchanged"}'
        )
        self.assertTrue(reuse)
        self.assertEqual(decision["source"], "llm")
        self.assertEqual(decision["software_change"], "not_required")
        self.assertTrue(decision["calls"])
        self.assertEqual(decision["calls"][0]["name"], "cfu_compute")
        self.assertTrue(raw)

    def test_required_answer_keeps_the_software_rung(self) -> None:
        self._write_kernel(self.CFU_KERNEL)
        reuse, decision, raw = self._choose(
            self._state(), '{"software_change": "required", "reason": "the kernel must call the new command"}'
        )
        self.assertFalse(reuse)
        self.assertEqual(decision["source"], "llm")
        self.assertEqual(decision["software_change"], "required")
        self.assertTrue(raw)

    def test_an_unusable_answer_falls_back_to_the_software_rung(self) -> None:
        self._write_kernel(self.CFU_KERNEL)
        reuse, decision, _ = self._choose(self._state(), "sure, keep it")
        self.assertFalse(reuse)
        self.assertEqual(decision["source"], "fallback")
        self.assertIn("usable software_change", decision["reason"])

    def test_always_pins_the_software_rung(self) -> None:
        self._write_kernel(self.CFU_KERNEL)

        def _boom(_state):  # pragma: no cover - `always` never consults the model
            raise AssertionError("--software-change always must not call the router")

        state = self._state(software_change="always")
        with mock.patch.object(graph, "_llm", _boom):
            reuse, decision, raw = graph._choose_software_change(
                state, graph._toolbox_for_stage(state, "implementation")
            )
        self.assertFalse(reuse)
        self.assertEqual(decision["source"], "fixed")
        self.assertEqual(raw, "")

    def test_a_changed_isa_contract_forces_the_software_rung(self) -> None:
        self._write_kernel(self.CFU_KERNEL)
        state = self._state(
            gates={"implementation_sw": {"ok": True, "applied": True, "isa_hash": "0000000000000000"}}
        )

        def _boom(_state):  # pragma: no cover - a moved ABI is not a judgement call
            raise AssertionError("a changed ISA contract must not be routed by the model")

        with mock.patch.object(graph, "_llm", _boom):
            reuse, decision, _ = graph._choose_software_change(
                state, graph._toolbox_for_stage(state, "implementation")
            )
        self.assertFalse(reuse)
        self.assertEqual(decision["source"], "guard")
        self.assertIn("ISA contract changed", decision["reason"])

    def test_a_matching_isa_digest_allows_reuse(self) -> None:
        self._write_kernel(self.CFU_KERNEL)
        state = self._state()
        current = graph._isa_digest(state)
        state = self._state(
            gates={"implementation_sw": {"ok": True, "applied": True, "isa_hash": current}}
        )
        reuse, decision, _ = self._choose(
            state, '{"software_change": "not_required", "reason": "same contract"}'
        )
        self.assertTrue(reuse)
        self.assertEqual(decision["source"], "llm")

    def test_harness_counter_calls_are_not_datapath_evidence(self) -> None:
        # The generated benchmark wrapper always calls cfu_agent_counter_read();
        # that reports cycles and must not make an untouched kernel look
        # accelerated.
        self._write_kernel(
            '#include "cfu_agent_counter.h"\n'
            "int kernel(void) { cfu_agent_counter_read(); return 7; }\n"
        )
        state = self._state()
        toolbox = graph._toolbox_for_stage(state, "implementation")
        targets = graph._software_targets(state, toolbox)
        self.assertFalse(graph._software_exercises_cfu(state, toolbox, targets)[0])
        self.assertFalse(graph._software_reuse_evidence(state, toolbox)[0])

    def test_control_intrinsic_named_like_a_helper_does_not_count(self) -> None:
        self._write_kernel("int kernel(void) { return cfu_enable(); }\n")
        isa = {
            **self.ISA,
            "commands": [
                {
                    "function_id": 2,
                    "name": "enable",
                    "operand": "a0",
                    "result": "a0",
                    "semantics": "declared by this contract",
                    "response": "value",
                }
            ],
        }
        state = self._state(isa_spec=isa)
        toolbox = graph._toolbox_for_stage(state, "implementation")
        targets = graph._software_targets(state, toolbox)
        self.assertFalse(graph._software_exercises_cfu(state, toolbox, targets)[0])

    def test_a_broken_router_never_breaks_the_ladder(self) -> None:
        self._write_kernel(self.CFU_KERNEL)

        def _boom(*args, **kwargs):
            raise RuntimeError("gateway exploded")

        with mock.patch.object(graph, "_complete_llm", _boom):
            reuse, decision, raw = graph._choose_software_change(
                self._state(), graph._toolbox_for_stage(self._state(), "implementation")
            )
        self.assertFalse(reuse)
        self.assertEqual(decision["source"], "fallback")
        self.assertIn("gateway exploded", decision["reason"])
        self.assertEqual(raw, "")

    def _generate(self, state: dict, decision: tuple) -> dict:
        with mock.patch.object(graph, "_missing_design_files", lambda s: []), \
                mock.patch.object(graph, "agent_cfu_overlay_commands", lambda s, isa: [["fake-gen"]]), \
                mock.patch.object(graph, "_run_check", lambda *a, **k: {"returncode": 0, "command": "fake-gen"}), \
                mock.patch.object(graph, "_soc_rtl_path", lambda s: None), \
                mock.patch.object(graph, "_choose_software_change", lambda s, tb: decision), \
                mock.patch.object(graph, "_write_intrinsics_header", lambda s, tb: "kernel/include/cfu_intrinsics.h"):
            return graph.generate_soc(state)

    def test_generate_soc_skips_to_the_software_build_when_reusing(self) -> None:
        self._write_kernel(self.CFU_KERNEL)
        decision = {
            "software_change": "not_required",
            "reason": "the kernel already calls cfu_compute",
            "source": "llm",
            "evidence": "the current kernel already calls cfu_compute",
            "files": [f"agents/generated/runs/{self.run_id}/workspace/embench_depthconv/cfu/src/depthconv.c"],
            "calls": [{"file": "x.c", "line": 1, "name": "cfu_compute", "text": "cfu_compute(1, 2);"}],
        }
        update = self._generate(self._state(), (True, decision, "09_implementation/llm_raw/software_route_000.txt"))
        gate = update["gates"]["hw_build"]
        self.assertTrue(gate["ok"])
        # The conformance micro-test runs before the software rung; the reuse decision
        # is preserved and applied once conformance passes.
        self.assertEqual(gate["next"], "check_conformance")
        self.assertEqual(
            graph._route_after_hw_conformance({"gates": update["gates"]}), "build_sw"
        )
        self.assertEqual(gate["software_decision"]["software_change"], "not_required")
        sw_gate = update["gates"]["implementation_sw"]
        self.assertTrue(sw_gate["ok"])
        self.assertTrue(sw_gate["reused"])
        self.assertFalse(sw_gate["skipped"])
        self.assertTrue(sw_gate["intrinsics_header"])
        self.assertIn("reused unchanged", update["implementation_sw_summary"])
        self.assertEqual(update["llm_raw_outputs"], ["09_implementation/llm_raw/software_route_000.txt"])
        history = "\n".join(update["iteration_history"])
        self.assertIn("Generated the SoC from the agent design.", history)
        self.assertIn("Software reused unchanged", history)
        self.assertTrue((self.run_dir / "09_implementation/software_decision.yaml").exists())

    def test_generate_soc_runs_the_software_rung_when_required(self) -> None:
        decision = {
            "software_change": "required",
            "reason": "the kernel must be updated",
            "source": "llm",
            "evidence": "the current kernel already calls cfu_compute",
            "files": [],
            "calls": [],
        }
        update = self._generate(self._state(), (False, decision, ""))
        self.assertEqual(update["gates"]["hw_build"]["next"], "check_conformance")
        self.assertNotIn("implementation_sw", update["gates"])
        self.assertNotIn("implementation_sw_summary", update)

    def test_build_sw_compiles_the_reused_kernel(self) -> None:
        state = self._state(
            gates={"implementation_sw": {"ok": True, "reused": True, "files": ["k.c"]}},
        )
        ran: list[list[str]] = []

        def _run(*args, **kwargs):
            ran.append(list(args[2]))
            return {"returncode": 0, "command": "fake-make"}

        with mock.patch.object(graph, "_toolbox_for_stage", lambda s, stage: object()), \
                mock.patch.object(graph, "_software_build_commands", lambda s, tb: [["fake-make"]]), \
                mock.patch.object(graph, "_run_check", _run), \
                mock.patch.object(graph, "_software_elf", lambda s: ""):
            update = graph.build_sw(state)
        self.assertTrue(update["gates"]["sw_build"]["ok"])
        self.assertTrue(update["gates"]["sw_build"]["software_reused"])
        self.assertEqual(ran, [["fake-make"]])
        self.assertIn("reused software unchanged", "\n".join(update["iteration_history"]))
        self.assertNotIn(
            "did not apply a change", "\n".join(update["errors"])
        )

    def test_review_still_reports_complete_with_a_reused_software_gate(self) -> None:
        state = self._state(
            gates={
                "workload_spec": {"ok": True},
                "isa": {"ok": True},
                "implementation_hw": {"ok": True},
                "hw_build": {"ok": True},
                "hw_conformance": {"ok": True},
                "implementation_sw": {"ok": True, "reused": True},
                "sw_build": {"ok": True, "software_reused": True},
                "sim": {"ok": True},
                "cost": {"ok": True},
            }
        )
        self.assertEqual(graph._review_decision(state), ("final_report", "complete"))


class LoopOptimizationTest(unittest.TestCase):
    """--loop spends leftover --max-iters budget on deliberate improvements."""

    def setUp(self) -> None:
        self.run_id = "gate-loop-00000000"
        self.run_dir = REPO_ROOT / f"agents/generated/runs/{self.run_id}"
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.run_dir, True)

    def _gates(self, **overrides) -> dict:
        gates = {
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "implementation_hw": {"ok": True},
            "hw_build": {"ok": True},
            "hw_conformance": {"ok": True},
            "implementation_sw": {"ok": True},
            "sw_build": {"ok": True},
            "sim": {"ok": True, "speedup_x100": 200, "reference_cycles": 100, "cfu_cycles": 50},
            "cost": {"ok": True},
        }
        gates.update(overrides)
        return gates

    def _state(self, **overrides) -> dict:
        fields = {
            "loop_optimize": True,
            "max_iters": 3,
            "gates": self._gates(),
            "cost_results": [
                {
                    "returncode": 0,
                    "parsed": {
                        "cost": {
                            "top": "AgentCfu",
                            "cells": 1200,
                            "wire_bits": 10,
                            "memories": 0,
                            "memory_bits": 0,
                        }
                    },
                }
            ],
        }
        fields.update(overrides)
        return _base_state(self.run_id, **fields)

    def test_a_round_only_counts_as_improved_when_it_is_pareto_better(self) -> None:
        base = {"speedup_x100": 200, "cost": {"cells": 1200}}
        self.assertTrue(graph._round_improved({"speedup_x100": 250, "cost": {"cells": 1200}}, base))
        self.assertTrue(graph._round_improved({"speedup_x100": 200, "cost": {"cells": 900}}, base))
        # Slower, or faster but bigger: neither is an improvement.
        self.assertFalse(graph._round_improved({"speedup_x100": 150, "cost": {"cells": 1200}}, base))
        self.assertFalse(graph._round_improved({"speedup_x100": 250, "cost": {"cells": 1500}}, base))
        self.assertFalse(graph._round_improved({"speedup_x100": 200, "cost": {"cells": 1200}}, base))
        self.assertTrue(graph._round_improved({"speedup_x100": 200, "cost": {"cells": 900}}, None))

    def _choose(self, state: dict, text: str) -> tuple[str, dict, str]:
        with mock.patch.object(graph, "_llm", lambda s: _UsedLlm(text)):
            return graph._choose_optimization(state, state.get("optimization_rounds") or [])

    def test_the_agent_may_start_another_round(self) -> None:
        target, decision, raw = self._choose(
            self._state(),
            '{"continue": true, "target": "implement_hw", "focus": "cost", '
            '"reason": "replace the vector RF with a streamed datapath"}',
        )
        self.assertEqual(target, "implement_hw")
        self.assertEqual(decision["source"], "llm")
        self.assertEqual(decision["focus"], "cost")
        self.assertTrue(raw)

    def test_the_agent_may_stop_in_conservative_mode(self) -> None:
        target, decision, _ = self._choose(
            self._state(loop_policy="conservative"), '{"continue": false, "reason": "no concrete change left"}'
        )
        self.assertEqual(target, "")
        self.assertFalse(decision["continue"])
        self.assertEqual(decision["source"], "llm")

    def test_spending_budget_requires_a_concrete_change(self) -> None:
        target, decision, _ = self._choose(
            self._state(), '{"continue": true, "target": "implement_hw", "reason": ""}'
        )
        self.assertEqual(target, "")
        self.assertIn("concrete change", decision["reason"])

    def test_an_unusable_target_stops_the_loop(self) -> None:
        target, decision, _ = self._choose(
            self._state(),
            '{"continue": true, "target": "implement_ql", "reason": "try something"}',
        )
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "fallback")
        self.assertIn("unusable target", decision["reason"])

    def test_a_model_that_did_not_answer_cannot_spend_budget(self) -> None:
        # The deterministic fallback is "stop", and it is not a model decision.
        target, decision, _ = self._choose(self._state(), "not json at all")
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "fallback")

    def test_a_broken_router_never_extends_the_run(self) -> None:
        def _boom(*args, **kwargs):
            raise RuntimeError("gateway exploded")

        with mock.patch.object(graph, "_complete_llm", _boom):
            target, decision, raw = graph._choose_optimization(self._state(), [])
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "fallback")
        self.assertIn("gateway exploded", decision["reason"])
        self.assertEqual(raw, "")

    def test_the_loop_is_off_without_the_flag(self) -> None:
        def _boom(_state):  # pragma: no cover - --loop is off
            raise AssertionError("the optimizer must not be consulted without --loop")

        with mock.patch.object(graph, "_llm", _boom):
            target, decision, _ = graph._choose_optimization(
                self._state(loop_optimize=False), []
            )
        self.assertEqual((target, decision), ("", {}))

    def test_no_budget_means_no_round(self) -> None:
        target, decision, _ = self._choose(self._state(max_iters=1), "{}")
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "budget")

    def test_review_records_the_baseline_round_even_without_the_loop(self) -> None:
        update = graph.review_and_route(self._state(loop_optimize=False))
        rounds = update["optimization_rounds"]
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["target"], "baseline")
        self.assertTrue(rounds[0]["improved"])
        self.assertEqual(rounds[0]["speedup_x100"], 200)
        self.assertEqual(rounds[0]["cost"]["cells"], 1200)
        self.assertEqual(update["gates"]["review"]["next"], "final_report")
        self.assertEqual(update["optimization_plan"], {})
        self.assertFalse((self.run_dir / "12_review/optimization_rounds.yaml").exists())

    def test_review_starts_a_round_and_spends_budget(self) -> None:
        state = self._state()
        with mock.patch.object(graph, "_llm", lambda s: _UsedLlm(
            '{"continue": true, "target": "implement_sw", "focus": "cost", '
            '"reason": "block the loop to cut scalar traffic"}'
        )):
            update = graph.review_and_route(state)
        self.assertEqual(update["gates"]["review"]["next"], "implement_sw")
        self.assertEqual(update["gates"]["review"]["decision"], "optimize:implement_sw")
        self.assertEqual(update["status"], "complete")
        self.assertEqual(update["repairs_used"], 1)
        self.assertEqual(update["iteration"], 1)
        self.assertEqual(update["attempts"]["optimize"], 1)
        self.assertEqual(update["stage_jobs"], "implement_sw")
        plan = update["optimization_plan"]
        self.assertEqual((plan["round"], plan["target"], plan["focus"]), (1, "implement_sw", "cost"))
        self.assertIn("Started optimization round 1", "\n".join(update["iteration_history"]))
        self.assertEqual(len(update["optimization_rounds"]), 1)
        self.assertTrue((self.run_dir / "12_review/optimization_rounds.yaml").is_file())
        self.assertTrue((self.run_dir / "12_review/optimization_decision.yaml").is_file())
        raw = update["llm_raw_outputs"]
        self.assertEqual(len(raw), 1)
        self.assertTrue((REPO_ROOT / raw[0]).is_file())

    def test_review_stops_at_the_budget_and_records_why(self) -> None:
        update = graph.review_and_route(self._state(max_iters=1))
        self.assertEqual(update["gates"]["review"]["next"], "final_report")
        self.assertEqual(update["gates"]["review"]["optimization"]["source"], "budget")
        self.assertEqual(update["optimization_plan"], {})
        self.assertNotIn("repairs_used", update)

    def test_a_failed_run_never_starts_an_optimization_round(self) -> None:
        def _boom(_state):  # pragma: no cover - only a complete run may loop
            raise AssertionError("an incomplete run must not be optimized")

        state = self._state(gates=self._gates(sim={"ok": False, "skipped": False}))
        with mock.patch.object(graph, "_llm", _boom):
            update = graph.review_and_route(state)
        self.assertEqual(update["status"], "failed")
        self.assertEqual(update["gates"]["review"]["next"], "final_report")

    def test_the_route_reads_the_review_gate(self) -> None:
        self.assertEqual(
            graph._route_after_review({"gates": {"review": {"next": "implement_hw"}}}),
            "implement_hw",
        )
        self.assertEqual(graph._route_after_review({"gates": {}}), "final_report")

    def test_the_directive_tells_the_rung_what_to_optimize(self) -> None:
        state = self._state(
            optimization_plan={
                "round": 2,
                "target": "implement_hw",
                "focus": "cost",
                "reason": "shrink the RF",
            }
        )
        text = graph.optimization_directive(state)
        self.assertIn("reduce the synthesized CFU cost", text)
        self.assertIn("shrink the RF", text)
        self.assertIn("ISA contract is frozen", text)
        self.assertIn("2.00x", text)
        self.assertEqual(graph.optimization_directive(self._state()), "")

    def test_the_optimization_prompt_is_not_a_repair_prompt(self) -> None:
        from agents.cfu_design_agent.prompts import hardware_prompt, software_prompt

        optimization = graph.optimization_directive(
            self._state(
                optimization_plan={
                    "round": 1,
                    "target": "implement_hw",
                    "focus": "performance",
                    "reason": "widen",
                }
            )
        )
        hw = hardware_prompt("t", {"name": "w"}, "", "", {}, optimization=optimization)
        self.assertIn("=== OPTIMIZATION ROUND ===", hw)
        self.assertNotIn("HARDWARE REPAIR MODE", hw)
        sw = software_prompt("t", {"name": "w"}, "", "", {}, optimization=optimization)
        self.assertIn("=== OPTIMIZATION ROUND ===", sw)
        self.assertNotIn("SOFTWARE REPAIR MODE", sw)
        self.assertNotIn("first software attempt", sw)

    def test_the_report_lists_the_rounds_and_flags_a_regression(self) -> None:
        rounds = [
            {"round": 0, "target": "baseline", "improved": True, "measured": True,
             "speedup_x100": 200, "cost": {"cells": 1200}, "reason": "initial ladder"},
            {"round": 1, "target": "implement_hw", "improved": True, "measured": True,
             "speedup_x100": 250, "cost": {"cells": 1300}, "reason": "widen the datapath"},
            {"round": 2, "target": "implement_sw", "improved": False, "measured": True,
             "speedup_x100": 210, "cost": {"cells": 1300}, "reason": "block the loop"},
        ]
        report = graph.render_report(self._state(optimization_rounds=rounds))
        self.assertIn("## Optimization Rounds", report)
        self.assertIn("best round 1, final round 2", report)
        self.assertIn("WARNING: the final round is not the best round", report)
        self.assertIn("speedup=2.50x cells=1300 (+0.50x, +100 cells)", report)
        self.assertIn("speedup=2.10x cells=1300 (-0.40x, +0 cells)", report)
        self.assertIn("(best)", report)

    def test_a_round_without_its_own_measurement_reports_not_measured(self) -> None:
        rounds = [
            {"round": 0, "target": "baseline", "improved": True, "measured": True,
             "speedup_x100": 200, "cost": {"cells": 1200}, "reason": "initial ladder"},
            {"round": 1, "target": "implement_hw", "improved": False, "measured": False,
             "speedup_x100": 0, "cost": {}, "reason": "the SoC build failed"},
        ]
        report = graph.render_report(self._state(optimization_rounds=rounds))
        # The failed round must not repeat the baseline numbers as its own.
        self.assertIn("round 1: implement_hw not measured", report)
        self.assertNotIn("speedup=2.00x cells=1200 improved=False", report)
        self.assertFalse(graph._round_improved(rounds[1], rounds[0]))

    def test_the_graph_step_budget_grows_with_the_iteration_budget(self) -> None:
        # The loop makes the ladder a cycle, so LangGraph's default step limit
        # (25) is not enough once several rounds run.
        self.assertGreater(graph.recursion_limit(1), 25)
        self.assertGreater(graph.recursion_limit(5), graph.recursion_limit(2))
        self.assertEqual(
            graph.graph_config("t", max_iters=4)["recursion_limit"],
            graph.recursion_limit(4),
        )

    def test_the_report_has_no_section_without_rounds(self) -> None:
        self.assertNotIn("## Optimization Rounds", graph.render_report(self._state()))


class _FakeTty(io.StringIO):
    def isatty(self) -> bool:  # noqa: D102
        return True


class StreamLogTest(unittest.TestCase):
    """Live streaming progress is one in-place line, not a flood of lines."""

    def setUp(self) -> None:
        graph.enable_progress(["analysis", "implementation"])
        graph._LIVE_LINE.clear()
        self.addCleanup(graph.disable_progress)

    def _capture(self, payload: dict, *, tty: bool = True) -> str:
        stream = _FakeTty() if tty else io.StringIO()
        with mock.patch.object(sys, "stdout", stream):
            graph._log_stream_progress(payload)
        return stream.getvalue()

    def test_tty_updates_rewrite_one_line_without_newlines(self) -> None:
        self.assertEqual(self._capture({"phase": "start"}), "")
        first = self._capture(
            {"phase": "content", "stage": "analyze_workload", "chars": 100}
        )
        second = self._capture(
            {"phase": "content", "stage": "analyze_workload", "chars": 200}
        )
        self.assertIn("\r", first)
        self.assertNotIn("\n", first + second)
        self.assertIn("analyze workload", first)
        self.assertIn("200 chars", second)
        # Clearing erases the line so the next normal print starts clean.
        stream = _FakeTty()
        with mock.patch.object(sys, "stdout", stream):
            graph._LIVE_LINE.clear()
        self.assertIn("\r", stream.getvalue())

    def test_no_intermediate_output_when_stdout_is_redirected(self) -> None:
        self.assertEqual(
            self._capture({"phase": "content", "stage": "analyze_workload", "chars": 100}, tty=False),
            "",
        )

    def test_no_repeat_when_nothing_advanced(self) -> None:
        # The reported bug: a long thinking phase repeated "0 chars".
        self.assertEqual(self._capture({"phase": "start"}), "")
        for _ in range(5):
            self.assertEqual(
                self._capture(
                    {
                        "phase": "content",
                        "stage": "analyze_workload",
                        "chars": 0,
                        "thinking_chars": 0,
                    }
                ),
                "",
            )

    def test_thinking_growth_is_reported_as_thinking(self) -> None:
        self._capture({"phase": "start"})
        printed = self._capture(
            {
                "phase": "content",
                "stage": "analyze_workload",
                "chars": 0,
                "thinking_chars": 400,
                "thinking_tokens": 100,
            }
        )
        self.assertIn("400 think chars", printed)
        self.assertNotIn("0 chars", printed)

    def test_start_resets_the_baseline_for_the_next_call(self) -> None:
        self._capture({"phase": "start"})
        self._capture({"phase": "content", "stage": "analyze_workload", "chars": 500})
        self.assertEqual(self._capture({"phase": "start"}), "")
        printed = self._capture(
            {"phase": "content", "stage": "implement_hw", "chars": 50}
        )
        self.assertIn("implement HW", printed)


if __name__ == "__main__":
    unittest.main()
