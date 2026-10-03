"""Gate honesty, state-merge semantics, and iteration behavior."""

from __future__ import annotations

import io
import shutil
import sys
import unittest
from unittest import mock

from agents.cfu_design_agent import graph
from agents.cfu_design_agent.runtime.llm import LlmResult
from agents.cfu_design_agent.flow import gates
from agents.cfu_design_agent.flow import instrument
from agents.cfu_design_agent.flow import ladder
from agents.cfu_design_agent.flow import optimize as flow_optimize
from agents.cfu_design_agent.runtime import checks
from agents.cfu_design_agent.runtime import commands
from agents.cfu_design_agent.runtime import llm_step
from agents.cfu_design_agent.runtime import stage_services as services
from agents.cfu_design_agent.stages import analysis
from agents.cfu_design_agent.stages import feedback
from agents.cfu_design_agent.stages import review
from agents.cfu_design_agent.stages.implement import delivery
from agents.cfu_design_agent.stages.implement import hw as impl_hw
from agents.cfu_design_agent.stages.implement import intrinsics
from agents.cfu_design_agent.stages.implement import sw as impl_sw
from agents.cfu_design_agent.stages.review import optimize as review_optimize
from agents.cfu_design_agent.stages.review import report
from agents.cfu_design_agent.stages.verify import soc as verify_soc
from agents.cfu_design_agent.stages.verify import software as verify_software
from agents.tests.common import REPO_ROOT, offline_llm
from agents.cfu_design_agent.flow import gates as gates_mod
from agents.cfu_design_agent.stages import feedback as feedback_mod
from agents.cfu_design_agent.stages.review import report as report_mod

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
        with mock.patch.object(llm_step, "_llm", lambda state: _FakeLlm(patch)):
            return impl_hw.implement_hw(state)

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
        feedback = feedback_mod.implementation_feedback({"gates": result["gates"]}, "implementation_hw")
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

        reason = feedback_mod.tool_intent_error(narration, "RuntimeError: tools rejected")
        self.assertIn("tool-intent preamble", reason)
        self.assertIn("RuntimeError: tools rejected", reason)
        self.assertEqual(feedback_mod.tool_intent_error("a finished design", ""), "")

    def test_repair_edits_compose_on_the_live_workspace(self) -> None:
        # The prompt shows, and the protocol edits, the live run-workspace copy.
        # A repair must therefore apply on top of the earlier attempt instead of
        # being reset against the tracked original (which the prompt never shows).
        self._seed()
        state = _base_state(self.run_id, iteration=0)
        with mock.patch.object(llm_step, "_llm", lambda state: _FakeLlm(GOOD_PATCH)):
            first = impl_hw.implement_hw(state)
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
        with mock.patch.object(llm_step, "_llm", lambda state: _FakeLlm(second_patch)):
            second = impl_hw.implement_hw(state2)
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
        with mock.patch.object(llm_step, "_llm", lambda state: _FakeLlm(THINKING_PATCH)):
            first = impl_hw.implement_hw(state)
        state2 = dict(state)
        state2.update(first)
        state2["iteration"] = 1
        with mock.patch.object(llm_step, "_llm", lambda state: _FakeLlm(THINKING_PATCH)):
            second = impl_hw.implement_hw(state2)
        self.assertTrue(first["llm_raw_outputs"][0].endswith("hardware_000.txt"))
        self.assertTrue(second["llm_raw_outputs"][0].endswith("hardware_001.txt"))

    def test_agent_tools_are_logged_and_audited(self) -> None:
        self._seed()
        state = _base_state(self.run_id)
        read_path = f"agents/generated/runs/{self.run_id}/workspace/design/AgentCfu.scala"
        llm = _ToolUsingLlm(GOOD_PATCH, read_path)
        with mock.patch.object(llm_step, "_llm", lambda state: llm):
            result = impl_hw.implement_hw(state)
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
        report = report_mod.render_report(state)
        self.assertIn("## Tool Calls", report)
        self.assertIn("read_file", report)

    def test_agent_content_is_not_written_without_apply_patch(self) -> None:
        # apply_patch=False (--no-apply-patch) must not let Protocols A/B/C
        # mutate the run workspace; the response has to stay a proposal even
        # when command execution is still allowed.
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
                with mock.patch.object(llm_step, "_llm", lambda state: _FakeLlm(text)):
                    result = impl_hw.implement_hw(state)
                gate = result["gates"]["implementation_hw"]
                self.assertFalse(gate["ok"], label)
                self.assertFalse(gate["applied"], label)
                self.assertEqual(result["changed_design_files"], [], label)
                self.assertTrue(
                    any("patch application is disabled" in error for error in result["errors"]),
                    result["errors"],
                )
                self.assertEqual(target.read_text(), before, label)


class ReviewDecisionTest(unittest.TestCase):
    def _gates(self, **overrides) -> dict:
        gates = {
            "workload_spec": {"ok": True},
            "analysis": {"ok": True},
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
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "complete"))

    def test_ir_require_failure_is_separate_from_validation(self) -> None:
        gates = self._gates(ir={"ok": False, "status": "unavailable", "required": True})
        state = _base_state("review-ir-required", gates=gates, ir_features="require")
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "failed"))
        self.assertFalse(gates_mod._is_complete(state))
        optional = _base_state("review-ir-optional", gates=gates, ir_features="auto")
        self.assertEqual(gates_mod._review_decision(optional), ("final_report", "complete"))

    def test_skipped_rung_reports_planned(self) -> None:
        gates = self._gates(cost={"ok": False, "skipped": True, "reason": "execution-disabled"})
        state = _base_state("review-skip", gates=gates)
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "planned"))

    def test_nothing_executed_reports_planned(self) -> None:
        gates = self._gates(
            hw_build={"ok": False, "skipped": True, "reason": "dry-run"},
            sw_build={"ok": False, "skipped": True, "reason": "dry-run"},
            sim={"ok": False, "skipped": True, "reason": "dry-run"},
            cost={"ok": False, "skipped": True, "reason": "dry-run"},
        )
        state = _base_state("review-noexec", gates=gates)
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "planned"))

    def test_executed_failure_is_failed(self) -> None:
        gates = self._gates(sim={"ok": False, "skipped": False})
        state = _base_state("review-fail", gates=gates)
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "failed"))

    def test_invalid_workload_is_blocked(self) -> None:
        gates = self._gates(workload_spec={"ok": False, "problems": ["missing"]})
        state = _base_state("review-blocked", gates=gates)
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "blocked"))

    def test_missing_rung_cannot_be_complete(self) -> None:
        gates = self._gates()
        gates.pop("sim")
        state = _base_state("review-missing", gates=gates)
        self.assertEqual(gates_mod._review_decision(state), ("final_report", "blocked"))
        self.assertFalse(gates_mod._is_complete(state))

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
    """--no-apply-patch --no-run-commands must not report failed/blocked."""

    def _run(self) -> dict:
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
        with mock.patch.object(llm_step, "_llm", lambda state: _ContractLlm()):
            return graph.run_agent(state)

    def test_run_agent_reports_planned_when_nothing_runs(self) -> None:
        final = self._run()
        self.assertEqual(final["status"], "planned")
        self.assertEqual(final.get("iteration", 0), 0)
        self.assertEqual(final["gates"]["hw_build"]["skipped"], True)

    def test_failed_rung_with_budget_repairs_instead_of_reviewing(self) -> None:
        gates = {
            "workload_spec": {"ok": True},
            "isa": {"ok": True},
            "implementation_hw": {"ok": True},
            "hw_build": {"ok": False, "skipped": False, "next": "implement"},
        }
        state = _base_state("review-repair", gates=gates, repairs_used=0, max_iters=3)
        self.assertEqual(graph._route_after_hw_build(state), "implement")

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

    def test_analysis_accepts_a_fenced_answer_with_a_preamble(self) -> None:
        # Observed 2026-09-29 (twice): the model prefixed a sentence and wrapped
        # the mapping in a ```yaml fence, and the whole analysis was discarded.
        state = _base_state(self.run_id, workload_spec={"name": "depthconv"})
        llm = _ToolUsingLlm(
            "Here is the analysis.\n\n```yaml\n" + self.ANALYSIS_YAML + "```\n",
            "benchmarks/embench-iot/src/depthconv/depthconv.c",
        )
        with mock.patch.object(llm_step, "_llm", lambda state: llm):
            result = analysis.analyze_workload(state)
        self.assertIn("hot_function: depthconv", result["workload_analysis"])
        self.assertTrue(result["gates"]["analysis"]["ok"])
        self.assertEqual(result["gates"]["analysis"]["next"], "design_contract")

    def test_an_unparseable_analysis_is_repaired_not_handed_on_as_empty(self) -> None:
        # The empty mapping used to reach every later stage as "{}".
        state = _base_state(self.run_id, workload_spec={"name": "depthconv"})
        llm = _ToolUsingLlm("I could not produce YAML this time.", "unused")
        with mock.patch.object(llm_step, "_llm", lambda state: llm):
            result = analysis.analyze_workload(state)
        gate = result["gates"]["analysis"]
        self.assertFalse(gate["ok"])
        self.assertEqual(gate["next"], "analyze_workload")
        self.assertEqual(result["stage_jobs"], "analyze_workload")
        self.assertEqual(result["answer_repairs_used"], 1)
        self.assertIn("did not deliver a valid YAML mapping", " ".join(gate["problems"]))
        # The repair feedback quotes what the model actually sent.
        self.assertIn("could not produce YAML", " ".join(gate["problems"]))
        self.assertEqual(result["workload_analysis"].strip(), "{}")
        self.assertEqual(
            graph._route_after_analysis({**state, "gates": result["gates"]}),
            "analyze_workload",
        )

    def test_an_unparseable_analysis_blocks_the_run_when_repairs_are_spent(self) -> None:
        state = _base_state(self.run_id, workload_spec={"name": "depthconv"},
                            max_answer_repairs=0)
        llm = _ToolUsingLlm("still not YAML", "unused")
        with mock.patch.object(llm_step, "_llm", lambda state: llm):
            result = analysis.analyze_workload(state)
        gate = result["gates"]["analysis"]
        self.assertFalse(gate["ok"])
        self.assertEqual(gate["next"], "review_and_route")
        self.assertEqual(result["stopped_at"], "analysis")
        self.assertEqual(
            graph._route_after_analysis({**state, "gates": result["gates"]}),
            "review_and_route",
        )

    def test_analysis_offers_observation_tool_for_applied_profile_id(self) -> None:
        ident = "obs_loop_517c0167_md5_3"
        state = _base_state(
            self.run_id,
            benchmark_name="md5sum",
            benchmark_project={"kind": "embench_benchmark", "benchmark": "md5sum"},
            observation_report={"applied": [ident], "points": [{
                "id": ident, "file": "md5.c", "function": "md5", "kind": "loop",
                "line": 119, "ordinal": 3,
            }]},
            embench_profiles={"profile": {ident: {"cycles": 1200, "calls": 12}}},
        )
        class LookupLlm(_ToolUsingLlm):
            def complete_with_tools(self, system, user, *, tools, tool_runner, **kwargs):
                names = [item["function"]["name"] for item in tools]
                self_tools = "inspect_observation" in names
                assert self_tools, names
                assert ident in user and "inspect_observation" in user
                answer = tool_runner("inspect_observation", {"id": ident, "view": "summary"})
                assert f"Observation {ident}" in answer, answer
                return LlmResult(self.final_text, used_llm=True, tool_calls=({
                    "round": 1, "name": "inspect_observation", "arguments": {"id": ident},
                    "result": answer, "ok": True,
                },), tool_rounds=1)
        llm = LookupLlm(self.ANALYSIS_YAML, "unused")
        with mock.patch.object(llm_step, "_llm", lambda state: llm):
            result = analysis.analyze_workload(state)
        self.assertEqual(result["tool_calls"][0]["tool"], "inspect_observation")

    def test_analysis_records_tool_calls_and_parses_the_answer(self) -> None:
        state = _base_state(
            self.run_id,
            workload_spec={"name": "depthconv", "operation": "embench_workload"},
            benchmark_name="depthconv",
        )
        llm = _ToolUsingLlm(
            self.ANALYSIS_YAML, "benchmarks/embench-iot/src/depthconv/depthconv.c"
        )
        with mock.patch.object(llm_step, "_llm", lambda state: llm):
            result = analysis.analyze_workload(state)
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

    def test_analysis_tool_loop_fails_closed_when_the_endpoint_rejects_tools(self) -> None:
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
        with mock.patch.object(llm_step, "_llm", lambda state: _NoToolsLlm(self.ANALYSIS_YAML)):
            with self.assertRaisesRegex(RuntimeError, "tools not supported"):
                analysis.analyze_workload(state)
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
        self.assertIn("REPEATED IDENTICAL ATTEMPT", feedback_mod.software_stuck_directive(identical))
        self.assertEqual(feedback_mod.software_stuck_directive({"sw_attempts": []}), "")
        self.assertEqual(
            feedback_mod.software_stuck_directive(
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
        self.assertEqual(feedback_mod.software_stuck_directive(changed), "")

    def test_software_uses_read_only_stage_tools_and_records_them(self) -> None:
        target = (
            f"agents/generated/runs/{self.run_id}/workspace/embench_depthconv/cfu/"
            "src/depthconv.c"
        )
        body = "#ifdef USE_AGENT_CFU\nint kernel(void){ return 0; }\n#endif\n"
        patch = (
            "EDIT_BEGIN\n"
            f"PATH: {target}\n"
            "SEARCH\n"
            "int kernel(void){ return 0; }\n"
            "REPLACE\n"
            "int kernel(void){ return cfu_compute(1, 2); }\n"
            "EDIT_END"
        )
        state = self._state()
        state.update({"agent_tools": True, "apply_patch": True})
        llm = _ToolUsingLlm(patch, target)
        with mock.patch.object(services, "_software_targets", return_value=[(target, body)]), \
                mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=""), \
                mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
                mock.patch.object(llm_step, "_llm", lambda s: llm):
            result = impl_sw.implement_sw(state)
        self.assertEqual(result["tool_calls"][0]["tool"], "read_file")
        self.assertEqual(result["gates"]["implementation_sw"]["tool_calls"], 1)
        log = self.run_dir / "09_implementation/sw_tool_calls.yaml"
        self.assertTrue(log.is_file())
        self.assertIn("read_file", log.read_text())

    def test_software_coding_tools_are_capability_gated_and_do_not_set_sw_build(self) -> None:
        target = (
            f"agents/generated/runs/{self.run_id}/workspace/embench_depthconv/cfu/"
            "src/depthconv.c"
        )
        path = REPO_ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int kernel(void){ return 0; }\n")

        class CodingLlm(_ToolUsingLlm):
            def complete_with_tools(self, system, user, *, tools, tool_runner, **kwargs):
                names = [item["function"]["name"] for item in tools]
                assert "edit_software" in names, names
                edit = tool_runner("edit_software", {
                    "path": target,
                    "search": "int kernel(void){ return 0; }",
                    "replace": "int kernel(void){ return cfu_compute(1, 2); }",
                })
                assert "Agent" not in edit  # only bounded diff/result text
                check = tool_runner("check_software_compile", {})
                return LlmResult(
                    "EDIT_BEGIN\nPATH: " + target + "\nSEARCH\nint kernel(void){ return 0; }\n"
                    "REPLACE\nint kernel(void){ return cfu_compute(1, 2); }\nEDIT_END",
                    used_llm=True,
                    tool_calls=(
                        {"round": 1, "name": "edit_software", "arguments": {}, "result": edit, "ok": True},
                        {"round": 2, "name": "check_software_compile", "arguments": {}, "result": check, "ok": True},
                    ),
                )

        state = self._state()
        state.update({"agent_tools": True, "apply_patch": True, "run_commands": True, "dry_run": False})
        check_result = {"returncode": 0, "full_log_path": "check.log"}
        with mock.patch.object(services, "_software_targets", return_value=[(target, path.read_text())]), \
                mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=""), \
                mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
                mock.patch.object(commands, "_isolated_software_check_commands", return_value=[["fake"]]), \
                mock.patch.object(checks, "_run_check", return_value=check_result) as check, \
                mock.patch.object(llm_step, "_llm", lambda s: CodingLlm("", target)):
            result = impl_sw.implement_sw(state)
        check.assert_called_once()
        self.assertTrue(result["gates"]["implementation_sw"]["ok"])
        self.assertNotIn("sw_build", result["gates"])
        self.assertIn("cfu_compute", path.read_text())

    def test_software_repair_requires_diagnosis_before_editing(self) -> None:
        target = (
            f"agents/generated/runs/{self.run_id}/workspace/embench_depthconv/cfu/"
            "src/depthconv.c"
        )
        path = REPO_ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int kernel(void){ return 0; }\n")

        class RepairLlm(_ToolUsingLlm):
            def complete_with_tools(self, system, user, *, tools, tool_runner, **kwargs):
                names = [item["function"]["name"] for item in tools]
                assert "submit_repair_diagnosis" in names, names
                assert "edit_software" in names, names
                locked = tool_runner("edit_software", {
                    "path": target, "search": "int kernel(void){ return 0; }", "replace": "int kernel(void){ return 1; }",
                })
                assert "diagnosis" in locked
                failure_ids = next(
                    item["function"]["parameters"]["properties"]["failure_ids"]
                    for item in tools if item["function"]["name"] == "submit_repair_diagnosis"
                )
                del failure_ids  # schema presence is the contract; use known state ID below
                diagnosis = tool_runner("submit_repair_diagnosis", {
                    "failure_ids": ["fail_sw"], "observed_errors": ["compile error"],
                    "root_cause": "scalar placeholder", "repair_mapping": ["call CFU"],
                    "regression_risks": ["result semantics"],
                })
                assert "accepted" in diagnosis
                edit = tool_runner("edit_software", {
                    "path": target, "search": "int kernel(void){ return 0; }", "replace": "int kernel(void){ return cfu_compute(1, 2); }",
                })
                return LlmResult(
                    "EDIT_BEGIN\nPATH: " + target + "\nSEARCH\nreturn 0;\nREPLACE\n"
                    "return cfu_compute(1, 2);\nEDIT_END",
                    used_llm=True,
                    tool_calls=({"round": 1, "name": "edit_software", "arguments": {}, "result": edit, "ok": True},),
                )
        state = self._state()
        state.update({
            "agent_tools": True, "apply_patch": True, "run_commands": True, "dry_run": False,
            "validation_results": [{
                "rung": "sw_build", "returncode": 1, "stderr_tail": "error",
                "failure_observation": {
                    "schema_version": "1", "id": "fail_sw", "rung": "sw_build",
                    "category": "compile", "command": "make", "cwd": str(REPO_ROOT),
                    "returncode": 1, "timed_out": False, "summary": "compile error",
                    "source_locations": [], "signature": "failure-v1:" + "0" * 64,
                    "log_path": "",
                },
            }],
        })
        with mock.patch.object(services, "_software_targets", return_value=[(target, path.read_text())]), \
                mock.patch.object(intrinsics, "_write_intrinsics_header", return_value=""), \
                mock.patch.object(intrinsics, "_software_exercises_cfu", return_value=(True, "")), \
                mock.patch.object(commands, "_isolated_software_check_commands", return_value=[["fake"]]), \
                mock.patch.object(llm_step, "_llm", lambda s: RepairLlm("", target)):
            result = impl_sw.implement_sw(state)
        self.assertTrue(result["gates"]["implementation_sw"]["ok"], result["gates"]["implementation_sw"])
        self.assertIsInstance(result["gates"]["implementation_sw"]["repair_diagnosis"], dict)

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
        with mock.patch.object(llm_step, "_llm", lambda s: llm):
            first = impl_sw.implement_sw(state)
        self.assertEqual(len(first["sw_attempts"]), 1)
        gate = first["gates"]["implementation_sw"]
        self.assertTrue(gate["response_hash"])
        self.assertTrue(gate["error_hash"])
        self.assertNotIn("REPEATED IDENTICAL ATTEMPT", llm.prompts[0])

        state2 = {**state, **first}
        with mock.patch.object(llm_step, "_llm", lambda s: llm):
            second = impl_sw.implement_sw(state2)
        # Only one prior attempt exists, so the second try is not yet flagged.
        self.assertNotIn("REPEATED IDENTICAL ATTEMPT", llm.prompts[1])

        state3 = {
            **state2,
            **second,
            "sw_attempts": [*first["sw_attempts"], *second["sw_attempts"]],
        }
        with mock.patch.object(llm_step, "_llm", lambda s: llm):
            impl_sw.implement_sw(state3)
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
        text = feedback_mod.sim_attribution(results, log)
        self.assertIn("verify failed for depthconv", text)
        self.assertIn("verify_benchmark returned 0", text)
        self.assertIn("wrong-result", text)

    def test_fail_symbol_without_the_diagnostic_is_still_attributed(self) -> None:
        log = "SimFailure: Software reached the fail symbol :("
        results = [{"returncode": 1, "stdout_tail": log, "stderr_tail": ""}]
        self.assertIn("fail symbol", feedback_mod.sim_attribution(results, log))

    def test_timeout_is_attributed_to_a_hang(self) -> None:
        results = [{"returncode": -9, "timed_out": True, "stdout_tail": "", "stderr_tail": ""}]
        self.assertIn("timed out", feedback_mod.sim_attribution(results, ""))

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
        self.assertEqual(feedback_mod.sim_attribution(results, log), "")

    def test_sim_feedback_puts_the_attribution_with_the_log(self) -> None:
        state = {
            "gates": {"sim": {"attribution": "ATTRIBUTION_TEXT"}},
            "validation_results": [],
        }
        text = feedback_mod.sim_feedback(state)
        self.assertIn("ATTRIBUTION_TEXT", text)
        self.assertIn("sim (simulation) errors", text)

    def test_sim_feedback_carries_the_agent_routing_diagnosis(self) -> None:
        state = {
            "gates": {
                "sim": {
                    "attribution": "ATTRIBUTION_TEXT",
                    "repair_decision": {
                        "target": "implement",
                        "root_cause": "the CFU never returns a response",
                        "source": "llm",
                    },
                }
            },
            "validation_results": [],
        }
        text = feedback_mod.sim_feedback(state)
        self.assertIn("assigned this failure to implement", text)
        self.assertIn("never returns a response", text)
        # A deterministic decision is bookkeeping, not a diagnosis to inject.
        fallback = {
            "gates": {
                "sim": {
                    "repair_decision": {"target": "implement", "source": "fallback", "reason": "x"}
                }
            },
            "validation_results": [],
        }
        self.assertNotIn("assigned this failure", feedback_mod.sim_feedback(fallback))


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
            "analysis": {"ok": True},
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
        self.assertTrue(flow_optimize._round_improved({"speedup_x100": 250, "cost": {"cells": 1200}}, base))
        self.assertTrue(flow_optimize._round_improved({"speedup_x100": 200, "cost": {"cells": 900}}, base))
        # Slower, or faster but bigger: neither is an improvement.
        self.assertFalse(flow_optimize._round_improved({"speedup_x100": 150, "cost": {"cells": 1200}}, base))
        self.assertFalse(flow_optimize._round_improved({"speedup_x100": 250, "cost": {"cells": 1500}}, base))
        self.assertFalse(flow_optimize._round_improved({"speedup_x100": 200, "cost": {"cells": 1200}}, base))
        self.assertTrue(flow_optimize._round_improved({"speedup_x100": 200, "cost": {"cells": 900}}, None))

    def _choose(self, state: dict, text: str) -> tuple[str, dict, str]:
        with mock.patch.object(llm_step, "_llm", lambda s: _UsedLlm(text)):
            return review_optimize._choose_optimization(state, state.get("optimization_rounds") or [])

    def test_the_agent_may_start_another_round(self) -> None:
        target, decision, raw = self._choose(
            self._state(),
            '{"continue": true, "target": "implement", "focus": "cost", '
            '"reason": "replace the vector RF with a streamed datapath"}',
        )
        self.assertEqual(target, "implement")
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
            self._state(), '{"continue": true, "target": "implement", "reason": ""}'
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

        with mock.patch.object(llm_step, "_complete_llm", _boom):
            target, decision, raw = review_optimize._choose_optimization(self._state(), [])
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "fallback")
        self.assertIn("gateway exploded", decision["reason"])
        self.assertEqual(raw, "")

    def test_the_loop_is_off_without_the_flag(self) -> None:
        def _boom(_state):  # pragma: no cover - --loop is off
            raise AssertionError("the optimizer must not be consulted without --loop")

        with mock.patch.object(llm_step, "_llm", _boom):
            target, decision, _ = review_optimize._choose_optimization(
                self._state(loop_optimize=False), []
            )
        self.assertEqual((target, decision), ("", {}))

    def test_no_budget_means_no_round(self) -> None:
        target, decision, _ = self._choose(self._state(max_iters=1), "{}")
        self.assertEqual(target, "")
        self.assertEqual(decision["source"], "budget")

    def test_review_records_the_baseline_round_even_without_the_loop(self) -> None:
        update = review.review_and_route(self._state(loop_optimize=False))
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
        with mock.patch.object(llm_step, "_llm", lambda s: _UsedLlm(
            '{"continue": true, "target": "implement", "focus": "cost", '
            '"reason": "block the loop to cut scalar traffic"}'
        )):
            update = review.review_and_route(state)
        self.assertEqual(update["gates"]["review"]["next"], "implement")
        self.assertEqual(update["gates"]["review"]["decision"], "optimize:implement")
        self.assertEqual(update["status"], "complete")
        self.assertEqual(update["repairs_used"], 1)
        self.assertEqual(update["iteration"], 1)
        self.assertEqual(update["attempts"]["optimize"], 1)
        self.assertEqual(update["stage_jobs"], "implement")
        plan = update["optimization_plan"]
        self.assertEqual((plan["round"], plan["target"], plan["focus"]), (1, "implement", "cost"))
        self.assertIn("Started optimization round 1", "\n".join(update["iteration_history"]))
        self.assertEqual(len(update["optimization_rounds"]), 1)
        self.assertTrue((self.run_dir / "12_review/optimization_rounds.yaml").is_file())
        self.assertTrue((self.run_dir / "12_review/optimization_decision.yaml").is_file())
        raw = update["llm_raw_outputs"]
        self.assertEqual(len(raw), 1)
        self.assertTrue((REPO_ROOT / raw[0]).is_file())

    def test_review_stops_at_the_budget_and_records_why(self) -> None:
        update = review.review_and_route(self._state(max_iters=1))
        self.assertEqual(update["gates"]["review"]["next"], "final_report")
        self.assertEqual(update["gates"]["review"]["optimization"]["source"], "budget")
        self.assertEqual(update["optimization_plan"], {})
        self.assertNotIn("repairs_used", update)

    def test_a_failed_run_never_starts_an_optimization_round(self) -> None:
        def _boom(_state):  # pragma: no cover - only a complete run may loop
            raise AssertionError("an incomplete run must not be optimized")

        state = self._state(gates=self._gates(sim={"ok": False, "skipped": False}))
        with mock.patch.object(llm_step, "_llm", _boom):
            update = review.review_and_route(state)
        self.assertEqual(update["status"], "failed")
        self.assertEqual(update["gates"]["review"]["next"], "final_report")

    def test_the_route_reads_the_review_gate(self) -> None:
        self.assertEqual(
            graph._route_after_review({"gates": {"review": {"next": "implement"}}}),
            "implement",
        )
        self.assertEqual(graph._route_after_review({"gates": {}}), "final_report")

    def test_the_directive_tells_the_rung_what_to_optimize(self) -> None:
        state = self._state(
            optimization_plan={
                "round": 2,
                "target": "implement",
                "focus": "cost",
                "reason": "shrink the RF",
            }
        )
        text = feedback_mod.optimization_directive(state)
        self.assertIn("reduce the synthesized CFU cost", text)
        self.assertIn("shrink the RF", text)
        self.assertIn("ISA contract is frozen", text)
        self.assertIn("2.00x", text)
        self.assertEqual(feedback_mod.optimization_directive(self._state()), "")

    def test_the_optimization_prompt_is_not_a_repair_prompt(self) -> None:
        from agents.cfu_design_agent.prompts import hardware_prompt, software_prompt

        optimization = feedback_mod.optimization_directive(
            self._state(
                optimization_plan={
                    "round": 1,
                    "target": "implement",
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
            {"round": 1, "target": "implement", "improved": True, "measured": True,
             "speedup_x100": 250, "cost": {"cells": 1300}, "reason": "widen the datapath"},
            {"round": 2, "target": "implement", "improved": False, "measured": True,
             "speedup_x100": 210, "cost": {"cells": 1300}, "reason": "block the loop"},
        ]
        report = report_mod.render_report(self._state(optimization_rounds=rounds))
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
            {"round": 1, "target": "implement", "improved": False, "measured": False,
             "speedup_x100": 0, "cost": {}, "reason": "the SoC build failed"},
        ]
        report = report_mod.render_report(self._state(optimization_rounds=rounds))
        # The failed round must not repeat the baseline numbers as its own.
        self.assertIn("round 1: implement not measured", report)
        self.assertNotIn("speedup=2.00x cells=1200 improved=False", report)
        self.assertFalse(flow_optimize._round_improved(rounds[1], rounds[0]))

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
        self.assertNotIn("## Optimization Rounds", report_mod.render_report(self._state()))


class RepairBudgetSplitTest(unittest.TestCase):
    """Answer-only repairs draw from their own pool (2026-09-28 audit fix).

    A malformed conformance oracle or ISA contract must never exhaust the
    design-repair budget the run needs for real hardware/software fixes.
    """

    def _transition(self, state, rung, target):
        return ladder._ladder_transition(
            state, rung, ok=False, default="implement",
            repair_target="implement" if target == "check_conformance" else target,
            repair_kind="answer" if target == "check_conformance" else None,
        )

    def test_answer_repair_spends_only_the_answer_pool(self) -> None:
        state = _base_state("budget-split-00000000", max_iters=2, repairs_used=1)
        update, target = self._transition(state, "hw_conformance", "check_conformance")
        self.assertEqual(target, "implement")
        self.assertEqual(update.get("answer_repairs_used"), 1)
        self.assertNotIn("repairs_used", update)

    def test_answer_repair_works_with_an_exhausted_design_pool(self) -> None:
        state = _base_state("budget-split-00000001", max_iters=1, repairs_used=0)
        update, target = self._transition(state, "hw_conformance", "check_conformance")
        self.assertEqual(target, "implement")

    def test_answer_pool_exhaustion_stops_with_a_named_budget(self) -> None:
        state = _base_state(
            "budget-split-00000002", max_iters=5, repairs_used=0,
            max_answer_repairs=1, answer_repairs_used=1,
        )
        update, target = self._transition(state, "hw_conformance", "check_conformance")
        self.assertEqual(target, "review_and_route")
        self.assertEqual(update.get("stopped_at"), "hw_conformance")
        self.assertIn("answer-repair budget", update["errors"][0])

    def test_design_repair_still_spends_only_the_design_pool(self) -> None:
        # max_iters=3 gives the design pool 2 rounds; one is already spent.
        state = _base_state("budget-split-00000003", max_iters=3, repairs_used=1)
        update, target = self._transition(state, "sim", "implement")
        self.assertEqual(target, "implement")
        self.assertEqual(update.get("repairs_used"), 2)
        self.assertNotIn("answer_repairs_used", update)


class RepairDiagnosisEnforcementTest(unittest.TestCase):
    """A repair answer without REPAIR_DIAGNOSIS is rejected, not just logged."""

    TEXT = (
        "EDIT_BEGIN a.scala\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\nEDIT_END"
    )

    def test_repair_without_diagnosis_is_rejected(self) -> None:
        outcome = delivery._apply_agent_output(
            {}, self.TEXT, toolbox=object(), require_diagnosis=True
        )
        self.assertFalse(outcome["applied"])
        self.assertFalse(outcome["gate_ok"])
        self.assertIn("REPAIR_DIAGNOSIS", outcome["errors"][0])

    def test_repair_with_diagnosis_passes_the_contract(self) -> None:
        text = "REPAIR_DIAGNOSIS\nObserved errors: x\n" + self.TEXT
        outcome = delivery._apply_agent_output(
            {"dry_run": True}, text, toolbox=object(), require_diagnosis=True
        )
        self.assertTrue(outcome["gate_ok"])
        self.assertTrue(outcome["proposed"])

    def test_first_attempt_needs_no_diagnosis(self) -> None:
        outcome = delivery._apply_agent_output(
            {"dry_run": True}, self.TEXT, toolbox=object(), require_diagnosis=False
        )
        self.assertTrue(outcome["gate_ok"])


class AttemptMemoryTest(unittest.TestCase):
    """Repair prompts must see what was already tried; repeats must be caught."""

    A = {"response_hash": "aaa", "error_hash": "e1", "applied": False, "error_text": "SEARCH not found"}
    B = {"response_hash": "bbb", "error_hash": "e2", "applied": False, "error_text": "no deliverable"}

    def test_a_b_a_oscillation_is_detected(self) -> None:
        directive = feedback_mod._repeated_attempt_directive([self.A, self.B, dict(self.A)])
        self.assertIn("REPEATED IDENTICAL ATTEMPT", directive)

    def test_distinct_attempts_produce_no_directive(self) -> None:
        self.assertEqual(feedback_mod._repeated_attempt_directive([self.A, self.B]), "")

    def test_ledger_lists_every_try_with_its_rejection(self) -> None:
        ledger = feedback_mod.attempt_ledger_block([self.A, self.B])
        self.assertIn("aaa", ledger)
        self.assertIn("SEARCH not found", ledger)
        self.assertIn("rejected", ledger)
        self.assertEqual(feedback_mod.attempt_ledger_block([]), "")

    def test_prior_attempts_alone_never_make_a_repair_prompt(self) -> None:
        # Regression: the attempt ledger must be repair-context only. Appending
        # it to a clean feedback string turned every re-run (e.g. an --loop
        # optimization round) into "REPAIR MODE" demanding a REPAIR_DIAGNOSIS.
        state = {"hw_attempts": [{"response_hash": "aaa", "error_hash": "e", "applied": True}]}
        self.assertEqual(feedback_mod.hardware_repair_feedback(state), "")
        self.assertEqual(feedback_mod._repeated_attempt_directive(state["hw_attempts"]), "")


class _FakeTty(io.StringIO):
    def isatty(self) -> bool:  # noqa: D102
        return True


class StreamLogTest(unittest.TestCase):
    """Live streaming progress is one in-place line, not a flood of lines."""

    def setUp(self) -> None:
        graph.enable_progress(["analysis", "implementation"])
        instrument._LIVE_LINE.clear()
        self.addCleanup(graph.disable_progress)

    def _capture(self, payload: dict, *, tty: bool = True) -> str:
        stream = _FakeTty() if tty else io.StringIO()
        with mock.patch.object(sys, "stdout", stream):
            instrument._log_stream_progress(payload)
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
            instrument._LIVE_LINE.clear()
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
            {"phase": "content", "stage": "implement", "chars": 50}
        )
        self.assertIn("implement HW", printed)


class AgentCfuCostParityTest(unittest.TestCase):
    """The cost rung must elaborate the SAME AgentCfu configuration as SoC/sim.

    Regression (2026-09-28): estimate_cost passed the WORKLOAD spec into the
    cost flag builder, which emitted no --agent-cfu-* flags at all; GenAgentCfu
    fell back to vlen=256 and a design that had passed generate_soc and
    simulation died on its own designRequire(vlen == 512).
    """

    ISA = {
        "datapath": {"vlen": 512, "xlen": 32, "regDepth": 2},
        "memory": {"load": True, "store": False},
    }

    def _state(self, **overrides) -> dict:
        state = _base_state(
            "cost-parity-00000000",
            isa_spec=self.ISA,
            context={"design_workspace": "agents/generated/runs/cost-parity-00000000/workspace/design"},
        )
        state.update(overrides)
        return state

    def test_cost_carries_the_same_agent_cfu_flags_as_the_soc_path(self) -> None:
        cost = " ".join(commands.agent_cfu_cost_commands(self._state())[0])
        for flag in commands.agent_cfu_datapath_flags(self.ISA):
            self.assertIn(flag, cost, f"cost command is missing {flag}")

    def test_cost_flags_come_from_the_isa_spec_not_a_workload_spec(self) -> None:
        # The exact bug: a workload spec (no datapath section) reached the flag
        # builder. The helper now always derives from state["isa_spec"].
        workload = {"benchmark_name": "md5sum", "operation": "md5"}
        cost = " ".join(commands.agent_cfu_cost_commands(self._state(workload_spec=workload))[0])
        self.assertIn("--agent-cfu-vlen 512", cost)
        self.assertIn("--agent-cfu-no-store", cost)

    def test_memory_options_are_forwarded(self) -> None:
        flags = commands.agent_cfu_datapath_flags(
            {"datapath": {"vlen": 128, "xlen": 32, "regDepth": 2},
             "memory": {"load": False, "store": False}}
        )
        self.assertIn("--agent-cfu-no-load", flags)
        self.assertIn("--agent-cfu-no-store", flags)
        self.assertIn("--agent-cfu-no-tilelink", flags)
        # A contract with load but no store must not lose the bus.
        flags = commands.agent_cfu_datapath_flags(
            {"datapath": {"vlen": 128, "xlen": 32, "regDepth": 2},
             "memory": {"load": True, "store": False}}
        )
        self.assertIn("--agent-cfu-no-store", flags)
        self.assertNotIn("--agent-cfu-no-tilelink", flags)

    def test_soc_and_cost_stay_in_lockstep_for_a_plain_contract(self) -> None:
        isa = {"datapath": {"vlen": 256, "xlen": 32, "regDepth": 2}}
        cost = " ".join(commands.agent_cfu_cost_commands(self._state(isa_spec=isa))[0])
        gen = " ".join(commands.agent_cfu_gen_flags(isa))
        shared = [t for t in gen.split() if t.startswith("--agent-cfu-") or t.isdigit()]
        for token in shared:
            self.assertIn(token, cost, f"cost/SoC flag divergence on {token}")


class ConformanceFailureAttributionTest(unittest.TestCase):
    """A conformance crash must be reported for what it is, and routed honestly."""

    def _sim_result(self, category: str) -> dict:
        return {"returncode": 1, "failure_observation": {"category": category}}

    def test_sim_error_wording_matches_the_evidence(self) -> None:
        harness = feedback_mod.conformance_sim_error(
            self._sim_result("sim_harness_crash"),
            "java.lang.ArrayIndexOutOfBoundsException: Index -1 out of bounds",
        )
        self.assertIn("simulation harness", harness)
        self.assertNotIn("reached the fail symbol", harness)

        assertion = feedback_mod.conformance_sim_error(
            self._sim_result("assertion"), "FAILURE config: defer requires a tick handler"
        )
        self.assertIn("HDL assertion failure", assertion)
        self.assertNotIn("reached the fail symbol", assertion)

        fail_symbol = feedback_mod.conformance_sim_error(
            self._sim_result("fail_symbol"), "FAILURE Software reached the fail symbol :("
        )
        self.assertIn("reached the fail symbol", fail_symbol)

        untyped = feedback_mod.conformance_sim_error({"returncode": 1}, "")
        self.assertIn("without reaching the pass or fail symbol", untyped)

    def test_commit_watchdog_routes_to_repair_with_progress_guidance(self) -> None:
        result = self._sim_result("liveness_timeout")
        self.assertEqual(feedback_mod.conformance_repair_target(False, [result]), "implement")
        message = feedback_mod.conformance_sim_error(result, "waiting on uop 0x6FD6")
        self.assertIn("repairable progress failure", message)
        self.assertIn("small bounded cases", message)
        self.assertIn("completion/backpressure", message)
        self.assertNotIn("this is not evidence about the CFU design", message)

    def test_harness_crash_does_not_route_a_design_repair(self) -> None:
        self.assertIsNone(feedback_mod.conformance_repair_target(
            False, [self._sim_result("sim_harness_crash")]
        ))
        self.assertIsNone(feedback_mod.conformance_repair_target(
            False, [self._sim_result("environment")]
        ))
        self.assertEqual(
            feedback_mod.conformance_repair_target(False, [self._sim_result("fail_symbol")]),
            "implement",
        )
        self.assertEqual(
            feedback_mod.conformance_repair_target(False, [self._sim_result("assertion")]),
            "implement",
        )
        # Host/harness faults are not repaired by re-authoring an oracle even
        # when the caller initially attributed the failed build to its C body.
        for category in ("sim_harness_crash", "environment", "policy"):
            with self.subTest(category=category):
                self.assertIsNone(feedback_mod.conformance_repair_target(
                    True, [self._sim_result(category)]))
        self.assertEqual(feedback_mod.conformance_repair_target(True, []), "implement")


class DeliveryLoopDetectionTest(unittest.TestCase):
    """Semantic repetition of one approach must stop before the budget does."""

    DELIVERY_A = (
        "EDIT_BEGIN a/AgentCfu.scala\n"
        "<<<<<<< SEARCH\n"
        "          val iMod16 = round(3 downto 0).asUInt\n"
        "=======\n"
        "          val iMod16 = round(3 downto 0)\n"
        ">>>>>>> REPLACE\n"
        "EDIT_END"
    )
    DELIVERY_A_REPHRASED = (
        "Here is the fix.\n" + DELIVERY_A + "\nLet me know if you need anything else."
    )
    DELIVERY_B = (
        "EDIT_BEGIN a/AgentCfu.scala\n"
        "<<<<<<< SEARCH\n"
        "          val rotated = sum.asBits.rotateLeft(roundShifts(round)).asUInt\n"
        "=======\n"
        "          val rotated = sum.asBits.rotateLeft(roundShifts(round))\n"
        ">>>>>>> REPLACE\n"
        "EDIT_END"
    )

    def test_search_key_tracks_the_targeted_line_not_the_prose(self) -> None:
        key = feedback_mod.delivery_search_key(self.DELIVERY_A)
        self.assertTrue(key)
        self.assertEqual(key, feedback_mod.delivery_search_key(self.DELIVERY_A_REPHRASED))
        self.assertNotEqual(key, feedback_mod.delivery_search_key(self.DELIVERY_B))
        self.assertEqual(feedback_mod.delivery_search_key("no delivery here"), "")

    def test_repeated_rejected_approach_is_rejected_before_application(self) -> None:
        attempts = [{"applied": False, "search_key": feedback_mod.delivery_search_key(self.DELIVERY_A)}]
        error = feedback_mod._repeated_delivery_error(attempts, self.DELIVERY_A_REPHRASED)
        self.assertIn("REPEATED APPROACH", error)
        self.assertEqual(feedback_mod._repeated_delivery_error(attempts, self.DELIVERY_B), "")
        # The same SEARCH line from an APPLIED attempt is stale, not a loop: it
        # is left to the deterministic SEARCH-not-found rejection.
        applied = [{"applied": True, "search_key": feedback_mod.delivery_search_key(self.DELIVERY_A)}]
        self.assertEqual(feedback_mod._repeated_delivery_error(applied, self.DELIVERY_A), "")

    def test_approach_directive_fires_across_rephrased_attempts(self) -> None:
        attempts = [
            {"applied": False, "search_key": feedback_mod.delivery_search_key(self.DELIVERY_A),
             "response_hash": "h1", "error_hash": "e1"},
            {"applied": False, "search_key": feedback_mod.delivery_search_key(self.DELIVERY_A_REPHRASED),
             "response_hash": "h2", "error_hash": "e2"},
        ]
        directive = feedback_mod._repeated_attempt_directive(attempts)
        self.assertIn("REPEATED APPROACH DETECTED", directive)
        attempts.append(
            {"applied": False, "search_key": feedback_mod.delivery_search_key(self.DELIVERY_B),
             "response_hash": "h3", "error_hash": "e3"}
        )
        self.assertEqual(feedback_mod._repeated_attempt_directive(attempts), "")

    def test_delivery_retry_does_not_touch_the_repair_budget(self) -> None:
        state = {"delivery_retries": 0, "iteration": 3}
        retry = ladder._delivery_retry_transition(state, "hw_build", "implement", False, True)
        self.assertIsNotNone(retry)
        update, target = retry
        self.assertEqual(target, "implement")
        self.assertEqual(update["delivery_retries"], 1)
        self.assertNotIn("repairs_used", update)
        self.assertEqual(update["iteration"], 4)
        # The free retry is capped; after it a rejection burns the normal budget.
        state["delivery_retries"] = gates_mod.DELIVERY_RETRY_MAX
        self.assertIsNone(ladder._delivery_retry_transition(state, "hw_build", "implement", False, True))
        # A delivered-but-broken design is a real repair, never a free retry.
        self.assertIsNone(ladder._delivery_retry_transition(state, "hw_build", "implement", False, False))
        self.assertIsNone(ladder._delivery_retry_transition(state, "hw_build", "implement", True, True))


if __name__ == "__main__":
    unittest.main()
